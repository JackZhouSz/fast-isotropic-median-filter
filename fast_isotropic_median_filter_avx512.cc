// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef __AVX2__

#include <emmintrin.h>
#include <immintrin.h>
#include <popcntintrin.h>
#include <x86intrin.h>
#include <xmmintrin.h>

#include <cstdint>
#include <cstdlib>

#include "absl/log/check.h"
#include "fast_isotropic_median_filter.h"
#include "image.h"

namespace fast_isotropic_median_filter {
namespace {
enum SearchDirection { kSearchUpward, kSearchDownward };

// Defines the stride for stepping along the compound histogram bitmask.
constexpr int kBitmaskStep = 64;

// Convenience shortcuts for vector loads.
template <typename T>
inline __m256i _mm256_loadu_si256_cast(T* ptr) {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
}
template <typename T>
inline __m128i _mm_loadu_si128_cast(T* ptr) {
  return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
}

// Constructs a 512-bit vector from 4 128-bit vectors.
inline __m512i _mm512_set_m128i(const __m128i& v3, const __m128i& v2,
                                const __m128i& v1, const __m128i& v0) {
  __m512i result = _mm512_castsi128_si512(v0);
  result = _mm512_inserti32x4(result, v1, 1);
  result = _mm512_inserti32x4(result, v2, 2);
  result = _mm512_inserti32x4(result, v3, 3);
  return result;
}

// Deposits contiguous low bits from `a` at the corresponding bit locations
// specified by `mask`; all other bits are zeroed.
inline uint64_t _pdep_u64(uint64_t a, uint64_t mask) {
  uint32_t res_lo = _pdep_u32(a, mask);
  a >>= _mm_popcnt_u32(mask);
  uint32_t res_hi = _pdep_u32(a, mask >> 32);
  return res_lo | (static_cast<uint64_t>(res_hi) << 32);
}

inline int _tzcnt_u64(uint64_t a) {
  return (a & 0xFFFFFFFF) ? _tzcnt_u32(a) : 32 + _tzcnt_u32(a >> 32);
}

inline int NthBitPos(uint64_t mask, int n) {
  const uint64_t nth_bit = _pdep_u64(1ULL << (n - 1), mask);
  return _tzcnt_u64(nth_bit);
}

// Returns a mask of which indices satisfy:
//   `(x - col)^2 + (y - row)^2 <= threshold`.
// This vectorized implementation is equivalent to the following scalar loop:
//
//   uint32_t result = 0;
//   for (int ind = 0; ind < 32; ++ind) {
//     const int x = src[ind * 2 + 0] - col;
//     const int y = src[ind * 2 + 1] - row;
//     if (x * x + y * y <= threshold) {
//       result |= 1 << ind;
//     }
//   }
//   return result;
//
// Note: 'thresh' is guaranteed to be in [1 .. 32767], so (dx^2 + dy^2) can be
// computed in 16-bit with saturation.
inline uint64_t BitmaskCircular64Avx512(const uint8_t* src, int col, int row,
                                        int thresh) {
  // Checks that the 128-byte buffer is 64-byte aligned.
  DCHECK_EQ(reinterpret_cast<uintptr_t>(src) & 0x3F, 0);

  const __m512i v00ff = _mm512_set1_epi16(0x00FF);
  const __m512i vcol = _mm512_set1_epi16(col);
  const __m512i vrow = _mm512_set1_epi16(row);
  const __m512i vthreshold = _mm512_set1_epi16(thresh);

  // Process first 32 coordinate pairs.
  const __m512i src0 = _mm512_load_si512(reinterpret_cast<const __m512i*>(src));
  const __m512i src0_x = _mm512_and_si512(src0, v00ff);
  const __m512i src0_y = _mm512_srli_epi16(src0, 8);
  const __m512i dx0 = _mm512_sub_epi16(src0_x, vcol);
  const __m512i dy0 = _mm512_sub_epi16(src0_y, vrow);
  const __m512i dx0_2 = _mm512_mullo_epi16(dx0, dx0);
  const __m512i dy0_2 = _mm512_mullo_epi16(dy0, dy0);
  const __m512i dr0_2 = _mm512_adds_epu16(dx0_2, dy0_2);
  const __mmask32 mask0 = _mm512_cmple_epu16_mask(dr0_2, vthreshold);

  // Processes second 32 coordinate pairs.
  const __m512i src1 =
      _mm512_load_si512(reinterpret_cast<const __m512i*>(src + 64));
  const __m512i src1_x = _mm512_and_si512(src1, v00ff);
  const __m512i src1_y = _mm512_srli_epi16(src1, 8);
  const __m512i dx1 = _mm512_sub_epi16(src1_x, vcol);
  const __m512i dy1 = _mm512_sub_epi16(src1_y, vrow);
  const __m512i dx1_2 = _mm512_mullo_epi16(dx1, dx1);
  const __m512i dy1_2 = _mm512_mullo_epi16(dy1, dy1);
  const __m512i dr1_2 = _mm512_adds_epu16(dx1_2, dy1_2);
  const __mmask32 mask1 = _mm512_cmple_epu16_mask(dr1_2, vthreshold);

  // Combines the two 32-bit masks into a final 64-bit mask.
  return static_cast<uint64_t>(mask0) | (static_cast<uint64_t>(mask1) << 32);
}

template <SearchDirection direction, typename ordinal_t>
uint16_t FindRankAvx512(const uint16_t* histc, int col, int row, int threshold,
                        int rank, uint16_t* count_col_ptr,
                        ordinal_t* pivot_col_ptr) {
  constexpr int kOrdinalShift = std::is_same_v<ordinal_t, uint8_t> ? 6 : 0;
  int count_val = *count_col_ptr;
  int pivot_val = *pivot_col_ptr << kOrdinalShift;
  const uint16_t* histp = histc + pivot_val;

  while (true) {
    if (direction == kSearchDownward) {
      DCHECK_GE(pivot_val, 0);
      histp -= kBitmaskStep;
    } else {
      DCHECK_LT(pivot_val, 65536);
    }

    const uint64_t mask = BitmaskCircular64Avx512(
        reinterpret_cast<const uint8_t*>(histp), col, row, threshold);
    const int hist_sum = _mm_popcnt_u64(mask);

    // Early continue if the solution is not in this block.
    if (direction == kSearchDownward) {
      count_val -= hist_sum;
      pivot_val -= kBitmaskStep;
      if (count_val >= rank) {
        continue;
      }
    } else {
      if (count_val + hist_sum < rank) {
        count_val += hist_sum;
        pivot_val += kBitmaskStep;
        histp += kBitmaskStep;
        continue;
      }
    }

    // Finds the exact solution within this 64-bit block.
    const int nth_bit_pos = NthBitPos(mask, rank - count_val);
    *count_col_ptr = count_val;
    *pivot_col_ptr = pivot_val >> kOrdinalShift;
    return pivot_val + nth_bit_pos;
  }

  // This function should always return from within the while(true) loop.
}

template <int output_width>
void UpdateCountsAvx512Impl(ReadViewGrayU16 ordinal_input, int radius, int row,
                            const uint8_t* inset, const uint16_t* pivot,
                            uint16_t* count) {
  static_assert(output_width % 16 == 0);
  static_assert(output_width >= 16);
  static_assert(output_width <= 144);
  DCHECK_EQ(ordinal_input.width() - 2 * radius, output_width);

  const __m512i v1_512 = _mm512_set1_epi16(1);
  const __m256i v1_256 = _mm256_set1_epi16(1);
  const int row_b = row + 2 * radius + 1;

  const __m512i p0 = output_width >= 32 ? _mm512_loadu_si512(pivot + 0)
                                        : _mm512_setzero_si512();
  const __m512i p1 = output_width >= 64 ? _mm512_loadu_si512(pivot + 32)
                                        : _mm512_setzero_si512();
  const __m512i p2 = output_width >= 96 ? _mm512_loadu_si512(pivot + 64)
                                        : _mm512_setzero_si512();
  const __m512i p3 = output_width >= 128 ? _mm512_loadu_si512(pivot + 96)
                                         : _mm512_setzero_si512();
  const __m256i pt = output_width % 32 != 0
                         ? _mm256_loadu_si256_cast(pivot + output_width - 16)
                         : _mm256_setzero_si256();

  __m512i d0 = _mm512_setzero_si512();
  __m512i d1 = _mm512_setzero_si512();
  __m512i d2 = _mm512_setzero_si512();
  __m512i d3 = _mm512_setzero_si512();
  __m256i dt = _mm256_setzero_si256();

  for (int col = 0; col <= radius * 2; ++col) {
    const int inset_v = inset[std::abs(radius - col)];

    const uint16_t* val_t_ptr = &ordinal_input(col, row + inset_v);
    const uint16_t* val_b_ptr = &ordinal_input(col, row_b - inset_v);

    if constexpr (output_width >= 32) {
      const __m512i vals_b0 = _mm512_loadu_si512(val_b_ptr + 0);
      const __m512i vals_t0 = _mm512_loadu_si512(val_t_ptr + 0);
      const __mmask32 mask_b0 = _mm512_cmplt_epu16_mask(vals_b0, p0);
      const __mmask32 mask_t0 = _mm512_cmplt_epu16_mask(vals_t0, p0);
      d0 = _mm512_mask_add_epi16(d0, mask_b0, d0, v1_512);
      d0 = _mm512_mask_sub_epi16(d0, mask_t0, d0, v1_512);
    }
    if constexpr (output_width >= 64) {
      const __m512i vals_b1 = _mm512_loadu_si512(val_b_ptr + 32);
      const __m512i vals_t1 = _mm512_loadu_si512(val_t_ptr + 32);
      const __mmask32 mask_b1 = _mm512_cmplt_epu16_mask(vals_b1, p1);
      const __mmask32 mask_t1 = _mm512_cmplt_epu16_mask(vals_t1, p1);
      d1 = _mm512_mask_add_epi16(d1, mask_b1, d1, v1_512);
      d1 = _mm512_mask_sub_epi16(d1, mask_t1, d1, v1_512);
    }

    if constexpr (output_width >= 96) {
      const __m512i vals_b2 = _mm512_loadu_si512(val_b_ptr + 64);
      const __m512i vals_t2 = _mm512_loadu_si512(val_t_ptr + 64);
      const __mmask32 mask_b2 = _mm512_cmplt_epu16_mask(vals_b2, p2);
      const __mmask32 mask_t2 = _mm512_cmplt_epu16_mask(vals_t2, p2);
      d2 = _mm512_mask_add_epi16(d2, mask_b2, d2, v1_512);
      d2 = _mm512_mask_sub_epi16(d2, mask_t2, d2, v1_512);
    }

    if constexpr (output_width >= 128) {
      const __m512i vals_b3 = _mm512_loadu_si512(val_b_ptr + 96);
      const __m512i vals_t3 = _mm512_loadu_si512(val_t_ptr + 96);
      const __mmask32 mask_b3 = _mm512_cmplt_epu16_mask(vals_b3, p3);
      const __mmask32 mask_t3 = _mm512_cmplt_epu16_mask(vals_t3, p3);
      d3 = _mm512_mask_add_epi16(d3, mask_b3, d3, v1_512);
      d3 = _mm512_mask_sub_epi16(d3, mask_t3, d3, v1_512);
    }

    // Tail.
    if constexpr (output_width % 32 != 0) {
      const auto vals_bt =
          _mm256_loadu_si256_cast(val_b_ptr + output_width - 16);
      const auto vals_tt =
          _mm256_loadu_si256_cast(val_t_ptr + output_width - 16);
      const __mmask16 mask_bt = _mm256_cmplt_epu16_mask(vals_bt, pt);
      const __mmask16 mask_tt = _mm256_cmplt_epu16_mask(vals_tt, pt);
      dt = _mm256_mask_add_epi16(dt, mask_bt, dt, v1_256);
      dt = _mm256_mask_sub_epi16(dt, mask_tt, dt, v1_256);
    }
  }
  auto* count_p512 = reinterpret_cast<__m512i*>(count);
  if constexpr (output_width >= 32) {
    count_p512[0] = _mm512_add_epi16(count_p512[0], d0);
  }
  if constexpr (output_width >= 64) {
    count_p512[1] = _mm512_add_epi16(count_p512[1], d1);
  }
  if constexpr (output_width >= 96) {
    count_p512[2] = _mm512_add_epi16(count_p512[2], d2);
  }
  if constexpr (output_width >= 128) {
    count_p512[3] = _mm512_add_epi16(count_p512[3], d3);
  }
  if constexpr (output_width % 32 != 0) {
    auto* count_p256 = reinterpret_cast<__m256i*>(count + output_width - 16);
    count_p256[0] = _mm256_add_epi16(count_p256[0], dt);
  }
}

__attribute__((target("avx512vbmi"))) inline void Transpose16x16Avx512Vbmi(
    const uint8_t** srcs, int col, int row, int stride, uint8_t* dst) {
  const __m128i s0 = _mm_loadu_si128_cast(srcs[0] + col);
  const __m128i s1 = _mm_loadu_si128_cast(srcs[1] + col);
  const __m128i s2 = _mm_loadu_si128_cast(srcs[2] + col);
  const __m128i s3 = _mm_loadu_si128_cast(srcs[3] + col);
  const __m128i s4 = _mm_loadu_si128_cast(srcs[4] + col);
  const __m128i s5 = _mm_loadu_si128_cast(srcs[5] + col);
  const __m128i s6 = _mm_loadu_si128_cast(srcs[6] + col);
  const __m128i s7 = _mm_loadu_si128_cast(srcs[7] + col);
  const __m128i s8 = _mm_loadu_si128_cast(srcs[8] + col);
  const __m128i s9 = _mm_loadu_si128_cast(srcs[9] + col);
  const __m128i sA = _mm_loadu_si128_cast(srcs[10] + col);
  const __m128i sB = _mm_loadu_si128_cast(srcs[11] + col);
  const __m128i sC = _mm_loadu_si128_cast(srcs[12] + col);
  const __m128i sD = _mm_loadu_si128_cast(srcs[13] + col);
  const __m128i sE = _mm_loadu_si128_cast(srcs[14] + col);
  const __m128i sF = _mm_loadu_si128_cast(srcs[15] + col);

  const __m512i a0 = _mm512_set_m128i(s3, s2, s1, s0);
  const __m512i a1 = _mm512_set_m128i(s7, s6, s5, s4);
  const __m512i a2 = _mm512_set_m128i(sB, sA, s9, s8);
  const __m512i a3 = _mm512_set_m128i(sF, sE, sD, sC);

  // clang-format off
  alignas(64) static constexpr uint8_t kTransposeIdxB0[] = {
      0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70,
      0x01, 0x11, 0x21, 0x31, 0x41, 0x51, 0x61, 0x71,
      0x02, 0x12, 0x22, 0x32, 0x42, 0x52, 0x62, 0x72,
      0x03, 0x13, 0x23, 0x33, 0x43, 0x53, 0x63, 0x73,
      0x04, 0x14, 0x24, 0x34, 0x44, 0x54, 0x64, 0x74,
      0x05, 0x15, 0x25, 0x35, 0x45, 0x55, 0x65, 0x75,
      0x06, 0x16, 0x26, 0x36, 0x46, 0x56, 0x66, 0x76,
      0x07, 0x17, 0x27, 0x37, 0x47, 0x57, 0x67, 0x77};
  alignas(64) static constexpr uint8_t kTransposeIdxB1[] = {
      0x08, 0x18, 0x28, 0x38, 0x48, 0x58, 0x68, 0x78,
      0x09, 0x19, 0x29, 0x39, 0x49, 0x59, 0x69, 0x79,
      0x0A, 0x1A, 0x2A, 0x3A, 0x4A, 0x5A, 0x6A, 0x7A,
      0x0B, 0x1B, 0x2B, 0x3B, 0x4B, 0x5B, 0x6B, 0x7B,
      0x0C, 0x1C, 0x2C, 0x3C, 0x4C, 0x5C, 0x6C, 0x7C,
      0x0D, 0x1D, 0x2D, 0x3D, 0x4D, 0x5D, 0x6D, 0x7D,
      0x0E, 0x1E, 0x2E, 0x3E, 0x4E, 0x5E, 0x6E, 0x7E,
      0x0F, 0x1F, 0x2F, 0x3F, 0x4F, 0x5F, 0x6F, 0x7F};
  // clang-format on

  const __m512i transpose_idx_b0 = _mm512_load_si512(kTransposeIdxB0);
  const __m512i transpose_idx_b1 = _mm512_load_si512(kTransposeIdxB1);

  // Permutes to construct 8-byte segments of the result:
  // b0 = {v00, v10, v20, .., v70, v01, v11, .., v71, .., v07, v17, .., v77 }
  // b1 = {v08, v18, v28, .., v78, v09, v19, .., v79, .., v0F, v1F, .., v7F }
  // b2 = {v80, v90, vA0, .., vF0, v81, v91, .., vF1, .., v87, v97, .., vF7 }
  // b3 = {v88, v98, vA8, .., vF8, v89, v99, .., vF9, .., v8F, v9F, .., vFF }
  const __m512i b0 = _mm512_permutex2var_epi8(a0, transpose_idx_b0, a1);
  const __m512i b1 = _mm512_permutex2var_epi8(a0, transpose_idx_b1, a1);
  const __m512i b2 = _mm512_permutex2var_epi8(a2, transpose_idx_b0, a3);
  const __m512i b3 = _mm512_permutex2var_epi8(a2, transpose_idx_b1, a3);

  // clang-format off
  alignas(64) static constexpr uint8_t kTransposeIdxC0[] = {
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
      0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
      0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
      0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F,
      0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
      0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57,
      0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
      0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F};
  alignas(64) static constexpr uint8_t kTransposeIdxC1[] = {
      0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
      0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67,
      0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F,
      0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F,
      0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
      0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77,
      0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F,
      0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E, 0x7F};
  // clang-format on

  const __m512i transpose_idx_c0 = _mm512_load_si512(kTransposeIdxC0);
  const __m512i transpose_idx_c1 = _mm512_load_si512(kTransposeIdxC1);

  // Interleaves 8-byte blocks to construct 16-byte blocks of the result:
  // c0 = {v00, v10, v20, .., vF0, v01, v11, .., vF1, .., v03, v13, .., vF3 }
  // c1 = {v04, v14, v24, .., vF4, v05, v15, .., vF5, .., v07, v17, .., vF7 }
  // c2 = {v08, v18, v28, .., vF8, v09, v19, .., vF9, .., v0B, v1B, .., vFB }
  // c3 = {v0C, v1C, v2C, .., vFC, v0D, v1D, .., vFD, .., v0F, v1F, .., vFF }
  const __m512i c0 = _mm512_permutex2var_epi8(b0, transpose_idx_c0, b2);
  const __m512i c1 = _mm512_permutex2var_epi8(b0, transpose_idx_c1, b2);
  const __m512i c2 = _mm512_permutex2var_epi8(b1, transpose_idx_c0, b3);
  const __m512i c3 = _mm512_permutex2var_epi8(b1, transpose_idx_c1, b3);

  // Destination pointers.
  __m128i* dst0 = reinterpret_cast<__m128i*>(&dst[(col + 0) * stride + row]);
  __m128i* dst1 = reinterpret_cast<__m128i*>(&dst[(col + 1) * stride + row]);
  __m128i* dst2 = reinterpret_cast<__m128i*>(&dst[(col + 2) * stride + row]);
  __m128i* dst3 = reinterpret_cast<__m128i*>(&dst[(col + 3) * stride + row]);
  __m128i* dst4 = reinterpret_cast<__m128i*>(&dst[(col + 4) * stride + row]);
  __m128i* dst5 = reinterpret_cast<__m128i*>(&dst[(col + 5) * stride + row]);
  __m128i* dst6 = reinterpret_cast<__m128i*>(&dst[(col + 6) * stride + row]);
  __m128i* dst7 = reinterpret_cast<__m128i*>(&dst[(col + 7) * stride + row]);
  __m128i* dst8 = reinterpret_cast<__m128i*>(&dst[(col + 8) * stride + row]);
  __m128i* dst9 = reinterpret_cast<__m128i*>(&dst[(col + 9) * stride + row]);
  __m128i* dstA = reinterpret_cast<__m128i*>(&dst[(col + 10) * stride + row]);
  __m128i* dstB = reinterpret_cast<__m128i*>(&dst[(col + 11) * stride + row]);
  __m128i* dstC = reinterpret_cast<__m128i*>(&dst[(col + 12) * stride + row]);
  __m128i* dstD = reinterpret_cast<__m128i*>(&dst[(col + 13) * stride + row]);
  __m128i* dstE = reinterpret_cast<__m128i*>(&dst[(col + 14) * stride + row]);
  __m128i* dstF = reinterpret_cast<__m128i*>(&dst[(col + 15) * stride + row]);

  _mm_storeu_si128(dst0, _mm512_extracti32x4_epi32(c0, 0));
  _mm_storeu_si128(dst1, _mm512_extracti32x4_epi32(c0, 1));
  _mm_storeu_si128(dst2, _mm512_extracti32x4_epi32(c0, 2));
  _mm_storeu_si128(dst3, _mm512_extracti32x4_epi32(c0, 3));
  _mm_storeu_si128(dst4, _mm512_extracti32x4_epi32(c1, 0));
  _mm_storeu_si128(dst5, _mm512_extracti32x4_epi32(c1, 1));
  _mm_storeu_si128(dst6, _mm512_extracti32x4_epi32(c1, 2));
  _mm_storeu_si128(dst7, _mm512_extracti32x4_epi32(c1, 3));
  _mm_storeu_si128(dst8, _mm512_extracti32x4_epi32(c2, 0));
  _mm_storeu_si128(dst9, _mm512_extracti32x4_epi32(c2, 1));
  _mm_storeu_si128(dstA, _mm512_extracti32x4_epi32(c2, 2));
  _mm_storeu_si128(dstB, _mm512_extracti32x4_epi32(c2, 3));
  _mm_storeu_si128(dstC, _mm512_extracti32x4_epi32(c3, 0));
  _mm_storeu_si128(dstD, _mm512_extracti32x4_epi32(c3, 1));
  _mm_storeu_si128(dstE, _mm512_extracti32x4_epi32(c3, 2));
  _mm_storeu_si128(dstF, _mm512_extracti32x4_epi32(c3, 3));
}

// Transposes 16 rows of 16 uint8_t pixels pointed to by 'srcs + col' into 'dst
// + row'. Uses basic AVX-512, without VBMI instructions.
inline void Transpose16x16Avx512(const uint8_t** srcs, int col, int row,
                                 int stride, uint8_t* dst) {
  const __m128i* sp0 = reinterpret_cast<const __m128i*>(srcs[0] + col);
  const __m128i* sp1 = reinterpret_cast<const __m128i*>(srcs[1] + col);
  const __m128i* sp2 = reinterpret_cast<const __m128i*>(srcs[2] + col);
  const __m128i* sp3 = reinterpret_cast<const __m128i*>(srcs[3] + col);
  const __m128i* sp4 = reinterpret_cast<const __m128i*>(srcs[4] + col);
  const __m128i* sp5 = reinterpret_cast<const __m128i*>(srcs[5] + col);
  const __m128i* sp6 = reinterpret_cast<const __m128i*>(srcs[6] + col);
  const __m128i* sp7 = reinterpret_cast<const __m128i*>(srcs[7] + col);
  const __m128i* sp8 = reinterpret_cast<const __m128i*>(srcs[8] + col);
  const __m128i* sp9 = reinterpret_cast<const __m128i*>(srcs[9] + col);
  const __m128i* sp10 = reinterpret_cast<const __m128i*>(srcs[10] + col);
  const __m128i* sp11 = reinterpret_cast<const __m128i*>(srcs[11] + col);
  const __m128i* sp12 = reinterpret_cast<const __m128i*>(srcs[12] + col);
  const __m128i* sp13 = reinterpret_cast<const __m128i*>(srcs[13] + col);
  const __m128i* sp14 = reinterpret_cast<const __m128i*>(srcs[14] + col);
  const __m128i* sp15 = reinterpret_cast<const __m128i*>(srcs[15] + col);

  const __m128i s0 = _mm_loadu_si128(sp0);
  const __m128i s1 = _mm_loadu_si128(sp1);
  const __m128i s2 = _mm_loadu_si128(sp2);
  const __m128i s3 = _mm_loadu_si128(sp3);
  const __m128i s4 = _mm_loadu_si128(sp4);
  const __m128i s5 = _mm_loadu_si128(sp5);
  const __m128i s6 = _mm_loadu_si128(sp6);
  const __m128i s7 = _mm_loadu_si128(sp7);
  const __m128i s8 = _mm_loadu_si128(sp8);
  const __m128i s9 = _mm_loadu_si128(sp9);
  const __m128i s10 = _mm_loadu_si128(sp10);
  const __m128i s11 = _mm_loadu_si128(sp11);
  const __m128i s12 = _mm_loadu_si128(sp12);
  const __m128i s13 = _mm_loadu_si128(sp13);
  const __m128i s14 = _mm_loadu_si128(sp14);
  const __m128i s15 = _mm_loadu_si128(sp15);

  // The high 128-bit lanes now contain the right 8x16 block.
  __m256i a0 = _mm256_setr_m128i(s0, s8);
  __m256i a1 = _mm256_setr_m128i(s1, s9);
  __m256i a2 = _mm256_setr_m128i(s2, s10);
  __m256i a3 = _mm256_setr_m128i(s3, s11);
  __m256i a4 = _mm256_setr_m128i(s4, s12);
  __m256i a5 = _mm256_setr_m128i(s5, s13);
  __m256i a6 = _mm256_setr_m128i(s6, s14);
  __m256i a7 = _mm256_setr_m128i(s7, s15);

  a0 = _mm256_permute4x64_epi64(a0, _MM_SHUFFLE(3, 1, 2, 0));
  a1 = _mm256_permute4x64_epi64(a1, _MM_SHUFFLE(3, 1, 2, 0));
  a2 = _mm256_permute4x64_epi64(a2, _MM_SHUFFLE(3, 1, 2, 0));
  a3 = _mm256_permute4x64_epi64(a3, _MM_SHUFFLE(3, 1, 2, 0));
  a4 = _mm256_permute4x64_epi64(a4, _MM_SHUFFLE(3, 1, 2, 0));
  a5 = _mm256_permute4x64_epi64(a5, _MM_SHUFFLE(3, 1, 2, 0));
  a6 = _mm256_permute4x64_epi64(a6, _MM_SHUFFLE(3, 1, 2, 0));
  a7 = _mm256_permute4x64_epi64(a7, _MM_SHUFFLE(3, 1, 2, 0));

  // clang-format off
  const __m256i shuffle1 =
      _mm256_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15,
                       0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
  const __m256i shuffle2 =
      _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
                       0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
  // clang-format on

  // Now each half of [a0 .. a7] contains two interleaved 8x8 matrices.
  a0 = _mm256_shuffle_epi8(a0, shuffle1);
  a1 = _mm256_shuffle_epi8(a1, shuffle1);
  a2 = _mm256_shuffle_epi8(a2, shuffle1);
  a3 = _mm256_shuffle_epi8(a3, shuffle1);
  a4 = _mm256_shuffle_epi8(a4, shuffle1);
  a5 = _mm256_shuffle_epi8(a5, shuffle1);
  a6 = _mm256_shuffle_epi8(a6, shuffle1);
  a7 = _mm256_shuffle_epi8(a7, shuffle1);

  const __m256i b0 = _mm256_unpacklo_epi16(a0, a1);
  const __m256i b1 = _mm256_unpackhi_epi16(a0, a1);
  const __m256i b2 = _mm256_unpacklo_epi16(a2, a3);
  const __m256i b3 = _mm256_unpackhi_epi16(a2, a3);
  const __m256i b4 = _mm256_unpacklo_epi16(a4, a5);
  const __m256i b5 = _mm256_unpackhi_epi16(a4, a5);
  const __m256i b6 = _mm256_unpacklo_epi16(a6, a7);
  const __m256i b7 = _mm256_unpackhi_epi16(a6, a7);

  const __m256i c0 = _mm256_unpacklo_epi32(b0, b2);
  const __m256i c1 = _mm256_unpackhi_epi32(b0, b2);
  const __m256i c2 = _mm256_unpacklo_epi32(b1, b3);
  const __m256i c3 = _mm256_unpackhi_epi32(b1, b3);
  const __m256i c4 = _mm256_unpacklo_epi32(b4, b6);
  const __m256i c5 = _mm256_unpackhi_epi32(b4, b6);
  const __m256i c6 = _mm256_unpacklo_epi32(b5, b7);
  const __m256i c7 = _mm256_unpackhi_epi32(b5, b7);

  __m256i d0 = _mm256_unpacklo_epi64(c0, c4);
  __m256i d1 = _mm256_unpackhi_epi64(c0, c4);
  __m256i d2 = _mm256_unpacklo_epi64(c1, c5);
  __m256i d3 = _mm256_unpackhi_epi64(c1, c5);
  __m256i d4 = _mm256_unpacklo_epi64(c2, c6);
  __m256i d5 = _mm256_unpackhi_epi64(c2, c6);
  __m256i d6 = _mm256_unpacklo_epi64(c3, c7);
  __m256i d7 = _mm256_unpackhi_epi64(c3, c7);

  // Deinterleaves the 8x8 matrices.
  d0 = _mm256_shuffle_epi8(d0, shuffle2);
  d1 = _mm256_shuffle_epi8(d1, shuffle2);
  d2 = _mm256_shuffle_epi8(d2, shuffle2);
  d3 = _mm256_shuffle_epi8(d3, shuffle2);
  d4 = _mm256_shuffle_epi8(d4, shuffle2);
  d5 = _mm256_shuffle_epi8(d5, shuffle2);
  d6 = _mm256_shuffle_epi8(d6, shuffle2);
  d7 = _mm256_shuffle_epi8(d7, shuffle2);

  __m128i* dst0 = reinterpret_cast<__m128i*>(&dst[(col + 0) * stride + row]);
  __m128i* dst1 = reinterpret_cast<__m128i*>(&dst[(col + 1) * stride + row]);
  __m128i* dst2 = reinterpret_cast<__m128i*>(&dst[(col + 2) * stride + row]);
  __m128i* dst3 = reinterpret_cast<__m128i*>(&dst[(col + 3) * stride + row]);
  __m128i* dst4 = reinterpret_cast<__m128i*>(&dst[(col + 4) * stride + row]);
  __m128i* dst5 = reinterpret_cast<__m128i*>(&dst[(col + 5) * stride + row]);
  __m128i* dst6 = reinterpret_cast<__m128i*>(&dst[(col + 6) * stride + row]);
  __m128i* dst7 = reinterpret_cast<__m128i*>(&dst[(col + 7) * stride + row]);
  __m128i* dst8 = reinterpret_cast<__m128i*>(&dst[(col + 8) * stride + row]);
  __m128i* dst9 = reinterpret_cast<__m128i*>(&dst[(col + 9) * stride + row]);
  __m128i* dstA = reinterpret_cast<__m128i*>(&dst[(col + 10) * stride + row]);
  __m128i* dstB = reinterpret_cast<__m128i*>(&dst[(col + 11) * stride + row]);
  __m128i* dstC = reinterpret_cast<__m128i*>(&dst[(col + 12) * stride + row]);
  __m128i* dstD = reinterpret_cast<__m128i*>(&dst[(col + 13) * stride + row]);
  __m128i* dstE = reinterpret_cast<__m128i*>(&dst[(col + 14) * stride + row]);
  __m128i* dstF = reinterpret_cast<__m128i*>(&dst[(col + 15) * stride + row]);

  _mm_storeu_si128(dst0, _mm256_extracti128_si256(d0, 0));
  _mm_storeu_si128(dst1, _mm256_extracti128_si256(d1, 0));
  _mm_storeu_si128(dst2, _mm256_extracti128_si256(d2, 0));
  _mm_storeu_si128(dst3, _mm256_extracti128_si256(d3, 0));
  _mm_storeu_si128(dst4, _mm256_extracti128_si256(d4, 0));
  _mm_storeu_si128(dst5, _mm256_extracti128_si256(d5, 0));
  _mm_storeu_si128(dst6, _mm256_extracti128_si256(d6, 0));
  _mm_storeu_si128(dst7, _mm256_extracti128_si256(d7, 0));
  _mm_storeu_si128(dst8, _mm256_extracti128_si256(d0, 1));
  _mm_storeu_si128(dst9, _mm256_extracti128_si256(d1, 1));
  _mm_storeu_si128(dstA, _mm256_extracti128_si256(d2, 1));
  _mm_storeu_si128(dstB, _mm256_extracti128_si256(d3, 1));
  _mm_storeu_si128(dstC, _mm256_extracti128_si256(d4, 1));
  _mm_storeu_si128(dstD, _mm256_extracti128_si256(d5, 1));
  _mm_storeu_si128(dstE, _mm256_extracti128_si256(d6, 1));
  _mm_storeu_si128(dstF, _mm256_extracti128_si256(d7, 1));
}

// Transposes 16 rows of 8 uint16_t pixels pointed to by 'srcs + col' into 'dst
// + row'.
inline void Transpose16x8Avx512(const uint16_t** srcs, int col, int row,
                                int stride, uint16_t* dst) {
  const __m128i* sp0 = reinterpret_cast<const __m128i*>(srcs[0] + col);
  const __m128i* sp1 = reinterpret_cast<const __m128i*>(srcs[1] + col);
  const __m128i* sp2 = reinterpret_cast<const __m128i*>(srcs[2] + col);
  const __m128i* sp3 = reinterpret_cast<const __m128i*>(srcs[3] + col);
  const __m128i* sp4 = reinterpret_cast<const __m128i*>(srcs[4] + col);
  const __m128i* sp5 = reinterpret_cast<const __m128i*>(srcs[5] + col);
  const __m128i* sp6 = reinterpret_cast<const __m128i*>(srcs[6] + col);
  const __m128i* sp7 = reinterpret_cast<const __m128i*>(srcs[7] + col);
  const __m128i* sp8 = reinterpret_cast<const __m128i*>(srcs[8] + col);
  const __m128i* sp9 = reinterpret_cast<const __m128i*>(srcs[9] + col);
  const __m128i* sp10 = reinterpret_cast<const __m128i*>(srcs[10] + col);
  const __m128i* sp11 = reinterpret_cast<const __m128i*>(srcs[11] + col);
  const __m128i* sp12 = reinterpret_cast<const __m128i*>(srcs[12] + col);
  const __m128i* sp13 = reinterpret_cast<const __m128i*>(srcs[13] + col);
  const __m128i* sp14 = reinterpret_cast<const __m128i*>(srcs[14] + col);
  const __m128i* sp15 = reinterpret_cast<const __m128i*>(srcs[15] + col);

  const __m128i s0 = _mm_loadu_si128(sp0);
  const __m128i s1 = _mm_loadu_si128(sp1);
  const __m128i s2 = _mm_loadu_si128(sp2);
  const __m128i s3 = _mm_loadu_si128(sp3);
  const __m128i s4 = _mm_loadu_si128(sp4);
  const __m128i s5 = _mm_loadu_si128(sp5);
  const __m128i s6 = _mm_loadu_si128(sp6);
  const __m128i s7 = _mm_loadu_si128(sp7);
  const __m128i s8 = _mm_loadu_si128(sp8);
  const __m128i s9 = _mm_loadu_si128(sp9);
  const __m128i s10 = _mm_loadu_si128(sp10);
  const __m128i s11 = _mm_loadu_si128(sp11);
  const __m128i s12 = _mm_loadu_si128(sp12);
  const __m128i s13 = _mm_loadu_si128(sp13);
  const __m128i s14 = _mm_loadu_si128(sp14);
  const __m128i s15 = _mm_loadu_si128(sp15);

  const __m256i a0 = _mm256_setr_m128i(s0, s8);
  const __m256i a1 = _mm256_setr_m128i(s1, s9);
  const __m256i a2 = _mm256_setr_m128i(s2, s10);
  const __m256i a3 = _mm256_setr_m128i(s3, s11);
  const __m256i a4 = _mm256_setr_m128i(s4, s12);
  const __m256i a5 = _mm256_setr_m128i(s5, s13);
  const __m256i a6 = _mm256_setr_m128i(s6, s14);
  const __m256i a7 = _mm256_setr_m128i(s7, s15);

  const __m256i b0 = _mm256_unpacklo_epi16(a0, a1);
  const __m256i b1 = _mm256_unpackhi_epi16(a0, a1);
  const __m256i b2 = _mm256_unpacklo_epi16(a2, a3);
  const __m256i b3 = _mm256_unpackhi_epi16(a2, a3);
  const __m256i b4 = _mm256_unpacklo_epi16(a4, a5);
  const __m256i b5 = _mm256_unpackhi_epi16(a4, a5);
  const __m256i b6 = _mm256_unpacklo_epi16(a6, a7);
  const __m256i b7 = _mm256_unpackhi_epi16(a6, a7);

  const __m256i c0 = _mm256_unpacklo_epi32(b0, b2);
  const __m256i c1 = _mm256_unpackhi_epi32(b0, b2);
  const __m256i c2 = _mm256_unpacklo_epi32(b1, b3);
  const __m256i c3 = _mm256_unpackhi_epi32(b1, b3);
  const __m256i c4 = _mm256_unpacklo_epi32(b4, b6);
  const __m256i c5 = _mm256_unpackhi_epi32(b4, b6);
  const __m256i c6 = _mm256_unpacklo_epi32(b5, b7);
  const __m256i c7 = _mm256_unpackhi_epi32(b5, b7);

  const __m256i d0 = _mm256_unpacklo_epi64(c0, c4);
  const __m256i d1 = _mm256_unpackhi_epi64(c0, c4);
  const __m256i d2 = _mm256_unpacklo_epi64(c1, c5);
  const __m256i d3 = _mm256_unpackhi_epi64(c1, c5);
  const __m256i d4 = _mm256_unpacklo_epi64(c2, c6);
  const __m256i d5 = _mm256_unpackhi_epi64(c2, c6);
  const __m256i d6 = _mm256_unpacklo_epi64(c3, c7);
  const __m256i d7 = _mm256_unpackhi_epi64(c3, c7);

  __m256i* dst0 = reinterpret_cast<__m256i*>(&dst[(col + 0) * stride + row]);
  __m256i* dst1 = reinterpret_cast<__m256i*>(&dst[(col + 1) * stride + row]);
  __m256i* dst2 = reinterpret_cast<__m256i*>(&dst[(col + 2) * stride + row]);
  __m256i* dst3 = reinterpret_cast<__m256i*>(&dst[(col + 3) * stride + row]);
  __m256i* dst4 = reinterpret_cast<__m256i*>(&dst[(col + 4) * stride + row]);
  __m256i* dst5 = reinterpret_cast<__m256i*>(&dst[(col + 5) * stride + row]);
  __m256i* dst6 = reinterpret_cast<__m256i*>(&dst[(col + 6) * stride + row]);
  __m256i* dst7 = reinterpret_cast<__m256i*>(&dst[(col + 7) * stride + row]);

  _mm256_storeu_si256(dst0, d0);
  _mm256_storeu_si256(dst1, d1);
  _mm256_storeu_si256(dst2, d2);
  _mm256_storeu_si256(dst3, d3);
  _mm256_storeu_si256(dst4, d4);
  _mm256_storeu_si256(dst5, d5);
  _mm256_storeu_si256(dst6, d6);
  _mm256_storeu_si256(dst7, d7);
}

// Transposes 8 rows of 8 pixels pointed to by 'srcs + col' into 'dst + row'.
inline void Transpose8x8Avx512(const uint16_t** srcs, int col, int row,
                               int stride, uint16_t* dst) {
  const __m128i* sp0 = reinterpret_cast<const __m128i*>(srcs[0] + col);
  const __m128i* sp1 = reinterpret_cast<const __m128i*>(srcs[1] + col);
  const __m128i* sp2 = reinterpret_cast<const __m128i*>(srcs[2] + col);
  const __m128i* sp3 = reinterpret_cast<const __m128i*>(srcs[3] + col);
  const __m128i* sp4 = reinterpret_cast<const __m128i*>(srcs[4] + col);
  const __m128i* sp5 = reinterpret_cast<const __m128i*>(srcs[5] + col);
  const __m128i* sp6 = reinterpret_cast<const __m128i*>(srcs[6] + col);
  const __m128i* sp7 = reinterpret_cast<const __m128i*>(srcs[7] + col);

  const __m128i a0 = _mm_loadu_si128(sp0);
  const __m128i a1 = _mm_loadu_si128(sp1);
  const __m128i a2 = _mm_loadu_si128(sp2);
  const __m128i a3 = _mm_loadu_si128(sp3);
  const __m128i a4 = _mm_loadu_si128(sp4);
  const __m128i a5 = _mm_loadu_si128(sp5);
  const __m128i a6 = _mm_loadu_si128(sp6);
  const __m128i a7 = _mm_loadu_si128(sp7);

  const __m128i b0 = _mm_unpacklo_epi16(a0, a1);
  const __m128i b1 = _mm_unpackhi_epi16(a0, a1);
  const __m128i b2 = _mm_unpacklo_epi16(a2, a3);
  const __m128i b3 = _mm_unpackhi_epi16(a2, a3);
  const __m128i b4 = _mm_unpacklo_epi16(a4, a5);
  const __m128i b5 = _mm_unpackhi_epi16(a4, a5);
  const __m128i b6 = _mm_unpacklo_epi16(a6, a7);
  const __m128i b7 = _mm_unpackhi_epi16(a6, a7);

  const __m128i c0 = _mm_unpacklo_epi32(b0, b2);
  const __m128i c1 = _mm_unpackhi_epi32(b0, b2);
  const __m128i c2 = _mm_unpacklo_epi32(b1, b3);
  const __m128i c3 = _mm_unpackhi_epi32(b1, b3);
  const __m128i c4 = _mm_unpacklo_epi32(b4, b6);
  const __m128i c5 = _mm_unpackhi_epi32(b4, b6);
  const __m128i c6 = _mm_unpacklo_epi32(b5, b7);
  const __m128i c7 = _mm_unpackhi_epi32(b5, b7);

  const __m128i d0 = _mm_unpacklo_epi64(c0, c4);
  const __m128i d1 = _mm_unpackhi_epi64(c0, c4);
  const __m128i d2 = _mm_unpacklo_epi64(c1, c5);
  const __m128i d3 = _mm_unpackhi_epi64(c1, c5);
  const __m128i d4 = _mm_unpacklo_epi64(c2, c6);
  const __m128i d5 = _mm_unpackhi_epi64(c2, c6);
  const __m128i d6 = _mm_unpacklo_epi64(c3, c7);
  const __m128i d7 = _mm_unpackhi_epi64(c3, c7);

  __m128i* dst0 = reinterpret_cast<__m128i*>(&dst[(col + 0) * stride + row]);
  __m128i* dst1 = reinterpret_cast<__m128i*>(&dst[(col + 1) * stride + row]);
  __m128i* dst2 = reinterpret_cast<__m128i*>(&dst[(col + 2) * stride + row]);
  __m128i* dst3 = reinterpret_cast<__m128i*>(&dst[(col + 3) * stride + row]);
  __m128i* dst4 = reinterpret_cast<__m128i*>(&dst[(col + 4) * stride + row]);
  __m128i* dst5 = reinterpret_cast<__m128i*>(&dst[(col + 5) * stride + row]);
  __m128i* dst6 = reinterpret_cast<__m128i*>(&dst[(col + 6) * stride + row]);
  __m128i* dst7 = reinterpret_cast<__m128i*>(&dst[(col + 7) * stride + row]);

  _mm_storeu_si128(dst0, d0);
  _mm_storeu_si128(dst1, d1);
  _mm_storeu_si128(dst2, d2);
  _mm_storeu_si128(dst3, d3);
  _mm_storeu_si128(dst4, d4);
  _mm_storeu_si128(dst5, d5);
  _mm_storeu_si128(dst6, d6);
  _mm_storeu_si128(dst7, d7);
}

inline int8_t HorizontalSumS8(__m128i a) {
  a = _mm_add_epi8(a, _mm_srli_si128(a, 8));
  a = _mm_add_epi8(a, _mm_srli_si128(a, 4));
  a = _mm_add_epi8(a, _mm_srli_si128(a, 2));
  a = _mm_add_epi8(a, _mm_srli_si128(a, 1));
  return _mm_cvtsi128_si32(a);
}

inline int16_t HorizontalSumS16(__m128i a) {
  a = _mm_add_epi16(a, _mm_srli_si128(a, 8));
  a = _mm_add_epi16(a, _mm_srli_si128(a, 4));
  a = _mm_add_epi16(a, _mm_srli_si128(a, 2));
  return _mm_cvtsi128_si32(a);
}
}  // namespace

namespace internal {
uint16_t SearchUpDownAvx512(int col, int row, int threshold, int rank,
                            const uint16_t* histc, uint8_t* pivot_col_ptr,
                            uint16_t* count_col_ptr) {
  return (*count_col_ptr < rank)
             ? FindRankAvx512<kSearchUpward>(histc, col, row, threshold, rank,
                                             count_col_ptr, pivot_col_ptr)
             : FindRankAvx512<kSearchDownward>(histc, col, row, threshold, rank,
                                               count_col_ptr, pivot_col_ptr);
}

uint16_t SearchUpDownAvx512(int col, int row, int threshold, int rank,
                            const uint16_t* histc, uint16_t* pivot_col_ptr,
                            uint16_t* count_col_ptr) {
  return (*count_col_ptr < rank)
             ? FindRankAvx512<kSearchUpward>(histc, col, row, threshold, rank,
                                             count_col_ptr, pivot_col_ptr)
             : FindRankAvx512<kSearchDownward>(histc, col, row, threshold, rank,
                                               count_col_ptr, pivot_col_ptr);
}

void UpdateCountsAvx512(ReadViewGrayU8 ordinal_input, int radius, int row,
                        const uint8_t* inset, const uint8_t* pivot,
                        uint16_t* count) {
  const int output_width = ordinal_input.width() - 2 * radius;
  DCHECK_EQ(output_width % 16, 0);
  DCHECK_GE(output_width, 16);
  DCHECK_LE(output_width, 64);
  const __m128i v1_128 = _mm_set1_epi8(1);
  const __m256i v1_256 = _mm256_set1_epi8(1);
  const __m512i v1_512 = _mm512_set1_epi8(1);
  const int row_b = row + 2 * radius + 1;
  // The four cases ({16, 32, 48, 64}) are manually unrolled for performance.
  if (output_width == 64) {
    const __m512i* pivot_p = reinterpret_cast<const __m512i*>(pivot);
    const __m512i v_pivot = _mm512_loadu_si512(pivot_p);
    __m512i delta = _mm512_setzero_si512();
    for (int col = 0; col <= radius * 2; ++col) {
      const int inset_v = inset[std::abs(radius - col)];
      const void* val_t_ptr = &ordinal_input(col, row + inset_v);
      const void* val_b_ptr = &ordinal_input(col, row_b - inset_v);
      const __m512i vals_t = _mm512_loadu_si512(val_t_ptr);
      const __m512i vals_b = _mm512_loadu_si512(val_b_ptr);
      const __mmask64 mask_b = _mm512_cmplt_epu8_mask(vals_b, v_pivot);
      const __mmask64 mask_t = _mm512_cmplt_epu8_mask(vals_t, v_pivot);
      delta = _mm512_mask_add_epi8(delta, mask_b, delta, v1_512);
      delta = _mm512_mask_sub_epi8(delta, mask_t, delta, v1_512);
    }
    __m512i* count_p = reinterpret_cast<__m512i*>(count);
    const __m256i delta8_lo256 = _mm512_castsi512_si256(delta);
    const __m256i delta8_hi256 = _mm512_extracti64x4_epi64(delta, 1);
    const __m512i delta16_lo512 = _mm512_cvtepi8_epi16(delta8_lo256);
    const __m512i delta16_hi512 = _mm512_cvtepi8_epi16(delta8_hi256);
    count_p[0] = _mm512_add_epi16(count_p[0], delta16_lo512);
    count_p[1] = _mm512_add_epi16(count_p[1], delta16_hi512);
  } else if (output_width == 48) {  // 512-bit, masked to 384-bit.
    constexpr __mmask64 k48 = 0x0000FFFFFFFFFFFFULL;
    __m512i delta = _mm512_setzero_si512();
    const __m512i v1_512 = _mm512_set1_epi8(1);
    const __m512i v_pivot = _mm512_maskz_loadu_epi8(k48, pivot);
    for (int col = 0; col <= radius * 2; ++col) {
      const int inset_v = inset[std::abs(radius - col)];
      const void* val_t_ptr = &ordinal_input(col, row + inset_v);
      const void* val_b_ptr = &ordinal_input(col, row_b - inset_v);
      const __m512i vals_t = _mm512_maskz_loadu_epi8(k48, val_t_ptr);
      const __m512i vals_b = _mm512_maskz_loadu_epi8(k48, val_b_ptr);
      const __mmask64 mask_b = _mm512_cmplt_epu8_mask(vals_b, v_pivot);
      const __mmask64 mask_t = _mm512_cmplt_epu8_mask(vals_t, v_pivot);
      delta = _mm512_mask_add_epi8(delta, mask_b, delta, v1_512);
      delta = _mm512_mask_sub_epi8(delta, mask_t, delta, v1_512);
    }
    const __m256i delta8_lo256 = _mm512_castsi512_si256(delta);
    const __m256i delta8_hi256 = _mm512_extracti64x4_epi64(delta, 1);
    const __m128i delta8_hi128 = _mm256_extracti128_si256(delta8_hi256, 0);
    const __m512i delta16_512 = _mm512_cvtepi8_epi16(delta8_lo256);
    const __m256i delta16_256 = _mm256_cvtepi8_epi16(delta8_hi128);
    __m512i* count_p512 = reinterpret_cast<__m512i*>(count);
    __m256i* count_p256 = reinterpret_cast<__m256i*>(count + 32);
    count_p512[0] = _mm512_add_epi16(count_p512[0], delta16_512);
    count_p256[0] = _mm256_add_epi16(count_p256[0], delta16_256);
  } else if (output_width == 32) {
    __m256i delta = _mm256_setzero_si256();
    const __m256i* pivot_p = reinterpret_cast<const __m256i*>(pivot);
    const __m256i pivot = pivot_p[0];
    for (int col = 0; col <= radius * 2; ++col) {
      const int inset_v = inset[std::abs(radius - col)];
      const void* val_t_ptr = &ordinal_input(col, row + inset_v);
      const void* val_b_ptr = &ordinal_input(col, row_b - inset_v);
      const __m256i vals_t = _mm256_loadu_si256_cast(val_t_ptr);
      const __m256i vals_b = _mm256_loadu_si256_cast(val_b_ptr);
      const __mmask32 mask_b = _mm256_cmplt_epu8_mask(vals_b, pivot);
      const __mmask32 mask_t = _mm256_cmplt_epu8_mask(vals_t, pivot);
      delta = _mm256_mask_add_epi8(delta, mask_b, delta, v1_256);
      delta = _mm256_mask_sub_epi8(delta, mask_t, delta, v1_256);
    }
    __m512i* count_p = reinterpret_cast<__m512i*>(count);
    const __m512i delta16 = _mm512_cvtepi8_epi16(delta);
    count_p[0] = _mm512_add_epi16(count_p[0], delta16);
  } else {  // output_width == 16
    __m128i delta = _mm_setzero_si128();
    const __m128i* pivot_p = reinterpret_cast<const __m128i*>(pivot);
    const __m128i pivot = pivot_p[0];
    for (int col = 0; col <= radius * 2; ++col) {
      const int inset_v = inset[std::abs(radius - col)];
      const void* val_t_ptr = &ordinal_input(col, row + inset_v);
      const void* val_b_ptr = &ordinal_input(col, row_b - inset_v);
      const __m128i vals_t = _mm_loadu_si128_cast(val_t_ptr);
      const __m128i vals_b = _mm_loadu_si128_cast(val_b_ptr);
      const __mmask16 mask_b = _mm_cmplt_epu8_mask(vals_b, pivot);
      const __mmask16 mask_t = _mm_cmplt_epu8_mask(vals_t, pivot);
      delta = _mm_mask_add_epi8(delta, mask_b, delta, v1_128);
      delta = _mm_mask_sub_epi8(delta, mask_t, delta, v1_128);
    }
    __m256i* count_p = reinterpret_cast<__m256i*>(count);
    const __m256i delta16 = _mm256_cvtepi8_epi16(delta);
    count_p[0] = _mm256_add_epi16(count_p[0], delta16);
  }
}

void UpdateCountsAvx512(ReadViewGrayU16 ordinal_input, int radius, int row,
                        const uint8_t* inset, const uint16_t* pivot,
                        uint16_t* count) {
  const int output_width = ordinal_input.width() - 2 * radius;
  DCHECK_EQ(output_width % 16, 0);
  DCHECK_GE(output_width, 16);
  DCHECK_LE(output_width, 144);
  switch (output_width) {
    case 16:
      UpdateCountsAvx512Impl<16>(ordinal_input, radius, row, inset, pivot,
                                 count);
      break;
    case 32:
      UpdateCountsAvx512Impl<32>(ordinal_input, radius, row, inset, pivot,
                                 count);
      break;
    case 48:
      UpdateCountsAvx512Impl<48>(ordinal_input, radius, row, inset, pivot,
                                 count);
      break;
    case 64:
      UpdateCountsAvx512Impl<64>(ordinal_input, radius, row, inset, pivot,
                                 count);
      break;
    case 80:
      UpdateCountsAvx512Impl<80>(ordinal_input, radius, row, inset, pivot,
                                 count);
      break;
    case 96:
      UpdateCountsAvx512Impl<96>(ordinal_input, radius, row, inset, pivot,
                                 count);
      break;
    case 112:
      UpdateCountsAvx512Impl<112>(ordinal_input, radius, row, inset, pivot,
                                  count);
      break;
    case 128:
      UpdateCountsAvx512Impl<128>(ordinal_input, radius, row, inset, pivot,
                                  count);
      break;
    case 144:
    default:
      UpdateCountsAvx512Impl<144>(ordinal_input, radius, row, inset, pivot,
                                  count);
      break;
  };
}

// Equivalent to TransposeFirstRowsAvx512, but using the AVX512 BMI intrinsics.
__attribute__((target("avx512vbmi"))) void TransposeFirstRowsAvx512Vbmi(
    ReadViewGrayU8 ordinal_input, int radius, const uint8_t* inset,
    uint8_t* pixels_left, uint8_t* pixels_right) {
  const int kernel_width_ceil = (2 * radius + 16) & ~15;
  const int output_width = ordinal_input.width() - 2 * radius;
  CHECK_EQ(output_width % 16, 0);
  alignas(16) static const uint8_t zeros[256] = {0};

  // Processes pixels in 16x16 blocks.
  int row = 0;
  for (; row < kernel_width_ceil; row += 16) {
    const uint8_t* pl[16];
    const uint8_t* pr[16];
    // Sets up pointers.(Tail rows are pointed to an array of zeros.)
    for (int dy = 0; dy < 16; ++dy) {
      if (row + dy > 2 * radius) {
        pl[dy] = zeros;
        pr[dy] = zeros;
        continue;
      }
      const int dr = std::abs(row + dy - radius);
      const int inset_v = inset[dr];
      pl[dy] = &ordinal_input(inset_v, row + dy) - 1;
      pr[dy] = &ordinal_input(2 * radius - inset_v, row + dy);
    }
    // Steps through blocks and transposes into buffers.
    for (int col = 0; col < output_width; col += 16) {
      Transpose16x16Avx512Vbmi(pl, col, row, kernel_width_ceil, pixels_left);
      Transpose16x16Avx512Vbmi(pr, col, row, kernel_width_ceil, pixels_right);
    }
  }
}

void TransposeFirstRowsAvx512(ReadViewGrayU8 ordinal_input, int radius,
                              const uint8_t* inset, uint8_t* pixels_left,
                              uint8_t* pixels_right) {
  static const bool has_vbmi = UseAvx512Vbmi();
  if (has_vbmi) {
    TransposeFirstRowsAvx512Vbmi(ordinal_input, radius, inset, pixels_left,
                                 pixels_right);
    return;
  }

  const int kernel_width_ceil = (2 * radius + 16) & ~15;
  const int output_width = ordinal_input.width() - 2 * radius;
  CHECK_EQ(output_width % 16, 0);
  alignas(16) static const uint8_t zeros[256] = {0};

  // Processes pixels in 16x16 blocks.
  int row = 0;
  for (; row < kernel_width_ceil; row += 16) {
    const uint8_t* pl[16];
    const uint8_t* pr[16];
    // Sets up pointers.(Tail rows are pointed to an array of zeros.)
    for (int dy = 0; dy < 16; ++dy) {
      if (row + dy > 2 * radius) {
        pl[dy] = zeros;
        pr[dy] = zeros;
        continue;
      }
      const int dr = std::abs(row + dy - radius);
      const int inset_v = inset[dr];
      pl[dy] = &ordinal_input(inset_v, row + dy) - 1;
      pr[dy] = &ordinal_input(2 * radius - inset_v, row + dy);
    }
    // Steps through blocks and transposes into buffers.
    for (int col = 0; col < output_width; col += 16) {
      Transpose16x16Avx512(pl, col, row, kernel_width_ceil, pixels_left);
      Transpose16x16Avx512(pr, col, row, kernel_width_ceil, pixels_right);
    }
  }
}

void TransposeFirstRowsAvx512(ReadViewGrayU16 ordinal_input, int radius,
                              const uint8_t* inset, uint16_t* pixels_left,
                              uint16_t* pixels_right) {
  const int kernel_width_ceil = (2 * radius + 16) & ~15;
  const int output_width = ordinal_input.width() - 2 * radius;
  alignas(16) static const uint16_t zeros[256] = {0};

  // Processes pixels in 8x16 blocks.
  int row = 0;
  for (; row < kernel_width_ceil - 8; row += 16) {
    const uint16_t* pl[16];
    const uint16_t* pr[16];
    // Sets up pointers.(Tail rows are pointed to an array of zeros.)
    for (int dy = 0; dy < 16; ++dy) {
      if (row + dy > 2 * radius) {
        pl[dy] = zeros;
        pr[dy] = zeros;
        continue;
      }
      const int dr = std::abs(row + dy - radius);
      const int inset_v = inset[dr];
      pl[dy] = &ordinal_input(inset_v, row + dy) - 1;
      pr[dy] = &ordinal_input(2 * radius - inset_v, row + dy);
    }
    // Steps through blocks and transposes into buffers.
    for (int col = 0; col < output_width; col += 8) {
      Transpose16x8Avx512(pl, col, row, kernel_width_ceil, pixels_left);
      Transpose16x8Avx512(pr, col, row, kernel_width_ceil, pixels_right);
    }
  }

  // Processes pixels in 8x8 blocks.
  for (; row < kernel_width_ceil; row += 8) {
    const uint16_t* pl[8];
    const uint16_t* pr[8];

    // Sets up pointers.(Tail rows are pointed to an array of zeros.)
    for (int dy = 0; dy < 8; ++dy) {
      if (row + dy > 2 * radius) {
        pl[dy] = zeros;
        pr[dy] = zeros;
        continue;
      }
      const int dr = std::abs(row + dy - radius);
      const int inset_v = inset[dr];
      pl[dy] = &ordinal_input(inset_v, row + dy) - 1;
      pr[dy] = &ordinal_input(2 * radius - inset_v, row + dy);
    }

    // Steps through blocks and transposes into buffers.
    for (int col = 0; col < output_width; col += 8) {
      Transpose8x8Avx512(pl, col, row, kernel_width_ceil, pixels_left);
      Transpose8x8Avx512(pr, col, row, kernel_width_ceil, pixels_right);
    }
  }
}

int StepHorizontalAvx512(const uint8_t* pixels_left,
                         const uint8_t* pixels_right, int pivot_val,
                         int kernel_width_ceil) {
  CHECK_EQ(kernel_width_ceil % 16, 0);
  CHECK_LE(kernel_width_ceil, 112);
  const __m128i v1_128 = _mm_set1_epi8(1);
  const __m128i* pixels_left_ptr =
      reinterpret_cast<const __m128i*>(pixels_left);
  const __m128i* pixels_right_ptr =
      reinterpret_cast<const __m128i*>(pixels_right);
  const __m128i v_pivot = _mm_set1_epi8(pivot_val);
  __m128i delta_l = _mm_setzero_si128();
  __m128i delta_r = _mm_setzero_si128();
  for (int ind = 0; ind < kernel_width_ceil / 16; ++ind) {
    const __m128i v_left = _mm_loadu_si128(pixels_left_ptr + ind);
    const __m128i v_right = _mm_loadu_si128(pixels_right_ptr + ind);
    const __mmask16 mask_l = _mm_cmplt_epu8_mask(v_left, v_pivot);
    const __mmask16 mask_r = _mm_cmplt_epu8_mask(v_right, v_pivot);
    delta_l = _mm_mask_add_epi8(delta_l, mask_l, delta_l, v1_128);
    delta_r = _mm_mask_add_epi8(delta_r, mask_r, delta_r, v1_128);
  }

  const __m128i delta_v = _mm_sub_epi8(delta_r, delta_l);

  return HorizontalSumS8(delta_v);
}

int StepHorizontalAvx512(const uint16_t* pixels_left,
                         const uint16_t* pixels_right, int pivot_val,
                         int kernel_width_ceil) {
  const __m128i v1_128 = _mm_set1_epi16(1);
  const __m128i* pixels_left_ptr =
      reinterpret_cast<const __m128i*>(pixels_left);
  const __m128i* pixels_right_ptr =
      reinterpret_cast<const __m128i*>(pixels_right);
  const __m128i v_pivot = _mm_set1_epi16(pivot_val);
  __m128i v_count = _mm_setzero_si128();
  for (int ind = 0; ind < kernel_width_ceil / 8; ++ind) {
    const __m128i v_left = _mm_loadu_si128(pixels_left_ptr + ind);
    const __m128i v_right = _mm_loadu_si128(pixels_right_ptr + ind);
    const __mmask8 mask_l = _mm_cmplt_epu16_mask(v_left, v_pivot);
    const __mmask8 mask_r = _mm_cmplt_epu16_mask(v_right, v_pivot);
    v_count = _mm_mask_sub_epi16(v_count, mask_l, v_count, v1_128);
    v_count = _mm_mask_add_epi16(v_count, mask_r, v_count, v1_128);
  }
  // Sums all 8 int16 elements of v_count.
  return HorizontalSumS16(v_count);
}
}  // namespace internal
}  // namespace fast_isotropic_median_filter

#endif  // __AVX2__
