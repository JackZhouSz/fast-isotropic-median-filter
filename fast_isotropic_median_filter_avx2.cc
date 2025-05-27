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
#include <fstream>
#include <sstream>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "image.h"
#include "nth_set_bit_index.h"

namespace fast_isotropic_median_filter {
namespace {
enum SearchDirection { kSearchUpward, kSearchDownward };

// Defines the stride for stepping along the compound histogram bitmask.
constexpr int kBitmaskStep = 64;

// If we're building for 32-bit AVX2 (e.g. --config=android_x86), we need
// to manually emulate some 64-bit intrinsics.
#ifndef __x86_64__
inline int _mm_popcnt_u64(uint64_t a) {
  return _mm_popcnt_u32(a) + _mm_popcnt_u32(a >> 32);
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
#endif

inline int NthBitPosPdep(uint64_t mask, int n) {
  const uint64_t nth_bit = _pdep_u64(1ULL << (n - 1), mask);
  return _tzcnt_u64(nth_bit);
}

inline int NthBitPosPopct(uint64_t m, int n) {
  int res = 0;
  int ct;
  // clang-format off
  if ((ct = _mm_popcnt_u32(m         )) < n) { n -= ct; res += 32; m >>= 32; }
  if ((ct = _mm_popcnt_u32(m & 0xFFFF)) < n) { n -= ct; res += 16; m >>= 16; }
  if ((ct = _mm_popcnt_u32(m & 0x00FF)) < n) { n -= ct; res +=  8; m >>=  8; }
  // clang-format on
  res += nth_set_bit_index[n - 1][m & 0xFF];
  return res;
}

// Returns a mask of which indices satisfy:
//   `(x - col)^2 + (y - row)^2 <= threshold`.
// This vectorized implementation is equivalent to the following scalar loop:
//
//   uint32_t result = 0;
//   for (int ind = 0; ind < 32; ++ind) {
//     const int x = src[ind * 2 + 0] - col;
//     const int y = src[ind * 2 + 1] - row;
//     if (x * x + y * y <= thresh) {
//       result |= 1 << ind;
//     }
//   }
//   return result;
//
// TODO: Investigate using `vpmaddubsw` to do the dot-products more
// efficiently.
inline uint32_t BitmaskCircular32Avx2(const uint8_t* src, int col, int row,
                                      int thresh) {
  // Checks that src is 32-byte aligned. Note: if this is triggered, it can be
  // worked around by disabling the check and using "mm256_loadu_si256" in place
  // of "mm256_load_si256" below.
  DCHECK_EQ(reinterpret_cast<uintptr_t>(src) & 0x1F, 0);
  const __m256i v00FF = _mm256_set1_epi16(0x00FF);
  const __m256i v0 = _mm256_set1_epi16(0x0);
  const __m256i vcol = _mm256_set1_epi16(col);
  const __m256i vrow = _mm256_set1_epi16(row);
  const __m256i vthreshold = _mm256_set1_epi16(thresh);
  const __m256i* src_p = reinterpret_cast<const __m256i*>(src);
  const __m256i src0 = _mm256_load_si256(src_p);
  const __m256i src1 = _mm256_load_si256(src_p + 1);
  const __m256i src0_x = _mm256_and_si256(src0, v00FF);
  const __m256i src0_y = _mm256_srli_epi16(src0, 8);
  const __m256i src1_x = _mm256_and_si256(src1, v00FF);
  const __m256i src1_y = _mm256_srli_epi16(src1, 8);
  const __m256i src0_dx = _mm256_sub_epi16(src0_x, vcol);
  const __m256i src0_dy = _mm256_sub_epi16(src0_y, vrow);
  const __m256i src1_dx = _mm256_sub_epi16(src1_x, vcol);
  const __m256i src1_dy = _mm256_sub_epi16(src1_y, vrow);
  const __m256i src0_dx2 = _mm256_mullo_epi16(src0_dx, src0_dx);
  const __m256i src0_dy2 = _mm256_mullo_epi16(src0_dy, src0_dy);
  const __m256i src1_dx2 = _mm256_mullo_epi16(src1_dx, src1_dx);
  const __m256i src1_dy2 = _mm256_mullo_epi16(src1_dy, src1_dy);
  const __m256i src0_dr2 = _mm256_adds_epu16(src0_dx2, src0_dy2);
  const __m256i src1_dr2 = _mm256_adds_epu16(src1_dx2, src1_dy2);
  // There is no 16-bit unsigned compare, so we subtract and compare with zero.
  const __m256i src0_cmp = _mm256_subs_epu16(src0_dr2, vthreshold);
  const __m256i src1_cmp = _mm256_subs_epu16(src1_dr2, vthreshold);
  const __m256i m0 = _mm256_cmpeq_epi16(src0_cmp, v0);
  const __m256i m1 = _mm256_cmpeq_epi16(src1_cmp, v0);
  // Note: `_mm256_packs_epi16` operates in 128-bit lanes, so the result must
  // be unshuffled to put it in logical order.
  __m256i mask = _mm256_packs_epi16(m0, m1);
  mask = _mm256_permute4x64_epi64(mask, _MM_SHUFFLE(3, 1, 2, 0));
  return _mm256_movemask_epi8(mask);
}

inline uint64_t BitmaskCircular64Avx2(const uint8_t* src, int col, int row,
                                      int threshold) {
  const auto mask0 = BitmaskCircular32Avx2(src + 0 * 64, col, row, threshold);
  const auto mask1 = BitmaskCircular32Avx2(src + 1 * 64, col, row, threshold);
  return mask0 | static_cast<uint64_t>(mask1) << 32;
}

template <SearchDirection direction, bool use_pdep, typename ordinal_t>
uint16_t FindRankAvx2(const uint16_t* histc, int col, int row, int threshold,
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

    const uint64_t mask = BitmaskCircular64Avx2(
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

    // The solution is in this 64-bit block, so we need to identify it. The
    // `pdep` instruction scatters the low bits in the first argument to
    // sequential set bit positions in `mask`, so the `1` in position n ends
    // up in the position of the nth set bit of the mask. From there, counting
    // trailing zeros yields the position of that bit in the mask. On some CPU's
    // (older AMD's) where the `pdep` instruction is very slow (microcoded), a
    // hand-written popcount-based version is called instead.
    const int nth_bit_pos = use_pdep ? NthBitPosPdep(mask, rank - count_val)
                                     : NthBitPosPopct(mask, rank - count_val);
    *count_col_ptr = count_val;
    *pivot_col_ptr = pivot_val >> kOrdinalShift;
    return pivot_val + nth_bit_pos;
  }
}

// Transposes 16 rows of 16 uint8_t pixels pointed to by 'srcs + col' into 'dst
// + row'.
inline void Transpose16x16Avx2(const uint8_t** srcs, int col, int row,
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
inline void Transpose16x8Avx2(const uint16_t** srcs, int col, int row,
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
// Also flips the sign bit, to facilitate signed comparisons later.
inline void Transpose8x8Avx2(const uint16_t** srcs, int col, int row,
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
#ifdef __linux__
bool IsAMDCPU() {
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuinfo, line)) {
    if (line.find("vendor_id") == 0) {
      if (absl::StrContains(line, "AuthenticAMD")) {
        return true;
      }
    }
  }
  return false;
}

int CPUFamily() {
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  int family = 0;
  while (std::getline(cpuinfo, line)) {
    if (line.find("cpu family") == 0) {
      size_t colon_pos = line.find(':');
      if (colon_pos != std::string::npos) {
        std::stringstream ss(line.substr(colon_pos + 1));
        if (ss >> family) {
          return family;
        }
      }
    }
  }
  return 0;
}

bool IsPreZen3AMD() { return IsAMDCPU() && CPUFamily() < 25; }
#endif  // __linux__
bool UsePdep() {
#ifdef __linux__
  static bool inited = false;
  static bool use_pdep = false;
  if (inited) return use_pdep;
  use_pdep = __builtin_cpu_supports("bmi2") && !IsPreZen3AMD();
  inited = true;
  return use_pdep;
#else
  return false;  // TODO: implement this check for e.g. Windows.
#endif
}

uint16_t SearchUpDownAvx2Pdep(int col, int row, int threshold, int rank,
                              const uint16_t* histc, uint8_t* pivot_col_ptr,
                              uint16_t* count_col_ptr) {
  return (*count_col_ptr < rank)
             ? FindRankAvx2<kSearchUpward, /*use_pdep=*/true>(
                   histc, col, row, threshold, rank, count_col_ptr,
                   pivot_col_ptr)
             : FindRankAvx2<kSearchDownward, /*use_pdep=*/true>(
                   histc, col, row, threshold, rank, count_col_ptr,
                   pivot_col_ptr);
}

uint16_t SearchUpDownAvx2Pdep(int col, int row, int threshold, int rank,
                              const uint16_t* histc, uint16_t* pivot_col_ptr,
                              uint16_t* count_col_ptr) {
  return (*count_col_ptr < rank)
             ? FindRankAvx2<kSearchUpward, /*use_pdep=*/true>(
                   histc, col, row, threshold, rank, count_col_ptr,
                   pivot_col_ptr)
             : FindRankAvx2<kSearchDownward, /*use_pdep=*/true>(
                   histc, col, row, threshold, rank, count_col_ptr,
                   pivot_col_ptr);
}

uint16_t SearchUpDownAvx2Popct(int col, int row, int threshold, int rank,
                               const uint16_t* histc, uint8_t* pivot_col_ptr,
                               uint16_t* count_col_ptr) {
  return (*count_col_ptr < rank)
             ? FindRankAvx2<kSearchUpward, /*use_pdep=*/false>(
                   histc, col, row, threshold, rank, count_col_ptr,
                   pivot_col_ptr)
             : FindRankAvx2<kSearchDownward, /*use_pdep=*/false>(
                   histc, col, row, threshold, rank, count_col_ptr,
                   pivot_col_ptr);
}

uint16_t SearchUpDownAvx2Popct(int col, int row, int threshold, int rank,
                               const uint16_t* histc, uint16_t* pivot_col_ptr,
                               uint16_t* count_col_ptr) {
  return (*count_col_ptr < rank)
             ? FindRankAvx2<kSearchUpward, /*use_pdep=*/false>(
                   histc, col, row, threshold, rank, count_col_ptr,
                   pivot_col_ptr)
             : FindRankAvx2<kSearchDownward, /*use_pdep=*/false>(
                   histc, col, row, threshold, rank, count_col_ptr,
                   pivot_col_ptr);
}

void UpdateCountsAvx2(ReadViewGrayU8 ordinal_input, int radius, int row,
                      const uint8_t* inset, const uint8_t* pivot,
                      uint16_t* count) {
  const int output_width = ordinal_input.width() - 2 * radius;
  CHECK_EQ(output_width % 16, 0);
  CHECK_LE(output_width, 64);
  // The four cases ({16, 32, 48, 64}) are manually unrolled for performance.
  // TODO: See if there's a more concise way to write this.
  if (output_width == 64) {
    const __m256i v80 = _mm256_set1_epi8(0x80);
    __m256i delta0 = _mm256_setzero_si256();
    __m256i delta1 = _mm256_setzero_si256();
    const __m256i* pivot_p = reinterpret_cast<const __m256i*>(pivot);
    const __m256i pivot0 = pivot_p[0] ^ v80;
    const __m256i pivot1 = pivot_p[1] ^ v80;
    for (int col = 0; col <= radius * 2; ++col) {
      const int inset_v = inset[std::abs(radius - col)];
      const __m256i* val_t =
          reinterpret_cast<const __m256i*>(&ordinal_input(col, row + inset_v));
      const __m256i* val_b = reinterpret_cast<const __m256i*>(
          &ordinal_input(col, row + 2 * radius + 1 - inset_v));
      const __m256i vals_t0 = _mm256_loadu_si256(&val_t[0]);
      const __m256i vals_t1 = _mm256_loadu_si256(&val_t[1]);
      const __m256i vals_b0 = _mm256_loadu_si256(&val_b[0]);
      const __m256i vals_b1 = _mm256_loadu_si256(&val_b[1]);
      delta0 = _mm256_sub_epi8(delta0, _mm256_cmpgt_epi8(pivot0, vals_b0));
      delta0 = _mm256_add_epi8(delta0, _mm256_cmpgt_epi8(pivot0, vals_t0));
      delta1 = _mm256_sub_epi8(delta1, _mm256_cmpgt_epi8(pivot1, vals_b1));
      delta1 = _mm256_add_epi8(delta1, _mm256_cmpgt_epi8(pivot1, vals_t1));
    }
    // Expands deltas to 16-bit and adds to counts.
    __m256i* count_p = reinterpret_cast<__m256i*>(count);
    const __m128i delta8_0 = _mm256_extracti128_si256(delta0, 0);
    const __m128i delta8_1 = _mm256_extracti128_si256(delta0, 1);
    const __m128i delta8_2 = _mm256_extracti128_si256(delta1, 0);
    const __m128i delta8_3 = _mm256_extracti128_si256(delta1, 1);
    const __m256i delta16_0 = _mm256_cvtepi8_epi16(delta8_0);
    const __m256i delta16_1 = _mm256_cvtepi8_epi16(delta8_1);
    const __m256i delta16_2 = _mm256_cvtepi8_epi16(delta8_2);
    const __m256i delta16_3 = _mm256_cvtepi8_epi16(delta8_3);
    count_p[0] = _mm256_add_epi16(count_p[0], delta16_0);
    count_p[1] = _mm256_add_epi16(count_p[1], delta16_1);
    count_p[2] = _mm256_add_epi16(count_p[2], delta16_2);
    count_p[3] = _mm256_add_epi16(count_p[3], delta16_3);
  } else if (output_width == 48) {
    const __m256i v80_256 = _mm256_set1_epi8(0x80);
    const __m128i v80_128 = _mm_set1_epi8(0x80);
    __m256i delta01 = _mm256_setzero_si256();
    __m128i delta2 = _mm_setzero_si128();
    const __m256i* pivot_p_256 = reinterpret_cast<const __m256i*>(pivot);
    const __m128i* pivot_p_128 = reinterpret_cast<const __m128i*>(pivot);
    const __m256i pivot01 = pivot_p_256[0] ^ v80_256;
    const __m128i pivot2 = pivot_p_128[2] ^ v80_128;
    for (int col = 0; col <= radius * 2; ++col) {
      const int inset_v = inset[std::abs(radius - col)];
      const __m256i* valt_256 =
          reinterpret_cast<const __m256i*>(&ordinal_input(col, row + inset_v));
      const __m256i* valb_256 = reinterpret_cast<const __m256i*>(
          &ordinal_input(col, row + 2 * radius + 1 - inset_v));
      const __m128i* valt_128 = reinterpret_cast<const __m128i*>(valt_256);
      const __m128i* valb_128 = reinterpret_cast<const __m128i*>(valb_256);
      const __m256i vals_t01 = _mm256_loadu_si256(&valt_256[0]);
      const __m128i vals_t2 = _mm_loadu_si128(&valt_128[2]);
      const __m256i vals_b01 = _mm256_loadu_si256(&valb_256[0]);
      const __m128i vals_b2 = _mm_loadu_si128(&valb_128[2]);
      delta01 = _mm256_sub_epi8(delta01, _mm256_cmpgt_epi8(pivot01, vals_b01));
      delta01 = _mm256_add_epi8(delta01, _mm256_cmpgt_epi8(pivot01, vals_t01));
      delta2 = _mm_sub_epi8(delta2, _mm_cmpgt_epi8(pivot2, vals_b2));
      delta2 = _mm_add_epi8(delta2, _mm_cmpgt_epi8(pivot2, vals_t2));
    }
    // Expands deltas to 16-bit and adds to counts.
    __m256i* count_p = reinterpret_cast<__m256i*>(count);
    const __m128i delta0 = _mm256_extracti128_si256(delta01, 0);
    const __m128i delta1 = _mm256_extracti128_si256(delta01, 1);
    const __m256i delta16_0 = _mm256_cvtepi8_epi16(delta0);
    const __m256i delta16_1 = _mm256_cvtepi8_epi16(delta1);
    const __m256i delta16_2 = _mm256_cvtepi8_epi16(delta2);
    count_p[0] = _mm256_add_epi16(count_p[0], delta16_0);
    count_p[1] = _mm256_add_epi16(count_p[1], delta16_1);
    count_p[2] = _mm256_add_epi16(count_p[2], delta16_2);
  } else if (output_width == 32) {
    const __m256i v80 = _mm256_set1_epi8(0x80);
    __m256i delta = _mm256_setzero_si256();
    const __m256i* pivot_p = reinterpret_cast<const __m256i*>(pivot);
    const __m256i pivot = pivot_p[0] ^ v80;
    for (int col = 0; col <= radius * 2; ++col) {
      const int inset_v = inset[std::abs(radius - col)];
      const __m256i* val_t =
          reinterpret_cast<const __m256i*>(&ordinal_input(col, row + inset_v));
      const __m256i* val_b = reinterpret_cast<const __m256i*>(
          &ordinal_input(col, row + 2 * radius + 1 - inset_v));
      const __m256i vals_t = _mm256_loadu_si256(val_t);
      const __m256i vals_b = _mm256_loadu_si256(val_b);
      delta = _mm256_sub_epi8(delta, _mm256_cmpgt_epi8(pivot, vals_b));
      delta = _mm256_add_epi8(delta, _mm256_cmpgt_epi8(pivot, vals_t));
    }
    // Expands deltas to 16-bit and adds to counts.
    __m256i* count_p = reinterpret_cast<__m256i*>(count);
    const __m128i delta8_0 = _mm256_extracti128_si256(delta, 0);
    const __m128i delta8_1 = _mm256_extracti128_si256(delta, 1);
    const __m256i delta16_0 = _mm256_cvtepi8_epi16(delta8_0);
    const __m256i delta16_1 = _mm256_cvtepi8_epi16(delta8_1);
    count_p[0] = _mm256_add_epi16(count_p[0], delta16_0);
    count_p[1] = _mm256_add_epi16(count_p[1], delta16_1);
  } else {  // output_width == 16
    const __m128i v80 = _mm_set1_epi8(0x80);
    __m128i delta = _mm_setzero_si128();
    const __m128i* pivot_p = reinterpret_cast<const __m128i*>(pivot);
    const __m128i pivot = pivot_p[0] ^ v80;
    for (int col = 0; col <= radius * 2; ++col) {
      const int inset_v = inset[std::abs(radius - col)];
      const __m128i* valt =
          reinterpret_cast<const __m128i*>(&ordinal_input(col, row + inset_v));
      const __m128i* valb = reinterpret_cast<const __m128i*>(
          &ordinal_input(col, row + 2 * radius + 1 - inset_v));
      const __m128i vals_t = _mm_loadu_si128(valt);
      const __m128i vals_b = _mm_loadu_si128(valb);
      delta = _mm_sub_epi8(delta, _mm_cmpgt_epi8(pivot, vals_b));
      delta = _mm_add_epi8(delta, _mm_cmpgt_epi8(pivot, vals_t));
    }
    // Expands deltas to 16-bit and adds to counts.
    __m256i* count_p = reinterpret_cast<__m256i*>(count);
    const __m256i delta16 = _mm256_cvtepi8_epi16(delta);
    count_p[0] = _mm256_add_epi16(count_p[0], delta16);
  }
}

// TODO: Optimize to reduce memory accesses in cases where consecutive
// insets are the same.
// TODO: Is there a reasonable way to avoid requiring GrayReadView here?
void UpdateCountsAvx2(ReadViewGrayU16 ordinal_input, int radius, int row,
                      const uint8_t* inset, const uint16_t* pivot,
                      uint16_t* count) {
  const __m128i v8000 = _mm_set1_epi16(0x8000);
  const int output_width = ordinal_input.width() - 2 * radius;
  __m128i* count_p = reinterpret_cast<__m128i*>(count);
  const __m128i* pivot_p = reinterpret_cast<const __m128i*>(pivot);
  // TODO: Unroll this like the 8-bit version, if a concise way can be
  // found to do so. (16 cases.)
  for (int col = 0; col <= radius * 2; ++col) {
    const int inset_v = inset[std::abs(radius - col)];
    const __m128i* val_t =
        reinterpret_cast<const __m128i*>(&ordinal_input(col, row + inset_v));
    const __m128i* val_b = reinterpret_cast<const __m128i*>(
        &ordinal_input(col, row + 2 * radius + 1 - inset_v));
    for (int ind = 0; ind < output_width / 8; ++ind) {
      const __m128i vals_t = _mm_loadu_si128(&val_t[ind]);
      const __m128i vals_b = _mm_loadu_si128(&val_b[ind]);
      count_p[ind] = _mm_sub_epi16(
          count_p[ind], _mm_cmplt_epi16(vals_b, pivot_p[ind] ^ v8000));
      count_p[ind] = _mm_add_epi16(
          count_p[ind], _mm_cmplt_epi16(vals_t, pivot_p[ind] ^ v8000));
    }
  }
}

void TransposeFirstRowsAvx2(ReadViewGrayU8 ordinal_input, int radius,
                            const uint8_t* inset, uint8_t* pixels_left,
                            uint8_t* pixels_right) {
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
      Transpose16x16Avx2(pl, col, row, kernel_width_ceil, pixels_left);
      Transpose16x16Avx2(pr, col, row, kernel_width_ceil, pixels_right);

      // For reference, scalar equivalent to the Transpose16x16 calls above:
      //
      //     for (int dy = 0; dy < 16; ++dy) {
      //       for (int dx = 0; dx < 16; ++dx) {
      //         pixels_left[(col + dx) * kernel_width_ceil + (row + dy)] =
      //             pl[dy][col + dx];
      //         pixels_right[(col + dx) * kernel_width_ceil + (row + dy)] =
      //             pr[dy][col + dx];
      //       }
      //     }
    }
  }
}

void TransposeFirstRowsAvx2(ReadViewGrayU16 ordinal_input, int radius,
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
      Transpose16x8Avx2(pl, col, row, kernel_width_ceil, pixels_left);
      Transpose16x8Avx2(pr, col, row, kernel_width_ceil, pixels_right);
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
      Transpose8x8Avx2(pl, col, row, kernel_width_ceil, pixels_left);
      Transpose8x8Avx2(pr, col, row, kernel_width_ceil, pixels_right);
    }
  }
}

int StepHorizontalAvx2(const uint8_t* pixels_left, const uint8_t* pixels_right,
                       int pivot_val, int kernel_width_ceil) {
  CHECK_EQ(kernel_width_ceil % 16, 0);
  CHECK_LE(kernel_width_ceil, 112);
  pivot_val ^= 0x80;  // Offset pivot to signed range.
  const __m128i* pixels_left_ptr =
      reinterpret_cast<const __m128i*>(pixels_left);
  const __m128i* pixels_right_ptr =
      reinterpret_cast<const __m128i*>(pixels_right);
  // Note: SSE/AVX doesn't have uint8 compare instructions, so we use the
  // signed compare instructions and xor the pivot with 0x80 to offset it
  // to the signed range. [pixels_left and pixels_right have already been
  // offset.]
  const __m128i v_pivot = _mm_set1_epi8(pivot_val);
  __m128i delta_left = _mm_setzero_si128();
  __m128i delta_right = _mm_setzero_si128();
  for (int ind = 0; ind < kernel_width_ceil / 16; ++ind) {
    const __m128i v_left = _mm_loadu_si128(pixels_left_ptr + ind);
    const __m128i v_right = _mm_loadu_si128(pixels_right_ptr + ind);
    delta_left = _mm_sub_epi8(delta_left, _mm_cmplt_epi8(v_left, v_pivot));
    delta_right = _mm_sub_epi8(delta_right, _mm_cmplt_epi8(v_right, v_pivot));
  }

  const __m128i delta_v = _mm_sub_epi8(delta_right, delta_left);

  return HorizontalSumS8(delta_v);
}

int StepHorizontalAvx2(const uint16_t* pixels_left,
                       const uint16_t* pixels_right, int pivot_val,
                       int kernel_width_ceil) {
  const __m128i* pixels_left_ptr =
      reinterpret_cast<const __m128i*>(pixels_left);
  const __m128i* pixels_right_ptr =
      reinterpret_cast<const __m128i*>(pixels_right);
  // Note: SSE/AVX doesn't have uint16 compare instructions, so we use the
  // signed compare instructions and xor the pivot with 0x8000 to offset it
  // to the signed range. [pixels_left and pixels_right have already been
  // offset.]
  const __m128i v_pivot = _mm_set1_epi16(pivot_val ^ 0x8000);
  __m128i v_count = _mm_setzero_si128();
  for (int ind = 0; ind < kernel_width_ceil / 8; ++ind) {
    const __m128i v_left = _mm_loadu_si128(pixels_left_ptr + ind);
    const __m128i v_right = _mm_loadu_si128(pixels_right_ptr + ind);
    v_count = _mm_add_epi16(v_count, _mm_cmplt_epi16(v_left, v_pivot));
    v_count = _mm_sub_epi16(v_count, _mm_cmplt_epi16(v_right, v_pivot));
  }
  // Sums all 8 int16 elements of v_count.
  return HorizontalSumS16(v_count);
}
}  // namespace internal
}  // namespace fast_isotropic_median_filter
#endif  // __AVX2__
