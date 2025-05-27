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

#if defined(__aarch64__)
#include <arm_neon.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "absl/log/check.h"
#include "image.h"
#include "nth_set_bit_index.h"

// Alternative transpose implementations which may be faster on some chipsets.
#define TRANSPOSE_TABLE 0
#define TRANSPOSE_ST2 0
namespace fast_isotropic_median_filter {
namespace {
enum SearchDirection { kSearchUpward, kSearchDownward };

// Defines the stride for stepping along the compound histogram bitmask.
constexpr int kBitmaskStep = 64;

// Returns a mask of which indices satisfy: 'x^2 + y^2 <= threshold'.
// Note: the 'vmlal' instructions (multiply-accumulate-long) do not saturate,
// so care must be taken not to overflow the 16-bit results. The input tile size
// must be limited to ensure that the squared distance between points compared
// here will not exceed 0xFFFF. In practice, a square input tile must not exceed
// '182 + radius' on a side.
inline uint8x16_t BytemaskCircular16Neon(const uint8_t* src,
                                         const uint8x16_t& vcol,
                                         const uint8x16_t& vrow,
                                         const uint8x16_t& vthresh) {
  const uint8x16_t bitmask = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                              0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
  const uint8x16x2_t& xy = vld2q_u8(src);
  const uint8x16_t x = vabdq_u8(xy.val[0], vcol);
  const uint8x16_t y = vabdq_u8(xy.val[1], vrow);
  const uint8x8_t x_lo = vget_low_u8(x);
  const uint8x8_t y_lo = vget_low_u8(y);
  const uint16x8_t r0 = vmlal_u8(vmull_u8(x_lo, x_lo), y_lo, y_lo);
  const uint16x8_t r1 = vmlal_high_u8(vmull_high_u8(x, x), y, y);
  const uint16x8_t m0 = vcleq_u16(r0, vthresh);
  const uint16x8_t m1 = vcleq_u16(r1, vthresh);
  return vuzp1q_u8(m0, m1) & bitmask;
}

inline uint8x8_t BitmaskCircular64Neon(const uint8_t* src, int col, int row,
                                       int threshold) {
  const uint16x8_t vthresh = vdupq_n_u16(threshold);
  const uint8x16_t vrow = vdupq_n_u8(row);
  const uint8x16_t vcol = vdupq_n_u8(col);

  const auto mask0 = BytemaskCircular16Neon(src + 0 * 32, vcol, vrow, vthresh);
  const auto mask1 = BytemaskCircular16Neon(src + 1 * 32, vcol, vrow, vthresh);
  const auto mask2 = BytemaskCircular16Neon(src + 2 * 32, vcol, vrow, vthresh);
  const auto mask3 = BytemaskCircular16Neon(src + 3 * 32, vcol, vrow, vthresh);

  const uint8x16_t mask0123 =
      vpaddq_u8(vpaddq_u8(mask0, mask1), vpaddq_u8(mask2, mask3));
  return vget_low_u8(vpaddq_u8(mask0123, mask0123));
}

template <SearchDirection direction, typename ordinal_t>
uint16_t FindRankNeon(const uint16_t* histc, int col, int row, int threshold,
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

    uint8x8_t mask = BitmaskCircular64Neon(
        reinterpret_cast<const uint8_t*>(histp), col, row, threshold);
    const uint8x8_t hist = vcnt_u8(mask);  // Popcount.
    const int hist_sum = vaddv_u8(hist);   // Total # of '1' bits: [0 .. 64].

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

    const int thresh = rank - count_val;
    uint8x8_t vthresh = vdup_n_u8(thresh);

    // Constructs prefix sum from 'hist', shifted one byte with leading zero.
    const uint8x8_t v0 = vdup_n_u8(0);
    uint8x8_t prefix_sum = vext_u8(v0, hist, 7) + vext_u8(v0, hist, 6);
    prefix_sum += vext_u8(v0, prefix_sum, 6);
    prefix_sum += vext_u8(v0, prefix_sum, 4);
    vthresh = vclt_u8(prefix_sum, vthresh);

    const int ind8 = -vaddv_s8(vthresh) - 1;  // in [0 .. 7]
    const int thresh_adj = prefix_sum[ind8];
    const int bitmask = mask[ind8];  // 8-bit bitmask to search for solution.

    if constexpr (std::is_same_v<ordinal_t, uint8_t>) {
      *count_col_ptr = count_val;
      *pivot_col_ptr = pivot_val >> kOrdinalShift;
      return (*pivot_col_ptr << kOrdinalShift) + ind8 * 8 +
             nth_set_bit_index[thresh - thresh_adj - 1][bitmask];
    } else {
      // Updates the pivot and count to the nearest 8-element-aligned index.
      // Note: the AVX2 implementation keeps the pivots quantized to multiples
      // of 64.
      *count_col_ptr = count_val + thresh_adj;
      *pivot_col_ptr = pivot_val + ind8 * 8;

      // Uses a lookup table to find the last 3 bits of the result.
      return *pivot_col_ptr +
             nth_set_bit_index[thresh - thresh_adj - 1][bitmask];
    }
  }
}
inline void Transpose16x16Neon(const uint8_t** srcs, int col, int row,
                               ptrdiff_t stride, uint8_t* dst) {
  const uint8x16_t a0 = vld1q_u8(srcs[0] + col);
  const uint8x16_t a1 = vld1q_u8(srcs[1] + col);
  const uint8x16_t a2 = vld1q_u8(srcs[2] + col);
  const uint8x16_t a3 = vld1q_u8(srcs[3] + col);
  const uint8x16_t a4 = vld1q_u8(srcs[4] + col);
  const uint8x16_t a5 = vld1q_u8(srcs[5] + col);
  const uint8x16_t a6 = vld1q_u8(srcs[6] + col);
  const uint8x16_t a7 = vld1q_u8(srcs[7] + col);
  const uint8x16_t a8 = vld1q_u8(srcs[8] + col);
  const uint8x16_t a9 = vld1q_u8(srcs[9] + col);
  const uint8x16_t a10 = vld1q_u8(srcs[10] + col);
  const uint8x16_t a11 = vld1q_u8(srcs[11] + col);
  const uint8x16_t a12 = vld1q_u8(srcs[12] + col);
  const uint8x16_t a13 = vld1q_u8(srcs[13] + col);
  const uint8x16_t a14 = vld1q_u8(srcs[14] + col);
  const uint8x16_t a15 = vld1q_u8(srcs[15] + col);

  const uint8x16x2_t b0 = vtrnq_u8(a0, a1);
  const uint8x16x2_t b1 = vtrnq_u8(a2, a3);
  const uint8x16x2_t b2 = vtrnq_u8(a4, a5);
  const uint8x16x2_t b3 = vtrnq_u8(a6, a7);
  const uint8x16x2_t b4 = vtrnq_u8(a8, a9);
  const uint8x16x2_t b5 = vtrnq_u8(a10, a11);
  const uint8x16x2_t b6 = vtrnq_u8(a12, a13);
  const uint8x16x2_t b7 = vtrnq_u8(a14, a15);

  const uint16x8x2_t c0 = vtrnq_u16(b0.val[0], b1.val[0]);
  const uint16x8x2_t c1 = vtrnq_u16(b0.val[1], b1.val[1]);
  const uint16x8x2_t c2 = vtrnq_u16(b2.val[0], b3.val[0]);
  const uint16x8x2_t c3 = vtrnq_u16(b2.val[1], b3.val[1]);
  const uint16x8x2_t c4 = vtrnq_u16(b4.val[0], b5.val[0]);
  const uint16x8x2_t c5 = vtrnq_u16(b4.val[1], b5.val[1]);
  const uint16x8x2_t c6 = vtrnq_u16(b6.val[0], b7.val[0]);
  const uint16x8x2_t c7 = vtrnq_u16(b6.val[1], b7.val[1]);

  const uint32x4x2_t d0 = vtrnq_u32(c0.val[0], c2.val[0]);
  const uint32x4x2_t d1 = vtrnq_u32(c0.val[1], c2.val[1]);
  const uint32x4x2_t d2 = vtrnq_u32(c1.val[0], c3.val[0]);
  const uint32x4x2_t d3 = vtrnq_u32(c1.val[1], c3.val[1]);
  const uint32x4x2_t d4 = vtrnq_u32(c4.val[0], c6.val[0]);
  const uint32x4x2_t d5 = vtrnq_u32(c4.val[1], c6.val[1]);
  const uint32x4x2_t d6 = vtrnq_u32(c5.val[0], c7.val[0]);
  const uint32x4x2_t d7 = vtrnq_u32(c5.val[1], c7.val[1]);

  const uint64x2_t e0 = vtrn1q_u64(d0.val[0], d4.val[0]);
  const uint64x2_t e1 = vtrn1q_u64(d2.val[0], d6.val[0]);
  const uint64x2_t e2 = vtrn1q_u64(d1.val[0], d5.val[0]);
  const uint64x2_t e3 = vtrn1q_u64(d3.val[0], d7.val[0]);
  const uint64x2_t e4 = vtrn1q_u64(d0.val[1], d4.val[1]);
  const uint64x2_t e5 = vtrn1q_u64(d2.val[1], d6.val[1]);
  const uint64x2_t e6 = vtrn1q_u64(d1.val[1], d5.val[1]);
  const uint64x2_t e7 = vtrn1q_u64(d3.val[1], d7.val[1]);
  const uint64x2_t e8 = vtrn2q_u64(d0.val[0], d4.val[0]);
  const uint64x2_t e9 = vtrn2q_u64(d2.val[0], d6.val[0]);
  const uint64x2_t e10 = vtrn2q_u64(d1.val[0], d5.val[0]);
  const uint64x2_t e11 = vtrn2q_u64(d3.val[0], d7.val[0]);
  const uint64x2_t e12 = vtrn2q_u64(d0.val[1], d4.val[1]);
  const uint64x2_t e13 = vtrn2q_u64(d2.val[1], d6.val[1]);
  const uint64x2_t e14 = vtrn2q_u64(d1.val[1], d5.val[1]);
  const uint64x2_t e15 = vtrn2q_u64(d3.val[1], d7.val[1]);

  vst1q_u8(&dst[(col + 0) * stride + row], e0);
  vst1q_u8(&dst[(col + 1) * stride + row], e1);
  vst1q_u8(&dst[(col + 2) * stride + row], e2);
  vst1q_u8(&dst[(col + 3) * stride + row], e3);
  vst1q_u8(&dst[(col + 4) * stride + row], e4);
  vst1q_u8(&dst[(col + 5) * stride + row], e5);
  vst1q_u8(&dst[(col + 6) * stride + row], e6);
  vst1q_u8(&dst[(col + 7) * stride + row], e7);
  vst1q_u8(&dst[(col + 8) * stride + row], e8);
  vst1q_u8(&dst[(col + 9) * stride + row], e9);
  vst1q_u8(&dst[(col + 10) * stride + row], e10);
  vst1q_u8(&dst[(col + 11) * stride + row], e11);
  vst1q_u8(&dst[(col + 12) * stride + row], e12);
  vst1q_u8(&dst[(col + 13) * stride + row], e13);
  vst1q_u8(&dst[(col + 14) * stride + row], e14);
  vst1q_u8(&dst[(col + 15) * stride + row], e15);
}

// Transposes 8 rows of 8 pixels pointed to by 'srcs + col' into 'dst + row'.
inline void Transpose8x8Neon(const uint16_t** srcs, int col, int row,
                             ptrdiff_t stride, uint16_t* dst) {
#if TRANSPOSE_TABLE
#if TRANSPOSE_ST2
  const uint8x16_t vtable0 = {0, 1, 16, 17, 32, 33, 48, 49,
                              2, 3, 18, 19, 34, 35, 50, 51};
  const uint8x16_t vtable1 = {4, 5, 20, 21, 36, 37, 52, 53,
                              6, 7, 22, 23, 38, 39, 54, 55};
  const uint8x16_t vtable2 = {8,  9,  24, 25, 40, 41, 56, 57,
                              10, 11, 26, 27, 42, 43, 58, 59};
  const uint8x16_t vtable3 = {12, 13, 28, 29, 44, 45, 60, 61,
                              14, 15, 30, 31, 46, 47, 62, 63};
#else
  const uint8x16_t vtable0 = {0,  1,  2,  3,  16, 17, 18, 19,
                              32, 33, 34, 35, 48, 49, 50, 51};
  const uint8x16_t vtable1 = {4,  5,  6,  7,  20, 21, 22, 23,
                              36, 37, 38, 39, 52, 53, 54, 55};
  const uint8x16_t vtable2 = {8,  9,  10, 11, 24, 25, 26, 27,
                              40, 41, 42, 43, 56, 57, 58, 59};
  const uint8x16_t vtable3 = {12, 13, 14, 15, 28, 29, 30, 31,
                              44, 45, 46, 47, 60, 61, 62, 63};
#endif  // TRANSPOSE_ST2

  const uint8x16x4_t a0246 = {
      vld1q_u16(srcs[0] + col), vld1q_u16(srcs[2] + col),
      vld1q_u16(srcs[4] + col), vld1q_u16(srcs[6] + col)};
  const uint8x16x4_t a1357 = {
      vld1q_u16(srcs[1] + col), vld1q_u16(srcs[3] + col),
      vld1q_u16(srcs[5] + col), vld1q_u16(srcs[7] + col)};

  const uint16x8_t c0 = vqtbl4q_u8(a0246, vtable0);
  const uint16x8_t c2 = vqtbl4q_u8(a0246, vtable1);
  const uint16x8_t c4 = vqtbl4q_u8(a0246, vtable2);
  const uint16x8_t c6 = vqtbl4q_u8(a0246, vtable3);

  const uint16x8_t c1 = vqtbl4q_u8(a1357, vtable0);
  const uint16x8_t c3 = vqtbl4q_u8(a1357, vtable1);
  const uint16x8_t c5 = vqtbl4q_u8(a1357, vtable2);
  const uint16x8_t c7 = vqtbl4q_u8(a1357, vtable3);

#if TRANSPOSE_ST2
  const uint16x4x2_t c01_lo = {vget_low_u16(c0), vget_low_u16(c1)};
  const uint16x4x2_t c01_hi = {vget_high_u16(c0), vget_high_u16(c1)};
  const uint16x4x2_t c23_lo = {vget_low_u16(c2), vget_low_u16(c3)};
  const uint16x4x2_t c23_hi = {vget_high_u16(c2), vget_high_u16(c3)};
  const uint16x4x2_t c45_lo = {vget_low_u16(c4), vget_low_u16(c5)};
  const uint16x4x2_t c45_hi = {vget_high_u16(c4), vget_high_u16(c5)};
  const uint16x4x2_t c67_lo = {vget_low_u16(c6), vget_low_u16(c7)};
  const uint16x4x2_t c67_hi = {vget_high_u16(c6), vget_high_u16(c7)};

  vst2_u16(&dst[(col + 0) * stride + row], c01_lo);
  vst2_u16(&dst[(col + 1) * stride + row], c01_hi);
  vst2_u16(&dst[(col + 2) * stride + row], c23_lo);
  vst2_u16(&dst[(col + 3) * stride + row], c23_hi);
  vst2_u16(&dst[(col + 4) * stride + row], c45_lo);
  vst2_u16(&dst[(col + 5) * stride + row], c45_hi);
  vst2_u16(&dst[(col + 6) * stride + row], c67_lo);
  vst2_u16(&dst[(col + 7) * stride + row], c67_hi);
#else   // !TRANSPOSE_ST2
  const uint16x8x2_t d01 = vtrnq_u16(c0, c1);
  const uint16x8x2_t d23 = vtrnq_u16(c2, c3);
  const uint16x8x2_t d45 = vtrnq_u16(c4, c5);
  const uint16x8x2_t d67 = vtrnq_u16(c6, c7);

  vst1q_u16(&dst[(col + 0) * stride + row], d01.val[0]);
  vst1q_u16(&dst[(col + 1) * stride + row], d01.val[1]);
  vst1q_u16(&dst[(col + 2) * stride + row], d23.val[0]);
  vst1q_u16(&dst[(col + 3) * stride + row], d23.val[1]);
  vst1q_u16(&dst[(col + 4) * stride + row], d45.val[0]);
  vst1q_u16(&dst[(col + 5) * stride + row], d45.val[1]);
  vst1q_u16(&dst[(col + 6) * stride + row], d67.val[0]);
  vst1q_u16(&dst[(col + 7) * stride + row], d67.val[1]);
#endif  // TRANSPOSE_ST2

#else   // !TRANSPOSE_TABLE
  const uint16x8_t a0 = vld1q_u16(srcs[0] + col);
  const uint16x8_t a1 = vld1q_u16(srcs[1] + col);
  const uint16x8_t a2 = vld1q_u16(srcs[2] + col);
  const uint16x8_t a3 = vld1q_u16(srcs[3] + col);
  const uint16x8_t a4 = vld1q_u16(srcs[4] + col);
  const uint16x8_t a5 = vld1q_u16(srcs[5] + col);
  const uint16x8_t a6 = vld1q_u16(srcs[6] + col);
  const uint16x8_t a7 = vld1q_u16(srcs[7] + col);

  const uint16x8_t b0 = vtrn1q_u64(a0, a4);
  const uint16x8_t b4 = vtrn2q_u64(a0, a4);
  const uint16x8_t b1 = vtrn1q_u64(a1, a5);
  const uint16x8_t b5 = vtrn2q_u64(a1, a5);
  const uint16x8_t b2 = vtrn1q_u64(a2, a6);
  const uint16x8_t b6 = vtrn2q_u64(a2, a6);
  const uint16x8_t b3 = vtrn1q_u64(a3, a7);
  const uint16x8_t b7 = vtrn2q_u64(a3, a7);

  const uint32x4x2_t c02 = vtrnq_u32(b0, b2);
  const uint32x4x2_t c13 = vtrnq_u32(b1, b3);
  const uint32x4x2_t c46 = vtrnq_u32(b4, b6);
  const uint32x4x2_t c57 = vtrnq_u32(b5, b7);

  const uint16x8x2_t d01 = vtrnq_u16(c02.val[0], c13.val[0]);
  const uint16x8x2_t d23 = vtrnq_u16(c02.val[1], c13.val[1]);
  const uint16x8x2_t d45 = vtrnq_u16(c46.val[0], c57.val[0]);
  const uint16x8x2_t d67 = vtrnq_u16(c46.val[1], c57.val[1]);

  vst1q_u16(&dst[(col + 0) * stride + row], d01.val[0]);
  vst1q_u16(&dst[(col + 1) * stride + row], d01.val[1]);
  vst1q_u16(&dst[(col + 2) * stride + row], d23.val[0]);
  vst1q_u16(&dst[(col + 3) * stride + row], d23.val[1]);
  vst1q_u16(&dst[(col + 4) * stride + row], d45.val[0]);
  vst1q_u16(&dst[(col + 5) * stride + row], d45.val[1]);
  vst1q_u16(&dst[(col + 6) * stride + row], d67.val[0]);
  vst1q_u16(&dst[(col + 7) * stride + row], d67.val[1]);
#endif  // TRANSPOSE_TABLE
}
}  // namespace

namespace internal {
uint16_t SearchUpDownNeon(int col, int row, int threshold, int rank,
                          const uint16_t* histc, uint8_t* pivot_col_ptr,
                          uint16_t* count_col_ptr) {
  return (*count_col_ptr < rank)
             ? FindRankNeon<kSearchUpward>(histc, col, row, threshold, rank,
                                           count_col_ptr, pivot_col_ptr)
             : FindRankNeon<kSearchDownward>(histc, col, row, threshold, rank,
                                             count_col_ptr, pivot_col_ptr);
}

uint16_t SearchUpDownNeon(int col, int row, int threshold, int rank,
                          const uint16_t* histc, uint16_t* pivot_col_ptr,
                          uint16_t* count_col_ptr) {
  return (*count_col_ptr < rank)
             ? FindRankNeon<kSearchUpward>(histc, col, row, threshold, rank,
                                           count_col_ptr, pivot_col_ptr)
             : FindRankNeon<kSearchDownward>(histc, col, row, threshold, rank,
                                             count_col_ptr, pivot_col_ptr);
}

// TODO: Optimize to reduce memory accesses in cases where consecutive
// insets are the same.
void UpdateCountsNeon(ReadViewGrayU8 ordinal_input, int radius, int row,
                      const uint8_t* inset, const uint8_t* pivot,
                      uint16_t* count) {
  const int output_width = ordinal_input.width() - 2 * radius;
  uint16x8_t* count_p = reinterpret_cast<uint16x8_t*>(count);
  const uint8x16_t* pivot_p = reinterpret_cast<const uint8x16_t*>(pivot);

  // Enumerated special-cased loops.
  CHECK_GE(output_width, 16);
  CHECK_LE(output_width, 64);
  CHECK_EQ(output_width % 16, 0);
  // The four cases ({16, 32, 48, 64}) are manually unrolled for performance.
  // TODO: See if there's a more concise way to write this.
  if (output_width == 64) {
    const uint8x16_t pivot0 = pivot_p[0];
    const uint8x16_t pivot1 = pivot_p[1];
    const uint8x16_t pivot2 = pivot_p[2];
    const uint8x16_t pivot3 = pivot_p[3];
    uint8x16_t delta0 = vdupq_n_u8(0);
    uint8x16_t delta1 = vdupq_n_u8(0);
    uint8x16_t delta2 = vdupq_n_u8(0);
    uint8x16_t delta3 = vdupq_n_u8(0);
    for (int col = 0; col <= radius * 2; ++col) {
      const int inset_v = inset[std::abs(radius - col)];
      const uint8_t* val_t = &ordinal_input(col, row + inset_v);
      const uint8_t* val_b =
          &ordinal_input(col, row + 2 * radius + 1 - inset_v);
      delta0 += vcltq_u8(vld1q_u8(&val_t[0]), pivot0) -
                vcltq_u8(vld1q_u8(&val_b[0]), pivot0);
      delta1 += vcltq_u8(vld1q_u8(&val_t[16]), pivot1) -
                vcltq_u8(vld1q_u8(&val_b[16]), pivot1);
      delta2 += vcltq_u8(vld1q_u8(&val_t[32]), pivot2) -
                vcltq_u8(vld1q_u8(&val_b[32]), pivot2);
      delta3 += vcltq_u8(vld1q_u8(&val_t[48]), pivot3) -
                vcltq_u8(vld1q_u8(&val_b[48]), pivot3);
    }
    count_p[0] += vmovl_s8(vget_low_s8(delta0));
    count_p[1] += vmovl_s8(vget_high_s8(delta0));
    count_p[2] += vmovl_s8(vget_low_s8(delta1));
    count_p[3] += vmovl_s8(vget_high_s8(delta1));
    count_p[4] += vmovl_s8(vget_low_s8(delta2));
    count_p[5] += vmovl_s8(vget_high_s8(delta2));
    count_p[6] += vmovl_s8(vget_low_s8(delta3));
    count_p[7] += vmovl_s8(vget_high_s8(delta3));
  } else if (output_width == 48) {
    const uint8x16_t pivot0 = pivot_p[0];
    const uint8x16_t pivot1 = pivot_p[1];
    const uint8x16_t pivot2 = pivot_p[2];
    uint8x16_t delta0 = vdupq_n_u8(0);
    uint8x16_t delta1 = vdupq_n_u8(0);
    uint8x16_t delta2 = vdupq_n_u8(0);
    for (int col = 0; col <= radius * 2; ++col) {
      const int inset_v = inset[std::abs(radius - col)];
      const uint8_t* val_t = &ordinal_input(col, row + inset_v);
      const uint8_t* val_b =
          &ordinal_input(col, row + 2 * radius + 1 - inset_v);
      delta0 += vcltq_u8(vld1q_u8(&val_t[0]), pivot0) -
                vcltq_u8(vld1q_u8(&val_b[0]), pivot0);
      delta1 += vcltq_u8(vld1q_u8(&val_t[16]), pivot1) -
                vcltq_u8(vld1q_u8(&val_b[16]), pivot1);
      delta2 += vcltq_u8(vld1q_u8(&val_t[32]), pivot2) -
                vcltq_u8(vld1q_u8(&val_b[32]), pivot2);
    }
    count_p[0] += vmovl_s8(vget_low_s8(delta0));
    count_p[1] += vmovl_s8(vget_high_s8(delta0));
    count_p[2] += vmovl_s8(vget_low_s8(delta1));
    count_p[3] += vmovl_s8(vget_high_s8(delta1));
    count_p[4] += vmovl_s8(vget_low_s8(delta2));
    count_p[5] += vmovl_s8(vget_high_s8(delta2));
  } else if (output_width == 32) {
    const uint8x16_t pivot0 = pivot_p[0];
    const uint8x16_t pivot1 = pivot_p[1];
    uint8x16_t delta0 = vdupq_n_u8(0);
    uint8x16_t delta1 = vdupq_n_u8(0);
    for (int col = 0; col <= radius * 2; ++col) {
      const int inset_v = inset[std::abs(radius - col)];
      const uint8_t* val_t = &ordinal_input(col, row + inset_v);
      const uint8_t* val_b =
          &ordinal_input(col, row + 2 * radius + 1 - inset_v);
      delta0 += vcltq_u8(vld1q_u8(&val_t[0]), pivot0) -
                vcltq_u8(vld1q_u8(&val_b[0]), pivot0);
      delta1 += vcltq_u8(vld1q_u8(&val_t[16]), pivot1) -
                vcltq_u8(vld1q_u8(&val_b[16]), pivot1);
    }
    count_p[0] += vmovl_s8(vget_low_s8(delta0));
    count_p[1] += vmovl_s8(vget_high_s8(delta0));
    count_p[2] += vmovl_s8(vget_low_s8(delta1));
    count_p[3] += vmovl_s8(vget_high_s8(delta1));
  } else {
    const uint8x16_t pivot = pivot_p[0];
    uint8x16_t delta = vdupq_n_u8(0);
    for (int col = 0; col <= radius * 2; ++col) {
      const int inset_v = inset[std::abs(radius - col)];
      const uint8_t* val_t = &ordinal_input(col, row + inset_v);
      const uint8_t* val_b =
          &ordinal_input(col, row + 2 * radius + 1 - inset_v);
      delta += vcltq_u8(vld1q_u8(&val_t[0]), pivot) -
               vcltq_u8(vld1q_u8(&val_b[0]), pivot);
    }
    count_p[0] += vmovl_s8(vget_low_s8(delta));
    count_p[1] += vmovl_s8(vget_high_s8(delta));
  }

  // For reference, an equivalent variable-width loop:
  //
  //    for (int col = 0; col <= radius * 2; ++col) {
  //      const int inset_v = inset[std::abs(radius - col)];
  //      const uint8_t* val_t = &ordinal_input(col, row + inset_v);
  //      const uint8_t* val_b = &ordinal_input(col,
  //                                            row + 2 * radius + 1 - inset_v);
  //      for (int ind = 0; ind < output_width / 16; ++ind) {
  //        const int8x16_t delta =
  //            vcltq_u8(vld1q_u8(&val_t[ind * 16]), pivot_p[ind]) -
  //            vcltq_u8(vld1q_u8(&val_b[ind * 16]), pivot_p[ind]);
  //        // Sign-extends the 8-bit delta to 16-bit.
  //        // TODO: Invert to reduce the amount of sign-extension.
  //        count_p[ind * 2] += vmovl_s8(vget_low_s8(delta));
  //        count_p[ind * 2 + 1] += vmovl_s8(vget_high_s8(delta));
  //      }
  //    }
}

void UpdateCountsNeon(ReadViewGrayU16 ordinal_input, int radius, int row,
                      const uint8_t* inset, const uint16_t* pivot,
                      uint16_t* count) {
  const int output_width = ordinal_input.width() - 2 * radius;
  uint16x8_t* count_p = reinterpret_cast<uint16x8_t*>(count);
  const uint16x8_t* pivot_p = reinterpret_cast<const uint16x8_t*>(pivot);
  for (int col = 0; col <= radius * 2; ++col) {
    const int inset_v = inset[std::abs(radius - col)];
    const uint16_t* val_t = &ordinal_input(col, row + inset_v);
    const uint16_t* val_b = &ordinal_input(col, row + 2 * radius + 1 - inset_v);
    for (int ind = 0; ind < output_width / 8; ++ind) {
      count_p[ind] += vcltq_u16(vld1q_u16(&val_t[ind * 8]), pivot_p[ind]) -
                      vcltq_u16(vld1q_u16(&val_b[ind * 8]), pivot_p[ind]);
    }
  }
}

void TransposeFirstRowsNeon(ReadViewGrayU8 ordinal_input, int radius,
                            const uint8_t* inset, uint8_t* pixels_left,
                            uint8_t* pixels_right) {
  constexpr int kBlockSize = 16;
  const int kernel_width_ceil = (2 * radius + kBlockSize) & ~(kBlockSize - 1);
  const int output_width = ordinal_input.width() - 2 * radius;
  alignas(16) static const uint8_t zeros[256] = {0};

  // Processes pixels in 16x16 blocks.
  for (int row = 0; row < kernel_width_ceil; row += kBlockSize) {
    const uint8_t* pl[kBlockSize];
    const uint8_t* pr[kBlockSize];
    // Sets up pointers. (Tail rows are pointed to an array of zeros.)
    for (int dy = 0; dy < kBlockSize; ++dy) {
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
    for (int col = 0; col < output_width; col += kBlockSize) {
      Transpose16x16Neon(pl, col, row, kernel_width_ceil, pixels_left);
      Transpose16x16Neon(pr, col, row, kernel_width_ceil, pixels_right);
    }
  }
}

void TransposeFirstRowsNeon(ReadViewGrayU16 ordinal_input, int radius,
                            const uint8_t* inset, uint16_t* pixels_left,
                            uint16_t* pixels_right) {
  constexpr int kBlockSize = 8;
  // const int kernel_width_ceil = (2 * radius + kBlockSize) & ~(kBlockSize -
  // 1);
  // TODO: Clean up the discrepancy between the 8 above and 16 below?
  const int kernel_width_ceil = (2 * radius + 16) & ~(16 - 1);
  const int output_width = ordinal_input.width() - 2 * radius;
  alignas(16) static const uint16_t zeros[256] = {0};

  // Processes pixels in 8x8 blocks.
  for (int row = 0; row < kernel_width_ceil; row += kBlockSize) {
    const uint16_t* pl[kBlockSize];
    const uint16_t* pr[kBlockSize];
    // Sets up pointers. (Tail rows are pointed to an array of zeros.)
    for (int dy = 0; dy < kBlockSize; ++dy) {
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
    for (int col = 0; col < output_width; col += kBlockSize) {
      Transpose8x8Neon(pl, col, row, kernel_width_ceil, pixels_left);
      Transpose8x8Neon(pr, col, row, kernel_width_ceil, pixels_right);
    }
  }
}

int StepHorizontalNeon(const uint8_t* pixels_left, const uint8_t* pixels_right,
                       int pivot_val, int kernel_width_ceil) {
  const uint8x16_t v_pivot = vdupq_n_u8(static_cast<uint8_t>(pivot_val));
  int16x8_t v_count = vdupq_n_s16(0);
  for (int row = 0; row < kernel_width_ceil; row += 16) {
    const uint8x16_t v_left = vld1q_u8(pixels_left + row);
    const uint8x16_t v_right = vld1q_u8(pixels_right + row);
    v_count = vaddq_s8(v_count, vcltq_u8(v_left, v_pivot));
    v_count = vsubq_s8(v_count, vcltq_u8(v_right, v_pivot));
  }
  return vaddvq_s8(v_count);
}

int StepHorizontalNeon(const uint16_t* pixels_left,
                       const uint16_t* pixels_right, int pivot_val,
                       int kernel_width_ceil) {
  const uint16x8_t v_pivot = vdupq_n_u16(static_cast<uint16_t>(pivot_val));
  int16x8_t v_count = vdupq_n_s16(0);
  for (int row = 0; row < kernel_width_ceil; row += 8) {
    const uint16x8_t v_left = vld1q_u16(pixels_left + row);
    const uint16x8_t v_right = vld1q_u16(pixels_right + row);
    v_count = vaddq_s16(v_count, vcltq_u16(v_left, v_pivot));
    v_count = vsubq_s16(v_count, vcltq_u16(v_right, v_pivot));
  }
  return vaddvq_s16(v_count);
}
}  // namespace internal
}  // namespace fast_isotropic_median_filter
#endif  // defined(__aarch64__)
