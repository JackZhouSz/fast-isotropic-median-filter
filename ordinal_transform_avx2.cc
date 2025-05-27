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
#include "ordinal_transform.h"

#include <immintrin.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "image.h"

namespace fast_isotropic_median_filter {
namespace {
// Bitwise rotation of lowest 8 bits by 4. Equivalent to:
//     v = (v & ~0xFF) | (v & 0xF0) >> 4 | (v & 0x0F) << 4;
// Note: the compiler seems unable to generate a 'rorb' instruction from C code,
// and the performance impact is very significant, so assembly is required here.
// NOLINTBEGIN(google3-runtime-inline-assembly): b/382732417
template <typename T>
static inline T HistIndex16(T v) {
  asm volatile("rorb $4, %b0" : "=r"(v) : "0"(v) :);
  return v;
}
// NOLINTEND(google3-runtime-inline-assembly): b/382732417

// Converts vector {v0, v1, .., v15} to {0, v0, v0 + v1, v0 + v1 + v2, ...} in
// 'prefix_sum', and 'v0 + .. + v15' in 'total_sum'.
static inline void PrefixSum16Avx2(__m256i v, __m256i& prefix_sum,
                                   __m256i& total_sum) {
  // Broadcasts element 15 into positions [8 .. 15]; zeroes [0 .. 7].
  const __m256i v_broadcast_15_to_high_8 =
      _mm256_setr_epi64x(-1, -1, 0x0F0E0F0E0F0E0F0ELL, 0x0F0E0F0E0F0E0F0ELL);
  const __m256i v_broadcast_highest_to_all =
      _mm256_set1_epi64x(0x0F0E0F0E0F0E0F0ELL);
  const __m256i vzero = _mm256_setzero_si256();

  // Constructs prefix sum {{v0, v0 + v1, .., v0 + .. + v7}, {v8, v8 + v9, ..}}
  v = _mm256_add_epi16(v, _mm256_alignr_epi8(v, vzero, 16 - 2));
  v = _mm256_add_epi16(v, _mm256_alignr_epi8(v, vzero, 16 - 4));
  v = _mm256_add_epi16(v, _mm256_alignr_epi8(v, vzero, 16 - 8));

  // Swaps low and high 128-bit lanes.
  __m256i v_swapped = _mm256_permute2x128_si256(v, v, _MM_SHUFFLE(0, 0, 0, 1));
  total_sum = _mm256_add_epi16(v, v_swapped);
  total_sum = _mm256_shuffle_epi8(total_sum, v_broadcast_highest_to_all);

  // Shifts all elements to the left by 16 bits, within 128-bit lanes:
  // {0, v0, v0 + v1, .., v0 + .. + v6}, {0, v8, v8 + v9, ..}}
  v = _mm256_alignr_epi8(v, vzero, 14);

  // Adds (v0 + .. + v7) to last 8 elements to construct final prefix sum.
  const __m256i add = _mm256_shuffle_epi8(v_swapped, v_broadcast_15_to_high_8);
  prefix_sum = _mm256_add_epi16(v, add);
}

// Computes the prefix sum of an interleaved 65536-element histogram, in-place.
// The "channels" are summed in parallel, then the sums of channels are
// prefix-summed and added back to all channels to complete the prefix sum.
static inline void PrefixSum16x65536Avx2(__m256i* hist) {
  __m256i total_sum = _mm256_setzero_si256();
  for (int block = 0; block < 256; ++block) {
    __m256i sum = _mm256_setzero_si256();
    for (int ind = 0; ind < 16; ++ind) {
      __m256i val = hist[block * 16 + ind];
      hist[block * 16 + ind] = sum;
      sum = _mm256_add_epi16(sum, val);
    }
    __m256i prefix_sum, block_sum;
    PrefixSum16Avx2(sum, prefix_sum, block_sum);
    sum = _mm256_add_epi16(prefix_sum, total_sum);
    for (int ind = 0; ind < 16; ++ind) {
      hist[block * 16 + ind] = _mm256_add_epi16(hist[block * 16 + ind], sum);
    }
    total_sum = _mm256_add_epi16(total_sum, block_sum);
  }
}

// Helper function for the uint16 version of OrdinalTransformStdSortAvx2.
// Appends the (x, y) position of each pixel to the pixel value, and returns a
// sortable array of uint32's.
static std::unique_ptr<uint32_t[]> PackTuplesAvx2(
    ReadViewGrayU16 cardinal_input) {
  const int width = cardinal_input.width();
  const int height = cardinal_input.height();
  auto v = std::make_unique<uint32_t[]>(width * height);

  const __m256i dy = _mm256_set1_epi16(1 << 8);
  __m256i y_vec =
      _mm256_setr_epi16(0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15);
  for (int y = 0; y < height; ++y) {
    __m256i y_x = y_vec;
    const __m256i* vsrc_ptr =
        reinterpret_cast<const __m256i*>(&cardinal_input(0, y));
    __m256i* vdst_ptr = reinterpret_cast<__m256i*>(&v[width * y]);
    int x = 0;
    for (x = 0; x < width - 15; x += 16) {
      __m256i v = _mm256_loadu_si256(vsrc_ptr++);
      v = _mm256_permute4x64_epi64(v, _MM_SHUFFLE(3, 1, 2, 0));
      __m256i v_lo = _mm256_unpacklo_epi16(y_x, v);
      __m256i v_hi = _mm256_unpackhi_epi16(y_x, v);
      _mm256_storeu_si256(vdst_ptr++, v_lo);
      _mm256_storeu_si256(vdst_ptr++, v_hi);
      y_x = _mm256_add_epi16(y_x, _mm256_set1_epi16(16));
    }
    // Tail.
    uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(vdst_ptr);
    const uint16_t* row_ptr = &cardinal_input(0, y);
    for (; x < width; ++x) {
      *dst_ptr++ = static_cast<int64_t>(row_ptr[x]) << 16 | y << 8 | x;
    }
    y_vec = _mm256_add_epi16(y_vec, dy);
  }
  return v;
}

// Helper function for the float version of OrdinalTransformStdSortAvx2.
// Appends the (x, y) position of each pixel to the adjusted pixel value, and
// returns a sortable array of int64's.
static std::unique_ptr<int64_t[]> PackTuplesAvx2(ReadViewGrayF cardinal_input) {
  const int width = cardinal_input.width();
  const int height = cardinal_input.height();
  auto v = std::make_unique<int64_t[]>(width * height);
  const __m256i dy = _mm256_set1_epi32(1 << 8);
  __m256i y_vec = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
  for (int y = 0; y < height; ++y) {
    __m256i y_x = y_vec;
    const __m256i* vsrc_ptr =
        reinterpret_cast<const __m256i*>(&cardinal_input(0, y));
    __m256i* vdst_ptr = reinterpret_cast<__m256i*>(&v[width * y]);
    int x = 0;
    for (x = 0; x < width - 7; x += 8) {
      __m256i v = _mm256_loadu_si256(vsrc_ptr++);
      v = _mm256_permute4x64_epi64(v, _MM_SHUFFLE(3, 1, 2, 0));
      v = _mm256_xor_si256(v, _mm256_srli_epi32(_mm256_srai_epi32(v, 30), 1));
      __m256i v_lo = _mm256_unpacklo_epi32(y_x, v);
      __m256i v_hi = _mm256_unpackhi_epi32(y_x, v);
      _mm256_storeu_si256(vdst_ptr++, v_lo);
      _mm256_storeu_si256(vdst_ptr++, v_hi);
      y_x = _mm256_add_epi32(y_x, _mm256_set1_epi32(8));
    }
    // Tail.
    int64_t* dst_ptr = reinterpret_cast<int64_t*>(vdst_ptr);
    const int32_t* row_ptr =
        reinterpret_cast<const int32_t*>(&cardinal_input(0, y));
    for (; x < width; ++x) {
      int32_t float_as_int = row_ptr[x];
      float_as_int ^= float_as_int >> 31 & 0x7FFFFFFF;
      *dst_ptr++ = static_cast<int64_t>(float_as_int) << 32 | y << 8 | x;
    }
    y_vec = _mm256_add_epi32(y_vec, dy);
  }
  return v;
}

// Helper function for the uint16 version of OrdinalTransformStdSortAvx2.
// Unpacks the sorted tuples into `output`, `sorted_values`, and optionally
// `sorted_coords`.
template <typename ordinal_t>
static void UnpackSortedTuplesAvx2(
    const std::unique_ptr<uint32_t[]>& v,
    WriteViewGray<ordinal_t> ordinal_output, absl::Span<uint16_t> sorted_values,
    std::optional<absl::Span<uint16_t>> sorted_coords) {
  constexpr int kOrdinalShift = std::is_same_v<ordinal_t, uint8_t> ? 6 : 0;
  const int width = ordinal_output.width();
  const int height = ordinal_output.height();
  // Unpacks into reverse_lut.
  const __m256i* vsrc_ptr = reinterpret_cast<const __m256i*>(v.get());
  __m256i* vdst_values_ptr = reinterpret_cast<__m256i*>(sorted_values.data());
  int i = 0;
  if (sorted_coords.has_value()) {
    __m256i* vdst_coords_ptr =
        reinterpret_cast<__m256i*>(sorted_coords->data());
    const __m256i vdeinterleave =
        _mm256_set_epi8(15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0,
                        15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);
    for (i = 0; i < width * height - 31; i += 32) {
      __m256i v0 = _mm256_loadu_si256(vsrc_ptr++);
      __m256i v1 = _mm256_loadu_si256(vsrc_ptr++);
      __m256i v2 = _mm256_loadu_si256(vsrc_ptr++);
      __m256i v3 = _mm256_loadu_si256(vsrc_ptr++);
      v0 = _mm256_shuffle_epi8(v0, vdeinterleave);
      v1 = _mm256_shuffle_epi8(v1, vdeinterleave);
      v2 = _mm256_shuffle_epi8(v2, vdeinterleave);
      v3 = _mm256_shuffle_epi8(v3, vdeinterleave);
      v0 = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));
      v1 = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));
      v2 = _mm256_permute4x64_epi64(v2, _MM_SHUFFLE(3, 1, 2, 0));
      v3 = _mm256_permute4x64_epi64(v3, _MM_SHUFFLE(3, 1, 2, 0));
      const __m256i v01c = _mm256_permute2x128_si256(v0, v1, 0x20);
      const __m256i v23c = _mm256_permute2x128_si256(v2, v3, 0x20);
      const __m256i v01v = _mm256_permute2x128_si256(v0, v1, 0x31);
      const __m256i v23v = _mm256_permute2x128_si256(v2, v3, 0x31);
      _mm256_storeu_si256(vdst_coords_ptr++, v01c);
      _mm256_storeu_si256(vdst_coords_ptr++, v23c);
      _mm256_storeu_si256(vdst_values_ptr++, v01v);
      _mm256_storeu_si256(vdst_values_ptr++, v23v);
    }
    // Tail.
    for (; i < width * height; ++i) {
      sorted_values[i] = v[i] >> 16;
      (*sorted_coords)[i] = v[i] & 0xFFFF;
    }
  } else {
    for (i = 0; i < width * height - 31; i += 32) {
      __m256i v0 = _mm256_loadu_si256(vsrc_ptr++);
      __m256i v1 = _mm256_loadu_si256(vsrc_ptr++);
      __m256i v2 = _mm256_loadu_si256(vsrc_ptr++);
      __m256i v3 = _mm256_loadu_si256(vsrc_ptr++);
      v0 = _mm256_srli_epi32(v0, 16);
      v1 = _mm256_srli_epi32(v1, 16);
      v2 = _mm256_srli_epi32(v2, 16);
      v3 = _mm256_srli_epi32(v3, 16);
      __m256i v01 = _mm256_packus_epi32(v0, v1);
      __m256i v23 = _mm256_packus_epi32(v2, v3);
      v01 = _mm256_permute4x64_epi64(v01, _MM_SHUFFLE(3, 1, 2, 0));
      v23 = _mm256_permute4x64_epi64(v23, _MM_SHUFFLE(3, 1, 2, 0));
      _mm256_storeu_si256(vdst_values_ptr++, v01);
      _mm256_storeu_si256(vdst_values_ptr++, v23);
    }
    // Tail.
    for (; i < width * height; ++i) {
      sorted_values[i] = v[i] >> 16;
    }
  }

  // Constructs ordinal output image.
  for (int i = 0; i < width * height; ++i) {
    const uint32_t tuple = v[i];
    const uint8_t y = tuple >> 8 & 0xFF;
    const uint8_t x = tuple & 0xFF;
    ordinal_output(x, y) = i >> kOrdinalShift;
  }
}

// Helper function for the float version of OrdinalTransformStdSortAvx2.
// Unpacks the sorted tuples into `reverse_lut` and `output`.
template <typename ordinal_t>
static void UnpackSortedTuplesAvx2(
    const std::unique_ptr<int64_t[]>& v, WriteViewGray<ordinal_t> output,
    absl::Span<float> sorted_values,
    std::optional<absl::Span<uint16_t>> sorted_coords) {
  constexpr int kOrdinalShift = std::is_same_v<ordinal_t, uint8_t> ? 6 : 0;
  const int width = output.width();
  const int height = output.height();
  // Unpacks into sorted_values.
  int i = 0;
  const __m256i* vsrc_ptr = reinterpret_cast<const __m256i*>(v.get());
  __m256i* vdst_values_ptr = reinterpret_cast<__m256i*>(sorted_values.data());
  const __m256i vpermute = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
  if (sorted_coords.has_value()) {
    __m256i* vdst_coords_ptr =
        reinterpret_cast<__m256i*>(sorted_coords->data());
    for (; i < width * height - 15; i += 16) {
      const __m256i vpermute = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
      __m256i v0 = _mm256_loadu_si256(vsrc_ptr++);
      __m256i v1 = _mm256_loadu_si256(vsrc_ptr++);
      __m256i v2 = _mm256_loadu_si256(vsrc_ptr++);
      __m256i v3 = _mm256_loadu_si256(vsrc_ptr++);
      v0 = _mm256_permutevar8x32_epi32(v0, vpermute);
      v1 = _mm256_permutevar8x32_epi32(v1, vpermute);
      v2 = _mm256_permutevar8x32_epi32(v2, vpermute);
      v3 = _mm256_permutevar8x32_epi32(v3, vpermute);
      __m256 c01 = _mm256_permute2x128_si256(v0, v1, 0x20);
      __m256 c23 = _mm256_permute2x128_si256(v2, v3, 0x20);
      __m256 v01 = _mm256_permute2x128_si256(v0, v1, 0x31);
      __m256 v23 = _mm256_permute2x128_si256(v2, v3, 0x31);
      const __m256i v01_xor = _mm256_srli_epi32(_mm256_srai_epi32(v01, 30), 1);
      const __m256i v23_xor = _mm256_srli_epi32(_mm256_srai_epi32(v23, 30), 1);
      v01 = _mm256_xor_si256(v01, v01_xor);
      v23 = _mm256_xor_si256(v23, v23_xor);
      __m256 c0123 = _mm256_packus_epi32(c01, c23);
      c0123 = _mm256_permute4x64_epi64(c0123, _MM_SHUFFLE(3, 1, 2, 0));
      _mm256_storeu_si256(vdst_values_ptr++, v01);
      _mm256_storeu_si256(vdst_values_ptr++, v23);
      _mm256_storeu_si256(vdst_coords_ptr++, c0123);
    }
    // Tail.
    for (; i < width * height; ++i) {
      const int64_t tuple = v[i];
      int32_t float_as_int = static_cast<int32_t>(tuple >> 32);
      float_as_int ^= float_as_int >> 31 & 0x7FFFFFFF;
      *reinterpret_cast<int32_t*>(&sorted_values[i]) = float_as_int;
      (*sorted_coords)[i] = tuple & 0xFFFF;
    }
  } else {
    for (; i < width * height - 15; i += 16) {
      __m256i v0 = _mm256_loadu_si256(vsrc_ptr++);
      __m256i v1 = _mm256_loadu_si256(vsrc_ptr++);
      __m256i v2 = _mm256_loadu_si256(vsrc_ptr++);
      __m256i v3 = _mm256_loadu_si256(vsrc_ptr++);
      v0 = _mm256_srli_epi64(v0, 32);
      v2 = _mm256_srli_epi64(v2, 32);
      __m256i v01 = _mm256_blend_epi32(v1, v0, 0b01010101);
      __m256i v23 = _mm256_blend_epi32(v3, v2, 0b01010101);
      v01 = _mm256_permutevar8x32_epi32(v01, vpermute);
      v23 = _mm256_permutevar8x32_epi32(v23, vpermute);
      const __m256i v01_xor = _mm256_srli_epi32(_mm256_srai_epi32(v01, 30), 1);
      const __m256i v23_xor = _mm256_srli_epi32(_mm256_srai_epi32(v23, 30), 1);
      v01 = _mm256_xor_si256(v01, v01_xor);
      v23 = _mm256_xor_si256(v23, v23_xor);
      _mm256_storeu_si256(vdst_values_ptr++, v01);
      _mm256_storeu_si256(vdst_values_ptr++, v23);
    }
    // Tail.
    for (; i < width * height; ++i) {
      const int64_t tuple = v[i];
      int32_t float_as_int = static_cast<int32_t>(tuple >> 32);
      float_as_int ^= float_as_int >> 31 & 0x7FFFFFFF;
      *reinterpret_cast<int32_t*>(&sorted_values[i]) = float_as_int;
    }
  }
  // Constructs ordinal output image.
  for (int i = 0; i < width * height; ++i) {
    const uint64_t tuple = v[i];
    const uint8_t y = tuple >> 8 & 0xFF;
    const uint8_t x = tuple & 0xFF;
    output(x, y) = i >> kOrdinalShift;
  }
}

// Floating-point radix sort. Image values are bucket-sorted by the top 16 bits,
// then the remaining 16 bits are sorted 8 bits at a time by subroutines. Fully
// sorted pixel coordinates are placed as packed {uint8, uint8} into
// `sorted_coords_ptr`, which must be pre-allocated with as many elements as the
// input image.
void RadixSortFAvx2(ReadViewGrayF cardinal_input,
                    absl::Span<float> sorted_values,
                    absl::Span<uint16_t> sorted_coords) {
  const int width = cardinal_input.width();
  const int height = cardinal_input.height();
  CHECK_LE(width, 256);
  CHECK_LE(height, 256);
  CHECK_EQ(sorted_coords.size(), width * height);

  // Allocates scratch space for mid-level radix sort. This memory is used
  // sparsely and doesn't need to be pre-zeroed.
  auto scratch32 = std::make_unique<uint32_t[]>(width * height);

  // Allocates a zeroed 16-bit interleaved histogram. The SIMD type is used to
  // ensure vector alignment.
  std::vector<__m256i> hist_v(65536 / 16);
  uint16_t* hist = reinterpret_cast<uint16_t*>(hist_v.data());

  // Pre-scans the image to construct a histogram of the top 16 bits.
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      uint32_t val = *reinterpret_cast<const int32_t*>(&cardinal_input(x, y));
      val ^= 0x80000000 | (static_cast<int32_t>(val) >> 31);
      const int top_bits = val >> 16;
      hist[HistIndex16(top_bits)]++;
    }
  }

  // Edge case: if the input is 256x256 pixels and contains a uniform value in
  // the most significant 16 bits, the histogram will overflow. This is tested
  // for and handled here. Note that even in this case, the low 16 bits may
  // not all match (e.g. if all values are in [128.0 .. 129.0)), so the low bits
  // still need to be sorted.
  if (width * height == 65536) {
    int32_t float_as_int =
        *reinterpret_cast<const int32_t*>(&cardinal_input(0, 0));
    float_as_int ^= 0x80000000 | float_as_int >> 31;
    const uint16_t float_high_bits = static_cast<uint32_t>(float_as_int) >> 16;
    if (hist[HistIndex16(float_high_bits)] ==
        0) {  // Incremented 2^16 times and overflowed!
      // Constructs an array of 32-bit tuples, with the low float bits + coords.
      auto tuples = std::make_unique<uint32_t[]>(width * height);
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          int32_t float_as_int =
              *reinterpret_cast<const int32_t*>(&cardinal_input(x, y));
          // No need to flip the sign bit, since we're only using the low bits.
          float_as_int ^= float_as_int >> 31;
          tuples[y * width + x] = float_as_int << 16 | y << 8 | x;
        }
      }
      // Sorts the tuples in-place.
      std::sort(tuples.get(), tuples.get() + width * height);

      // Unpacks the sorted tuples into sorted_values (re-appending the high
      // bits) and sorted_coords.
      for (int ind = 0; ind < width * height; ++ind) {
        uint32_t tuple = tuples[ind];
        const int float_low_bits = tuple >> 16;
        int32_t float_as_int = float_high_bits << 16 | float_low_bits;
        float_as_int ^= 0x80000000;                       // Restores sign bit.
        float_as_int ^= float_as_int >> 31 & 0x7FFFFFFF;  // Restores low bits.
        *reinterpret_cast<int32_t*>(&sorted_values[ind]) = float_as_int;
        (sorted_coords)[ind] = tuple & 0xFFFF;
      }
      return;
    }
  }

  // Constructs prefix sum.
  PrefixSum16x65536Avx2(hist_v.data());

  // Bucket-sorts the top 16 bits. Note: incrementing the prefix-sum values has
  // the overall effect of "sliding" the prefix sum by one index, such that in
  // the subsequent pass the prefix sum begins at hist[0] instead of hist[1].
  auto tuples32 = std::make_unique<uint32_t[]>(width * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      uint32_t val = *reinterpret_cast<const int32_t*>(&cardinal_input(x, y));
      val ^= 0x80000000 | (static_cast<int32_t>(val) >> 31);
      const int top_bits = val >> 16;
      tuples32[hist[HistIndex16(top_bits)]++] = val << 16 | y << 8 | x;
    }
  }

  // Restricts scan to buckets that are actually used.
  uint16_t ind_min = 0;
  uint16_t ind_max = 0xFFFF;

  {
    uint32_t val = *reinterpret_cast<const int32_t*>(&cardinal_input(0, 0));
    val ^= 0x80000000 | static_cast<int32_t>(val) >> 31;
    const uint16_t pivot = val >> 16;
    for (int step = 32768; step > 0; step /= 2) {
      if (ind_min + step < pivot) {
        if (hist[HistIndex16(ind_min + step)] == 0) {
          ind_min += step;
        }
      }
      if (ind_max - step > pivot) {
        if (hist[HistIndex16(ind_max - step)] == hist[0xFFFF]) {
          ind_max -= step;
        }
      }
    }
  }

  // Sorts the individual buckets.
  uint16_t begin = 0, end = 0;
  for (int i = ind_min; i <= ind_max; ++i) {
    begin = end;
    end = hist[HistIndex16(i)];
    if (end == begin) continue;
    const int32_t float_high_bits = (i ^ 0x8000) << 16;
    const uint16_t count = end - begin;
    if (count <= internal::kSortThreshold32Avx2) {
      if (count == 1) {
        sorted_coords[begin] = tuples32[begin] & 0xFFFF;
        *reinterpret_cast<int32_t*>(&sorted_values[begin]) =
            float_high_bits | (tuples32[begin] >> 16);
      } else {
        std::sort(&tuples32[begin], &tuples32[begin] + count);
        for (int j = 0; j < count; ++j) {
          *reinterpret_cast<int32_t*>(&sorted_values[begin + j]) =
              float_high_bits | (tuples32[begin + j] >> 16);
          sorted_coords[begin + j] = tuples32[begin + j] & 0xFFFF;
        }
      }
    } else {
      if (count < 256) {
        internal::RadixSort32<uint8_t>(
            absl::MakeConstSpan(&tuples32[begin], count), float_high_bits,
            absl::MakeSpan(&scratch32[0], count),
            sorted_values.subspan(begin, count),
            sorted_coords.subspan(begin, count));
      } else {
        internal::RadixSort32<uint16_t>(
            absl::MakeConstSpan(&tuples32[begin], count), float_high_bits,
            absl::MakeSpan(&scratch32[0], count),
            sorted_values.subspan(begin, count),
            sorted_coords.subspan(begin, count));
      }
    }
  }

  // Un-twiddles the exponent and mantissa bits for negative values. The values
  // are already sorted, so as soon as we find a positive value we can break.
  // TODO: Vectorize, and binary search for the first positive value.
  // May be diminishing returns, since this is an unlikely edge case for photos.
  for (int i = 0; i < width * height; ++i) {
    int32_t float_as_int = *reinterpret_cast<int32_t*>(&sorted_values[i]);
    if (float_as_int >= 0) {
      break;
    }
    float_as_int ^= 0x7FFFFFFF;
    *reinterpret_cast<int32_t*>(&sorted_values[i]) = float_as_int;
  }
}
template <typename T, typename ordinal_t>
void OrdinalTransformStdSortAvx2(
    ReadViewGray<T> cardinal_input, WriteViewGray<ordinal_t> ordinal_output,
    absl::Span<T>& sorted_values,
    std::optional<absl::Span<uint16_t>> sorted_coords) {
  const int width = ordinal_output.width();
  const int height = ordinal_output.height();
  CHECK_LE(width, 256);
  CHECK_LE(height, 256);
  CHECK_EQ(cardinal_input.width(), width);
  CHECK_EQ(cardinal_input.height(), height);
  CHECK_GE(sorted_values.size(), width * height);
  // Constructs a vector of (value, y, x) tuples, encoded as sortable int.
  // Concatenating the y and x coordinates yields a tuple that can be sorted
  // in place, while also providing uniqueness and a well-defined ordering
  // (for duplicate cardinal input values), avoiding the need for stable_sort.
  auto tuples = PackTuplesAvx2(cardinal_input);

  // Sorts the tuples in-place.
  std::sort(tuples.get(), tuples.get() + width * height);

  // Unpacks the sorted tuples into reverse_lut and output.
  UnpackSortedTuplesAvx2(tuples, ordinal_output, sorted_values, sorted_coords);
}
}  // namespace

namespace internal {
template <typename ordinal_t>
void OrdinalTransformAvx2Impl(ReadViewGrayU16 cardinal_input,
                              WriteViewGray<ordinal_t> ordinal_output,
                              absl::Span<uint16_t> sorted_values,
                              std::optional<absl::Span<uint16_t>> sorted_coords,
                              const OrdinalTransformOptions& options) {
  constexpr int kOrdinalShift = std::is_same_v<ordinal_t, uint8_t> ? 6 : 0;
  constexpr int kMaxTileSize = 256;
  constexpr int kMaxTilePixels =
      std::is_same_v<ordinal_t, uint8_t> ? 16384 : 65536;
  const int width = ordinal_output.width();
  const int height = ordinal_output.height();
  // Note: a pixel count of exactly 65536 will overflow the 16-bit histogram
  // if the image is a solid color, but the end result is still correct.
  CHECK_LE(width, kMaxTileSize);
  CHECK_LE(height, kMaxTileSize);
  CHECK_LE(width * height, kMaxTilePixels);
  CHECK_EQ(cardinal_input.width(), width);
  CHECK_EQ(cardinal_input.height(), height);
  CHECK_EQ(sorted_values.size(), width * height);
  if (sorted_coords.has_value()) {
    CHECK_EQ(sorted_coords->size(), width * height);
  }

  const int sort_threshold = options.sort_threshold == kSortThresholdDefault
                                 ? kSortThresholdU16Avx2
                                 : options.sort_threshold;

  // For small tiles, uses std::sort.
  if (width * height < sort_threshold) {
    OrdinalTransformStdSortAvx2(cardinal_input, ordinal_output, sorted_values,
                                sorted_coords);
    return;
  }

  // Otherwise, uses an interleaved histogram.
  std::vector<__m256i> hist_v(65536 / 16);
  memset(hist_v.data(), 0, 65536 * sizeof(uint16_t));
  uint16_t* hist_ptr = reinterpret_cast<uint16_t*>(hist_v.data());
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      hist_ptr[HistIndex16(cardinal_input(col, row))]++;
    }
  }

  // Constructs prefix sum.
  PrefixSum16x65536Avx2(hist_v.data());

  // Cardinal->Ordinal mapping, and construction of reverse map.
  for (int row = 0; row < height; ++row) {
    if (sorted_coords.has_value()) {
      uint16_t* sorted_coords_ptr = sorted_coords->data();
      for (int col = 0; col < width; ++col) {
        const uint16_t val = cardinal_input(col, row);
        const uint16_t val_ind = HistIndex16(val);
        const uint16_t hist_val = hist_ptr[val_ind]++;
        ordinal_output(col, row) = hist_val >> kOrdinalShift;
        sorted_values[hist_val] = val;
        sorted_coords_ptr[hist_val] = row << 8 | col;
      }
    } else {
      for (int col = 0; col < width; ++col) {
        const uint16_t val = cardinal_input(col, row);
        const uint16_t val_ind = HistIndex16(val);
        const uint16_t hist_val = hist_ptr[val_ind]++;
        ordinal_output(col, row) = hist_val >> kOrdinalShift;
        sorted_values[hist_val] = val;
      }
    }
  }
}

template <typename ordinal_t>
void OrdinalTransformAvx2Impl(ReadViewGrayF cardinal_input,
                              WriteViewGray<ordinal_t> ordinal_output,
                              absl::Span<float> sorted_values,
                              std::optional<absl::Span<uint16_t>> sorted_coords,
                              const OrdinalTransformOptions& options) {
  constexpr int kOrdinalShift = std::is_same_v<ordinal_t, uint8_t> ? 6 : 0;
  constexpr int kMaxTileSize = 256;
  constexpr int kMaxTilePixels =
      std::is_same_v<ordinal_t, uint8_t> ? 16384 : 65536;
  const int width = ordinal_output.width();
  const int height = ordinal_output.height();
  CHECK_LE(width, kMaxTileSize);
  CHECK_LE(height, kMaxTileSize);
  CHECK_LE(width * height, kMaxTilePixels);
  CHECK_EQ(cardinal_input.width(), width);
  CHECK_EQ(cardinal_input.height(), height);
  CHECK_GE(sorted_values.size(), width * height);
  if (sorted_coords.has_value()) {
    CHECK_GE(sorted_coords->size(), width * height);
  }

  const int sort_threshold = options.sort_threshold == kSortThresholdDefault
                                 ? kSortThresholdFAvx2
                                 : options.sort_threshold;
  // For small tiles, uses std::sort.
  if (width * height < sort_threshold) {
    OrdinalTransformStdSortAvx2(cardinal_input, ordinal_output, sorted_values,
                                sorted_coords);
    return;
  }

  // Otherwise, uses radix sort. Note: `sorted_coords_ptr` doesn't need to be
  // pre-zeroed.
  absl::Span<uint16_t> sorted_coords_span;
  std::unique_ptr<uint16_t[]> sorted_coords_ptr;
  if (sorted_coords.has_value()) {
    sorted_coords_span = *sorted_coords;
    RadixSortFAvx2(cardinal_input, sorted_values, *sorted_coords);
  } else {
    sorted_coords_ptr = std::make_unique<uint16_t[]>(width * height);
    sorted_coords_span =
        absl::Span<uint16_t>(sorted_coords_ptr.get(), width * height);
    RadixSortFAvx2(cardinal_input, sorted_values, sorted_coords_span);
  }

  // Constructs ordinal output image.
  for (int i = 0; i < width * height; ++i) {
    const uint16_t coords = sorted_coords_span[i];
    const uint8_t y = coords >> 8;
    const uint8_t x = coords & 0xFF;
    ordinal_output(x, y) = i >> kOrdinalShift;
  }
}
void OrdinalTransformAvx2(ReadViewGrayU16 cardinal_input,
                          WriteViewGrayU16 ordinal_output,
                          absl::Span<uint16_t> sorted_values,
                          std::optional<absl::Span<uint16_t>> sorted_coords,
                          const OrdinalTransformOptions& options) {
  OrdinalTransformAvx2Impl(cardinal_input, ordinal_output, sorted_values,
                           sorted_coords, options);
}
void OrdinalTransformAvx2(ReadViewGrayU16 cardinal_input,
                          WriteViewGrayU8 ordinal_output,
                          absl::Span<uint16_t> sorted_values,
                          absl::Span<uint16_t> sorted_coords,
                          const OrdinalTransformOptions& options) {
  OrdinalTransformAvx2Impl(cardinal_input, ordinal_output, sorted_values,
                           sorted_coords, options);
}
void OrdinalTransformAvx2(ReadViewGrayF cardinal_input,
                          WriteViewGrayU16 ordinal_output,
                          absl::Span<float> sorted_values,
                          std::optional<absl::Span<uint16_t>> sorted_coords,
                          const OrdinalTransformOptions& options) {
  OrdinalTransformAvx2Impl(cardinal_input, ordinal_output, sorted_values,
                           sorted_coords, options);
}
void OrdinalTransformAvx2(ReadViewGrayF cardinal_input,
                          WriteViewGrayU8 ordinal_output,
                          absl::Span<float> sorted_values,
                          absl::Span<uint16_t> sorted_coords,
                          const OrdinalTransformOptions& options) {
  OrdinalTransformAvx2Impl(cardinal_input, ordinal_output, sorted_values,
                           sorted_coords, options);
}
}  // namespace internal
}  // namespace fast_isotropic_median_filter
#endif  // __AVX2__
