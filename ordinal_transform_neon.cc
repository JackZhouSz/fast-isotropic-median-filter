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
#include "ordinal_transform.h"

#include <arm_neon.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "image.h"

namespace fast_isotropic_median_filter {
namespace {
// Computes the prefix sum of a `length`-element 16-bit histogram, in-place.
// The instruction sequence is manually "pipelined" to maximize throughput.
// The staggering is necessary because each result element is partly dependent
// on the previous one.
static inline void PrefixSumNeon(uint16_t* hist, int length) {
  int ind = 0;
  DCHECK_GE(length, 32);
  DCHECK_EQ(length % 32, 0);
#define LD8(a, offset) a = vld1q_u16(&hist[ind + offset]);
#define SL1(a, b) a = vextq_u16(v0, b, 8 - 1);  // Shift left 1 element.
#define SL2(a, b) a = vextq_u16(v0, b, 8 - 2);  // Shift left 2 elements.
#define SL4(a, b) a = vextq_u16(v0, b, 8 - 4);  // Shift left 4 elements.
#define ADD(a, b) a = vaddq_u16(a, b);
#define ST8(a, offset) vst1q_u16(&hist[ind + offset], a);
#define DP7(a) a = vdupq_laneq_u16(a, 7);
  const uint16x8_t v0 = vdupq_n_u16(0);
  uint16x8_t sm = v0;
  uint16x8_t va, xa, vb, xb, vc, xc, vd, xd;
  // clang-format off
    LD8(va, 0);
    SL1(xa, va);
    ADD(va, xa);    LD8(vb, 8);
    SL2(xa, va);    SL1(xb, vb);
    ADD(va, xa);    ADD(vb, xb);    LD8(vc, 16);
    SL4(xa, va);    SL2(xb, vb);    SL1(xc, vc);
    ADD(va, xa);    ADD(vb, xb);    ADD(vc, xc);    LD8(vd, 24);
    ADD(sm, va);    SL4(xb, vb);    SL2(xc, vc);    SL1(xd, vd);
    ST8(sm, 0);     ADD(vb, xb);    ADD(vc, xc);    ADD(vd, xd);
    DP7(sm);        ADD(sm, vb);    SL4(xc, vc);    SL2(xd, vd);
  for (; ind < length - 32; ind += 32) {
    LD8(va, 32);    ST8(sm, 8);     ADD(vc, xc);    ADD(vd, xd);
    SL1(xa, va);    DP7(sm);        ADD(sm, vc);    SL4(xd, vd);
    ADD(va, xa);    LD8(vb, 40);    ST8(sm, 16);    ADD(vd, xd);
    SL2(xa, va);    SL1(xb, vb);    DP7(sm);        ADD(sm, vd);
    ADD(va, xa);    ADD(vb, xb);    LD8(vc, 48);    ST8(sm, 24);
    SL4(xa, va);    SL2(xb, vb);    SL1(xc, vc);    DP7(sm);
    ADD(va, xa);    ADD(vb, xb);    ADD(vc, xc);    LD8(vd, 56);
    ADD(sm, va);    SL4(xb, vb);    SL2(xc, vc);    SL1(xd, vd);
    ST8(sm, 32);    ADD(vb, xb);    ADD(vc, xc);    ADD(vd, xd);
    DP7(sm);        ADD(sm, vb);    SL4(xc, vc);    SL2(xd, vd);
  }
                    ST8(sm, 8);     ADD(vc, xc);    ADD(vd, xd);
                    DP7(sm);        ADD(sm, vc);    SL4(xd, vd);
                                    ST8(sm, 16);    ADD(vd, xd);
                                    DP7(sm);        ADD(sm, vd);
                                                    ST8(sm, 24);
                                                    DP7(sm);
  // clang-format on
#undef LD8
#undef SL1
#undef SL2
#undef SL4
#undef ADD
#undef ST8
#undef DP7

  // Non-pipelined loop, for reference (this runs ~10% slower):
  //
  // uint16x8_t vsum = v0;
  // for (int ind = 8; ind < 65536 + 8; ind += 8) {
  //   uint16x8_t v = vld1q_u16(&hist[ind]);
  //   v += vextq_u16(v0, v, 7);
  //   v += vextq_u16(v0, v, 6);
  //   v += vextq_u16(v0, v, 4);
  //   vsum += v;
  //   vst1q_u16(&hist[ind], vsum);
  //   vsum = vdupq_laneq_u16(vsum, 7);
  // }
}

// Helper function for the uint16 version of OrdinalTransformStdSortNeon.
// Appends the (x, y) position of each pixel to the pixel value, and returns a
// sortable array of uint32's.
static inline std::unique_ptr<uint32_t[]> PackTuplesNeon(
    ReadViewGrayU16 cardinal_input) {
  const int width = cardinal_input.width();
  const int height = cardinal_input.height();
  auto tuples = std::make_unique<uint32_t[]>(width * height);

  const uint16x8_t dy = vdupq_n_u16(1 << 8);
  uint16x8_t y_vec = {0, 1, 2, 3, 4, 5, 6, 7};
  for (int y = 0; y < height; ++y) {
    uint16x8x2_t res;
    res.val[0] = y_vec;
    const uint16_t* vsrc_ptr = &cardinal_input(0, y);
    uint16_t* vdst_ptr = reinterpret_cast<uint16_t*>(&tuples[width * y]);
    int x = 0;
    for (x = 0; x < width - 7; x += 8) {
      res.val[1] = vld1q_u16(vsrc_ptr);
      vsrc_ptr += 8;
      vst2q_u16(vdst_ptr, res);  // Stores interleaved.
      vdst_ptr += 16;
      res.val[0] += vdupq_n_u16(8);
    }
    // Tail.
    uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(vdst_ptr);
    const uint16_t* row_ptr = &cardinal_input(0, y);
    for (; x < width; ++x) {
      *dst_ptr++ = static_cast<int64_t>(row_ptr[x]) << 16 | y << 8 | x;
    }
    y_vec += dy;
  }
  return tuples;
}

// Helper function for the float version of OrdinalTransformStdSortNeon.
// Appends the (x, y) position of each pixel to the adjusted pixel value, and
// returns a sortable array of int64's.
static std::unique_ptr<int64_t[]> PackTuplesNeon(ReadViewGrayF cardinal_input) {
  const int width = cardinal_input.width();
  const int height = cardinal_input.height();
  auto tuples = std::make_unique<int64_t[]>(width * height);

  const uint32x4_t vmask = vdupq_n_u32(0x7FFFFFFF);
  const uint32x4_t dy = vdupq_n_u32(1 << 8);
  uint32x4_t y_vec = {0, 1, 2, 3};
  for (int y = 0; y < height; ++y) {
    uint32x4x2_t res;
    res.val[0] = y_vec;
    const uint32_t* vsrc_ptr =
        reinterpret_cast<const uint32_t*>(&cardinal_input(0, y));
    uint32_t* vdst_ptr = reinterpret_cast<uint32_t*>(&tuples[width * y]);
    int x = 0;
    for (x = 0; x < width - 3; x += 4) {
      res.val[1] = vld1q_u32(vsrc_ptr);
      vsrc_ptr += 4;
      res.val[1] ^= vshrq_n_s32(res.val[1], 31) & vmask;
      vst2q_u32(vdst_ptr, res);  // Stores interleaved.
      vdst_ptr += 8;
      res.val[0] += vdupq_n_u32(4);
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
    y_vec += dy;
  }
  return tuples;
}

// Helper function for the uint16 version of OrdinalTransformStdSortNeon.
// Unpacks the sorted tuples into `reverse_lut` and `output`.
template <typename ordinal_t>
static inline void UnpackSortedTuplesNeon(
    const std::unique_ptr<uint32_t[]>& tuples,
    WriteViewGray<ordinal_t> ordinal_output, absl::Span<uint16_t> sorted_values,
    std::optional<absl::Span<uint16_t>> sorted_coords) {
  constexpr int kOrdinalShift = std::is_same_v<ordinal_t, uint8_t> ? 6 : 0;
  const int width = ordinal_output.width();
  const int height = ordinal_output.height();
  // Unpacks into sorted_values.
  const uint16_t* src_ptr = reinterpret_cast<const uint16_t*>(tuples.get());
  uint16_t* dst_values_ptr = sorted_values.data();
  int i = 0;
  if (sorted_coords.has_value()) {
    uint16_t* dst_coords_ptr = sorted_coords->data();
    for (i = 0; i < width * height - 15; i += 16) {
      uint16x8x2_t v01, v23;
      v01 = vld2q_u16(src_ptr);
      v23 = vld2q_u16(src_ptr + 16);
      src_ptr += 32;
      vst1q_u16(dst_coords_ptr, v01.val[0]);
      vst1q_u16(dst_coords_ptr + 8, v23.val[0]);
      vst1q_u16(dst_values_ptr, v01.val[1]);
      vst1q_u16(dst_values_ptr + 8, v23.val[1]);
      dst_coords_ptr += 16;
      dst_values_ptr += 16;
    }
    // Tail.
    for (; i < width * height; ++i) {
      const uint32_t tuple = tuples[i];
      sorted_values[i] = tuple >> 16;
      (*sorted_coords)[i] = tuple & 0xFFFF;
    }
  } else {
    for (i = 0; i < width * height - 15; i += 16) {
      uint16x8x2_t v01, v23;
      v01 = vld2q_u16(src_ptr);
      v23 = vld2q_u16(src_ptr + 16);
      src_ptr += 32;
      vst1q_u16(dst_values_ptr, v01.val[1]);
      vst1q_u16(dst_values_ptr + 8, v23.val[1]);
      dst_values_ptr += 16;
    }
    // Tail.
    for (; i < width * height; ++i) {
      sorted_values[i] = tuples[i] >> 16;
    }
  }

  // Unpacks into output.
  for (int i = 0; i < width * height; ++i) {
    const uint32_t tuple = tuples[i];
    const int y = tuple >> 8 & 0xFF;
    const int x = tuple & 0xFF;
    ordinal_output(x, y) = i >> kOrdinalShift;
  }
}

// Helper function for the float version of OrdinalTransformStdSortNeon.
// Unpacks sorted tuples into `sorted_values`, `sorted_coords`, and `output`.
template <typename ordinal_t>
static inline void UnpackSortedTuplesNeon(
    const std::unique_ptr<int64_t[]>& v,
    WriteViewGray<ordinal_t> ordinal_output, absl::Span<float> sorted_values,
    std::optional<absl::Span<uint16_t>> sorted_coords) {
  constexpr int kOrdinalShift = std::is_same_v<ordinal_t, uint8_t> ? 6 : 0;
  const int width = ordinal_output.width();
  const int height = ordinal_output.height();
  // Unpacks into sorted_values.
  const uint32_t* src_ptr = reinterpret_cast<const uint32_t*>(v.get());
  uint32_t* dst_values_ptr = reinterpret_cast<uint32_t*>(sorted_values.data());
  const uint32x4_t vmask = vdupq_n_u32(0x7FFFFFFF);
  int i = 0;
  if (sorted_coords.has_value()) {
    uint16_t* dst_coords_ptr = sorted_coords->data();
    for (; i < width * height - 7; i += 8) {
      uint32x4x2_t v01, v23;
      v01 = vld2q_u32(src_ptr);
      v23 = vld2q_u32(src_ptr + 8);
      src_ptr += 16;
      v01.val[1] ^= vshrq_n_s32(v01.val[1], 31) & vmask;
      v23.val[1] ^= vshrq_n_s32(v23.val[1], 31) & vmask;
      vst1q_u16(dst_coords_ptr, vuzp1q_u16(v01.val[0], v23.val[0]));
      vst1q_u32(dst_values_ptr, v01.val[1]);
      vst1q_u32(dst_values_ptr + 4, v23.val[1]);
      dst_coords_ptr += 8;
      dst_values_ptr += 8;
    }
    // Tail.
    for (; i < width * height; ++i) {
      const int64_t tuple = v[i];
      (*sorted_coords)[i] = tuple & 0xFFFF;
      int32_t float_as_int = static_cast<int32_t>(tuple >> 32);
      float_as_int ^= float_as_int >> 31 & 0x7FFFFFFF;
      *reinterpret_cast<int32_t*>(&sorted_values[i]) = float_as_int;
    }
  } else {
    for (; i < width * height - 7; i += 8) {
      uint32x4x2_t v01, v23;
      v01 = vld2q_u32(src_ptr);
      v23 = vld2q_u32(src_ptr + 8);
      src_ptr += 16;
      v01.val[1] ^= vshrq_n_s32(v01.val[1], 31) & vmask;
      v23.val[1] ^= vshrq_n_s32(v23.val[1], 31) & vmask;
      vst1q_u32(dst_values_ptr, v01.val[1]);
      vst1q_u32(dst_values_ptr + 4, v23.val[1]);
      dst_values_ptr += 8;
    }
    // Tail.
    for (; i < width * height; ++i) {
      const int64_t tuple = v[i];
      int32_t float_as_int = static_cast<int32_t>(tuple >> 32);
      float_as_int ^= float_as_int >> 31 & 0x7FFFFFFF;
      *reinterpret_cast<int32_t*>(&sorted_values[i]) = float_as_int;
    }
  }

  // Unpacks into ordinal output.
  for (int i = 0; i < width * height; ++i) {
    const int64_t tuple = v[i];
    const int y = tuple >> 8 & 0xFF;
    const int x = tuple & 0xFF;
    ordinal_output(x, y) = i >> kOrdinalShift;
  }
}

// Floating-point radix sort. Image values are bucket-sorted by the top 16 bits,
// then the remaining 16 bits are sorted 8 bits at a time by subroutines. Fully
// sorted pixel coordinates are placed as packed {uint8, uint8} into
// `sorted_coords_ptr`, which must be pre-allocated with as many elements as the
// input image.
void RadixSortFNeon(ReadViewGrayF cardinal_input,
                    absl::Span<float> sorted_values,
                    absl::Span<uint16_t> sorted_coords) {
  const int width = cardinal_input.width();
  const int height = cardinal_input.height();
  CHECK_LE(width, 256);
  CHECK_LE(height, 256);
  CHECK_EQ(sorted_coords.size(), width * height);

  // Allocates a zeroed 16-bit interleaved histogram. The SIMD type is used to
  // ensure vector alignment.
  std::vector<uint16x8_t> hist_v(65536 / 8 + 1);
  uint16_t* hist = reinterpret_cast<uint16_t*>(&hist_v[1]);

  // Constructs a histogram of the top 16 bits of the float bit-pattern,
  // bit-twiddled to enable sorting as int.
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      uint32_t float_as_int =
          *reinterpret_cast<const int32_t*>(&cardinal_input(x, y));
      float_as_int ^= 0x80000000 | (static_cast<int32_t>(float_as_int) >> 31);
      const int float_top_bits = float_as_int >> 16;
      hist[float_top_bits]++;
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
    if (hist[float_high_bits] == 0) {  // Incremented 2^16 times and overflowed!
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

  // Constructs prefix sum. The prefix sum is constructed as { v0, v0 + v1,
  // ... }, but the loop below requires { 0, v0, v0 + v1, ... }, so the pointer
  // is decremented after prefix-summing.
  PrefixSumNeon(hist, /*length=*/65536);
  --hist;

  // Bucket-sorts the top 16 bits. Note: incrementing the prefix-sum values has
  // the overall effect of "sliding" the prefix sum by one index, such that in
  // the subsequent pass the prefix sum begins at hist[0] instead of hist[1].
  auto tuples32 = std::make_unique<uint32_t[]>(width * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      uint32_t val = *reinterpret_cast<const int32_t*>(&cardinal_input(x, y));
      val ^= 0x80000000 | (static_cast<int32_t>(val) >> 31);
      const int top_bits = val >> 16;
      tuples32[hist[top_bits]++] = val << 16 | y << 8 | x;
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
        if (hist[ind_min + step] == 0) ind_min += step;
      }
      if (ind_max - step > pivot) {
        if (hist[ind_max - step] == hist[0xFFFF]) ind_max -= step;
      }
    }
  }

  // Allocates scratch space for mid-level radix sort. This memory is used
  // sparsely and doesn't need to be pre-zeroed.
  auto tuples24 = std::make_unique<uint32_t[]>(width * height);

  // Sorts the individual buckets.
  uint16_t begin = 0, end = 0;
  for (int i = ind_min; i <= ind_max; ++i) {
    begin = end;
    end = hist[i];
    if (end == begin) continue;
    const int32_t float_high_bits = (i ^ 0x8000) << 16;
    const uint16_t count = end - begin;
    if (count <= internal::kSortThreshold32Neon) {
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
        internal::RadixSort32<uint8_t>(absl::MakeSpan(&tuples32[begin], count),
                                       float_high_bits,
                                       absl::MakeSpan(&tuples24[0], count),
                                       sorted_values.subspan(begin, count),
                                       sorted_coords.subspan(begin, count));
      } else {
        internal::RadixSort32<uint16_t>(absl::MakeSpan(&tuples32[begin], count),
                                        float_high_bits,
                                        absl::MakeSpan(&tuples24[0], count),
                                        sorted_values.subspan(begin, count),
                                        sorted_coords.subspan(begin, count));
      }
    }
  }

  // Un-twiddles the exponent and mantissa bits for negative values. The values
  // are already sorted, so as soon as we find a positive value we can break.
  // TODO: Vectorize this, and smartly find the first positive value.
  // (Though there may be diminishing returns; this is an unlikely edge case.)
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
void OrdinalTransformStdSortNeon(
    ReadViewGray<T> cardinal_input, WriteViewGray<ordinal_t> ordinal_output,
    absl::Span<T> sorted_values,
    std::optional<absl::Span<uint16_t>> sorted_coords) {
  const int width = ordinal_output.width();
  const int height = ordinal_output.height();
  CHECK_EQ(cardinal_input.width(), width);
  CHECK_EQ(cardinal_input.height(), height);
  CHECK_GE(sorted_values.size(), width * height);
  if (sorted_coords.has_value()) {
    CHECK_GE(sorted_coords->size(), width * height);
  }

  // Constructs a vector of (value, y, x) tuples, encoded as sortable int64.
  auto tuples = PackTuplesNeon(cardinal_input);

  // Sorts the tuples in-place.
  std::sort(&tuples[0], &tuples[width * height]);

  // Unpacks the sorted tuples into sorted_values and output.
  UnpackSortedTuplesNeon(tuples, ordinal_output, sorted_values, sorted_coords);
}
}  // namespace

namespace internal {
template <typename ordinal_t>
void OrdinalTransformNeonImpl(ReadViewGrayU16 cardinal_input,
                              WriteViewGray<ordinal_t> ordinal_output,
                              absl::Span<uint16_t> sorted_values,
                              std::optional<absl::Span<uint16_t>> sorted_coords,
                              const OrdinalTransformOptions& options) {
  constexpr int kOrdinalShift = std::is_same_v<ordinal_t, uint8_t> ? 6 : 0;
  const int width = ordinal_output.width();
  const int height = ordinal_output.height();
  // Note: a pixel count of exactly 65536 will overflow the 16-bit histogram
  // if the image is a solid color, but the end result is still correct.
  CHECK_LE(width, 256);
  CHECK_LE(height, 256);
  CHECK_EQ(cardinal_input.width(), width);
  CHECK_EQ(cardinal_input.height(), height);
  CHECK_EQ(sorted_values.size(), width * height);
  if (sorted_coords.has_value()) {
    CHECK_EQ(sorted_coords->size(), width * height);
  }

  const int sort_threshold = options.sort_threshold == kSortThresholdDefault
                                 ? kSortThresholdU16Neon
                                 : options.sort_threshold;

  // For small tiles, uses std::sort.
  if (width * height < sort_threshold && width <= 256 && height <= 256) {
    OrdinalTransformStdSortNeon(cardinal_input, ordinal_output, sorted_values,
                                sorted_coords);
    return;
  }

  // Otherwise, uses a histogram. An extra zero vector is appended to the
  // beginning.
  std::vector<uint16x8_t> hist(65536 / 8 + 1);
  uint16_t* hist_ptr = reinterpret_cast<uint16_t*>(&hist[1]);

  // Constructs histogram.
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const int val = cardinal_input(col, row);
      hist_ptr[val]++;
    }
  }
  PrefixSumNeon(hist_ptr, /*length=*/65536);

  // The prefix sum is computed above as { v0, v0 + v1, ... }, but the loop
  // below requires { 0, v0, v0 + v1, ... }, so the pointer is decremented.
  --hist_ptr;

  // Cardinal->Ordinal mapping, and construction of reverse map.
  for (int row = 0; row < height; ++row) {
    if (sorted_coords.has_value()) {
      uint16_t* sorted_coords_ptr = sorted_coords->data();
      for (int col = 0; col < width; ++col) {
        const int val = cardinal_input(col, row);
        const int hist_val = hist_ptr[val]++;
        ordinal_output(col, row) = hist_val >> kOrdinalShift;
        sorted_coords_ptr[hist_val] = row << 8 | col;
        sorted_values[hist_val] = val;
      }
    } else {
      for (int col = 0; col < width; ++col) {
        const int val = cardinal_input(col, row);
        const int hist_val = hist_ptr[val]++;
        ordinal_output(col, row) = hist_val >> kOrdinalShift;
        sorted_values[hist_val] = val;
      }
    }
  }
}

template <typename ordinal_t>
void OrdinalTransformNeonImpl(ReadViewGrayF cardinal_input,
                              WriteViewGray<ordinal_t> ordinal_output,
                              absl::Span<float> sorted_values,
                              std::optional<absl::Span<uint16_t>> sorted_coords,
                              const OrdinalTransformOptions& options) {
  constexpr int kOrdinalShift = std::is_same_v<ordinal_t, uint8_t> ? 6 : 0;
  const int width = ordinal_output.width();
  const int height = ordinal_output.height();
  CHECK_LE(width, 256);
  CHECK_LE(height, 256);
  CHECK_EQ(cardinal_input.width(), width);
  CHECK_EQ(cardinal_input.height(), height);
  CHECK_GE(sorted_values.size(), width * height);
  if (sorted_coords.has_value()) {
    CHECK_GE(sorted_coords->size(), width * height);
  }
  const int sort_threshold = options.sort_threshold == kSortThresholdDefault
                                 ? kSortThresholdFNeon
                                 : options.sort_threshold;

  // For small tiles, uses std::sort.
  if (width * height < sort_threshold) {
    OrdinalTransformStdSortNeon(cardinal_input, ordinal_output, sorted_values,
                                sorted_coords);
    return;
  }

  // Otherwise, uses radix sort. Note: `sorted_coords_ptr` doesn't need to be
  // pre-zeroed.
  absl::Span<uint16_t> sorted_coords_span;
  std::unique_ptr<uint16_t[]> sorted_coords_ptr;
  if (sorted_coords.has_value()) {
    sorted_coords_span = *sorted_coords;
    RadixSortFNeon(cardinal_input, sorted_values, *sorted_coords);
  } else {
    sorted_coords_ptr = std::make_unique<uint16_t[]>(width * height);
    sorted_coords_span =
        absl::Span<uint16_t>(sorted_coords_ptr.get(), width * height);
    RadixSortFNeon(cardinal_input, sorted_values, sorted_coords_span);
  }

  // Constructs ordinal output image.
  for (int i = 0; i < width * height; ++i) {
    const uint16_t coords = sorted_coords_span[i];
    const uint8_t y = coords >> 8;
    const uint8_t x = coords & 0xFF;
    ordinal_output(x, y) = i >> kOrdinalShift;
  }
}
void OrdinalTransformNeon(ReadViewGrayU16 cardinal_input,
                          WriteViewGrayU16 ordinal_output,
                          absl::Span<uint16_t> sorted_values,
                          std::optional<absl::Span<uint16_t>> sorted_coords,
                          const OrdinalTransformOptions& options) {
  OrdinalTransformNeonImpl(cardinal_input, ordinal_output, sorted_values,
                           sorted_coords, options);
}
void OrdinalTransformNeon(ReadViewGrayU16 cardinal_input,
                          WriteViewGrayU8 ordinal_output,
                          absl::Span<uint16_t> sorted_values,
                          absl::Span<uint16_t> sorted_coords,
                          const OrdinalTransformOptions& options) {
  OrdinalTransformNeonImpl(cardinal_input, ordinal_output, sorted_values,
                           sorted_coords, options);
}
void OrdinalTransformNeon(ReadViewGrayF cardinal_input,
                          WriteViewGrayU16 ordinal_output,
                          absl::Span<float> sorted_values,
                          std::optional<absl::Span<uint16_t>> sorted_coords,
                          const OrdinalTransformOptions& options) {
  OrdinalTransformNeonImpl(cardinal_input, ordinal_output, sorted_values,
                           sorted_coords, options);
}
void OrdinalTransformNeon(ReadViewGrayF cardinal_input,
                          WriteViewGrayU8 ordinal_output,
                          absl::Span<float> sorted_values,
                          absl::Span<uint16_t> sorted_coords,
                          const OrdinalTransformOptions& options) {
  OrdinalTransformNeonImpl(cardinal_input, ordinal_output, sorted_values,
                           sorted_coords, options);
}
}  // namespace internal
}  // namespace fast_isotropic_median_filter
#endif  // defined(__aarch64__)
