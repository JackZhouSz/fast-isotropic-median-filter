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

#include "ordinal_transform.h"

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
// Floating-point radix sort. Image values are bucket-sorted by the top 16 bits,
// then the remaining 16 bits are sorted 8 bits at a time by subroutines. Fully
// sorted pixel coordinates are placed as packed {uint8, uint8} into
// `sorted_coords`. Sorted values are placed into `sorted_values`.
static void RadixSortF(ReadViewGrayF cardinal_input,
                       absl::Span<float> sorted_values,
                       absl::Span<uint16_t> sorted_coords) {
  const int width = cardinal_input.width();
  const int height = cardinal_input.height();
  CHECK_LE(width, 256);
  CHECK_LE(height, 256);
  CHECK_EQ(sorted_coords.size(), width * height);

  // Allocates a zeroed 16-bit histogram.
  std::vector<uint16_t> hist(65536);

  // Constructs a histogram of the top 16 bits of the float bit-pattern,
  // bit-twiddled to facilitate sorting as int.
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

  // Prefix-sums the histogram to create offsets into the sorted array.
  int sum = 0;
  for (int ind = 0; ind < 65536; ++ind) {
    const int val = hist[ind];
    hist[ind] = sum;
    sum += val;
  }

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
  auto scratch32 = std::make_unique<uint32_t[]>(width * height);

  // Sorts the individual buckets.
  uint16_t begin = 0, end = 0;
  for (int i = ind_min; i <= ind_max; ++i) {
    begin = end;
    end = hist[i];
    if (end == begin) continue;
    const int32_t float_high_bits = (i ^ 0x8000) << 16;
    const uint16_t count = end - begin;
    if (count <= internal::kSortThreshold32) {
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
  // May be diminishing returns, since most images are non-negative.
  for (int i = 0; i < width * height; ++i) {
    int32_t float_as_int = *reinterpret_cast<int32_t*>(&sorted_values[i]);
    if (float_as_int >= 0) {
      break;
    }
    float_as_int ^= 0x7FFFFFFF;
    *reinterpret_cast<int32_t*>(&sorted_values[i]) = float_as_int;
  }
}

template <typename ordinal_t>
static void OrdinalTransformStdSort(
    ReadViewGrayU16 cardinal_input, WriteViewGray<ordinal_t> ordinal_output,
    absl::Span<uint16_t> sorted_values,
    std::optional<absl::Span<uint16_t>> sorted_coords = std::nullopt) {
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
  // Constructs a vector of (value, y, x) tuples, encoded as sortable uint32.
  // Concatenating the y and x coordinates yields a tuple that can be sorted
  // in place, while also providing uniqueness and a well-defined ordering
  // (for non-unique values) that avoids the need for stable_sort.
  auto tuples = std::make_unique<uint32_t[]>(width * height);

  // Constructs the array of tuples.
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      tuples[y * width + x] = cardinal_input(x, y) << 16 | y << 8 | x;
    }
  }

  // Sorts the tuples in-place.
  std::sort(tuples.get(), tuples.get() + width * height);

  // Unpacks into sorted_values, sorted_coords, and output.
  if (sorted_coords.has_value()) {
    for (int i = 0; i < width * height; ++i) {
      const uint32_t tuple = tuples[i];
      sorted_values[i] = tuple >> 16;
      (*sorted_coords)[i] = tuple & 0xFFFF;
      const int y = tuple >> 8 & 0xFF;
      const int x = tuple & 0xFF;
      ordinal_output(x, y) = i >> kOrdinalShift;
    }
  } else {
    for (int i = 0; i < width * height; ++i) {
      const uint32_t tuple = tuples[i];
      const int y = tuple >> 8 & 0xFF;
      const int x = tuple & 0xFF;
      sorted_values[i] = tuple >> 16;
      ordinal_output(x, y) = i >> kOrdinalShift;
    }
  }
}

template <typename ordinal_t>
static void OrdinalTransformStdSort(
    ReadViewGrayF cardinal_input, WriteViewGray<ordinal_t> ordinal_output,
    absl::Span<float> sorted_values,
    std::optional<absl::Span<uint16_t>> sorted_coords = std::nullopt) {
  // TODO: See if we can use other shifts for different tile sizes;
  // e.g. 7 for 181x181 tiles, 5 for 90x90 tiles, 4 for 64x64 tiles.
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

  // Constructs a vector of (value, y, x) tuples, encoded as sortable int64.
  // XOR-ing the exponent and mantissa bits by the sign bit constructs a bit
  // pattern that can be compared as int64, yielding the same sort order as the
  // original float. Concatenating the y and x coordinates yields a tuple that
  // can be sorted in place, while also providing uniqueness and a well-defined
  // ordering (for non-unique float values) that avoids the need for
  // stable_sort.
  auto tuples = std::make_unique<int64_t[]>(width * height);
  for (int y = 0; y < height; ++y) {
    const int32_t* row_ptr =
        reinterpret_cast<const int32_t*>(&cardinal_input(0, y));
    for (int x = 0; x < width; ++x) {
      int32_t float_as_int = row_ptr[x];
      float_as_int ^= (float_as_int >> 31 & 0x7FFFFFFF);
      tuples[y * width + x] =
          static_cast<int64_t>(float_as_int) << 32 | y << 8 | x;
    }
  }

  // Sorts the tuples in-place.
  std::sort(&tuples[0], &tuples[width * height]);

  // Unpacks into sorted_values and output.
  if (sorted_coords.has_value()) {
    for (int i = 0; i < width * height; ++i) {
      const int64_t tuple = tuples[i];
      int32_t float_as_int = static_cast<int32_t>(tuple >> 32);
      float_as_int ^= float_as_int >> 31 & 0x7FFFFFFF;
      *reinterpret_cast<int32_t*>(&sorted_values[i]) = float_as_int;
      (*sorted_coords)[i] = tuple & 0xFFFF;
      const int y = tuple >> 8 & 0xFF;
      const int x = tuple & 0xFF;
      ordinal_output(x, y) = i >> kOrdinalShift;
    }
  } else {
    for (int i = 0; i < width * height; ++i) {
      const int64_t tuple = tuples[i];
      int32_t float_as_int = static_cast<int32_t>(tuple >> 32);
      float_as_int ^= float_as_int >> 31 & 0x7FFFFFFF;
      *reinterpret_cast<int32_t*>(&sorted_values[i]) = float_as_int;
      const int y = tuple >> 8 & 0xFF;
      const int x = tuple & 0xFF;
      ordinal_output(x, y) = i >> kOrdinalShift;
    }
  }
}
}  // namespace

namespace internal {
template <typename ordinal_t>
void OrdinalTransformImpl(ReadViewGrayU8 cardinal_input,
                          WriteViewGray<ordinal_t> ordinal_output,
                          absl::Span<uint8_t> sorted_values,
                          std::optional<absl::Span<uint16_t>> sorted_coords,
                          const OrdinalTransformOptions& options) {
  constexpr int kOrdinalShift = std::is_same_v<ordinal_t, uint8_t> ? 6 : 0;
  constexpr int kMaxTileSize = 256;
  constexpr int kMaxTilePixels =
      std::is_same_v<ordinal_t, uint8_t> ? 16384 : 65536;
  const int width = ordinal_output.width();
  const int height = ordinal_output.height();
  // Note: a pixel count of exactly 65536 will overflow the 16-bit histogram
  // if the image is a solid color, but the end result will still be correct.
  CHECK_LE(width, kMaxTileSize);
  CHECK_LE(height, kMaxTileSize);
  CHECK_LE(width * height, kMaxTilePixels);
  CHECK_EQ(cardinal_input.width(), width);
  CHECK_EQ(cardinal_input.height(), height);
  CHECK_GE(sorted_values.size(), width * height);

  // Initializes a 256-element histogram, padded at the start.
  alignas(sizeof(uint64_t)) uint16_t hist_v[256 + 4] = {0};
  uint16_t* hist = &hist_v[4];

  // Constructs histogram.
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const int val = cardinal_input(col, row);
      hist[val]++;
    }
  }

  // Prefix-sums the histogram to create offsets into the sorted array.
  internal::PrefixSum256(hist);
  hist--;  // Points to an exclusive prefix sum.

  // Cardinal->Ordinal mapping, and construction of reverse map.
  for (int row = 0; row < height; ++row) {
    if (sorted_coords.has_value()) {
      uint16_t* sorted_coords_ptr = sorted_coords->data();
      for (int col = 0; col < width; ++col) {
        const int val = cardinal_input(col, row);
        const int hist_val = hist[val]++;
        ordinal_output(col, row) = hist_val >> kOrdinalShift;
        sorted_coords_ptr[hist_val] = row << 8 | col;
        sorted_values[hist_val] = val;
      }
    } else {
      for (int col = 0; col < width; ++col) {
        const int val = cardinal_input(col, row);
        const int hist_val = hist[val]++;
        ordinal_output(col, row) = hist_val >> kOrdinalShift;
        sorted_values[hist_val] = val;
      }
    }
  }
}

template <typename ordinal_t>
void OrdinalTransformImpl(ReadViewGrayU16 cardinal_input,
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

#ifdef __AVX2__
  if (options.use_simd) {
    if constexpr (std::is_same_v<ordinal_t, uint8_t>) {
      CHECK(sorted_coords.has_value());
      OrdinalTransformAvx2(cardinal_input, ordinal_output, sorted_values,
                           sorted_coords.value(), options);
    } else {
      OrdinalTransformAvx2(cardinal_input, ordinal_output, sorted_values,
                           sorted_coords, options);
    }
    return;
  }
#elif defined(__aarch64__)
  if (options.use_simd) {
    if constexpr (std::is_same_v<ordinal_t, uint8_t>) {
      CHECK(sorted_coords.has_value());
      OrdinalTransformNeon(cardinal_input, ordinal_output, sorted_values,
                           sorted_coords.value(), options);
    } else {
      OrdinalTransformNeon(cardinal_input, ordinal_output, sorted_values,
                           sorted_coords, options);
    }
    return;
  }
#endif

  const int sort_threshold = options.sort_threshold == kSortThresholdDefault
                                 ? kSortThresholdU16
                                 : options.sort_threshold;

  // For small tiles, uses std::sort.
  if (width * height < sort_threshold) {
    OrdinalTransformStdSort(cardinal_input, ordinal_output, sorted_values,
                            sorted_coords);
    return;
  }

  // Otherwise, uses a histogram.
  std::vector<uint16_t> hist(65536);

  // Constructs histogram.
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const int val = cardinal_input(col, row);
      hist[val]++;
    }
  }

  // Constructs prefix sum.
  int sum = 0;
  for (int ind = 0; ind < 65536; ++ind) {
    const int val = hist[ind];
    hist[ind] = sum;
    sum += val;
  }

  // Cardinal->Ordinal mapping, and construction of reverse map.
  //
  // On entry, each element of the prefix-summed histogram contains the first
  // ordinal value associated with that cardinal value. For instance, if the
  // input image contains three instances each of values [0, 1, 2, ..], the
  // prefix-summed histogram would begin [0, 3, 6, ...], so the first instance
  // of cardinal value '2' would map to ordinal value '6'. The bucket is then
  // incremented so that the next instance of '2' will map to '7', and so on.
  // This ensures that each output value is unique.
  for (int row = 0; row < height; ++row) {
    if (sorted_coords.has_value()) {
      for (int col = 0; col < width; ++col) {
        const int val = cardinal_input(col, row);
        const int hist_val = hist[val]++;
        ordinal_output(col, row) = hist_val >> kOrdinalShift;
        (*sorted_coords)[hist_val] = row << 8 | col;
        sorted_values[hist_val] = val;
      }
    } else {
      for (int col = 0; col < width; ++col) {
        const int val = cardinal_input(col, row);
        const int hist_val = hist[val]++;
        ordinal_output(col, row) = hist_val >> kOrdinalShift;
        sorted_values[hist_val] = val;
      }
    }
  }
}

template <typename ordinal_t>
void OrdinalTransformImpl(ReadViewGrayF cardinal_input,
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

#ifdef __AVX2__
  if (options.use_simd) {
    if constexpr (std::is_same_v<ordinal_t, uint8_t>) {
      CHECK(sorted_coords.has_value());
      OrdinalTransformAvx2(cardinal_input, ordinal_output, sorted_values,
                           sorted_coords.value(), options);
    } else {
      OrdinalTransformAvx2(cardinal_input, ordinal_output, sorted_values,
                           sorted_coords, options);
    }
    return;
  }
#elif defined(__aarch64__)
  if (options.use_simd) {
    if constexpr (std::is_same_v<ordinal_t, uint8_t>) {
      CHECK(sorted_coords.has_value());
      OrdinalTransformNeon(cardinal_input, ordinal_output, sorted_values,
                           sorted_coords.value(), options);
    } else {
      OrdinalTransformNeon(cardinal_input, ordinal_output, sorted_values,
                           sorted_coords, options);
    }
    return;
  }
#endif

  const int sort_threshold = options.sort_threshold == kSortThresholdDefault
                                 ? kSortThresholdF
                                 : options.sort_threshold;
  // For small tiles, uses std::sort.
  if (width * height < sort_threshold) {
    OrdinalTransformStdSort(cardinal_input, ordinal_output, sorted_values,
                            sorted_coords);
    return;
  }

  // Otherwise, uses radix sort. Note: `sorted_coords_ptr` doesn't need to be
  // pre-zeroed.
  absl::Span<uint16_t> sorted_coords_span;
  std::unique_ptr<uint16_t[]> sorted_coords_ptr;
  if (sorted_coords.has_value()) {
    sorted_coords_span = *sorted_coords;
    RadixSortF(cardinal_input, sorted_values, *sorted_coords);
  } else {
    sorted_coords_ptr = std::make_unique<uint16_t[]>(width * height);
    sorted_coords_span =
        absl::Span<uint16_t>(sorted_coords_ptr.get(), width * height);
    RadixSortF(cardinal_input, sorted_values, sorted_coords_span);
  }

  // Constructs ordinal output image.
  for (int i = 0; i < width * height; ++i) {
    const uint16_t coords = sorted_coords_span[i];
    const uint8_t y = coords >> 8;
    const uint8_t x = coords & 0xFF;
    ordinal_output(x, y) = i >> kOrdinalShift;
  }
}

void OrdinalTransform(ReadViewGrayU8 cardinal_input,
                      WriteViewGrayU16 ordinal_output,
                      absl::Span<uint8_t> sorted_values,
                      std::optional<absl::Span<uint16_t>> sorted_coords,
                      const OrdinalTransformOptions& options) {
  OrdinalTransformImpl(cardinal_input, ordinal_output, sorted_values,
                       sorted_coords, options);
}

void OrdinalTransform(ReadViewGrayU16 cardinal_input,
                      WriteViewGrayU16 ordinal_output,
                      absl::Span<uint16_t> sorted_values,
                      std::optional<absl::Span<uint16_t>> sorted_coords,
                      const OrdinalTransformOptions& options) {
  OrdinalTransformImpl(cardinal_input, ordinal_output, sorted_values,
                       sorted_coords, options);
}

void OrdinalTransform(ReadViewGrayF cardinal_input,
                      WriteViewGrayU16 ordinal_output,
                      absl::Span<float> sorted_values,
                      std::optional<absl::Span<uint16_t>> sorted_coords,
                      const OrdinalTransformOptions& options) {
  OrdinalTransformImpl(cardinal_input, ordinal_output, sorted_values,
                       sorted_coords, options);
}

void OrdinalTransform(ReadViewGrayU8 cardinal_input,
                      WriteViewGrayU8 ordinal_output,
                      absl::Span<uint8_t> sorted_values,
                      absl::Span<uint16_t> sorted_coords,
                      const OrdinalTransformOptions& options) {
  OrdinalTransformImpl(cardinal_input, ordinal_output, sorted_values,
                       sorted_coords, options);
}

void OrdinalTransform(ReadViewGrayU16 cardinal_input,
                      WriteViewGrayU8 ordinal_output,
                      absl::Span<uint16_t> sorted_values,
                      absl::Span<uint16_t> sorted_coords,
                      const OrdinalTransformOptions& options) {
  OrdinalTransformImpl(cardinal_input, ordinal_output, sorted_values,
                       sorted_coords, options);
}

void OrdinalTransform(ReadViewGrayF cardinal_input,
                      WriteViewGrayU8 ordinal_output,
                      absl::Span<float> sorted_values,
                      absl::Span<uint16_t> sorted_coords,
                      const OrdinalTransformOptions& options) {
  OrdinalTransformImpl(cardinal_input, ordinal_output, sorted_values,
                       sorted_coords, options);
}
}  // namespace internal

std::vector<uint8_t> OrdinalTransform(
    ReadViewGrayU8 cardinal_input, WriteViewGrayU16 ordinal_output,
    std::optional<absl::Span<uint16_t>> sorted_coords) {
  const int width = ordinal_output.width();
  const int height = ordinal_output.height();
  CHECK_LE(width, 256);
  CHECK_LE(height, 256);
  CHECK_EQ(cardinal_input.width(), width);
  CHECK_EQ(cardinal_input.height(), height);
  if (sorted_coords.has_value()) {
    CHECK_EQ(sorted_coords->size(), width * height);
  }

  std::vector<uint8_t> sorted_values(width * height);
  internal::OrdinalTransform(cardinal_input, ordinal_output,
                             absl::MakeSpan(sorted_values), sorted_coords);
  return sorted_values;
}

std::vector<uint16_t> OrdinalTransform(
    ReadViewGrayU16 cardinal_input, WriteViewGrayU16 ordinal_output,
    std::optional<absl::Span<uint16_t>> sorted_coords) {
  const int width = ordinal_output.width();
  const int height = ordinal_output.height();
  CHECK_LE(width, 256);
  CHECK_LE(height, 256);
  CHECK_EQ(cardinal_input.width(), width);
  CHECK_EQ(cardinal_input.height(), height);
  if (sorted_coords.has_value()) {
    CHECK_EQ(sorted_coords->size(), width * height);
  }

  std::vector<uint16_t> sorted_values(width * height);
  internal::OrdinalTransform(cardinal_input, ordinal_output,
                             absl::MakeSpan(sorted_values), sorted_coords);
  return sorted_values;
}

std::vector<float> OrdinalTransform(
    ReadViewGrayF cardinal_input, WriteViewGrayU16 ordinal_output,
    std::optional<absl::Span<uint16_t>> sorted_coords) {
  const int width = ordinal_output.width();
  const int height = ordinal_output.height();
  CHECK_LE(width, 256);
  CHECK_LE(height, 256);
  CHECK_EQ(cardinal_input.width(), width);
  CHECK_EQ(cardinal_input.height(), height);
  if (sorted_coords.has_value()) {
    CHECK_EQ(sorted_coords->size(), width * height);
  }

  std::vector<float> sorted_values(width * height);
  internal::OrdinalTransform(cardinal_input, ordinal_output,
                             absl::MakeSpan(sorted_values), sorted_coords);
  return sorted_values;
}

void OrdinalTransformInverse(ReadViewGrayU16 ordinal_input,
                             absl::Span<const uint8_t> sorted_values,
                             WriteViewGrayU8 cardinal_output) {
  const int width = cardinal_output.width();
  const int height = cardinal_output.height();
  CHECK_EQ(ordinal_input.width(), width);
  CHECK_EQ(ordinal_input.height(), height);
  CHECK_EQ(sorted_values.size(), width * height);
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      DCHECK_LT(ordinal_input(col, row), sorted_values.size());
      cardinal_output(col, row) = sorted_values[ordinal_input(col, row)];
    }
  }
}

void OrdinalTransformInverse(ReadViewGrayU16 ordinal_input,
                             absl::Span<const uint16_t> sorted_values,
                             WriteViewGrayU16 cardinal_output) {
  const int width = cardinal_output.width();
  const int height = cardinal_output.height();
  CHECK_EQ(ordinal_input.width(), width);
  CHECK_EQ(ordinal_input.height(), height);
  CHECK_EQ(sorted_values.size(), width * height);
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      DCHECK_LT(ordinal_input(col, row), sorted_values.size());
      cardinal_output(col, row) = sorted_values[ordinal_input(col, row)];
    }
  }
}

void OrdinalTransformInverse(ReadViewGrayU16 ordinal_input,
                             absl::Span<const float> sorted_values,
                             WriteViewGrayF cardinal_output) {
  const int width = cardinal_output.width();
  const int height = cardinal_output.height();
  CHECK_EQ(ordinal_input.width(), width);
  CHECK_EQ(ordinal_input.height(), height);
  CHECK_EQ(sorted_values.size(), width * height);
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      DCHECK_LT(ordinal_input(col, row), sorted_values.size());
      cardinal_output(col, row) = sorted_values[ordinal_input(col, row)];
    }
  }
}
}  // namespace fast_isotropic_median_filter
