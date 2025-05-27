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

#ifndef FAST_ISOTROPIC_MEDIAN_FILTER_ORDINAL_TRANSFORM_H_
#define FAST_ISOTROPIC_MEDIAN_FILTER_ORDINAL_TRANSFORM_H_
#include <stdint.h>

#include <algorithm>
#include <optional>
#include <vector>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "image.h"

namespace fast_isotropic_median_filter {
namespace internal {
constexpr int kSortThresholdDefault = -1;
constexpr int kSortThresholdStdSortNever = 0;
constexpr int kSortThresholdStdSortAlways = 256 * 256 + 1;

// Internal options, for benchmarking purposes.
struct OrdinalTransformOptions {
  bool use_simd = true;
  int sort_threshold = kSortThresholdDefault;
};

// Empirical default thresholds for switching between std::sort (num elements <
// threshold) and bucket-sort (num_elements >= threshold). Bucket-sort can
// outperform std::sort on this data because the low 16 bits (which encodes
// positional information) is already sorted. The '24' and '32' refer to
// inner-loop phases of the radix sort for 24-bit and 32-bit arrays,
// respectively. 'F' refers to the outer-loop phase where the full 32-bit
// floating-point value concatenated with positional data is sorted. The
// relative performance of radix sort and std::sort is quite different between
// CPU architectures, so these thresholds are set differently per-architecture.
// They are overrideable in the internal API for benchmarking purposes.
#if defined(__arm64__) || defined(__aarch64__)
constexpr int kSortThreshold24 = 32;
constexpr int kSortThreshold32 = 64;
constexpr int kSortThresholdU16 = 88 * 88;
constexpr int kSortThresholdF = 100 * 100;
#else
constexpr int kSortThreshold24 = 32;
constexpr int kSortThreshold32 = 64;
constexpr int kSortThresholdU16 = 52 * 52;
constexpr int kSortThresholdF = 78 * 78;
#endif

#ifdef __aarch64__
constexpr int kSortThreshold24Neon = 48;
constexpr int kSortThreshold32Neon = 96;
constexpr int kSortThresholdU16Neon = 76 * 76;
constexpr int kSortThresholdFNeon = 92 * 92;

void OrdinalTransformNeon(
    ReadViewGrayU16 cardinal_input, WriteViewGrayU16 ordinal_output,
    absl::Span<uint16_t> sorted_values,
    std::optional<absl::Span<uint16_t>> sorted_coords = std::nullopt,
    const OrdinalTransformOptions& options = {});
void OrdinalTransformNeon(ReadViewGrayU16 cardinal_input,
                          WriteViewGrayU8 ordinal_output,
                          absl::Span<uint16_t> sorted_values,
                          absl::Span<uint16_t> sorted_coords,
                          const OrdinalTransformOptions& options = {});

void OrdinalTransformNeon(
    ReadViewGrayF cardinal_input, WriteViewGrayU16 ordinal_output,
    absl::Span<float> sorted_values,
    std::optional<absl::Span<uint16_t>> sorted_coords = std::nullopt,
    const OrdinalTransformOptions& options = {});
void OrdinalTransformNeon(ReadViewGrayF cardinal_input,
                          WriteViewGrayU8 ordinal_output,
                          absl::Span<float> sorted_values,
                          absl::Span<uint16_t> sorted_coords,
                          const OrdinalTransformOptions& options = {});
#endif

#ifdef __AVX2__
constexpr int kSortThreshold24Avx2 = 32;
constexpr int kSortThreshold32Avx2 = 64;
constexpr int kSortThresholdU16Avx2 = 26 * 26;
constexpr int kSortThresholdFAvx2 = 74 * 74;

void OrdinalTransformAvx2(
    ReadViewGrayU16 cardinal_input, WriteViewGrayU16 ordinal_output,
    absl::Span<uint16_t> sorted_values,
    std::optional<absl::Span<uint16_t>> sorted_coords = std::nullopt,
    const OrdinalTransformOptions& options = {});
void OrdinalTransformAvx2(ReadViewGrayU16 cardinal_input,
                          WriteViewGrayU8 ordinal_output,
                          absl::Span<uint16_t> sorted_values,
                          absl::Span<uint16_t> sorted_coords,
                          const OrdinalTransformOptions& options = {});
void OrdinalTransformAvx2(
    ReadViewGrayF cardinal_input, WriteViewGrayU16 ordinal_output,
    absl::Span<float> sorted_values,
    std::optional<absl::Span<uint16_t>> sorted_coords = std::nullopt,
    const OrdinalTransformOptions& options = {});
void OrdinalTransformAvx2(ReadViewGrayF cardinal_input,
                          WriteViewGrayU8 ordinal_output,
                          absl::Span<float> sorted_values,
                          absl::Span<uint16_t> sorted_coords,
                          const OrdinalTransformOptions& options = {});
#endif

// Alternate APIs that writes the sorted values into a preallocated buffer, and
// that also accepts an options struct for benchmarking and testing.
void OrdinalTransform(
    ReadViewGrayU8 cardinal_input, WriteViewGrayU16 ordinal_output,
    absl::Span<uint8_t> sorted_values,
    std::optional<absl::Span<uint16_t>> sorted_coords = std::nullopt,
    const OrdinalTransformOptions& options = {});
void OrdinalTransform(
    ReadViewGrayU16 cardinal_input, WriteViewGrayU16 ordinal_output,
    absl::Span<uint16_t> sorted_values,
    std::optional<absl::Span<uint16_t>> sorted_coords = std::nullopt,
    const OrdinalTransformOptions& options = {});
void OrdinalTransform(
    ReadViewGrayF cardinal_input, WriteViewGrayU16 ordinal_output,
    absl::Span<float> sorted_values,
    std::optional<absl::Span<uint16_t>> sorted_coords = std::nullopt,
    const OrdinalTransformOptions& options = {});

// Alternate APIs that write a quantized version of the ordinal output
// (right-shifted by 6 bits) into an 8-bit buffer. Max input size is 128x128.
void OrdinalTransform(ReadViewGrayU8 cardinal_input,
                      WriteViewGrayU8 ordinal_output,
                      absl::Span<uint8_t> sorted_values,
                      absl::Span<uint16_t> sorted_coords,
                      const OrdinalTransformOptions& options = {});
void OrdinalTransform(ReadViewGrayU16 cardinal_input,
                      WriteViewGrayU8 ordinal_output,
                      absl::Span<uint16_t> sorted_values,
                      absl::Span<uint16_t> sorted_coords,
                      const OrdinalTransformOptions& options = {});
void OrdinalTransform(ReadViewGrayF cardinal_input,
                      WriteViewGrayU8 ordinal_output,
                      absl::Span<float> sorted_values,
                      absl::Span<uint16_t> sorted_coords,
                      const OrdinalTransformOptions& options = {});

// Performs an inclusive prefix sum on a 256-element histogram of uint8 or
// uint16 elements. I.e. {1, 2, 3, 4, ...} becomes {1, 3, 6, 10, ...}.
// The data must be 8-byte aligned.
// TODO: See if 4-up parallelism works better than 2-up.
template <typename T>
inline void PrefixSum256(T* hist) {
  static_assert(std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t>);
  DCHECK(reinterpret_cast<uintptr_t>(hist) % sizeof(uint64_t) == 0);
  constexpr uint64_t kMult = std::is_same_v<T, uint8_t> ? 0x0101010101010101ULL
                                                        : 0x0001000100010001ULL;
  uint64_t* hist64 = reinterpret_cast<uint64_t*>(hist);
  uint64_t sum0 = 0, sum1 = 0;
  constexpr int kNumBlocks = 256 * sizeof(T) / sizeof(uint64_t);
  constexpr int kStride64 = kNumBlocks / 2;
  constexpr int kShift = 64 - 8 * sizeof(T);
  for (int ind = 0; ind < kStride64; ++ind) {
    const uint64_t val0 = hist64[ind + kStride64 * 0] * kMult + sum0;
    const uint64_t val1 = hist64[ind + kStride64 * 1] * kMult + sum1;
    hist64[ind + kStride64 * 0] = val0;
    hist64[ind + kStride64 * 1] = val1;
    sum0 = (val0 >> kShift) * kMult;
    sum1 = (val1 >> kShift) * kMult;
  }
  for (int ind = 0; ind < kStride64; ++ind) {
    hist64[ind + kStride64] += sum0;
  }
}

// Sorts the high 8 bits an array of 24-bit values, placing the sorted "payload"
// (lower 16 bits) into `sorted_coords_ptr`. The low 16 bits are assumed to
// already be in sorted order. The bucket offsets are determined through a
// histogramming pre-pass, to reduce memory use and enable writing the payload
// directly into `sorted_coords_ptr` instead of using a scratch buffer.
template <typename hist_t>
static void RadixSort24(absl::Span<const uint32_t> tuples24,
                        uint32_t float_high_bits,
                        absl::Span<float> sorted_values,
                        absl::Span<uint16_t> sorted_coords) {
  // Initializes a 256-element histogram, padded at the start.
  alignas(sizeof(uint64_t)) hist_t hist_v[256 + sizeof(uint64_t)] = {0};
  hist_t* hist = &hist_v[sizeof(uint64_t)];
  const uint8_t* float_low_byte_ptr =
      reinterpret_cast<const uint8_t*>(tuples24.data()) + 2;
  for (int ind = 0; ind < tuples24.size(); ++ind) {
    const uint8_t float_low_byte = float_low_byte_ptr[ind * sizeof(uint32_t)];
    hist[float_low_byte]++;
  }
  // Prefix-sums the histogram to create offsets into the sorted array.
  internal::PrefixSum256(hist);
  hist--;  // Points to an exclusive prefix sum.

  // Bucket-sorts bits directly into the final array(s).
  for (int ind = 0; ind < tuples24.size(); ++ind) {
    const uint32_t tuple = tuples24[ind];
    const uint32_t float_low_byte = tuple >> 16;
    const int offset = hist[float_low_byte]++;
    sorted_coords[offset] = tuple & 0xFFFF;
    *reinterpret_cast<int32_t*>(&sorted_values[offset]) =
        float_high_bits | float_low_byte;
  }
}

// Radix-sorts the top 16 bits of a large 32-bit array, placing the "payload"
// (bottom 16 bits) into `sorted_coords_ptr`. A dense bucket-sort is used for
// the top 8 bits, using the provided `scratch32` buffer. The `scratch16` buffer
// may be used for recursively sorting the next 8 bits.
template <typename hist_t>
static void RadixSort32(absl::Span<const uint32_t> tuples32,
                        int32_t float_high_bits, absl::Span<uint32_t> tuples24,
                        absl::Span<float> sorted_values,
                        absl::Span<uint16_t> sorted_coords) {
  // Initializes a 256-element histogram, padded at the start.
  alignas(sizeof(uint64_t)) hist_t hist_v[256 + sizeof(uint64_t)] = {0};
  hist_t* hist = &hist_v[sizeof(uint64_t)];
  for (int ind = 0; ind < tuples32.size(); ++ind) {
    const uint32_t tuple = tuples32[ind];
    const int float_mid_byte = tuple >> 24;
    hist[float_mid_byte]++;
  }
  // Prefix-sums the histogram to create offsets into the sorted array.
  internal::PrefixSum256(hist);
  --hist;  // Points to an exclusive prefix sum.

  // Bucket-sorts by the mid 8 bits, using the provided scratch buffer to hold
  // the intermediate 24-bit tuples (low byte of float value | y | x).
  for (int ind = 0; ind < tuples32.size(); ++ind) {
    const uint32_t tuple = tuples32[ind];
    const int float_mid_byte = tuple >> 24;
    tuples24[hist[float_mid_byte]++] = tuple & 0x00FFFFFF;
  }
  // Sorts each bucket into the final ordered buffer.
  int begin = 0, end = 0;
  for (int ind = 0; ind < 256; ++ind) {
    begin = end;
    end = hist[ind];
    if (begin == end) continue;
    const int bucket_size = end - begin;
    if (bucket_size <= internal::kSortThreshold24) {
      if (bucket_size == 1) {
        const uint32_t float_mid_byte = ind << 8;
        const uint32_t float_low_byte = tuples24[begin] >> 16;
        *reinterpret_cast<int32_t*>(&sorted_values[begin]) =
            float_high_bits | float_mid_byte | float_low_byte;
        sorted_coords[begin] = tuples24[begin] & 0xFFFF;
      } else {
        std::sort(&tuples24[begin], &tuples24[begin] + bucket_size);
        const uint32_t float_mid_byte = ind << 8;
        for (int j = 0; j < bucket_size; ++j) {
          const uint32_t tuple = tuples24[begin + j];
          const uint32_t float_low_byte = tuple >> 16;
          *reinterpret_cast<int32_t*>(&sorted_values[begin + j]) =
              float_high_bits | float_mid_byte | float_low_byte;
          sorted_coords[begin + j] = tuple & 0xFFFF;
        }
      }
    } else {
      const uint32_t float_mid_byte = ind << 8;
      if (bucket_size < 256) {
        RadixSort24<uint8_t>(tuples24.subspan(begin, bucket_size),
                             float_high_bits | float_mid_byte,
                             sorted_values.subspan(begin, bucket_size),
                             sorted_coords.subspan(begin, bucket_size));
      } else {
        RadixSort24<uint16_t>(tuples24.subspan(begin, bucket_size),
                              float_high_bits | float_mid_byte,
                              sorted_values.subspan(begin, bucket_size),
                              sorted_coords.subspan(begin, bucket_size));
      }
    }
  }
}
}  // namespace internal

// Ordinal Transform. For each value in the input, calculates its ordinal rank
// in the image and writes it to the corresponding pixel of the output, e.g.:
//
//   [ 314, 159, 265, 358 ]    [  4,  0,  2,  5 ]
//   [ 979, 777, 777, 264 ] -> [ 14,  9, 10,  1 ]
//   [ 440, 777, 777, 987 ]    [  6, 11, 12, 15 ]
//   [ 895, 293, 650, 498 ]    [ 13,  3,  8,  7 ]
//
// Duplicate inputs are assigned distinct consecutive ranks in scanline order,
// to ensure that the output rank of each pixel is unique. The return vector is
// a lookup table that maps output back to input. Images must be the same size,
// no larger than 65536 total pixels.
// If non-null, sorted coordinates (y << 8 | x) are written to `sorted_coords`.
// (This is used for the circular variant of the percentile filter.)
// TODO: Make versions that return 32-bit sorted_coords and quantized
// (right-shifted) ordinal output, supporting larger tiles up to e.g. 2048x2048,
// with either implicit or explicit quantization amount. Also, a version that
// returns 8-bit quantized ordinal output, for small tiles up to 128x128.
std::vector<uint8_t> OrdinalTransform(
    ReadViewGrayU8 cardinal_input, WriteViewGrayU16 ordinal_output,
    std::optional<absl::Span<uint16_t>> sorted_coords = std::nullopt);
std::vector<uint16_t> OrdinalTransform(
    ReadViewGrayU16 cardinal_input, WriteViewGrayU16 ordinal_output,
    std::optional<absl::Span<uint16_t>> sorted_coords = std::nullopt);
std::vector<float> OrdinalTransform(
    ReadViewGrayF cardinal_input, WriteViewGrayU16 ordinal_output,
    std::optional<absl::Span<uint16_t>> sorted_coords = std::nullopt);

// Inverts the Ordinal Transform, using the lookup table of sorted values
// computed in the forward transform.
void OrdinalTransformInverse(ReadViewGrayU16 ordinal_input,
                             absl::Span<const uint8_t> sorted_values,
                             WriteViewGrayU8 cardinal_output);
void OrdinalTransformInverse(ReadViewGrayU16 ordinal_input,
                             absl::Span<const uint16_t> sorted_values,
                             WriteViewGrayU16 cardinal_output);
void OrdinalTransformInverse(ReadViewGrayU16 ordinal_input,
                             absl::Span<const float> sorted_values,
                             WriteViewGrayF cardinal_output);
}  // namespace fast_isotropic_median_filter
#endif  // FAST_ISOTROPIC_MEDIAN_FILTER_ORDINAL_TRANSFORM_H_
