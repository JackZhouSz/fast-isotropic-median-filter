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

#ifndef FAST_ISOTROPIC_MEDIAN_FILTER_FAST_ISOTROPIC_MEDIAN_FILTER_H_
#define FAST_ISOTROPIC_MEDIAN_FILTER_FAST_ISOTROPIC_MEDIAN_FILTER_H_

#if defined(__AVX2__) || defined(__aarch64__)
#include <cstdint>
#endif

#include "absl/status/status.h"
#include "image.h"
#include "thread_pool.h"

namespace fast_isotropic_median_filter {
struct FastIsotropicMedianFilterOptions {
  int radius = 32;          // [1 .. 100]
  float percentile = 0.5f;  // [0.0 .. 1.0]
  bool use_simd = true;
};

namespace internal {
constexpr int kHistPad = 128;
constexpr int kMaximumFilterRadius = 100;

// The threshold is used to determine inclusion in the circular window.
// For a pixel at offset (dx, dy) from the center, it is included iff:
//   (dx * dx + dy * dy) <= threshold.
// Radius = 1 is special-cased to avoid a square window shape.
inline int ThresholdForRadius(int radius) {
  return radius == 1 ? 1 : static_cast<int>((radius + 0.5f) * (radius + 0.5f));
}
#ifdef __AVX2__
// The 'pdep' instruction (part of the BMI2 instruction set) is highly useful
// for the Percentile filter inner loop, except on certain AMD CPUs (Zen1, Zen2)
// where is has an extremely slow microcoded implementation. So we only use
// 'pdep' on micro-architectures that implement it efficiently.
bool UsePdep();
#endif  // __AVX2__

// Baseline slow implementation for correctness testing.
void BaselineIsotropicMedianFilter(ReadViewPlanarU8 input, float percentile,
                                   WriteViewPlanarU8 output);
void BaselineIsotropicMedianFilter(ReadViewPlanarU16 input, float percentile,
                                   WriteViewPlanarU16 output);

// Optimized implementation.
void FastIsotropicMedianFilter(ReadViewPlanarU8 input, float percentile,
                               bool use_simd, WriteViewPlanarU8 output);
void FastIsotropicMedianFilter(ReadViewPlanarU16 input, float percentile,
                               bool use_simd, WriteViewPlanarU16 output);
void FastIsotropicMedianFilter(ReadViewPlanarF input, float percentile,
                               bool use_simd, WriteViewPlanarF output);
#if defined(__aarch64__)
uint16_t SearchUpDownNeon(int col, int row, int threshold, int rank,
                          const uint16_t* histc, uint8_t* pivot_col_ptr,
                          uint16_t* count_col_ptr);
uint16_t SearchUpDownNeon(int col, int row, int threshold, int rank,
                          const uint16_t* histc, uint16_t* pivot_col_ptr,
                          uint16_t* count_col_ptr);
void UpdateCountsNeon(ReadViewGrayU8 ordinal_input, int radius, int row,
                      const uint8_t* inset, const uint8_t* pivot,
                      uint16_t* count);
void UpdateCountsNeon(ReadViewGrayU16 ordinal_input, int radius, int row,
                      const uint8_t* inset, const uint16_t* pivot,
                      uint16_t* count);
void TransposeFirstRowsNeon(ReadViewGrayU8 ordinal_input, int radius,
                            const uint8_t* inset, uint8_t* pixels_left,
                            uint8_t* pixels_right);
void TransposeFirstRowsNeon(ReadViewGrayU16 ordinal_input, int radius,
                            const uint8_t* inset, uint16_t* pixels_left,
                            uint16_t* pixels_right);
int StepHorizontalNeon(const uint8_t* pixels_left, const uint8_t* pixels_right,
                       int pivot_val, int kernel_width_ceil);
int StepHorizontalNeon(const uint16_t* pixels_left,
                       const uint16_t* pixels_right, int pivot_val,
                       int kernel_width_ceil);
#elif defined(__AVX2__)
uint16_t SearchUpDownAvx2Pdep(int col, int row, int threshold, int rank,
                              const uint16_t* histc, uint8_t* pivot_col_ptr,
                              uint16_t* count_col_ptr);
uint16_t SearchUpDownAvx2Pdep(int col, int row, int threshold, int rank,
                              const uint16_t* histc, uint16_t* pivot_col_ptr,
                              uint16_t* count_col_ptr);
uint16_t SearchUpDownAvx2Popct(int col, int row, int threshold, int rank,
                               const uint16_t* histc, uint8_t* pivot_col_ptr,
                               uint16_t* count_col_ptr);
uint16_t SearchUpDownAvx2Popct(int col, int row, int threshold, int rank,
                               const uint16_t* histc, uint16_t* pivot_col_ptr,
                               uint16_t* count_col_ptr);
void UpdateCountsAvx2(ReadViewGrayU8 ordinal_input, int radius, int row,
                      const uint8_t* inset, const uint8_t* pivot,
                      uint16_t* count);
void UpdateCountsAvx2(ReadViewGrayU16 ordinal_input, int radius, int row,
                      const uint8_t* inset, const uint16_t* pivot,
                      uint16_t* count);
void TransposeFirstRowsAvx2(ReadViewGrayU8 ordinal_input, int radius,
                            const uint8_t* inset, uint8_t* pixels_left,
                            uint8_t* pixels_right);
void TransposeFirstRowsAvx2(ReadViewGrayU16 ordinal_input, int radius,
                            const uint8_t* inset, uint16_t* pixels_left,
                            uint16_t* pixels_right);
int StepHorizontalAvx2(const uint8_t* pixels_left, const uint8_t* pixels_right,
                       int pivot_val, int kernel_width_ceil);
int StepHorizontalAvx2(const uint16_t* pixels_left,
                       const uint16_t* pixels_right, int pivot_val,
                       int kernel_width_ceil);
#endif
}  // namespace internal

// The thread pool must be set before any calls to FastIsotropicMedianFilter.
void SetThreadPool(ThreadPool* pool);

// Public API.
absl::Status FastIsotropicMedianFilter(
    ReadViewPlanarU8 input, const FastIsotropicMedianFilterOptions& options,
    WriteViewPlanarU8 output);
absl::Status FastIsotropicMedianFilter(
    ReadViewPlanarU16 input, const FastIsotropicMedianFilterOptions& options,
    WriteViewPlanarU16 output);
absl::Status FastIsotropicMedianFilter(
    ReadViewPlanarF input, const FastIsotropicMedianFilterOptions& options,
    WriteViewPlanarF output);
}  // namespace fast_isotropic_median_filter
#endif  // FAST_ISOTROPIC_MEDIAN_FILTER_FAST_ISOTROPIC_MEDIAN_FILTER_H_
