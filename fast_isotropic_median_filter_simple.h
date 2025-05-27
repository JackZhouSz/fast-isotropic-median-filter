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

#ifndef FAST_ISOTROPIC_MEDIAN_FILTER_FAST_ISOTROPIC_MEDIAN_FILTER_SIMPLE_H_
#define FAST_ISOTROPIC_MEDIAN_FILTER_FAST_ISOTROPIC_MEDIAN_FILTER_SIMPLE_H_

#include "absl/status/status.h"
#include "image.h"
#include "thread_pool.h"

namespace fast_isotropic_median_filter {
struct FastIsotropicMedianFilterSimpleOptions {
  int radius = 32;          // [1 .. 100]
  float percentile = 0.5f;  // [0.0 .. 1.0]
};

// The thread pool must be set before any calls to
// FastIsotropicMedianFilterSimple.
void SetThreadPoolSimple(ThreadPool* pool);

// Public API.
absl::Status FastIsotropicMedianFilterSimple(
    ReadViewGrayU8 input, const FastIsotropicMedianFilterSimpleOptions& options,
    WriteViewGrayU8 output);
absl::Status FastIsotropicMedianFilterSimple(
    ReadViewGrayU16 input,
    const FastIsotropicMedianFilterSimpleOptions& options,
    WriteViewGrayU16 output);
absl::Status FastIsotropicMedianFilterSimple(
    ReadViewGrayF input, const FastIsotropicMedianFilterSimpleOptions& options,
    WriteViewGrayF output);
}  // namespace fast_isotropic_median_filter
#endif  // FAST_ISOTROPIC_MEDIAN_FILTER_FAST_ISOTROPIC_MEDIAN_FILTER_SIMPLE_H_
