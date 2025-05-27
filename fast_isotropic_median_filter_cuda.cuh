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

#ifndef FAST_ISOTROPIC_MEDIAN_FILTER_FAST_ISOTROPIC_MEDIAN_FILTER_CUDA_H_
#define FAST_ISOTROPIC_MEDIAN_FILTER_FAST_ISOTROPIC_MEDIAN_FILTER_CUDA_H_

#include <cstddef>
#include <cstdint>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "cuda_utils.h"
#include <cuda_runtime_api.h>

namespace fast_isotropic_median_filter {
// Struct contains parameters for applying a percentile filter.
struct FastIsotropicMedianFilterCudaOptions {
  int radius = 32;          // [1 .. 96]
  float percentile = 0.5f;  // [0.0f .. 1.0f]
};

template <typename value_t>
struct FastIsotropicMedianFilterBuffers {
  const int input_width;
  const int input_height;
  const int output_width;
  const int output_height;

  // Note: some parameters are mutable because they are double-buffered, and may
  // be swapped.
  mutable value_t* d_packed_values = nullptr;
  mutable uint16_t* d_packed_coords = nullptr;
  int* d_begin_offsets = nullptr;
  int* d_end_offsets = nullptr;
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  mutable value_t* d_reverse_maps = nullptr;
  mutable uint16_t* d_omnigrams = nullptr;

  FastIsotropicMedianFilterBuffers(const CudaGraySurface<value_t>& input,
                          const FastIsotropicMedianFilterCudaOptions& options,
                          CudaGraySurface<value_t>& output,
                          absl::Status* status = nullptr,
                          int* bytes_allocated = nullptr)
      : input_width(input.Width()),
        input_height(input.Height()),
        output_width(output.Width()),
        output_height(output.Height()) {
    auto result = Allocate(input, options, output, bytes_allocated);
    if (status != nullptr) {
      *status = result;
    }
  }

  ~FastIsotropicMedianFilterBuffers(){
    cudaFree(d_packed_values);
    cudaFree(d_packed_coords);
    cudaFree(d_begin_offsets);
    cudaFree(d_end_offsets);
    cudaFree(d_temp_storage);
    cudaFree(d_reverse_maps);
    cudaFree(d_omnigrams);
  }

  absl::Status Allocate(const CudaGraySurface<value_t>& input,
                        const FastIsotropicMedianFilterCudaOptions& options,
                        CudaGraySurface<value_t>& output,
                        int* bytes_allocated = nullptr);
};

// Explicit template instantiations for the supported types.
template struct FastIsotropicMedianFilterBuffers<uint8_t>;
template struct FastIsotropicMedianFilterBuffers<uint16_t>;
template struct FastIsotropicMedianFilterBuffers<float>;

// Applies a circular-kernel radius-32 percentile filter to the input image.
// Input and output dimensions must be multiples of 64 pixels. The output image
// is inset by the radius on all sides with respect to the input (e.g. 512x512
// input -> 448x448 output).
absl::Status FastIsotropicMedianFilter(const CudaGraySurface<uint8_t>& input,
  const FastIsotropicMedianFilterCudaOptions& options,
  const FastIsotropicMedianFilterBuffers<uint8_t>& buffers,
  CudaGraySurface<uint8_t>& output, cudaStream_t stream);

absl::Status FastIsotropicMedianFilter(const CudaGraySurface<uint16_t>& input,
  const FastIsotropicMedianFilterCudaOptions& options,
  const FastIsotropicMedianFilterBuffers<uint16_t>& buffers,
  CudaGraySurface<uint16_t>& output, cudaStream_t stream);

absl::Status FastIsotropicMedianFilter(const CudaGraySurface<float>& input,
  const FastIsotropicMedianFilterCudaOptions& options,
  const FastIsotropicMedianFilterBuffers<float>& buffers,
   CudaGraySurface<float>& output, cudaStream_t stream);
}  // namespace fast_isotropic_median_filter

#endif  // FAST_ISOTROPIC_MEDIAN_FILTER_FAST_ISOTROPIC_MEDIAN_FILTER_CUDA_H_
