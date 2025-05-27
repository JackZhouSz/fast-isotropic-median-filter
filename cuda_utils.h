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

#ifndef FAST_ISOTROPIC_MEDIAN_FILTER_CUDA_UTILS_H_
#define FAST_ISOTROPIC_MEDIAN_FILTER_CUDA_UTILS_H_

#include <cstddef>
#include <cstring>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "image.h"
#include <cuda_runtime_api.h>

namespace fast_isotropic_median_filter {
template <typename T>
class CudaGraySurface {
 public:
  CudaGraySurface(int width, int height, cudaError_t* error = nullptr)
      : width_(width), height_(height) {
    cudaError_t res = cudaMallocPitch(reinterpret_cast<void**>(&base_), &pitch_,
                                      width * sizeof(T), height);
    if (error != nullptr) {
      *error = res;
    }
    if (res != cudaSuccess) {
      LOG(ERROR) << "Error allocating CUDA surface: "
                 << cudaGetErrorString(res);
    }
  }
  explicit CudaGraySurface(ReadViewGray<T> image,
                           cudaError_t* error = nullptr)
      : width_(image.width()), height_(image.height()) {
    cudaError_t res = cudaMallocPitch(reinterpret_cast<void**>(&base_), &pitch_,
                                      width_ * sizeof(T), height_);
    if (error != nullptr) {
      *error = res;
    }
    if (res != cudaSuccess) {
      LOG(ERROR) << "Error allocating CUDA surface: "
                 << cudaGetErrorString(res);
      return;
    }
    cudaMemcpy2D(base_, pitch_, image.base(), image.stride() * sizeof(T),
                 width_ * sizeof(T), height_, cudaMemcpyHostToDevice);
  }
  ~CudaGraySurface() { cudaFree(base_); }

  // Reads the contents of the CUDA surface into the provided image.
  absl::Status Readback(WriteViewGray<T> image) const {
    if (image.width() != width_ || image.height() != height_) {
      return absl::InvalidArgumentError(
          "Image dimensions must match CUDA surface.");
    }
    cudaError_t res = cudaMemcpy2D(
        image.base(),                    // Destination pointer (host)
        image.stride() * sizeof(T),      // Destination pitch
        base_,                           // Source CUDA array
        pitch_,                          // Pitch (in bytes)
        image.width() * sizeof(T),       // Width of copy region (in bytes)
        image.height(),                  // Height of copy region
        cudaMemcpyDeviceToHost);         // Kind of memory copy

    if (res != cudaSuccess) {
      return absl::InternalError(absl::StrCat("Error copying from CUDA array: ",
                                              cudaGetErrorString(res)));
    }
    return absl::OkStatus();
  }

  T* Base() const { return base_; }
  size_t Width() const { return width_; }
  size_t Height() const { return height_; }
  size_t Pitch() const { return pitch_; }

 private:
  T* base_;
  size_t width_;
  size_t height_;
  size_t pitch_;
};

}  // namespace fast_isotropic_median_filter

#endif  // FAST_ISOTROPIC_MEDIAN_FILTER_CUDA_UTILS_H_
