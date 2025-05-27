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

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fast_isotropic_median_filter.h"
#include "fast_isotropic_median_filter_simple.h"
#include "image.h"
#include "thread_pool.h"

using fast_isotropic_median_filter::FastIsotropicMedianFilterOptions;
using fast_isotropic_median_filter::FastIsotropicMedianFilterSimpleOptions;
using fast_isotropic_median_filter::ImagePlanar;
using fast_isotropic_median_filter::ReadViewPlanar;
using fast_isotropic_median_filter::WriteViewPlanar;

namespace {
constexpr int kMinRadius = 1;
constexpr int kMaxRadius = 100;

template <typename T>
absl::StatusOr<ImagePlanar<T>> ToImagePlanar(const cv::Mat& mat) {
  // Checks that the bit-depth of 'mat' matches 'T'.
  if (mat.elemSize1() != sizeof(T)) {
    return absl::InvalidArgumentError(
        "Image bit-depth does not match template type.");
  }
  ImagePlanar<T> image(mat.cols, mat.rows, mat.channels());
  for (int y = 0; y < mat.rows; ++y) {
    const T* row_ptr = reinterpret_cast<const T*>(mat.ptr(y));
    for (int x = 0; x < mat.cols; ++x) {
      for (int c = 0; c < mat.channels(); ++c) {
        image(x, y, c) = row_ptr[x * mat.channels() + c];
      }
    }
  }
  return image;
}

template <typename T>
absl::StatusOr<cv::Mat> ApplyFastIsotropicMedianFilter(
    const cv::Mat& src, const FastIsotropicMedianFilterOptions& options) {
  const int radius = options.radius;
  // Extends the input image by radius on each side, repeating edge pixels.
  cv::Mat src_padded;
  cv::copyMakeBorder(src, src_padded, radius, radius, radius, radius,
                     cv::BORDER_REPLICATE);

  auto input_result = ToImagePlanar<T>(src_padded);
  if (!input_result.ok()) {
    return input_result.status();
  }
  ImagePlanar<T> input = std::move(*input_result);

  ImagePlanar<T> output(src.cols, src.rows, input.planes());

  std::cout << "input.width(): " << input.width() << std::endl;
  std::cout << "input.height(): " << input.height() << std::endl;
  std::cout << "input planes(): " << input.planes() << std::endl;
  std::cout << "output.width(): " << output.width() << std::endl;
  std::cout << "output.height(): " << output.height() << std::endl;
  std::cout << "output planes(): " << output.planes() << std::endl;

  absl::Status status =
      FastIsotropicMedianFilter(input.ReadView(), options, output.WriteView());
  if (!status.ok()) {
    std::cerr << "FastIsotropicMedianFilter error: " << status << std::endl;
    return status;
  }

  // Converts the output image to a cv::Mat.
  cv::Mat dst;
  auto dst_result = ToCvMat(output.ReadView(), src.type());
  if (!dst_result.ok()) {
    return dst_result.status();
  }
  return *dst_result;
}

template <typename T>
bool ImageContentsMatch(ReadViewPlanar<T> a, ReadViewPlanar<T> b) {
  if (a.width() != b.width() || a.height() != b.height() ||
      a.planes() != b.planes()) {
    std::cerr << "Image dimensions do not match." << std::endl;
    return false;
  }
  for (int y = 0; y < a.height(); ++y) {
    for (int x = 0; x < a.width(); ++x) {
      for (int c = 0; c < a.planes(); ++c) {
        if (a(x, y, c) != b(x, y, c)) {
          return false;
        }
      }
    }
  }
  return true;
}

absl::Status FastIsotropicMedianFilterTestFloat(const cv::Mat& input_mat) {
  auto input_result = ToImagePlanar<float>(input_mat);
  if (!input_result.ok()) return input_result.status();
  ImagePlanar<float> input = std::move(*input_result);

  // For floating-point correctness testing, we find the min and max values of
  // the input image, and test against the 16-bit baseline filter run on that
  // range.
  float input_min = std::numeric_limits<float>::max();
  float input_max = std::numeric_limits<float>::min();
  for (int y = 0; y < input.height(); ++y) {
    for (int x = 0; x < input.width(); ++x) {
      for (int c = 0; c < input.planes(); ++c) {
        float val = input(x, y, c);
        // If value is not infinity or NaN, updates min and max.
        if (std::isfinite(val)) {
          if (val < input_min) input_min = val;
          if (val > input_max) input_max = val;
        } else if (std::isnan(val)) {
          std::cerr << "Input image has NaN values!" << std::endl;
          return absl::InternalError("Input image has NaN values.");
        }
      }
    }
  }
  if (input_min == input_max) {
    std::cerr << "Input image has no dynamic range!" << std::endl;
    return absl::InternalError("Input image has no dynamic range.");
  }

  // Converts the input image to 16-bit.
  const float scale = 65535.0f / (input_max - input_min);
  ImagePlanar<uint16_t> input_16(input.width(), input.height(), input.planes());
  for (int y = 0; y < input.height(); ++y) {
    for (int x = 0; x < input.width(); ++x) {
      for (int c = 0; c < input.planes(); ++c) {
        float val = input(x, y, c);
        if (std::isfinite(val)) {
          input_16(x, y, c) = (val - input_min) * scale;
        } else {
          input_16(x, y, c) = val > 0.0f ? 0xFFFF : 0;
        }
      }
    }
  }

  int max_radius = (std::min(input.width(), input.height()) - 1) / 2;
  max_radius = std::min(max_radius, kMaxRadius);
  for (int radius = kMinRadius; radius <= max_radius; ++radius) {
    const int output_width = input.width() - 2 * radius;
    const int output_height = input.height() - 2 * radius;
    FastIsotropicMedianFilterOptions options = {.radius = radius};
    ImagePlanar<float> output(output_width, output_height, input.planes());
    ImagePlanar<float> output_simple(output_width, output_height,
                                     input.planes());
    ImagePlanar<uint16_t> output_16(output_width, output_height,
                                    input.planes());
    for (float percentile : {0.0f, 0.01f, 0.1f, 0.5f, 0.9f, 0.99f, 1.0f}) {
      options.percentile = percentile;

      // Runs the optimized CPU implementation.
      absl::Status status = FastIsotropicMedianFilter(input.ReadView(), options,
                                                      output.WriteView());
      if (!status.ok()) {
        std::cerr << "FastIsotropicMedianFilter error at radius = " << radius
                  << ": " << status << std::endl;
        return status;
      }

      // Runs the simplified CPU implementation.
      FastIsotropicMedianFilterSimpleOptions options_simple = {
          .radius = radius, .percentile = percentile};
      for (int c = 0; c < input.planes(); ++c) {
        absl::Status status = FastIsotropicMedianFilterSimple(
            ToGray(input.ReadView(), c), options_simple,
            ToGray(output_simple.WriteView(), c));
        if (!status.ok()) {
          std::cerr << "FastIsotropicMedianFilterSimple error at radius = "
                    << radius << ", channel = " << c << ": " << status
                    << std::endl;
          return status;
        }
      }

      fast_isotropic_median_filter::internal::BaselineIsotropicMedianFilter(
          input_16.ReadView(), options.percentile, output_16.WriteView());

      for (int y = 0; y < output.height(); ++y) {
        for (int x = 0; x < output.width(); ++x) {
          for (int c = 0; c < output.planes(); ++c) {
            const float val = output(x, y, c);
            const int val_16 = output_16(x, y, c);
            const int val_q = std::isfinite(val) ? ((val - input_min) * scale)
                              : val > 0          ? 0xFFFF
                                                 : 0;
            if (val_16 != val_q) {
              std::cerr
                  << "Optimized vs Baseline: Discrepancy detected at radius = "
                  << radius << " percentile = " << percentile << " x = " << x
                  << " y = " << y << " c = " << c << std::endl;
              return absl::InternalError("Contents do not match.");
            }

            const float val_simple = output_simple(x, y, c);
            if (val_simple != val && std::isfinite(val_simple) &&
                std::isfinite(val)) {
              std::cerr
                  << "Optimized vs Simple: Discrepancy detected at radius = "
                  << radius << " percentile = " << percentile << " x = " << x
                  << " y = " << y << " c = " << c << std::endl;
              return absl::InternalError("Contents do not match.");
            }
          }
        }
      }
    }
    std::cout << "Testing radius = " << radius << " passed." << std::endl;
  }

  std::cout << "Test passed." << std::endl;
  return absl::OkStatus();
}

template <typename T>
absl::Status FastIsotropicMedianFilterTestInt(const cv::Mat& input_mat) {
  auto input_result = ToImagePlanar<T>(input_mat);
  if (!input_result.ok()) return input_result.status();
  ImagePlanar<T> input = std::move(*input_result);

  int max_radius = (std::min(input.width(), input.height()) - 1) / 2;
  max_radius = std::min(max_radius, kMaxRadius);
  for (int radius = kMinRadius; radius <= max_radius; ++radius) {
    const int output_width = input.width() - 2 * radius;
    const int output_height = input.height() - 2 * radius;
    FastIsotropicMedianFilterOptions options;
    options.radius = radius;
    ImagePlanar<T> output(output_width, output_height, input.planes());
    ImagePlanar<T> output_simple(output_width, output_height, input.planes());
    ImagePlanar<T> output_gt(output_width, output_height, input.planes());
    for (float percentile : {0.0f, 0.01f, 0.1f, 0.5f, 0.9f, 0.99f, 1.0f}) {
      options.percentile = percentile;

      // Runs the optimized CPU implementation.
      absl::Status status = FastIsotropicMedianFilter(input.ReadView(), options,
                                                      output.WriteView());
      if (!status.ok()) {
        std::cerr << "FastIsotropicMedianFilter error at radius = " << radius
                  << ": " << status << std::endl;
        return status;
      }

      // Runs the simplified CPU implementation.
      FastIsotropicMedianFilterSimpleOptions options_simple = {
          .radius = radius, .percentile = percentile};
      for (int c = 0; c < input.planes(); ++c) {
        absl::Status status = FastIsotropicMedianFilterSimple(
            ToGray(input.ReadView(), c), options_simple,
            ToGray(output_simple.WriteView(), c));
        if (!status.ok()) {
          std::cerr << "FastIsotropicMedianFilterSimple error at radius = "
                    << radius << ", channel = " << c << ": " << status
                    << std::endl;
          return status;
        }
      }

      // For ground-truth, we use a simple sliding-window implementation based
      // on the method of Huang et al. [1979], adapted for our circular kernel.
      fast_isotropic_median_filter::internal::BaselineIsotropicMedianFilter(
          input.ReadView(), options.percentile, output_gt.WriteView());

      if (!ImageContentsMatch(output.ReadView(), output_gt.ReadView())) {
        std::cerr
            << "Optimized vs Ground-truth: Discrepancy detected at radius = "
            << radius << ", percentile = " << percentile << std::endl;
        return absl::InternalError("Contents do not match.");
      }

      if (!ImageContentsMatch(output_simple.ReadView(), output_gt.ReadView())) {
        std::cerr << "Simple vs Ground-truth: Discrepancy detected at radius = "
                  << radius << ", percentile = " << percentile << std::endl;
        return absl::InternalError("Contents do not match.");
      }
    }
    std::cout << "Testing radius = " << radius << " passed." << std::endl;
  }
  return absl::OkStatus();
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <input_image>" << std::endl;
    std::cerr << "  <input_image>: Path to the input image (e.g., image.jpg, "
                 "image.hdr)."
              << std::endl;
    return 1;
  }

  std::string input_filename = argv[1];

  cv::Mat input_mat =
      cv::imread(input_filename, cv::IMREAD_ANYDEPTH | cv::IMREAD_COLOR);
  if (input_mat.empty()) {
    std::cerr << "Error: Could not open or find the image: " << input_filename
              << std::endl;
    return 1;
  }

  std::cout << "Input filename: " << input_filename << std::endl;
  std::cout << "Image width = " << input_mat.cols << std::endl;
  std::cout << "Image height = " << input_mat.rows << std::endl;
  std::cout << "Image channels = " << input_mat.channels() << std::endl;
  std::cout << "Image depth = " << input_mat.elemSize1() * 8
            << " bits per channel" << std::endl;

  fast_isotropic_median_filter::ThreadPool pool;
  fast_isotropic_median_filter::SetThreadPool(&pool);
  fast_isotropic_median_filter::SetThreadPoolSimple(&pool);

  absl::Status status;
  switch (input_mat.elemSize1()) {
    case 1:
      status = FastIsotropicMedianFilterTestInt<uint8_t>(input_mat);
      break;
    case 2:
      status = FastIsotropicMedianFilterTestInt<uint16_t>(input_mat);
      break;
    case 4:
      status = FastIsotropicMedianFilterTestFloat(input_mat);
      break;
    default:
      std::cerr << "Error: unsupported image bit-depth: "
                << input_mat.elemSize1() << std::endl;
      return 1;
  }
  if (!status.ok()) {
    std::cerr << "FastIsotropicMedianFilterTest error: " << status << std::endl;
    return 1;
  }

  std::cout << "Fast isotropic median filter test passed." << std::endl;
  return 0;
}