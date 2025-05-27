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

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fast_isotropic_median_filter.h"
#include "image.h"

using fast_isotropic_median_filter::FastIsotropicMedianFilterOptions;
using fast_isotropic_median_filter::ImagePlanar;
using fast_isotropic_median_filter::ReadViewPlanar;
using fast_isotropic_median_filter::WriteViewPlanar;

namespace {
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
absl::StatusOr<cv::Mat> ToCvMat(ReadViewPlanar<T> image, int cv_type) {
  if (image.base() == nullptr) {
    return absl::InvalidArgumentError("Input ReadViewPlanar is empty.");
  }

  cv::Mat mat(image.height(), image.width(), cv_type);
  if (mat.elemSize1() != sizeof(T)) {
    return absl::InvalidArgumentError(
        "Image bit-depth does not match template type.");
  }

  if (CV_MAT_CN(cv_type) != image.planes()) {
    return absl::InvalidArgumentError(
        "Incorrect number of channels. Expected: " +
        std::to_string(image.planes()) +
        ", got: " + std::to_string(CV_MAT_CN(cv_type)));
  }

  // Copies the data.
  for (int y = 0; y < image.height(); ++y) {
    T* row_ptr = reinterpret_cast<T*>(mat.ptr(y));
    for (int x = 0; x < image.width(); ++x) {
      for (int c = 0; c < image.planes(); ++c) {
        row_ptr[x * image.planes() + c] = image(x, y, c);
      }
    }
  }

  return mat;
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

  fast_isotropic_median_filter::ThreadPool pool;
  fast_isotropic_median_filter::SetThreadPool(&pool);

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
void Composite(const cv::Mat& input_mat, const cv::Mat& mask_mat,
               cv::Mat& output_mat) {
  const float scale =
      std::is_floating_point_v<T> ? 1.0f : 1.0 / std::numeric_limits<T>::max();
  for (int y = 0; y < output_mat.rows; ++y) {
    const T* input_row = reinterpret_cast<const T*>(input_mat.ptr(y));
    const T* mask_row = reinterpret_cast<const T*>(mask_mat.ptr(y));
    T* output_row = reinterpret_cast<T*>(output_mat.ptr(y));
    for (int x = 0; x < output_mat.cols; ++x) {
      const float mask_value = mask_row[x] * scale;
      if (mask_value > 0.0f) {
        for (int c = 0; c < output_mat.channels(); ++c) {
          const float v_in = input_row[x * output_mat.channels() + c];
          const float v_out = output_row[x * output_mat.channels() + c];
          output_row[x * output_mat.channels() + c] +=
              (v_in - v_out) * mask_value;
        }
      }
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: " << argv[0] << " <input_image> <radius> [percentile]"
              << std::endl;
    std::cerr << "  <input_image>: Path to the input image (e.g., image.jpg, "
                 "image.hdr)."
              << std::endl;
    std::cerr << "  <radius>: Integer radius for the percentile filter, in [1 "
                 ".. 100]."
              << std::endl;
    std::cerr << "  [percentile]: Optional floating-point percentile value in "
                 "[0.0 .. 1.0], default: 0.5."
              << std::endl;
    return 1;
  }

  std::string input_filename = argv[1];
  int radius = std::stoi(argv[2]);  // Convert radius to integer

  if (radius < 1 || radius > 100) {
    std::cerr << "Radius must be between 1 and 100." << std::endl;
    return 1;
  }

  float percentile = 0.5f;
  if (argc >= 4) {
    percentile = std::stof(argv[3]);
  }
  if (percentile < 0.0f || percentile > 1.0f) {
    std::cerr << "Percentile must be between 0.0 and 1.0." << std::endl;
    return 1;
  }

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
  std::cout << "Filter radius = " << radius << std::endl;
  std::cout << "Filter percentile = " << percentile << std::endl;

  absl::StatusOr<cv::Mat> output_mat_result;

  FastIsotropicMedianFilterOptions options = {.radius = radius,
                                              .percentile = percentile};

  switch (input_mat.elemSize1()) {
    case 1:
      output_mat_result =
          ApplyFastIsotropicMedianFilter<uint8_t>(input_mat, options);
      break;
    case 2:
      output_mat_result =
          ApplyFastIsotropicMedianFilter<uint16_t>(input_mat, options);
      break;
    case 4:
      output_mat_result =
          ApplyFastIsotropicMedianFilter<float>(input_mat, options);
      break;
    default:
      std::cerr << "Error: unsupported image bit-depth: "
                << input_mat.elemSize1() << std::endl;
      return 1;
  }

  if (!output_mat_result.ok()) {
    std::cerr << "ApplyFastIsotropicMedianFilter error: "
              << output_mat_result.status() << std::endl;
    return 1;
  }

  cv::Mat output_mat = output_mat_result.value();

  // If a corresponding mask file is found, composites through it.
  std::string mask_filename =
      input_filename.substr(0, input_filename.find_last_of(".")) + "_mask" +
      input_filename.substr(input_filename.find_last_of("."));
  if (std::filesystem::exists(mask_filename)) {
    cv::Mat mask_mat =
        cv::imread(mask_filename, cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
    if (!mask_mat.empty() && mask_mat.elemSize1() == input_mat.elemSize1() &&
        mask_mat.cols == input_mat.cols && mask_mat.rows == input_mat.rows) {
      switch (input_mat.elemSize1()) {
        case 1:
          Composite<uint8_t>(input_mat, mask_mat, output_mat);
          break;
        case 2:
          Composite<uint16_t>(input_mat, mask_mat, output_mat);
          break;
        case 4:
          Composite<float>(input_mat, mask_mat, output_mat);
          break;
        default:
          break;
      };
    }
  }

  // Writes the output image.
  std::string output_filename =
      input_filename.substr(0, input_filename.find_last_of(".")) + "_output" +
      input_filename.substr(input_filename.find_last_of("."));
  if (!cv::imwrite(output_filename, output_mat)) {
    std::cerr << "Error writing output image" << std::endl;
    return 1;
  }

  std::cout << "Image processed and saved to: " << output_filename << std::endl;

  return 0;
}