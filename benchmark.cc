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
#include <sys/types.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <regex>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "fast_isotropic_median_filter.h"
#include "image.h"

#define BENCHMARK_CPU 1

#ifdef __APPLE__
#define BENCHMARK_GPU 0  // Apple Silicon does not currently support CUDA.
#else
#define BENCHMARK_GPU 1
#endif

#if BENCHMARK_GPU
#include "cuda_utils.h"
#include "fast_isotropic_median_filter_cuda.cuh"
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

ABSL_FLAG(std::string, input_filename, "", "Input image filename (required)");
ABSL_FLAG(int, radius_min, 4, "Minimum radius for the filter");
ABSL_FLAG(int, radius_max, 100, "Maximum radius for the filter");
ABSL_FLAG(int, radius_step, 4, "Step size for the radius");
ABSL_FLAG(int, num_channels, 3, "Number of channels to benchmark");
ABSL_FLAG(int, num_threads, -1, "Number of threads to use");

namespace fast_isotropic_median_filter {
namespace {
// The minimum duration for each benchmark.
constexpr float kBenchmarkDurationSeconds = 2.0f;

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

// If the source image is not floating-point, higher bit-depth test images are
// constructed by adding uniform random noise to the source image.
absl::Status ConstructImagesFromU8(const cv::Mat& input_mat,
                                   ImagePlanar<uint8_t>& input8,
                                   ImagePlanar<uint16_t>& input16,
                                   ImagePlanar<float>& inputf) {
  auto input8_result = ToImagePlanar<uint8_t>(input_mat);
  if (!input8_result.ok()) {
    std::cerr << "Error: " << input8_result.status() << std::endl;
    return input8_result.status();
  }
  input8 = std::move(*input8_result);

  input16 =
      ImagePlanar<uint16_t>(input8.width(), input8.height(), input8.planes());
  inputf = ImagePlanar<float>(input8.width(), input8.height(), input8.planes());

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint16_t> dist8(0, 255);
  std::uniform_real_distribution<float> distf(0.0f, 1.0f);

  for (int y = 0; y < input8.height(); ++y) {
    for (int x = 0; x < input8.width(); ++x) {
      for (int c = 0; c < input8.planes(); ++c) {
        input16(x, y, c) = input8(x, y, c) << 8 | dist8(gen);  // [0 .. 0xFFFF]
        inputf(x, y, c) = (input8(x, y, c) + distf(gen)) / 256.0f;  // [0 .. 1]
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ConstructImagesFromU16(const cv::Mat& input_mat,
                                    ImagePlanar<uint8_t>& input8,
                                    ImagePlanar<uint16_t>& input16,
                                    ImagePlanar<float>& inputf) {
  auto input16_result = ToImagePlanar<uint16_t>(input_mat);
  if (!input16_result.ok()) {
    std::cerr << "Error: " << input16_result.status() << std::endl;
    return input16_result.status();
  }
  input16 = std::move(*input16_result);

  input8 =
      ImagePlanar<uint8_t>(input16.width(), input16.height(), input16.planes());
  inputf =
      ImagePlanar<float>(input16.width(), input16.height(), input16.planes());

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> distf(0.0f, 1.0f);

  for (int y = 0; y < input16.height(); ++y) {
    for (int x = 0; x < input16.width(); ++x) {
      for (int c = 0; c < input16.planes(); ++c) {
        input16(x, y, c) = input16(x, y, c) >> 8;  // [0 .. 255]
        inputf(x, y, c) = (input16(x, y, c) + distf(gen)) / 65536.0f;  // [0..1]
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ConstructImagesFromF(const cv::Mat& input_mat,
                                  ImagePlanar<uint8_t>& input8,
                                  ImagePlanar<uint16_t>& input16,
                                  ImagePlanar<float>& inputf) {
  auto inputf_result = ToImagePlanar<float>(input_mat);
  if (!inputf_result.ok()) {
    std::cerr << "Error: " << inputf_result.status() << std::endl;
    return inputf_result.status();
  }
  inputf = std::move(*inputf_result);

  input8 =
      ImagePlanar<uint8_t>(inputf.width(), inputf.height(), inputf.planes());
  input16 =
      ImagePlanar<uint16_t>(inputf.width(), inputf.height(), inputf.planes());
  for (int y = 0; y < input16.height(); ++y) {
    for (int x = 0; x < input16.width(); ++x) {
      for (int c = 0; c < input16.planes(); ++c) {
        const float v = std::clamp(inputf(x, y, c), 0.0f, 1.0f);
        input8(x, y, c) = v * 255.0f + 0.5f;
        input16(x, y, c) = v * 65535.0f + 0.5f;
      }
    }
  }
  return absl::OkStatus();
}

std::string GetCpuName() {
#ifdef __APPLE__
  size_t len;
  sysctlbyname("machdep.cpu.brand_string", nullptr, &len, nullptr, 0);
  if (len) {
    char* brand_string = new char[len];
    sysctlbyname("machdep.cpu.brand_string", brand_string, &len, nullptr, 0);
    std::string result(brand_string);
    delete[] brand_string;
    return result;
  }
#else
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  std::regex modelNameRegex("model name\\s*:\\s*(.*)");
  if (cpuinfo.is_open()) {
    while (std::getline(cpuinfo, line)) {
      std::smatch matches;
      if (std::regex_search(line, matches, modelNameRegex) &&
          matches.size() > 1) {
        return matches[1].str();
      }
    }
    cpuinfo.close();
  }
#endif
  return "CPU name not found";
}

#if BENCHMARK_GPU
std::string GetGpuName() {
  int deviceId;
  cudaGetDevice(&deviceId);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);
  return props.name;
}
#endif  // BENCHMARK_GPU

// Logs to a .csv file as well as to the console.
void CsvLog(FILE* file, const char* format, ...) {
  va_list args;
  va_start(args, format);
  char buffer[128];
  vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  std::cout << buffer;
  fprintf(file, "%s", buffer);
  fflush(file);
}
}  // namespace

int Benchmark(const std::string& input_filename, int radius_min, int radius_max,
              int radius_step, int num_channels, int num_threads) {
  if (num_threads < 1) {
    num_threads = ThreadPool::DefaultNumThreads();
  }

  ThreadPool pool(num_threads);
  SetThreadPool(&pool);

  const std::string results_filename = absl::StrCat(
      absl::StrReplaceAll(input_filename, {{".", "_"}}), "_results.csv");

  // If the file already exists, appends to it rather than overwriting it.
  FILE* results = fopen(results_filename.c_str(), "a");

  cv::Mat input_mat =
      cv::imread(input_filename, cv::IMREAD_ANYDEPTH | cv::IMREAD_COLOR);
  if (input_mat.empty()) {
    std::cerr << "Error: Could not open or find the image: " << input_filename
              << std::endl;
    return 1;
  }

  // If the image has more channels than the requested number of channels,
  // crops the image to the requested number of channels.
  if (input_mat.channels() > num_channels) {
    cv::Mat cropped_mat;
    cv::extractChannel(input_mat, cropped_mat, 0);
    input_mat = cropped_mat;
  }

  CsvLog(results, "#\n");
  CsvLog(results, "# Input filename: %s\n", input_filename.c_str());
  CsvLog(results,
         "# Image width = %d, height = %d, channels = %d, depth = %d bits per "
         "channel\n",
         input_mat.cols, input_mat.rows, input_mat.channels(),
         static_cast<int>(input_mat.elemSize1()) * 8);
  CsvLog(results, "# Radius range: min = %d, max = %d, step = %d\n", radius_min,
         radius_max, radius_step);
#if BENCHMARK_GPU
  CsvLog(results, "# Num Threads = %d, CPU = %s,,,GPU = %s\n", num_threads,
         GetCpuName().c_str(), GetGpuName().c_str());
#else
  CsvLog(results, "# Num Threads = %d, CPU = %s\n", num_threads,
         GetCpuName().c_str());
#endif

  // Extends the input image by 100 pixels on each side, repeating edge pixels.
  cv::Mat input_mat_padded;
  cv::copyMakeBorder(input_mat, input_mat_padded, 100, 100, 100, 100,
                     cv::BORDER_REPLICATE);

  ImagePlanar<uint8_t> input8;
  ImagePlanar<uint16_t> input16;
  ImagePlanar<float> inputf;

  // Constructs 8-bit, 16-bit, and float test images from the padded input.
  absl::Status status = absl::OkStatus();
  switch (input_mat.elemSize1()) {
    case 1:
      status = ConstructImagesFromU8(input_mat_padded, input8, input16, inputf);
      break;
    case 2:
      status =
          ConstructImagesFromU16(input_mat_padded, input8, input16, inputf);
      break;
    case 4:
      status = ConstructImagesFromF(input_mat_padded, input8, input16, inputf);
      break;
    default:
      std::cerr << "Error: unsupported image bit-depth: "
                << input_mat.elemSize1() << std::endl;
      return 1;
  }
  if (!status.ok()) {
    std::cerr << "Error constructing test images: " << status << std::endl;
    return 1;
  }

  const int output_width = input8.width() - 200;
  const int output_height = input8.height() - 200;

  // Note: for benchmarking purposes, we consider a "megapixel" to be a
  // grayscale megapixel. So e.g. a 1000x1000 RGB image counts as 3 megapixels.
  const float output_megapixels =
      output_width * output_height * input8.planes() / 1000000.0f;

  ImagePlanar<uint8_t> output8(output_width, output_height, input8.planes());
  ImagePlanar<uint16_t> output16(output_width, output_height, input8.planes());
  ImagePlanar<float> outputf(output_width, output_height, input8.planes());

#if BENCHMARK_GPU
  // The CUDA implementation is grayscale-only and requires the output image
  // size to be a multiple of 64 in each dimension.
  const int output_width_cuda = output_width & ~63;
  const int output_height_cuda = output_height & ~63;
  const float output_megapixels_cuda =
      output_width_cuda * output_height_cuda / 1000000.0f;
  CHECK_GE(output_width_cuda, 64);
  CHECK_GE(output_height_cuda, 64);

  // Constructs CUDA output surfaces.
  cudaError_t error;
  CudaGraySurface<uint8_t> output8_surface_cuda(output_width_cuda,
                                                output_height_cuda, &error);
  CHECK_EQ(error, cudaSuccess);
  CudaGraySurface<uint16_t> output16_surface_cuda(output_width_cuda,
                                                  output_height_cuda, &error);
  CHECK_EQ(error, cudaSuccess);
  CudaGraySurface<float> outputf_surface_cuda(output_width_cuda,
                                              output_height_cuda, &error);
  CHECK_EQ(error, cudaSuccess);

  cudaStream_t stream = nullptr;
  error = cudaStreamCreate(&stream);
  CHECK_EQ(error, cudaSuccess);
#endif

  CsvLog(results, "Radius,8-bit MP/s,16-bit MP/s,float MP/s\n");

  // Benchmark loop. (8-bit, 16-bit, and float.)
  for (int radius = radius_min; radius <= radius_max; radius += radius_step) {
    FastIsotropicMedianFilterOptions options;
    options.radius = radius;
    options.percentile = 0.5f;

    absl::Time start, end;
    int iters;

#if BENCHMARK_CPU
    // 8-bit.
    ReadViewPlanar<uint8_t> input8_cropped =
        FastCrop(input8.ReadView(), 100 - radius, 100 - radius,
                 output_width + 100 + radius, output_height + 100 + radius);
    start = absl::Now();
    end = start;
    iters = 0;
    while (absl::ToDoubleSeconds(end - start) < kBenchmarkDurationSeconds) {
      auto status = FastIsotropicMedianFilter(input8_cropped, options,
                                              output8.WriteView());
      if (!status.ok()) {
        std::cerr << "FastIsotropicMedianFilter error: " << status << std::endl;
        return 1;
      }
      end = absl::Now();
      iters++;
    }
    const float megapixels_per_sec_8 =
        output_megapixels * iters / absl::ToDoubleSeconds(end - start);

    // 16-bit.
    ReadViewPlanar<uint16_t> input16_cropped =
        FastCrop(input16.ReadView(), 100 - radius, 100 - radius,
                 output_width + 100 + radius, output_height + 100 + radius);
    start = absl::Now();
    end = start;
    iters = 0;
    while (absl::ToDoubleSeconds(end - start) < kBenchmarkDurationSeconds) {
      auto status = FastIsotropicMedianFilter(input16_cropped, options,
                                              output16.WriteView());
      if (!status.ok()) {
        std::cerr << "FastIsotropicMedianFilter error: " << status << std::endl;
        return 1;
      }
      end = absl::Now();
      iters++;
    }
    const float megapixels_per_sec_16 =
        output_megapixels * iters / absl::ToDoubleSeconds(end - start);

    // Floating-point.
    ReadViewPlanar<float> inputf_cropped =
        FastCrop(inputf.ReadView(), 100 - radius, 100 - radius,
                 output_width + 100 + radius, output_height + 100 + radius);
    start = absl::Now();
    end = start;
    iters = 0;
    while (absl::ToDoubleSeconds(end - start) < kBenchmarkDurationSeconds) {
      auto status = FastIsotropicMedianFilter(inputf_cropped, options,
                                              outputf.WriteView());
      if (!status.ok()) {
        std::cerr << "FastIsotropicMedianFilter error: " << status << std::endl;
        return 1;
      }
      end = absl::Now();
      iters++;
    }
    const float megapixels_per_sec_f =
        output_megapixels * iters / absl::ToDoubleSeconds(end - start);
#else
    const float megapixels_per_sec_8 = 0.0f;
    const float megapixels_per_sec_16 = 0.0f;
    const float megapixels_per_sec_f = 0.0f;
#endif

#if BENCHMARK_GPU
    float megapixels_per_sec_8_cuda = 0.0f;
    float megapixels_per_sec_16_cuda = 0.0f;
    float megapixels_per_sec_f_cuda = 0.0f;
    // The CUDA implementation supports radii up to 96.
    if (radius > 96) {
      // Logs combined results to file and console.
      CsvLog(results, "%d,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n", radius,
             megapixels_per_sec_8, megapixels_per_sec_16, megapixels_per_sec_f,
             0.0f, 0.0f, 0.0f);
      continue;
    }
    FastIsotropicMedianFilterCudaOptions options_cuda(
        {.radius = radius, .percentile = 0.5f});

    // The CUDA implementation is grayscale-only and requires the output image
    // size to be a multiple of 64 in each dimension.
    const int input_width_cuda = output_width_cuda + 2 * radius;
    const int input_height_cuda = output_height_cuda + 2 * radius;
    const int input_inset_cuda = (input8.width() - input_width_cuda) / 2;

    // Extracts first channel for benchmarking.
    ReadViewGray<uint8_t> input8_first_channel = ToGray(input8.ReadView(), 0);
    ReadViewGray<uint16_t> input16_first_channel =
        ToGray(input16.ReadView(), 0);
    ReadViewGray<float> inputf_first_channel = ToGray(inputf.ReadView(), 0);

    ReadViewGray<uint8_t> input8_cropped_cuda =
        FastCrop(input8_first_channel, input_inset_cuda, input_inset_cuda,
                 input_inset_cuda + input_width_cuda,
                 input_inset_cuda + input_height_cuda);
    ReadViewGray<uint16_t> input16_cropped_cuda =
        FastCrop(input16_first_channel, input_inset_cuda, input_inset_cuda,
                 input_inset_cuda + input_width_cuda,
                 input_inset_cuda + input_height_cuda);
    ReadViewGray<float> inputf_cropped_cuda =
        FastCrop(inputf_first_channel, input_inset_cuda, input_inset_cuda,
                 input_inset_cuda + input_width_cuda,
                 input_inset_cuda + input_height_cuda);

    // Construct CUDA input surfaces from input images.
    cudaError_t error;
    CudaGraySurface<uint8_t> input8_surface_cuda(input8_cropped_cuda, &error);
    CHECK_EQ(error, cudaSuccess);
    CudaGraySurface<uint16_t> input16_surface_cuda(input16_cropped_cuda,
                                                   &error);
    CHECK_EQ(error, cudaSuccess);
    CudaGraySurface<float> inputf_surface_cuda(inputf_cropped_cuda, &error);
    CHECK_EQ(error, cudaSuccess);

    // 8-bit CUDA benchmark.
    {
      absl::Status status;
      int bytes_allocated = 0;
      FastIsotropicMedianFilterBuffers<uint8_t> buffers_cuda(
          input8_surface_cuda, options_cuda, output8_surface_cuda, &status,
          &bytes_allocated);
      if (!status.ok()) {
        std::cerr << "FastIsotropicMedianFilterBuffers error: " << status
                  << std::endl;
        break;
      }

      start = absl::Now();
      end = start;
      iters = 0;
      while (absl::ToDoubleSeconds(end - start) < kBenchmarkDurationSeconds) {
        auto status = FastIsotropicMedianFilter(input8_surface_cuda,
                                                options_cuda, buffers_cuda,
                                                output8_surface_cuda, stream);
        cudaStreamSynchronize(stream);
        if (!status.ok()) {
          std::cerr << "FastIsotropicMedianFilter error: " << status
                    << std::endl;
          break;
        }
        end = absl::Now();
        iters++;
      }
      megapixels_per_sec_8_cuda =
          output_megapixels_cuda * iters / absl::ToDoubleSeconds(end - start);

      // Reads back result from GPU to validate against CPU implementation.
      ImageGray<uint8_t> output8_cuda({output_width_cuda, output_height_cuda});
      ImageGray<uint8_t> output8_cpu({output_width_cuda, output_height_cuda});
      status = output8_surface_cuda.Readback(output8_cuda.WriteView());
      if (!status.ok()) {
        std::cerr << "Error reading back CUDA output: " << status << std::endl;
        break;
      }
      // Calls the CPU implementation.
      status = FastIsotropicMedianFilter(ToPlanar(input8_cropped_cuda), options,
                                         ToPlanar(output8_cpu.WriteView()));
      if (!status.ok()) {
        std::cerr << "Error running 8-bit CPU implementation: " << status
                  << std::endl;
        break;
      }
      // Validates that the CUDA and CPU implementations match.
      bool all_equal = true;
      for (int y = 0; y < output8_cuda.height(); ++y) {
        for (int x = 0; x < output8_cuda.width(); ++x) {
          if (output8_cuda(x, y) != output8_cpu(x, y)) {
            all_equal = false;
            break;
          }
        }
        if (!all_equal) break;
      }
      if (!all_equal) {
        std::cerr << "Error: 8-bit CUDA result does not match CPU result."
                  << std::endl;
        break;
      }
    }

    // 16-bit CUDA benchmark.
    {
      absl::Status status;
      int bytes_allocated = 0;
      FastIsotropicMedianFilterBuffers<uint16_t> buffers_cuda(
          input16_surface_cuda, options_cuda, output16_surface_cuda, &status,
          &bytes_allocated);
      if (!status.ok()) {
        std::cerr << "FastIsotropicMedianFilterBuffers error: " << status
                  << std::endl;
        break;
      }

      start = absl::Now();
      end = start;
      iters = 0;
      while (absl::ToDoubleSeconds(end - start) < kBenchmarkDurationSeconds) {
        auto status = FastIsotropicMedianFilter(input16_surface_cuda,
                                                options_cuda, buffers_cuda,
                                                output16_surface_cuda, stream);
        cudaStreamSynchronize(stream);
        if (!status.ok()) {
          std::cerr << "FastIsotropicMedianFilter error: " << status
                    << std::endl;
          return 1;
        }
        end = absl::Now();
        iters++;
      }
      megapixels_per_sec_16_cuda =
          output_megapixels_cuda * iters / absl::ToDoubleSeconds(end - start);

      // Reads back result from GPU to validate against CPU implementation.
      ImageGray<uint16_t> output16_cuda({output_width_cuda,
                                         output_height_cuda});
      ImageGray<uint16_t> output16_cpu({output_width_cuda, output_height_cuda});
      status = output16_surface_cuda.Readback(output16_cuda.WriteView());
      if (!status.ok()) {
        std::cerr << "Error reading back CUDA output: " << status << std::endl;
        break;
      }
      // Calls the CPU implementation.
      status = FastIsotropicMedianFilter(ToPlanar(input16_cropped_cuda),
                                         options,
                                         ToPlanar(output16_cpu.WriteView()));
      if (!status.ok()) {
        std::cerr << "Error running 16-bit CPU implementation: " << status
                  << std::endl;
        break;
      }
      // Validates that the CUDA and CPU implementations match.
      bool all_equal = true;
      for (int y = 0; y < output16_cuda.height(); ++y) {
        for (int x = 0; x < output16_cuda.width(); ++x) {
          if (output16_cuda(x, y) != output16_cpu(x, y)) {
            all_equal = false;
            break;
          }
        }
        if (!all_equal) break;
      }
      if (!all_equal) {
        std::cerr << "Error: 16-bit CUDA result does not match CPU result."
                  << std::endl;
        break;
      }
    }

    // Floating-point CUDA benchmark.
    {
      absl::Status status;
      int bytes_allocated = 0;
      FastIsotropicMedianFilterBuffers<float> buffers_cuda(
          inputf_surface_cuda, options_cuda, outputf_surface_cuda, &status,
          &bytes_allocated);
      if (!status.ok()) {
        std::cerr << "FastIsotropicMedianFilterBuffers error: " << status
                  << std::endl;
        break;
      }

      start = absl::Now();
      end = start;
      iters = 0;
      while (absl::ToDoubleSeconds(end - start) < kBenchmarkDurationSeconds) {
        auto status = FastIsotropicMedianFilter(inputf_surface_cuda,
                                                options_cuda, buffers_cuda,
                                                outputf_surface_cuda, stream);
        cudaStreamSynchronize(stream);
        if (!status.ok()) {
          std::cerr << "FastIsotropicMedianFilter error: " << status
                    << std::endl;
          return 1;
        }
        end = absl::Now();
        iters++;
      }
      megapixels_per_sec_f_cuda =
          output_megapixels_cuda * iters / absl::ToDoubleSeconds(end - start);

      // Reads back result from GPU to validate against CPU implementation.
      ImageGray<float> outputf_cuda({output_width_cuda,
        output_height_cuda});
      ImageGray<float> outputf_cpu({output_width_cuda, output_height_cuda});
      status = outputf_surface_cuda.Readback(outputf_cuda.WriteView());
      if (!status.ok()) {
        std::cerr << "Error reading back CUDA output: " << status << std::endl;
        break;
      }
      // Calls the CPU implementation.
      status = FastIsotropicMedianFilter(ToPlanar(inputf_cropped_cuda),
          options, ToPlanar(outputf_cpu.WriteView()));
      if (!status.ok()) {
        std::cerr << "Error running float CPU implementation: " << status
                  << std::endl;
        break;
      }
      // Validates that the CUDA and CPU implementations match.
      bool all_equal = true;
      for (int y = 0; y < outputf_cuda.height(); ++y) {
        for (int x = 0; x < outputf_cuda.width(); ++x) {
          if (outputf_cuda(x, y) != outputf_cpu(x, y)) {
            all_equal = false;
            break;
          }
        }
        if (!all_equal) break;
      }
      if (!all_equal) {
        std::cerr << "Error: float CUDA result does not match CPU result."
                  << std::endl;
        break;
      }
    }

    // Logs combined results to file and console.
    CsvLog(results, "%d,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n", radius,
           megapixels_per_sec_8, megapixels_per_sec_16, megapixels_per_sec_f,
           megapixels_per_sec_8_cuda, megapixels_per_sec_16_cuda,
           megapixels_per_sec_f_cuda);
#else
    // Logs CPU-only results to file and console.
    CsvLog(results, "%d,%.6g,%.6g,%.6g\n", radius, megapixels_per_sec_8,
           megapixels_per_sec_16, megapixels_per_sec_f);
#endif
  }

#if BENCHMARK_GPU
  cudaStreamDestroy(stream);
#endif

  fclose(results);
  return 0;
}
}  // namespace fast_isotropic_median_filter

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  // Get the flag values.
  std::string input_filename = absl::GetFlag(FLAGS_input_filename);
  int radius_min = absl::GetFlag(FLAGS_radius_min);
  int radius_max = absl::GetFlag(FLAGS_radius_max);
  int radius_step = absl::GetFlag(FLAGS_radius_step);
  int num_channels = absl::GetFlag(FLAGS_num_channels);
  int num_threads = absl::GetFlag(FLAGS_num_threads);

  if (input_filename.empty()) {
    std::cerr << "Error: --input_filename is required." << std::endl;
    return 1;
  }
  if (radius_min < 1 || radius_min > 100) {
    std::cerr << "Error: --radius_min must be in [1 .. 100]." << std::endl;
    return 1;
  }
  if (radius_max < 1 || radius_max > 100) {
    std::cerr << "Error: --radius_max must be in [1 .. 100]." << std::endl;
    return 1;
  }
  if (radius_min > radius_max) {
    std::cerr << "Error: --radius_min must be <= radius_max." << std::endl;
    return 1;
  }
  if (num_channels < 1 || num_channels > 4) {
    std::cerr << "Error: --num_channels must be in [1 .. 4]." << std::endl;
  }
  if (radius_step <= 0) {
    std::cerr << "Error: --radius_step must be greater than 0." << std::endl;
    return 1;
  }

  return fast_isotropic_median_filter::Benchmark(input_filename, radius_min,
                                                 radius_max, radius_step,
                                                 num_channels, num_threads);
}
