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

#ifndef FAST_ISOTROPIC_MEDIAN_FILTER_IMAGE_H_
#define FAST_ISOTROPIC_MEDIAN_FILTER_IMAGE_H_

#include <stdlib.h>

#include <algorithm>
#include <cstdint>

namespace fast_isotropic_median_filter {
// Forward declarations.
template <typename T>
class WriteViewGray;
template <typename T>
class WriteViewPlanar;

// Aliases for read-only views.
template <typename T>
using ReadViewGray = WriteViewGray<const T>;
template <typename T>
using ReadViewPlanar = WriteViewPlanar<const T>;

// Simple gray image view. (Doesn't manage memory.)
template <typename T>
class WriteViewGray {
 public:
  WriteViewGray(T* base, int width, int height, int stride = 0)
      : base_(base),
        width_(width),
        height_(height),
        stride_(stride == 0 ? width : stride) {}
  WriteViewGray() : base_(nullptr), width_(0), height_(0), stride_(0) {}

  T& operator()(int x, int y) { return base_[y * stride_ + x]; }
  const T& operator()(int x, int y) const { return base_[y * stride_ + x]; }

  // Accessors.
  T* base() const { return base_; }
  int width() const { return width_; }
  int height() const { return height_; }
  int stride() const { return stride_; }

  ReadViewGray<T> ReadView() const {
    return ReadViewGray<T>(base_, width_, height_, stride_);
  }

 private:
  T* base_;
  int width_;
  int height_;
  int stride_;
};

// Basic implementation of a gray image buffer. The logical memory is allocated
// with 128-byte alignment, for optimal SIMD / caching performance.
template <typename T>
class ImageGray {
 public:
  ImageGray() : view_(), mem_(nullptr) {}
  ImageGray(int width, int height) {
    mem_ = static_cast<T*>(malloc(width * height * sizeof(T) + 127));
    T* base =
        reinterpret_cast<T*>((reinterpret_cast<intptr_t>(mem_) + 127) & ~127);
    view_ = WriteViewGray<T>(base, width, height);
  }
  ImageGray(int width, int height, T value) : ImageGray<T>(width, height) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        (*this)(x, y) = value;
      }
    }
  }

  // Enables move constructor and move assignment.
  ImageGray(ImageGray&& other)
      : view_(std::move(other.view_)), mem_(other.mem_) {
    other.mem_ = nullptr;
  }
  ImageGray& operator=(ImageGray&& other) {
    if (this != &other) {
      free(mem_);
      mem_ = other.mem_;
      view_ = std::move(other.view_);
      other.mem_ = nullptr;
    }
    return *this;
  }

  // Disallows copy constructor and copy assignment.
  ImageGray(const ImageGray& other) = delete;
  ImageGray& operator=(const ImageGray& other) = delete;

  virtual ~ImageGray() { free(mem_); }

  WriteViewGray<T> WriteView() const { return view_; }
  ReadViewGray<T> ReadView() const { return view_.ReadView(); }

  T& operator()(int x, int y) { return view_(x, y); }
  const T& operator()(int x, int y) const { return view_(x, y); }

  int base() const { return view_.base(); }
  int width() const { return view_.width(); }
  int height() const { return view_.height(); }
  int stride() const { return view_.stride(); }

 private:
  WriteViewGray<T> view_;
  T* mem_;
};

// Planar image classes.
template <typename T>
class WriteViewPlanar {
 public:
  WriteViewPlanar(T* base, int width, int height, int planes,
                  int row_stride = 0, int plane_stride = 0)
      : base_(base),
        width_(width),
        height_(height),
        planes_(planes),
        row_stride_(row_stride == 0 ? width : row_stride),
        plane_stride_(plane_stride == 0
                          ? (row_stride == 0 ? width : row_stride) * height
                          : plane_stride) {}
  WriteViewPlanar()
      : base_(nullptr),
        width_(0),
        height_(0),
        planes_(0),
        row_stride_(0),
        plane_stride_(0) {}

  T& operator()(int x, int y, int c) {
    return base_[c * plane_stride_ + y * row_stride_ + x];
  }
  const T& operator()(int x, int y, int c) const {
    return base_[c * plane_stride_ + y * row_stride_ + x];
  }

  // Accessors.
  T* base() const { return base_; }
  int width() const { return width_; }
  int height() const { return height_; }
  int planes() const { return planes_; }
  int row_stride() const { return row_stride_; }
  int plane_stride() const { return plane_stride_; }

  ReadViewPlanar<T> ReadView() const {
    return ReadViewPlanar<T>(base_, width_, height_, planes_, row_stride_,
                             plane_stride_);
  }

 protected:
  void SetBase(T* base) { base_ = base; }

 private:
  T* base_;
  int width_;
  int height_;
  int planes_;
  int row_stride_;
  int plane_stride_;
};

// Basic implementation of a Planar image. The logical memory is allocated with
// 128-byte alignment, for optimal SIMD / caching performance.
template <typename T>
class ImagePlanar {
 public:
  ImagePlanar() : view_(), mem_(nullptr) {}
  ImagePlanar(int width, int height, int planes) {
    mem_ = static_cast<T*>(malloc(width * height * planes * sizeof(T) + 127));
    T* base =
        reinterpret_cast<T*>((reinterpret_cast<intptr_t>(mem_) + 127) & ~127);
    view_ = WriteViewPlanar<T>(base, width, height, planes);
  }
  ImagePlanar(int width, int height, int planes, T value)
      : ImagePlanar<T>(width, height, planes) {
    for (int c = 0; c < planes; ++c) {
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          (*this)(x, y, c) = value;
        }
      }
    }
  }

  // Enables move constructor and move assignment.
  ImagePlanar(ImagePlanar&& other)
      : view_(std::move(other.view_)), mem_(other.mem_) {
    other.mem_ = nullptr;
  }
  ImagePlanar& operator=(ImagePlanar&& other) {
    if (this != &other) {
      free(mem_);
      mem_ = other.mem_;
      view_ = std::move(other.view_);
      other.mem_ = nullptr;
    }
    return *this;
  }

  // Disallows copy constructor and copy assignment.
  ImagePlanar(const ImagePlanar& other) = delete;
  ImagePlanar& operator=(const ImagePlanar& other) = delete;

  virtual ~ImagePlanar() { free(mem_); }

  WriteViewPlanar<T> WriteView() const { return view_; }
  ReadViewPlanar<T> ReadView() const { return view_.ReadView(); }

  T& operator()(int x, int y, int c) { return view_(x, y, c); }
  const T& operator()(int x, int y, int c) const { return view_(x, y, c); }

  // Accessors.
  int base() const { return view_.base(); }
  int width() const { return view_.width(); }
  int height() const { return view_.height(); }
  int planes() const { return view_.planes(); }
  int row_stride() const { return view_.row_stride(); }
  int plane_stride() const { return view_.plane_stride(); }

 private:
  WriteViewPlanar<T> view_;
  T* mem_;
};

// Convenience typedefs.
using WriteViewGrayU8 = WriteViewGray<uint8_t>;
using ReadViewGrayU8 = ReadViewGray<uint8_t>;
using ImageGrayU8 = ImageGray<uint8_t>;
using WriteViewGrayU16 = WriteViewGray<uint16_t>;
using ReadViewGrayU16 = ReadViewGray<uint16_t>;
using ImageGrayU16 = ImageGray<uint16_t>;
using WriteViewGrayF = WriteViewGray<float>;
using ReadViewGrayF = ReadViewGray<float>;
using ImageGrayF = ImageGray<float>;
using WriteViewPlanarU8 = WriteViewPlanar<uint8_t>;
using ReadViewPlanarU8 = ReadViewPlanar<uint8_t>;
using ImagePlanarU8 = ImagePlanar<uint8_t>;
using WriteViewPlanarU16 = WriteViewPlanar<uint16_t>;
using ReadViewPlanarU16 = ReadViewPlanar<uint16_t>;
using ImagePlanarU16 = ImagePlanar<uint16_t>;
using WriteViewPlanarF = WriteViewPlanar<float>;
using ReadViewPlanarF = ReadViewPlanar<float>;
using ImagePlanarF = ImagePlanar<float>;

// Conversion between Gray and Planar images.
template <typename T>
WriteViewGray<T> ToGray(WriteViewPlanar<T> image, int channel) {
  return WriteViewGray<T>(&image(0, 0, channel), image.width(), image.height(),
                          image.row_stride());
}
template <typename T>
WriteViewPlanar<T> ToPlanar(WriteViewGray<T> image) {
  return WriteViewPlanar<T>(image.base(), image.width(), image.height(), 1,
                            image.stride());
}

// Cropping of Gray and Planar images.
template <typename T>
WriteViewGray<T> FastCrop(WriteViewGray<T> image, int x0, int y0, int x1,
                          int y1) {
  x0 = std::clamp(x0, 0, image.width());
  y0 = std::clamp(y0, 0, image.height());
  x1 = std::clamp(x1, x0, image.width());
  y1 = std::clamp(y1, y0, image.height());
  return WriteViewGray<T>(&image(x0, y0), x1 - x0, y1 - y0, image.stride());
}
template <typename T>
WriteViewPlanar<T> FastCrop(WriteViewPlanar<T> image, int x0, int y0, int x1,
                            int y1) {
  x0 = std::clamp(x0, 0, image.width());
  y0 = std::clamp(y0, 0, image.height());
  x1 = std::clamp(x1, x0, image.width());
  y1 = std::clamp(y1, y0, image.height());
  return WriteViewPlanar<T>(&image(x0, y0, 0), x1 - x0, y1 - y0, image.planes(),
                            image.row_stride(), image.plane_stride());
}

// Copying data between images. The intersection of the source and destination
// images is copied.
template <typename T>
void CopyContents(ReadViewGray<T> src, WriteViewGray<T> dst) {
  const int width = std::min(src.width(), dst.width());
  const int height = std::min(src.height(), dst.height());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      dst(x, y) = src(x, y);
    }
  }
}
template <typename T>
void CopyContents(ReadViewPlanar<T> src, WriteViewPlanar<T> dst) {
  const int width = std::min(src.width(), dst.width());
  const int height = std::min(src.height(), dst.height());
  const int planes = std::min(src.planes(), dst.planes());
  for (int c = 0; c < planes; ++c) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        dst(x, y, c) = src(x, y, c);
      }
    }
  }
}
}  // namespace fast_isotropic_median_filter
#endif  // FAST_ISOTROPIC_MEDIAN_FILTER_IMAGE_H_
