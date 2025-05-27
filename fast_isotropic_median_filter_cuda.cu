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

#include "fast_isotropic_median_filter_cuda.cuh"

#include <cub/cub.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <cub/util_type.cuh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <type_traits>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "cuda_utils.h"
#include <cuda_runtime_api.h>

namespace fast_isotropic_median_filter {
namespace {
  // For optimal cacheline alignment, the omnigram is padded to a multiple of
  // 128 elements.
  constexpr int kOmnigramPad = 128;

  // A mask corresponding to all 32 threads in a warp.
  constexpr uint32_t kWarpAll = 0xFFFFFFFF;

// This informs the compiler that e.g.
// "*reinterpret_cast<aligned_uint32_t*>(&array[k])" can be loaded as an aligned
// uint32_t. Otherwise it may try to load it with e.g. four uint8_t loads.
using aligned_uint8_t = __align__(4) uint8_t;
using aligned_uint32_t = __align__(4) uint32_t;

struct FastIsotropicMedianFilterParams {
  int radius;
  int threshold;
  int percentile;  // target rank in [1 .. (kernel support)]
  int omnigram_elements;
  uint8_t num_offsets_h;
  uint8_t num_offsets_v;
  std::array<int16_t, 256> offsets_n;
  std::array<int16_t, 256> offsets_s;
  std::array<int16_t, 256> offsets_w;
  std::array<int16_t, 256> offsets_e;
  uint8_t pack_inset[256];
  uint16_t pack_offset[256];
};

__constant__ FastIsotropicMedianFilterParams c_params;

// For a two-element section of the omnigram, determines whether each element is
// inside the circular kernel centered at (kcenter_x, kcenter_y) with squared
// radius <= threshold. Returns two result bits.
// TODO: A generic version of this function that accepts a bitmap for
// the kernel shape, rather than computing the result through direct arithmetic.
__device__ __forceinline__ void InCircle(uint32_t xyxy, uint32_t kxyxy,
                                         int threshold, int& bit0, int& bit1) {
#if __CUDA_ARCH__ >= MIN_CC_DP4A  // sm61 or newer supports __dp4a intrinsics.
  // Computes the absolute difference between the window center coordinates and
  // the pixel coordinates.
  xyxy = __vabsdiffu4(xyxy, kxyxy);

  // Unpacks high and low words, to do two separate dot products.
  const int32_t xy0 = xyxy & 0xFFFF;
  const int32_t xy1 = xyxy >> 16;

  // Performs the dot products, and subtracts the threshold. If these values
  // become negative, the pixel is inside the kernel circle.
  const int32_t dot0 = __dp4a(xy0, xy0, -(threshold + 1));
  const int32_t dot1 = __dp4a(xy1, xy1, -(threshold + 1));
#else   // Scalar equivalent, if dot product intrinsics are not available.
  const int x0 = xyxy & 0xFF;
  const int y0 = (xyxy >> 8) & 0xFF;
  const int x1 = (xyxy >> 16) & 0xFF;
  const int y1 = (xyxy >> 24) & 0xFF;
  const int kcenter_x = kxyxy & 0xFF;
  const int kcenter_y = (kxyxy >> 8) & 0xFF;
  const int dx0 = x0 - kcenter_x;
  const int dy0 = y0 - kcenter_y;
  const int dx1 = x1 - kcenter_x;
  const int dy1 = y1 - kcenter_y;
  const int dot0 = dx0 * dx0 + dy0 * dy0 - (threshold + 1);
  const int dot1 = dx1 * dx1 + dy1 * dy1 - (threshold + 1);
#endif  // Dot product intrinsics vs scalar.

  // The sign bit of the offset dot product indicates whether the coordinate was
  // inside the kernel circle.
  bit0 = static_cast<uint32_t>(dot0) >> 31;
  bit1 = static_cast<uint32_t>(dot1) >> 31;
}

// Returns two concatenated result bits from InCircle().
__device__ __forceinline__ uint32_t InCircleBits(uint32_t xyxy, uint32_t kxyxy,
                                                 int threshold) {
  int bit0, bit1;
  InCircle(xyxy, kxyxy, threshold, bit0, bit1);
  return bit0 | (bit1 << 1);
}

// Returns the sum of the two result bits from InCircle().
__device__ __forceinline__ uint32_t InCircleSum(uint32_t xyxy, uint32_t kxyxy,
                                                int threshold) {
  int bit0, bit1;
  InCircle(xyxy, kxyxy, threshold, bit0, bit1);
  return bit0 + bit1;
}

// Blends two bitmasks together, using the mask to select between the two.
// On Maxwell+ this should compile down to a single 'LOP3.LUT' instruction
// (although there is no intrinsic for it).
__device__ __forceinline__ uint32_t Blend(uint32_t a, uint32_t b,
                                          uint32_t mask) {
  return (a & mask) | (b & ~mask);
}

// Each thread in a warp has processed two elements each from an omnigram scan
// of all 32 threads' windows, and the results are packed into a uint64_t. This
// function transposes these bits back to the window owners' respective threads.
// TODO: Convince NVIDIA to implement all this in a single hardware
// instruction (as e.g. "__multiballot_sync()", or "__shfl_transpose_sync()").
__device__ __forceinline__ uint64_t Transpose64x32Sync(uint64_t bits,
                                                       int threadIdx_x) {
  constexpr uint32_t kWarpAll = 0xFFFFFFFF;
  // Manually processes as 32-bit instead of 64-bit; the instructions are
  // natively 32-bit, and there is no 64-bit funnelshift intrinsic.
  uint32_t bits_lo = bits & 0xFFFFFFFF;
  uint32_t bits_hi = bits >> 32;

  // Transposes 2-bit blocks, 1 thread apart.
  uint32_t bits_x_lo = __shfl_xor_sync(kWarpAll, bits_lo, 1);
  uint32_t bits_x_hi = __shfl_xor_sync(kWarpAll, bits_hi, 1);
  int shift = 2 - (threadIdx.x << 2 & 4);  // { 2, -2 }
  uint32_t mask = (threadIdx.x & 1) ? 0xCCCCCCCC : 0x33333333;
  bits_lo = Blend(bits_lo, __funnelshift_l(bits_x_lo, bits_x_lo, shift), mask);
  bits_hi = Blend(bits_hi, __funnelshift_l(bits_x_hi, bits_x_hi, shift), mask);

  // Transposes 4-bit blocks, 2 threads apart.
  bits_x_lo = __shfl_xor_sync(kWarpAll, bits_lo, 2);
  bits_x_hi = __shfl_xor_sync(kWarpAll, bits_hi, 2);
  shift = 4 - (threadIdx.x << 2 & 8);  // { 4, -4 }
  mask = (threadIdx.x & 2) ? 0xF0F0F0F0 : 0x0F0F0F0F;
  bits_lo = Blend(bits_lo, __funnelshift_l(bits_x_lo, bits_x_lo, shift), mask);
  bits_hi = Blend(bits_hi, __funnelshift_l(bits_x_hi, bits_x_hi, shift), mask);

  // Transposes 8-bit blocks, 4 threads apart.
  bits_x_lo = __shfl_xor_sync(kWarpAll, bits_lo, 4);
  bits_x_hi = __shfl_xor_sync(kWarpAll, bits_hi, 4);
  mask = (threadIdx.x & 4) ? 0x3715 : 0x6240;
  bits_lo = __byte_perm(bits_lo, bits_x_lo, mask);
  bits_hi = __byte_perm(bits_hi, bits_x_hi, mask);

  // Transposes 16-bit blocks, 8 threads apart.
  bits_x_lo = __shfl_xor_sync(kWarpAll, bits_lo, 8);
  bits_x_hi = __shfl_xor_sync(kWarpAll, bits_hi, 8);
  mask = (threadIdx.x & 8) ? 0x3276 : 0x5410;
  bits_lo = __byte_perm(bits_lo, bits_x_lo, mask);
  bits_hi = __byte_perm(bits_hi, bits_x_hi, mask);

  // Transposes 32-bit blocks, 16 threads apart.
  const int bits_src = (threadIdx.x & 16) ? bits_lo : bits_hi;
  const int bits_dst = __shfl_xor_sync(kWarpAll, bits_src, 16);
  if (threadIdx.x & 16)
    bits_lo = bits_dst;
  else
    bits_hi = bits_dst;

  // Recombines into 64-bit int.
  return bits_lo | (static_cast<uint64_t>(bits_hi) << 32);
}

// PackTilesRounded kernel: Packs an in_tile_size x in_tile_size tile of an
// input `value_t` grayscale image into linear lists of values (brightnesses)
// and tile-relative coordinates (y << 8 | x), so that they can be key-value
// sorted by cub::DeviceSegmentedRadixSort(). Corner pixels (outside the rounded
// rectangle defined by the kernel shape) are excluded.
template <typename value_t, int kTileStride>
__global__ void PackTilesRounded(const value_t* input_image, int input_stride,
                                 int in_tile_size, value_t* d_values,
                                 uint16_t* d_coords) {
  const int out_tile_pixels = (c_params.omnigram_elements + kOmnigramPad - 1) &
                              ~(kOmnigramPad - 1);

  const int block_offset = (blockIdx.y * gridDim.x + blockIdx.x);
  value_t* values = d_values + block_offset * out_tile_pixels;
  uint16_t* coords = d_coords + block_offset * out_tile_pixels;

  const int x = threadIdx.x;

  const int global_x = blockIdx.x * kTileStride + x;
  for (int y = 0; y < in_tile_size; ++y) {
    const int global_y = blockIdx.y * kTileStride + y;
    const int inset = c_params.pack_inset[y];
    const int offset = c_params.pack_offset[y] - inset;
    if (x < inset || x >= (in_tile_size - inset)) continue;
    const value_t value = input_image[global_y * input_stride + global_x];
    values[offset + x] = value;
    coords[offset + x] = y << 8 | x;
  }
}

template <typename value_t, int kOutTileSize, int kOrdinalShift, int kSeedsX,
          int kSeedsY>
__device__ __forceinline__ void SolveSeedPixels(
    const aligned_uint32_t* omnigram32, const value_t* g_values,
    aligned_uint8_t* hist_mem, int seed_offset_x, int seed_offset_y,
    int in_tile_size, value_t* d_output_image, int output_stride,
    int block_offset, int global_x, int global_y, int threadIdx, int& seed_piv,
    int& seed_ct) {
  constexpr int kElementsPerBucket = 1 << kOrdinalShift;
  constexpr int kSteps64 = kElementsPerBucket / 64;
  constexpr int kNumThreads = kOutTileSize * kSeedsY * 2;
  constexpr int kNumSeeds = kSeedsX * kSeedsY;
  constexpr int kNumSections = kNumThreads / kNumSeeds;
  constexpr int kSeedSpacingX = kOutTileSize / kSeedsX;
  constexpr int kSeedSpacingY = kOutTileSize / kSeedsY;
  constexpr int kSeedOffsetX = kSeedSpacingX / 2;
  constexpr int kSeedOffsetY = kSeedSpacingY / 2;
  const int padding = (in_tile_size - kOutTileSize) / 2;

  const int threshold = c_params.threshold;
  const int percentile = c_params.percentile;
  const int seed = threadIdx & (kNumSeeds - 1);
  const int section = threadIdx / kNumSeeds;

  const int num_buckets =
      (c_params.omnigram_elements + (kElementsPerBucket - 1)) /
      kElementsPerBucket;
  const int buckets_per_section =
      (num_buckets + (kNumSections - 1)) / kNumSections;
  const int hist_stride = (num_buckets + 3) & ~3;  // uint32 alignment

  uint8_t* hist8 = hist_mem + seed * hist_stride;
  const int seed_x = seed & (kSeedsX - 1);
  const int seed_y = seed / kSeedsX;
  const int out_x = kSeedOffsetX + seed_x * kSeedSpacingX;
  const int out_y = kSeedOffsetY + seed_y * kSeedSpacingY;
  const int sx = padding + out_x;
  const int sy = padding + out_y;
  const int seed_xyxy = __byte_perm(sx, sy, 0x4040);
  // 'bucket' = histogram bucket index, in [0 .. num_buckets - 1].
  for (int bucket = section * buckets_per_section;
       bucket < std::min((section + 1) * buckets_per_section, num_buckets);
       ++bucket) {
    int sum = 0;

    for (int step = 0; step < kElementsPerBucket; step += 64) {
      uint32_t xyxy =
          omnigram32[(bucket * kElementsPerBucket + step) / 2 + (seed & 31)];
      sum += InCircleSum(xyxy, seed_xyxy, threshold);
      for (int i = 1; i < 32; ++i) {
        xyxy = __shfl_sync(kWarpAll, xyxy, seed - 1);
        sum += InCircleSum(xyxy, seed_xyxy, threshold);
      }
    }

    hist8[bucket] = sum;
  }

  __syncthreads();

  // Finds the coarse histogram bucket containing the median.
  if (threadIdx < kNumSeeds) {
    uint32_t h4;
    const aligned_uint32_t* h32 =
        reinterpret_cast<const aligned_uint32_t*>(hist8);
    for (; seed_piv < num_buckets; seed_piv += 4) {
      h4 = h32[seed_piv / 4];
      const int sum4 = __vsadu4(h4, 0);
      seed_ct += sum4;
      if (seed_ct >= percentile) {
        seed_ct -= sum4;
        break;
      }
    }

    // Walks through the last few bytes to find the exact bucket.
    if (seed_ct + (h4 & 0xFF) < percentile) {
      ++seed_piv;
      seed_ct += h4 & 0xFF;
      if (seed_ct + (h4 >> 8 & 0xFF) < percentile) {
        ++seed_piv;
        seed_ct += h4 >> 8 & 0xFF;
        if (seed_ct + (h4 >> 16 & 0xFF) < percentile) {
          ++seed_piv;
          seed_ct += h4 >> 16 & 0xFF;
        }
      }
    }

    // Finds the exact median solution.
    int pivot = seed_piv << kOrdinalShift;

    // Count should be less than percentile.
    int count = seed_ct;
    int nth = percentile - count;
    int piv = pivot;

    // Constructs 64-bit bitmasks.
    uint64_t bits[kSteps64] = {0};
    for (int step = 0; step < kSteps64; ++step) {
      for (int i = 0; i < 64; i += 2) {
        uint32_t xyxy = omnigram32[(pivot + 64 * step + i) / 2];
        uint64_t two_bits = InCircleBits(xyxy, seed_xyxy, threshold);
        bits[step] |= two_bits << i;
      }
    }
    if constexpr (kSteps64 == 2) {
      if (__popcll(bits[0]) < nth) {
        piv += 64;
        nth -= __popcll(bits[0]);
        bits[0] = bits[1];
      }
    }
    uint32_t bits_lo = bits[0] & 0xFFFFFFFF;
    uint32_t bits_hi = bits[0] >> 32;
    if (__popc(bits_lo) < nth) {
      piv += 32;
      nth -= __popc(bits_lo);
      bits_lo = bits_hi;
    }
    piv += __fns(bits_lo, 0, nth);

    // Writes solution to output.
    d_output_image[(global_y + out_y) * output_stride + (global_x + out_x)] =
        g_values[piv];
  }
}

template <typename value_t, int kSteps64>
__device__ __forceinline__ void SolveLastBits(
    uint8_t* row_pivot, uint16_t* row_count, int out_x, int out_y,
    int percentile, uint64_t refinement_bits[kSteps64], const value_t* g_values,
    value_t* d_output_image, int global_x, int global_y, int output_stride) {
  constexpr int kOrdinalShift = kSteps64 == 1 ? 6 : 7;
  int piv = *row_pivot;
  int count = *row_count;  // $$$ use a reference??
  if (count >= percentile) {
    --piv;
    for (int step = 0; step < kSteps64; ++step) {
      count -= __popcll(refinement_bits[step]);
    }
  }
  int solution = piv << kOrdinalShift;
  int piv_adj = piv;
  int count_adj = count;
  uint64_t ref_bits = refinement_bits[0];
  if constexpr (kSteps64 == 2) {
    if (count + __popcll(ref_bits) < percentile) {
      // Note: the omnigram size is always small enough that we don't have
      // to worry about piv_adj overflowing to 256 here.
      piv_adj++;
      count_adj += __popcll(ref_bits) + __popcll(refinement_bits[1]);
      count += __popcll(ref_bits);
      ref_bits = refinement_bits[1];
      solution += 64;
    }
  }

  int bits_lo = ref_bits & 0xFFFFFFFF;
  int bits_hi = ref_bits >> 32;
  if (count + __popc(bits_lo) < percentile) {
    // The solution is nearer to the upper end of the segment, so we
    // adjust the pivot and count to the higher end. (Note that we can't
    // do this if it would make the quantized pivot overflow to 256.)
    if constexpr (kSteps64 == 1) {
      if (piv_adj < 255) {
        piv_adj++;
        count_adj += __popcll(ref_bits);
      }
    }
    count += __popc(bits_lo);
    solution += 32 + __fns(bits_hi, 0, percentile - count);
  } else {
    solution += __fns(bits_lo, 0, percentile - count);
  }

  *row_pivot = piv_adj;
  *row_count = count_adj;

  // Writes to output.
  d_output_image[(global_y + out_y) * output_stride + (global_x + out_x)] =
      g_values[solution];
}

// From the Percentile kernel, sweeps the kernel windows horizontally left and
// right from the seed pixel locations, completing the seed rows of output.
template <typename value_t, int kOutTileSize, int kSteps64, int kSeedsX,
          int kSeedsY>
__device__ __forceinline__ void HorizontalSweep(
    const value_t* g_values, uint8_t row_pivots[kSeedsY][kOutTileSize],
    uint16_t row_counts[kSeedsY][kOutTileSize], const uint8_t* ordinal_image,
    const aligned_uint32_t* omnigram32, int in_tile_size, int seed_offset_x,
    int seed_offset_y, value_t* d_output_image, int output_stride,
    int block_offset, int global_x, int global_y, int threadIdx) {
  // Steps horizontally left and right. (2 * kSeedsX * kSeedsY threads.) Threads
  // with an ID < kSeedsX * kSeedsY will step left, and threads with an x-index
  // >= kSeedsX * kSeedsY will step right.
  const int threshold = c_params.threshold;
  const int percentile = c_params.percentile;
  const int16_t* offsets_e = c_params.offsets_e.data();
  const int16_t* offsets_w = c_params.offsets_w.data();

  // Seed solution location in the sparse 8x4 grid.
  int seed_x = threadIdx & (kSeedsX - 1);
  int seed_y = (threadIdx / kSeedsX) & (kSeedsY - 1);

  const int padding = (in_tile_size - kOutTileSize) / 2;
  constexpr int kOrdinalShift = kSteps64 + 5;
  constexpr int kSeedSpacingX = kOutTileSize / kSeedsX;
  constexpr int kSeedSpacingY = kOutTileSize / kSeedsY;

  // Output pixel coordinates, and kernel center coordinates [in the input]
  // of the seed solution. ('out_x' and 'kcenter_x' will slide left and
  // right.)
  int out_x = seed_offset_x + seed_x * kSeedSpacingX;
  const int out_y = seed_offset_y + seed_y * kSeedSpacingY;
  int kcenter_x = padding + out_x;
  const int kcenter_y = padding + out_y;

  const int h_direction = threadIdx < kSeedsX * kSeedsY ? -1 : 1;

  uint8_t pivot = row_pivots[seed_y][out_x];
  uint16_t count = row_counts[seed_y][out_x];

  for (int step = 0; step < kSeedSpacingX; ++step) {
    if (h_direction == 1 && step == kSeedSpacingX - 1 - seed_offset_x) break;
    if (h_direction == -1 && step == seed_offset_x) break;

    out_x += h_direction;      // New output pixel.
    kcenter_x += h_direction;  // New kernel center in input.
    const uint32_t kxyxy = __byte_perm(kcenter_x, kcenter_y, 0x4040);

    // Adds leading pixels at the edge of the circular window, and subtracts
    // trailing pixels. This determines the number of pixels in the new
    // circular window that are less than the quantized pivot.
    // TODO: Reduce bank conflicts by staggering indices per
    // thread? Or by using global memory for the ordinal image instead of
    // shared ?
    // TODO: Can we use more threads here, combining with atomic ops?
    if (h_direction == 1) {
      const uint8_t* center =
          &ordinal_image[kcenter_y * in_tile_size + kcenter_x];
      for (int krow = 0; krow < c_params.num_offsets_h; ++krow) {
        const uint8_t ordinal_leading = center[offsets_e[krow]];
        const uint8_t ordinal_trailing = center[offsets_w[krow]];
        if (ordinal_leading < pivot) ++count;
        if (ordinal_trailing < pivot) --count;
      }
    } else {
      const uint8_t* center =
          &ordinal_image[kcenter_y * in_tile_size + kcenter_x + 1];
      for (int krow = 0; krow < c_params.num_offsets_h; ++krow) {
        const uint8_t ordinal_leading = center[offsets_w[krow]];
        const uint8_t ordinal_trailing = center[offsets_e[krow]];
        if (ordinal_leading < pivot) ++count;
        if (ordinal_trailing < pivot) --count;
      }
    }

    uint64_t refinement_bits[kSteps64];
    int my_ballot = 1;

    // TODO: The first iteration always involves every thread, so
    // could be special-cased to avoid the cross-thread transpose overhead.
    uint32_t ballot = 0xFFFFFFFF;  // Which threads still need work.
    while (ballot != 0) {
      uint64_t bits[kSteps64] = {0};
      // Accumulates binary histogram bits, two at a time.
      int k = __fns(ballot, 0, 1);  // Finds first set bit.
      while (k != -1) {
        // Pivots / counts for column k being processed by all threads.
        int k_pivot = __shfl_sync(kWarpAll, pivot, k);
        const int k_count = __shfl_sync(kWarpAll, count, k);
        const uint32_t k_kxyxy = __shfl_sync(kWarpAll, kxyxy, k);
        if (k_count >= percentile) {
          // Decrements the pivot because we want to read the preceding block
          // in the omnigram if we're scanning downward.
          --k_pivot;
        }
// Each warp of 32 threads reads a 128-byte (64-element) chunk from
// the omnigram, in 32 uint32_t chunks. For larger kernels, two such
// chunks are read.
#pragma unroll
        for (int step = 0; step < kSteps64; ++step) {
          uint32_t xyxy = omnigram32[(k_pivot << (kOrdinalShift - 1)) +
                                     (threadIdx & 31) + step * 32];
          uint32_t two_bits = InCircleBits(xyxy, k_kxyxy, threshold);
          bits[step] |= static_cast<uint64_t>(two_bits) << (2 * k);
        }
        // Finds next set bit.
        k = __fns(ballot, k, 2);
      }  // end accumulating bits

      // Warp transpose, so that each thread now has its own bits.
      for (int step = 0; step < kSteps64; ++step) {
        bits[step] = Transpose64x32Sync(bits[step], threadIdx);
      }

      if (my_ballot == 1) {
        // Determines the omnigram scan direction.
        int scan_direction = (count < percentile) ? 1 : -1;
        int popct = 0;
        for (int step = 0; step < kSteps64; ++step) {
          popct += __popcll(bits[step]);
        }
        if (scan_direction == -1) {
          count -= popct;
          --pivot;
        }
        const bool solution_in_this_block = scan_direction == 1
                                                ? count + popct >= percentile
                                                : count < percentile;
        if (solution_in_this_block) {
          for (int step = 0; step < kSteps64; ++step) {
            refinement_bits[step] = bits[step];
          }
          my_ballot = 0;
        } else {
          if (scan_direction == 1) {
            count += popct;
            ++pivot;
          }
        }
      }
      ballot = __ballot_sync(kWarpAll, my_ballot);
    }
    SolveLastBits<value_t, kSteps64>(&pivot, &count, out_x, out_y, percentile,
                                     refinement_bits, g_values, d_output_image,
                                     global_x, global_y, output_stride);
    row_pivots[seed_y][out_x] = pivot;
    row_counts[seed_y][out_x] = count;
  }
}

// From the Percentile kernel, sweeps the kernel windows vertically up and down
// from the seed rows, completing the tile of output. (kOutTileSize threads in
// x, kSeedsY * 2 threads in y.)
template <typename value_t, int kOutTileSize, int kSteps64, int kSeedsX,
          int kSeedsY>
__device__ __forceinline__ void VerticalSweep(
    const value_t* g_values, uint8_t row_pivots[kSeedsY][kOutTileSize],
    uint16_t row_counts[kSeedsY][kOutTileSize], const uint8_t* ordinal_image,
    const aligned_uint32_t* omnigram32, int in_tile_size, int seed_offset_y,
    value_t* d_output_image, int output_stride, int block_offset, int global_x,
    int global_y, int threadIdx_x, int threadIdx_y) {
  const int threshold = c_params.threshold;
  const int percentile = c_params.percentile;
  const int16_t* offsets_n = c_params.offsets_n.data();
  const int16_t* offsets_s = c_params.offsets_s.data();

  const int padding = (in_tile_size - kOutTileSize) / 2;
  constexpr int kOrdinalShift = kSteps64 + 5;  // kPadding <= 32 ? 6 : 7;
  constexpr int kSeedSpacingY = kOutTileSize / kSeedsY;

  const int v_direction = threadIdx_y < kSeedsY ? -1 : 1;

  // With 64 output columns, one warp processes the odd columns and the other
  // processes the even columns. This substantially reduces bank conflicts
  // during the window-sliding phase.
  const int out_x = kOutTileSize == 64
                        ? (threadIdx_x << 1 & 62) | (threadIdx_x >> 5 & 1)
                        : threadIdx_x;
  int out_y = seed_offset_y + (threadIdx_y & (kSeedsY - 1)) * kSeedSpacingY;
  const int kcenter_x = padding + out_x;
  int kcenter_y = padding + out_y;
  uint32_t kxyxy;

  uint8_t pivot = row_pivots[threadIdx_y & (kSeedsY - 1)][out_x];
  uint16_t count = row_counts[threadIdx_y & (kSeedsY - 1)][out_x];

  const int kNumSteps = std::max(seed_offset_y, kSeedSpacingY - seed_offset_y);

  for (int step = 0; step < kNumSteps; ++step) {
    // One direction may finish processing before the other. We would normally
    // break out of the loop in these cases, but we need all threads to reach
    // all the __syncthreads() calls below to avoid deadlock.
    if (v_direction == 1) {
      if (step == kSeedSpacingY - seed_offset_y - 1) break;
    } else {
      if (step == seed_offset_y) break;
    }

    // Slides circular windows up/down one pixel.
    // Updates the center of the kernel window.
    out_y += v_direction;
    kcenter_y += v_direction;
    kxyxy = __byte_perm(kcenter_x, kcenter_y, 0x4040);

    // Adds leading pixels at the edge of the circular window, and subtracts
    // trailing pixels. This determines the number of pixels in the new
    // circular window that are less than the quantized pivot.
    if (v_direction == 1) {
      const uint8_t* center =
          &ordinal_image[kcenter_y * in_tile_size + kcenter_x];
      for (int kcol = 0; kcol < c_params.num_offsets_v; ++kcol) {
        const uint8_t ordinal_leading = center[offsets_s[kcol]];
        const uint8_t ordinal_trailing = center[offsets_n[kcol]];
        if (ordinal_leading < pivot) ++count;
        if (ordinal_trailing < pivot) --count;
      }
    } else {
      const uint8_t* center =
          &ordinal_image[(kcenter_y + 1) * in_tile_size + kcenter_x];
      for (int kcol = 0; kcol < c_params.num_offsets_v; ++kcol) {
        const uint8_t ordinal_leading = center[offsets_n[kcol]];
        const uint8_t ordinal_trailing = center[offsets_s[kcol]];
        if (ordinal_leading < pivot) ++count;
        if (ordinal_trailing < pivot) --count;
      }
    }

    // Scans the omnigram at full resolution.
    int my_ballot = 1;
    uint64_t refinement_bits[kSteps64];

    // TODO: The first iteration always involves every thread, so
    // could be special-cased to avoid the cross-thread transpose overhead.
    uint32_t ballot = 0xFFFFFFFF;  // Which threads still need work.
    while (ballot != 0) {
      uint64_t bits[kSteps64] = {0};
      // Accumulates binary histogram bits, two at a time.
      int k = __fns(ballot, 0, 1);
      while (k != -1) {  // Finds nth set bit.
                         // Each thread reads from all active columns.
        // Pivots / counts for column k being processed by all threads.
        int k_pivot = __shfl_sync(kWarpAll, pivot, k);
        int k_count = __shfl_sync(kWarpAll, count, k);

        if (k_count >= percentile) {
          // Decrements the pivot because we want to read the preceding block
          // in the omnigram if we're scanning downward.
          --k_pivot;
        }
// Each warp of 32 threads reads a 128-byte (64-element) chunk from
// the omnigram, in 32 uint32_t chunks. The '32' bit of threadIdx_x
// was used above to select the pivot and count from the left or right
// half of the output image respectively, so here we only need the
// "& 31" bits to select which of the 32 chunks to read.
#pragma unroll
        for (int step = 0; step < kSteps64; ++step) {
          uint32_t xyxy = omnigram32[(k_pivot << (kOrdinalShift - 1)) +
                                     (threadIdx_x & 31) + step * 32];
          const uint32_t k_kxyxy = __shfl_sync(kWarpAll, kxyxy, k);
          uint32_t two_bits = InCircleBits(xyxy, k_kxyxy, threshold);
          bits[step] |= static_cast<uint64_t>(two_bits) << (2 * k);
        }
        k = __fns(ballot, k, 2);  // Finds next set bit.
      }  // end accumulating bits

      // Warp transpose, so that each thread now has its own bits.
      for (int step = 0; step < kSteps64; ++step) {
        bits[step] = Transpose64x32Sync(bits[step], threadIdx_x);
      }

      if (my_ballot == 1) {
        // Determines the omnigram scan direction.
        int scan_direction = (count < percentile) ? 1 : -1;
        int popct = 0;
        for (int step = 0; step < kSteps64; ++step) {
          popct += __popcll(bits[step]);
        }
        if (scan_direction == -1) {
          count -= popct;
          --pivot;
        }
        const bool solution_in_this_block = scan_direction == 1
                                                ? count + popct >= percentile
                                                : count < percentile;
        if (solution_in_this_block) {
          for (int step = 0; step < kSteps64; ++step) {
            refinement_bits[step] = bits[step];
          }
          my_ballot = 0;
        } else {
          if (scan_direction == 1) {
            count += popct;
            ++pivot;
          }
        }
      }  // end if (my_ballot == 1)
      ballot = __ballot_sync(kWarpAll, my_ballot);
    }  // end while (ballot != 0)

    // Now we obtain the full-precision solution.
    SolveLastBits<value_t, kSteps64>(&pivot, &count, out_x, out_y, percentile,
                                     refinement_bits, g_values, d_output_image,
                                     global_x, global_y, output_stride);
  }  // end for step
}

// FastIsotropicMedianFilter kernel. Given key-value-sorted values
// (brightnesses) and coordinates (y << 8 | x) from the corresponding input
// tile, applies a circular median filter to a single-channel output tile.
// Thread configuration: kOutTileSize in X, 2 * kSeedsY in Y.
template <typename value_t, int kOutTileSize, int kOrdinalShift, int kSeedsX,
          int kSeedsY>
__global__ __launch_bounds__(2 * kSeedsY * kOutTileSize) void
    FastIsotropicMedianFilter(const value_t* __restrict__ d_values,
                              const uint16_t* __restrict__ d_coords,
                              int in_tile_size, value_t* d_output_image,
                              int output_stride) {
  constexpr int kOutTileWidth = kOutTileSize;
  constexpr int kOutTileHeight = kOutTileSize;
  constexpr int kSeedSpacingX = kOutTileSize / kSeedsX;
  constexpr int kSeedSpacingY = kOutTileSize / kSeedsY;
  constexpr int kSeedOffsetX = kSeedSpacingX / 2;
  constexpr int kSeedOffsetY = kSeedSpacingY / 2;
  constexpr int kSteps64 = (1 << kOrdinalShift) / 64;

  const int in_tile_pixels = (c_params.omnigram_elements + kOmnigramPad - 1) &
                             ~(kOmnigramPad - 1);
  const int block_offset = (blockIdx.y * gridDim.x + blockIdx.x);
  const int global_x = blockIdx.x * kOutTileWidth;  // top left of output tile.
  const int global_y = blockIdx.y * kOutTileHeight;

  // The omnigram stays in global memory; the ordinal image and pivots/counts
  // are placed in shared memory.
  // Dynamic size: [in_tile_width * in_tile_height]
  extern __shared__ __align__(128) uint8_t ordinal_image[];

  __shared__ __align__(128) uint8_t row_pivots[kSeedsY][kOutTileWidth];
  __shared__ __align__(128) uint16_t row_counts[kSeedsY][kOutTileWidth];

  // Pointers to this tile's reverse map (g_values) and omnigram (g_coords) in
  // global memory.
  const value_t* g_values = d_values + block_offset * in_tile_pixels;
  const uint16_t* g_coords = d_coords + block_offset * in_tile_pixels;

  // The inner loops read the omnigram in two-element (32-bit) chunks.
  // For convenience, we define a 32-bit pointer here, with explicit alignment.
  const aligned_uint32_t* omnigram32 =
      reinterpret_cast<const aligned_uint32_t*>(g_coords);

  // For the seed pixels, we remember the pivots and counts in registers, and
  // write them to the row_pivots and row_counts arrays after ordinal image
  // construction is complete.
  int seed_piv = 0;
  int seed_ct = 0;

  // The ordinal image is [in_tile_size * in_tile_size] bytes, which ranges from
  // 25600 to 65536 bytes. The omnigram size is a "round rect" of this size,
  // with a maximum size of roughly 65536 - 193 * 193 * (1 - pi / 4) = 57542
  // bytes, ~29255 of which cannot be solutions, leaving ~28247
  // possible solutions. The omnigram pointer will be offset to ~128 elements
  // before the first possible solution, and quantized ordinal image values
  // clamped to [0..255] based on that offset.
  SolveSeedPixels<value_t, kOutTileSize, kOrdinalShift, kSeedsX, kSeedsY>(
      omnigram32, g_values, /*hist_mem=*/ordinal_image, kSeedOffsetX,
      kSeedOffsetY, in_tile_size, d_output_image, output_stride, block_offset,
      global_x, global_y, threadIdx.x + threadIdx.y * blockDim.x, seed_piv,
      seed_ct);

  __syncthreads();  // TODO: necessary?

  // Writes remembered seed_piv and seed_ct values to row_pivots and row_counts.
  // (32 threads.)
  const int omnigram_offset = (c_params.percentile >> kOrdinalShift) - 1;
  if (threadIdx.y == 0 && threadIdx.x < kSeedsX * kSeedsY) {
    const int seed_x = threadIdx.x & (kSeedsX - 1);
    const int seed_y = threadIdx.x / kSeedsX;
    row_pivots[seed_y][kSeedOffsetX + seed_x * kSeedSpacingX] =
        seed_piv - omnigram_offset;
    row_counts[seed_y][kSeedOffsetX + seed_x * kSeedSpacingX] = seed_ct;
  }

  __syncthreads();  // TODO: necessary?

  // Constructs the ordinal map from the omnigram, overwriting the histograms.
  // Note that we construct it from the coords array in global memory, because
  // part of the shared-memory omnigram has been overwritten by the
  // row_pivots/row_counts arrays.
  // TODO: Could be marginally improved by using uint4 loads?
  {
    // Note that by symmetry, the number of omnigram elements is even, so we can
    // safely divide it by 2. And the shift is reduced by one since we're
    // indexing into uint32's instead of uint16's.
    constexpr int kShift = kOrdinalShift - 1;
    const int num_threads = blockDim.x * blockDim.y;
    const int num_uint32s = c_params.omnigram_elements / 2;
    int ind = threadIdx.y * blockDim.x + threadIdx.x;  // 0 .. num_threads - 1
    for (; ind < num_uint32s; ind += num_threads) {
      const int ordinal_q =
          std::clamp((ind >> kShift) - omnigram_offset, 0, 255);
      const uint32_t xyxy = omnigram32[ind];
      const int x0 = xyxy & 0xFF;
      const int y0 = (xyxy >> 8) & 0xFF;
      const int x1 = (xyxy >> 16) & 0xFF;
      const int y1 = (xyxy >> 24) & 0xFF;
      ordinal_image[y0 * in_tile_size + x0] = ordinal_q;
      ordinal_image[y1 * in_tile_size + x1] = ordinal_q;
    }

    // Adjusts the 'omnigram32' pointer to reflect the offset of the quantized
    // ordinal image values. Also adjusts the g_values (reverse map) pointer
    // to match.
    omnigram32 += omnigram_offset * (1 << kShift);
    g_values += omnigram_offset * (1 << kOrdinalShift);
  }

  __syncthreads();

  // Sweeps left and right from the seed pixels to complete the seed rows.
  // (32 or 64 threads.)
  if (threadIdx.y * blockDim.x + threadIdx.x < kSeedsX * kSeedsY * 2) {
    HorizontalSweep<value_t, kOutTileSize, kSteps64, kSeedsX, kSeedsY>(
        g_values, row_pivots, row_counts, ordinal_image, omnigram32,
        in_tile_size, kSeedOffsetX, kSeedOffsetY, d_output_image, output_stride,
        block_offset, global_x, global_y,
        threadIdx.y * blockDim.x + threadIdx.x);
  }

  __syncthreads();

  // Sweeps up and down from the seed rows to complete the tile.
  VerticalSweep<value_t, kOutTileSize, kSteps64, kSeedsX, kSeedsY>(
      g_values, row_pivots, row_counts, ordinal_image, omnigram32, in_tile_size,
      kSeedOffsetY, d_output_image, output_stride, block_offset, global_x,
      global_y, threadIdx.x, threadIdx.y);
}

template <typename T>
absl::StatusOr<T*> AllocateCudaBuffer(size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  T* buf;
  cudaError_t err = cudaMalloc(&buf, bytes);
  if (err != cudaSuccess) {
    return absl::InternalError("Error allocating CUDA buffer");
  }
  return buf;
}

bool IsValidDevicePointer(const void* ptr) {
  cudaPointerAttributes attributes;
  cudaError_t error = cudaPointerGetAttributes(&attributes, ptr);
  if (error == cudaSuccess) {
    return attributes.type == cudaMemoryTypeDevice;
  } else if (error == cudaErrorInvalidValue) {
    return false;
  } else {
    printf("Error getting pointer attributes: %s\n", cudaGetErrorString(error));
    return false;
  }
}

FastIsotropicMedianFilterParams ComputeFastIsotropicMedianFilterParams(
    const FastIsotropicMedianFilterCudaOptions& options, int in_tile_size,
    bool rounded) {
  FastIsotropicMedianFilterParams params;
  const int radius = options.radius;
  params.radius = radius;
  params.threshold = radius == 1 ? 1 : (radius + 0.5f) * (radius + 0.5f);
  params.omnigram_elements = in_tile_size * in_tile_size;
  params.num_offsets_h = 2 * radius + 1;
  params.num_offsets_v = 2 * radius + 1;

  int support = (2 * radius + 1) * (2 * radius + 1);

  uint8_t insets[256];
  for (int i = 0; i <= radius; ++i) {
    const int chord = sqrt(params.threshold - (radius - i) * (radius - i));
    insets[i] = radius - chord;
  }

  for (int i = 0; i <= 2 * radius; ++i) {
    const int inset = insets[radius - std::abs(radius - i)];

    support -= 2 * inset;
    params.offsets_n[i] = (inset - radius - 1) * in_tile_size + (i - radius);
    params.offsets_s[i] = (radius - inset - 0) * in_tile_size + (i - radius);
    params.offsets_w[i] = (inset - radius - 1) + in_tile_size * (i - radius);
    params.offsets_e[i] = (radius - inset - 0) + in_tile_size * (i - radius);
  }

  if (rounded) {
    int offset = 0;
    for (int i = 0; i < in_tile_size; ++i) {
      const int inset = i < radius ? insets[i]
                        : i > in_tile_size - 1 - radius
                            ? insets[in_tile_size - 1 - i]
                            : 0;
      params.omnigram_elements -= 2 * inset;
      params.pack_inset[i] = inset;
      params.pack_offset[i] = offset;
      offset += in_tile_size - 2 * inset;
    }
  }

  params.percentile = options.percentile * (support - 1) + 1.5f;
  return params;
}

template <typename value_t>
int IdealOutputTileSize(int radius) {
  if constexpr (std::is_same_v<value_t, uint8_t>) {
    return radius <= 20 ? 32 : 64;
  } else if constexpr (std::is_same_v<value_t, uint16_t>) {
    return radius < 16 ? 32 : 64;
  } else if constexpr (std::is_same_v<value_t, float>) {
    return 64;
  } else {
    // static_assert(false, "Unsupported type");
    LOG(FATAL) << "Unsupported type";
  }
}

template <typename value_t, int kOutTileSize, int kOrdinalShift, int kSeedsX,
          int kSeedsY>
void CallFastIsotropicMedianFilter(int num_tiles_x, int num_tiles_y,
    int in_tile_size, int shared_mem_bytes,
    const FastIsotropicMedianFilterBuffers<value_t>& buffers,
    value_t* output_base, int output_pitch, cudaStream_t stream) {
  cudaFuncSetAttribute(
      FastIsotropicMedianFilter<value_t, kOutTileSize, kOrdinalShift, kSeedsX,
                                kSeedsY>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_bytes);
  dim3 percentile_filter_block(kOutTileSize, kSeedsY * 2, 1);
  dim3 grid_output(num_tiles_x, num_tiles_y);
  FastIsotropicMedianFilter<value_t, kOutTileSize, kOrdinalShift, kSeedsX,
                            kSeedsY>
      <<<grid_output, percentile_filter_block, shared_mem_bytes, stream>>>(
          buffers.d_reverse_maps, buffers.d_omnigrams, in_tile_size,
          output_base, output_pitch / sizeof(value_t));
}

// Applies a percentile filter to an input image, producing an
// output image inset by the filter radius on all sides. Pointers must be device
// memory. Pitch is in bytes.The output image dimensions must be multiples of 64
// pixels.
template <typename value_t>
absl::Status FastIsotropicMedianFilter(const value_t* input_base,
    int input_width, int input_height, int input_pitch,
    const FastIsotropicMedianFilterCudaOptions& options,
    const FastIsotropicMedianFilterBuffers<value_t>& buffers,
    value_t* output_base, int output_width, int output_height, int output_pitch,
    cudaStream_t stream) {
  const int radius = options.radius;

  if (input_width != buffers.input_width ||
      input_height != buffers.input_height ||
      output_width != buffers.output_width ||
      output_height != buffers.output_height) {
    return absl::InvalidArgumentError(
        "Input and output image dimensions must match preallocated buffers.");
  }

  cudaStreamSynchronize(stream);

  const int out_tile_size = IdealOutputTileSize<value_t>(radius);
  const int out_tile_width = out_tile_size;
  const int out_tile_height = out_tile_size;

  const int in_tile_size = out_tile_size + 2 * radius;
  const int in_tile_width = in_tile_size;
  const int in_tile_height = in_tile_size;

  if (!IsValidDevicePointer(input_base) || !IsValidDevicePointer(output_base)) {
    return absl::InvalidArgumentError(
        "Input and output pointers must be device memory.");
  }

  if (output_width % out_tile_width != 0 ||
      output_height % out_tile_height != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Output image dimensions must be multiples of %d.", out_tile_size));
  }
  if (input_width != output_width + 2 * radius ||
      input_height != output_height + 2 * radius) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "FastIsotropicMedianFilter: output image dimensions must be inset by "
        "the filter radius = %d on all sides with respect to input.",
        radius));
  }


  const int num_tiles_x = output_width / out_tile_width;
  const int num_tiles_y = output_height / out_tile_height;
  const int num_tiles = num_tiles_x * num_tiles_y;

  // Computes and uploads kernel parameters.
  // TODO: Skip this if the parameters have not changed. (But we do
  // need to get the correct number of omnigram elements.)
  int omnigram_elements = 0;
  {
    FastIsotropicMedianFilterParams params =
        ComputeFastIsotropicMedianFilterParams(options, in_tile_size,
                                               /*rounded=*/true);

    cudaMemcpyToSymbol(c_params, &params,
                       sizeof(FastIsotropicMedianFilterParams));
    omnigram_elements = params.omnigram_elements;
  }

  // The log timestamps can be used to infer timing (useful during development),
  // but for as-yet-unknown reasons there are some unusual stalls the first time
  // through the kernel sequence, specifically with the VerticalMerge kernel. So
  // we can process the kernel sequence a few times during testing, to obtain
  // more accurate timing.
    const int threads_x = (in_tile_size + 31) & ~31;  // {192, 224, 256}
    dim3 packTiles_block(threads_x, 1, 1);
    dim3 grid_input(num_tiles_x, num_tiles_y);
    if (out_tile_size == 32) {
      PackTilesRounded<value_t, /*kOutTileSize=*/32>
          <<<grid_input, packTiles_block, 0, stream>>>(
              input_base, input_pitch / sizeof(value_t), in_tile_size,
              buffers.d_packed_values, buffers.d_packed_coords);
    } else {
      PackTilesRounded<value_t, /*kOutTileSize=*/64>
          <<<grid_input, packTiles_block, 0, stream>>>(
              input_base, input_pitch / sizeof(value_t), in_tile_size,
              buffers.d_packed_values, buffers.d_packed_coords);
    }

    cudaStreamSynchronize(stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return absl::InternalError(
          absl::StrCat("Error after PackTiles: ", cudaGetErrorString(err)));
    }

    const int num_segments = num_tiles;
    const int omnigram_elements_padded =
        (omnigram_elements + kOmnigramPad - 1) & ~(kOmnigramPad - 1);
    const int num_items = omnigram_elements_padded * num_segments;

    size_t temp_storage_bytes = buffers.temp_storage_bytes;

    value_t*& d_keys_in = buffers.d_packed_values;
    value_t*& d_keys_out = buffers.d_reverse_maps;
    uint16_t*& d_values_in = buffers.d_packed_coords;
    uint16_t*& d_values_out = buffers.d_omnigrams;

    // Key-value sorts the packed arrays for all tiles.
    cub::DoubleBuffer<value_t> d_keys(d_keys_in, d_keys_out);
    cub::DoubleBuffer<uint16_t> d_values(d_values_in, d_values_out);

    err = cub::DeviceSegmentedRadixSort::SortPairs(
        buffers.d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items,
        num_segments, buffers.d_begin_offsets, buffers.d_end_offsets, 0,
        8 * sizeof(value_t), stream);
    cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
      return absl::InternalError(
          absl::StrCat("Error from CUB DeviceSegmentedRadixSort::SortPairs: ",
                       cudaGetErrorString(err)));
    }

    // Keeps double-buffers in the correct logical order.
    if (d_keys.Current() != d_keys_out) {
      std::swap(d_keys_in, d_keys_out);
    }
    if (d_values.Current() != d_values_out) {
      std::swap(d_values_in, d_values_out);
    }

    // Calls the percentile filter kernel.
    const int shared_mem_bytes =
        in_tile_width * in_tile_height * sizeof(uint8_t);  // Ordinal image.
    const bool quantize_6 = options.radius <= 48;
    if (quantize_6) {
      if (out_tile_size == 32) {
        CallFastIsotropicMedianFilter<value_t, /*kOutTileSize=*/32,
                             /*kOrdinalShift=*/6, /*kSeedsX=*/8,
                             /*kSeedsY=*/4>(
            num_tiles_x, num_tiles_y, in_tile_size, shared_mem_bytes, buffers,
            output_base, output_pitch, stream);
      } else {
        CallFastIsotropicMedianFilter<value_t, /*kOutTileSize=*/64,
                             /*kOrdinalShift=*/6, /*kSeedsX=*/8,
                             /*kSeedsY=*/4>(
            num_tiles_x, num_tiles_y, in_tile_size, shared_mem_bytes, buffers,
            output_base, output_pitch, stream);
      }
    } else {
      if (out_tile_size == 32) {
        CallFastIsotropicMedianFilter<value_t, /*kOutTileSize=*/32,
                             /*kOrdinalShift=*/7, /*kSeedsX=*/8,
                             /*kSeedsY=*/4>(
            num_tiles_x, num_tiles_y, in_tile_size, shared_mem_bytes, buffers,
            output_base, output_pitch, stream);
      } else {
        CallFastIsotropicMedianFilter<value_t, /*kOutTileSize=*/64,
                             /*kOrdinalShift=*/7, /*kSeedsX=*/8,
                             /*kSeedsY=*/8>(
            num_tiles_x, num_tiles_y, in_tile_size, shared_mem_bytes, buffers,
            output_base, output_pitch, stream);
      }
    }
    cudaStreamSynchronize(stream);
  // }  // loop

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return absl::InternalError(absl::StrCat(
        "Error after FastIsotropicMedianFilter launch: ",
        cudaGetErrorString(err)));
  }

  return absl::OkStatus();
}
}  // namespace

template <typename value_t>
absl::Status FastIsotropicMedianFilterBuffers<value_t>::Allocate(
    const CudaGraySurface<value_t>& input,
    const FastIsotropicMedianFilterCudaOptions& options,
    CudaGraySurface<value_t>& output, int* bytes_allocated) {
  const int padding = (input.Width() - output.Width()) / 2;
  const int out_tile_size = IdealOutputTileSize<value_t>(options.radius);

  const int input_width = input.Width();
  const int input_height = input.Height();
  const int output_width = output.Width();
  const int output_height = output.Height();

  int bytes = 0;

  if (options.radius > padding) {
    return absl::InvalidArgumentError(
        "Radius must be less than or equal to padding.");
  }

  if (output_width % out_tile_size != 0 || output_height % out_tile_size != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Output image dimensions must be multiples of %d.", out_tile_size));
  }
  if (input_width != output_width + 2 * padding ||
      input_height != output_height + 2 * padding) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Allocate: Output image dimensions must be inset by %d pixels "
        "on all sides with respect to input.",
        padding));
  }
  const int in_tile_size = out_tile_size + 2 * options.radius;
  const int in_tile_width = in_tile_size;
  const int in_tile_height = in_tile_size;
  const int out_tile_width = out_tile_size;
  const int out_tile_height = out_tile_size;
  int in_tile_pixels = in_tile_width * in_tile_height;

  int omnigram_elements = in_tile_pixels;
  {
    // Pads omnigram buffers to 128-element (256-byte) alignment.
    FastIsotropicMedianFilterParams params =
        ComputeFastIsotropicMedianFilterParams(options, in_tile_size,
                                               /*rounded=*/true);

    const int omnigram_elements_padded =
        (params.omnigram_elements + kOmnigramPad - 1) & ~(kOmnigramPad - 1);
    in_tile_pixels = omnigram_elements_padded;
    omnigram_elements = params.omnigram_elements;
  }

  const int num_output_tiles_x = output_width / out_tile_width;
  const int num_output_tiles_y = output_height / out_tile_height;
  const int num_output_tiles = num_output_tiles_x * num_output_tiles_y;

  const int num_input_tiles_x = num_output_tiles_x;
  const int num_input_tiles_y = num_output_tiles_y;
  const int num_input_tiles = num_input_tiles_x * num_input_tiles_y;

  const int unsorted_elements = num_output_tiles * in_tile_pixels;
  const int unsorted_values_num_bytes = unsorted_elements * sizeof(value_t);
  const int unsorted_coords_num_bytes = unsorted_elements * sizeof(uint16_t);

  // Fully merged.
  const int merged_elements = num_output_tiles * in_tile_pixels;
  const int reverse_maps_num_bytes = merged_elements * sizeof(value_t);
  const int omnigrams_num_bytes = merged_elements * sizeof(uint16_t);

  auto status_or_d_packed_values =
    AllocateCudaBuffer<value_t>(unsorted_values_num_bytes);
  auto status_or_d_packed_coords =
    AllocateCudaBuffer<uint16_t>(unsorted_coords_num_bytes);
  auto status_or_d_reverse_maps =
    AllocateCudaBuffer<value_t>(reverse_maps_num_bytes);
  auto status_or_d_omnigrams =
    AllocateCudaBuffer<uint16_t>(omnigrams_num_bytes);

  if (!status_or_d_packed_values.ok() || !status_or_d_packed_coords.ok() ||
      !status_or_d_reverse_maps.ok() || !status_or_d_omnigrams.ok()) {
    return absl::InternalError(
        "Failed to allocate buffers for percentile filter.");
  }
  d_packed_values = status_or_d_packed_values.value();
  d_packed_coords = status_or_d_packed_coords.value();
  d_reverse_maps = status_or_d_reverse_maps.value();
  d_omnigrams = status_or_d_omnigrams.value();

  bytes = unsorted_values_num_bytes + unsorted_coords_num_bytes +
          reverse_maps_num_bytes + omnigrams_num_bytes;

  CHECK_NE(d_packed_values, nullptr);
  CHECK_NE(d_packed_coords, nullptr);
  CHECK_NE(d_reverse_maps, nullptr);
  CHECK_NE(d_omnigrams, nullptr);

  // CUB segmented radix sort setup.
  const int num_segments = num_input_tiles;
  const int sort_tile_pixels = in_tile_pixels;
  const int num_items = sort_tile_pixels * num_segments;
  std::vector<int> h_begin_offsets(num_segments + 1);
  for (int ind = 0; ind <= num_segments; ++ind) {
    h_begin_offsets[ind] = ind * sort_tile_pixels;
  }

  value_t*& d_keys_in = d_packed_values;
  value_t*& d_keys_out = d_reverse_maps;
  uint16_t*& d_values_in = d_packed_coords;
  uint16_t*& d_values_out = d_omnigrams;

  cudaMalloc(&d_begin_offsets, (num_segments + 1) * sizeof(int));
  bytes += (num_segments + 1) * sizeof(int);

  CHECK_NE(d_begin_offsets, nullptr);
  cudaMemcpy(d_begin_offsets, h_begin_offsets.data(),
             (num_segments + 1) * sizeof(int), cudaMemcpyHostToDevice);

  {
    std::vector<int> h_end_offsets(num_segments);
    for (int ind = 0; ind < num_segments; ++ind) {
      h_end_offsets[ind] = ind * sort_tile_pixels + omnigram_elements;
    }
    cudaMalloc(&d_end_offsets, num_segments * sizeof(int));
    bytes += num_segments * sizeof(int);
    CHECK_NE(d_end_offsets, nullptr);
    cudaMemcpy(d_end_offsets, h_end_offsets.data(), num_segments * sizeof(int),
               cudaMemcpyHostToDevice);
  }

  cub::DoubleBuffer<value_t> d_keys(d_keys_in, d_keys_out);
  cub::DoubleBuffer<uint16_t> d_values(d_values_in, d_values_out);

  cub::DeviceSegmentedRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items,
      num_segments, d_begin_offsets, d_end_offsets, 0, 8 * sizeof(value_t));

  // Allocates temporary storage for segmented radix sort.
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  bytes += temp_storage_bytes;

  if (bytes_allocated != nullptr) {
    *bytes_allocated = bytes;
  }
  return absl::OkStatus();
}

absl::Status FastIsotropicMedianFilter(const CudaGraySurface<uint8_t>& input,
  const FastIsotropicMedianFilterCudaOptions& options,
  const FastIsotropicMedianFilterBuffers<uint8_t>& buffers,
  CudaGraySurface<uint8_t>& output, cudaStream_t stream) {
  return FastIsotropicMedianFilter<uint8_t>(input.Base(), input.Width(),
    input.Height(), input.Pitch(), options, buffers, output.Base(),
    output.Width(), output.Height(), output.Pitch(), stream);
}

absl::Status FastIsotropicMedianFilter(const CudaGraySurface<uint16_t>& input,
  const FastIsotropicMedianFilterCudaOptions& options,
  const FastIsotropicMedianFilterBuffers<uint16_t>& buffers,
  CudaGraySurface<uint16_t>& output, cudaStream_t stream) {
  return FastIsotropicMedianFilter<uint16_t>(input.Base(), input.Width(),
    input.Height(), input.Pitch(), options, buffers, output.Base(),
    output.Width(), output.Height(), output.Pitch(), stream);
}

absl::Status FastIsotropicMedianFilter(const CudaGraySurface<float>& input,
  const FastIsotropicMedianFilterCudaOptions& options,
  const FastIsotropicMedianFilterBuffers<float>& buffers,
  CudaGraySurface<float>& output, cudaStream_t stream) {
  return FastIsotropicMedianFilter<float>(input.Base(), input.Width(),
  input.Height(), input.Pitch(), options, buffers, output.Base(),
  output.Width(), output.Height(), output.Pitch(), stream);
}

}  // namespace fast_isotropic_median_filter
