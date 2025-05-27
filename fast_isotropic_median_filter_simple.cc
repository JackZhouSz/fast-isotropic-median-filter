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

#include "fast_isotropic_median_filter_simple.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "image.h"
#include "thread_pool.h"

// The fast isotropic median filter is built on several techniques working
// together. First, tiles of the input 16-bit image are reversibly transformed
// to an "ordinal" space where each pixel value is unique. A lookup table is
// created which specifies where each ordinal value can be found within the
// tile. From this, the binary histogram of each circular window within the tile
// can be dynamically computed. After the solution is computed for a given
// output pixel, the circular window is "slid" down one pixel, and the histogram
// search direction and distance updated. (For instance, suppose the value 'p'
// is 50th of 99 values in a circular window. When we slide the window down one
// pixel we might find that 'p' is now 45th of the 99 values in the new window,
// so we must search the histogram upward to find 5 more values.) This window-
// sliding and updating can be processed efficiently for several adjacent
// columns at once. Finally, the ordinal transform is reversed to produce the
// final output for that tile.
namespace fast_isotropic_median_filter {
namespace {
constexpr int kHistPad = 128;
constexpr int kMaximumFilterRadius = 100;

// Empirical threshold for switching between quantized 8-bit and full 16-bit
// ordinal images.
constexpr int kMaximumQuantizedRadius = 32;

// When set, the last row of solutions for each tile is encoded and forwarded to
// the next tile, avoiding an expensive initialization step. Set up here as a
// constant, to enable easier future debugging if needed.
constexpr bool kFeedForward = true;

// In this simplified implementation the output tile size is fixed at 64x64
// pixels, but in principle it could go as high as 248x248.
constexpr int kMaxOutputTileCols = 248;

// The 'FindRank()' function that searches the compound histogram is templatized
// by search direction.
enum SearchDirection { kSearchUpward, kSearchDownward };

struct PixelOffset {
  int8_t dx;
  int8_t dy;
};

static ThreadPool* thread_pool;

template <typename cardinal_t, typename ordinal_t>
void OrdinalTransformSimple(ReadViewGray<cardinal_t> cardinal_input,
                            WriteViewGray<ordinal_t> ordinal_output,
                            absl::Span<cardinal_t> sorted_values,
                            absl::Span<uint16_t> sorted_coords) {
  constexpr int kOrdinalShift = std::is_same_v<ordinal_t, uint8_t> ? 6 : 0;
  const int width = ordinal_output.width();
  const int height = ordinal_output.height();
  CHECK_EQ(cardinal_input.width(), width);
  CHECK_EQ(cardinal_input.height(), height);
  CHECK_GE(sorted_values.size(), width * height);
  CHECK_EQ(sorted_coords.size(), width * height);

  std::vector<std::tuple<cardinal_t, uint16_t>> tuples(width * height);

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      tuples[row * width + col] =
          std::make_tuple(cardinal_input(col, row), row << 8 | col);
    }
  }

  // Note: since some of the cardinal values may be duplicates, and the relative
  // ordering must be consistent between overlapping tiles in order to enable
  // solution-forwarding from one to the next, we use stable_sort here.
  std::stable_sort(tuples.begin(), tuples.end());
  // std::stable_sort(tuples, tuples + width * height);

  for (int i = 0; i < width * height; ++i) {
    const uint16_t coords = std::get<1>(tuples[i]);
    const uint8_t y = coords >> 8;
    const uint8_t x = coords & 0xFF;
    sorted_values[i] = std::get<0>(tuples[i]);
    sorted_coords[i] = coords;
    ordinal_output(x, y) = i >> kOrdinalShift;
  }
}

// The threshold is used to determine inclusion in the circular window.
// For a pixel at offset (dx, dy) from the center, it is included iff:
//   (dx * dx + dy * dy) <= threshold.
// Radius = 1 is special-cased to avoid a square window shape.
static int ThresholdForRadius(int radius) {
  return radius == 1 ? 1 : static_cast<int>((radius + 0.5f) * (radius + 0.5f));
}

static int ValueInRange(uint16_t v, int col, int row, int threshold) {
  const int dx = (v & 0xFF) - col;
  const int dy = (v >> 8) - row;
  return dx * dx + dy * dy <= threshold ? 1 : 0;
}

template <SearchDirection direction, typename ordinal_t>
uint16_t FindRank(const uint16_t* omnigram, int col, int row, int threshold,
                  int rank, uint16_t* count_col_ptr, ordinal_t* pivot_col_ptr) {
  constexpr int kOrdinalShift = std::is_same_v<ordinal_t, uint8_t> ? 6 : 0;
  constexpr int kPivotStep = 64;
  int count_val = *count_col_ptr;
  int pivot_val = *pivot_col_ptr << kOrdinalShift;
  const uint16_t* histp = omnigram + pivot_val;

  if (direction == kSearchDownward) {
    CHECK_GE(pivot_val, kPivotStep)
        << ": col = " << col << ", rank = " << rank
        << ", count_val = " << count_val << ", pivot_val = " << pivot_val;
    while (count_val >= rank) {
      for (int ind = -1; ind >= -kPivotStep; --ind) {
        count_val -= ValueInRange(histp[ind], col, row, threshold);
        if (count_val < rank) {
          return pivot_val + ind;
        }
      }

      pivot_val -= kPivotStep;
      histp -= kPivotStep;
      CHECK_GT(pivot_val, 0)
          << ": col = " << col << ", rank = " << rank
          << ", count_val = " << count_val << ", pivot_val = " << pivot_val;

      *count_col_ptr = count_val;
      *pivot_col_ptr = pivot_val >> kOrdinalShift;
    }
  } else {
    CHECK_LE(pivot_val, 65536 - kPivotStep);
    while (count_val < rank) {
      for (int ind = 0; ind < kPivotStep; ++ind) {
        count_val += ValueInRange(histp[ind], col, row, threshold);
        if (count_val >= rank) {
          return pivot_val + ind;
        }
      }

      pivot_val += kPivotStep;
      histp += kPivotStep;
      CHECK_LT(pivot_val, 65536);
      *count_col_ptr = count_val;
      *pivot_col_ptr = pivot_val >> kOrdinalShift;
    }
  }
  // Either inner loop should eventually return; we shouldn't reach this point.
  LOG(FATAL) << "End of FindRank() unexpectedly reached!";
}

template <typename ordinal_t>
uint16_t SearchUpDown(int col, int row, int threshold, int rank,
                      const uint16_t* omnigram, ordinal_t* pivot_col_ptr,
                      uint16_t* count_col_ptr) {
  return (*count_col_ptr < rank)
             ? FindRank<kSearchUpward>(omnigram, col, row, threshold, rank,
                                       count_col_ptr, pivot_col_ptr)
             : FindRank<kSearchDownward>(omnigram, col, row, threshold, rank,
                                         count_col_ptr, pivot_col_ptr);
}

template <typename ordinal_t>
static void UpdateCounts(ReadViewGray<ordinal_t> ordinal_input, int radius,
                         int row, const uint8_t* inset, const ordinal_t* pivot,
                         uint16_t* count) {
  const int output_width = ordinal_input.width() - 2 * radius;
  for (int col = 0; col <= radius * 2; ++col) {
    const int inset_val = inset[std::abs(radius - col)];
    for (int ind = 0; ind < output_width; ++ind) {
      const int val_t = ordinal_input(col + ind, row + inset_val);
      const int val_b =
          ordinal_input(col + ind, row + 2 * radius + 1 - inset_val);
      if (val_b < pivot[ind]) count[ind]++;
      if (val_t < pivot[ind]) count[ind]--;
    }
  }
}

// The first output row is processed by stepping the circular kernel window
// horizontally, adding new pixels on the right and removing them on the left.
// By rearranging/"transposing" these pixels such that the added/subtracted
// pixels are contiguous, the stepping can subsequently be more efficiently
// processed using SIMD vectorization. Visualized below, where '*' illustrates
// the first few rows of the circular kernel shape, and [A .. H] shows pixels
// that will be added as the window is stepped to the right:
//
//           * * * * * A B C D E F G H                     A A A A A A A A
//       * * * * * * * * * A B C D E F G H                 B B B B B B B B
//     * * * * * * * * * * * A B C D E F G H               C C C C C C C C
//     * * * * * * * * * * * A B C D E F G H     ------>   D D D D D D D D
//   * * * * * * * * * * * * * A B C D E F G H   ------>   E E E E E E E E
//   * * * * * * * * * * * * * A B C D E F G H             F F F F F F F F
//   * * * * * * * * * * * * * A B C D E F G H             G G G G G G G G
//   * * * * * * * * * * * * * A B C D E F G H             H H H H H H H H
//
template <typename ordinal_t>
static void TransposeFirstRows(ReadViewGray<ordinal_t> ordinal_input,
                               int radius, const uint8_t* inset,
                               ordinal_t* pixels_left,
                               ordinal_t* pixels_right) {
  const int kernel_width_ceil = (2 * radius + 16) & ~15;

  const int output_width = ordinal_input.width() - 2 * radius;
  const ordinal_t zeros[256] = {0};
  constexpr int kBlockSize = std::is_same_v<ordinal_t, uint8_t> ? 16 : 8;

  // Processes pixels in (kBlockSize x kBlockSize) blocks.
  for (int row = 0; row < kernel_width_ceil; row += kBlockSize) {
    const ordinal_t* pl[kBlockSize];
    const ordinal_t* pr[kBlockSize];

    // Sets up pointers.
    for (int dy = 0; dy < kBlockSize; ++dy) {
      if (row + dy > 2 * radius) {
        pl[dy] = zeros;
        pr[dy] = zeros;
        continue;
      }
      const int dr = std::abs(row + dy - radius);
      const int inset_v = inset[dr];
      pl[dy] = &ordinal_input(inset_v, row + dy) - 1;
      pr[dy] = &ordinal_input(2 * radius - inset_v, row + dy);
    }

    // Steps through blocks and shears/transposes into buffers.
    for (int col = 0; col < output_width; col += kBlockSize) {
      for (int dy = 0; dy < kBlockSize; ++dy) {
        for (int dx = 0; dx < kBlockSize; ++dx) {
          pixels_left[(col + dx) * kernel_width_ceil + (row + dy)] =
              pl[dy][col + dx];
          pixels_right[(col + dx) * kernel_width_ceil + (row + dy)] =
              pr[dy][col + dx];
        }
      }
    }
  }
}

// Solves for the top left pixel in the 8-bit ordinal image, placing the exact
// cardinal solution in cardinal_output(0, 0), and the quantized pivot and count
// values in *pivot and *count, respectively.
template <typename cardinal_t>
void SolveFirstPixel(ReadViewGrayU8 ordinal_input, int rank, int support,
                     const uint16_t* omnigram, const uint8_t* inset,
                     absl::Span<const cardinal_t> ordinal_to_cardinal_lut,
                     uint8_t* pivot, uint16_t* count,
                     WriteViewGray<cardinal_t> cardinal_output, int col = 0) {
  const int radius = (ordinal_input.width() - cardinal_output.width()) / 2;

  // Makes a histogram of the top left circular ROI.
  std::vector<uint16_t> hist(256, 0);
  for (int y = 0; y <= 2 * radius; ++y) {
    const int inset_v = inset[std::abs(radius - y)];
    for (int x = inset_v; x <= 2 * radius - inset_v; ++x) {
      ++hist[ordinal_input(x + col, y)];
    }
  }
  // Scans the histogram to find the index of the 'rank'th value.
  int ct = 0;
  for (int i = 0; i < 256; ++i) {
    ct += hist[i];
    if (ct >= rank) {
      pivot[col] = i;
      count[col] = ct - hist[i];  // Count of values below 'i'.
      break;
    }
  }

  // Finds the exact solution in the omnigram and writes to output.
  // TODO: This is slight overkill. We know the solution is in the
  // next 64 values of the omnigram, so we could find it without the full
  // SearchUpDown logic. Also, we don't need to write the cardinal value to the
  // output, since it was already written by the previous tile.
  const int threshold = ThresholdForRadius(radius);

  const uint16_t ordinal_output = SearchUpDown(
      /*col=*/radius + col, /*row=*/radius, threshold, rank, omnigram,
      &pivot[col], &count[col]);
  cardinal_output(col, 0) = ordinal_to_cardinal_lut[ordinal_output];
}

// Solves for the top left pixel in the 16-bit ordinal image, placing the exact
// cardinal solution in cardinal_output(0, 0), and the quantized pivot and count
// values in *pivot and *count, respectively.
template <typename cardinal_t>
void SolveFirstPixel(ReadViewGrayU16 ordinal_input, int rank, int support,
                     const uint16_t* omnigram, const uint8_t* inset,
                     absl::Span<const cardinal_t> ordinal_to_cardinal_lut,
                     uint16_t* pivot, uint16_t* count,
                     WriteViewGray<cardinal_t> cardinal_output, int col = 0) {
  const int radius = (ordinal_input.width() - cardinal_output.width()) / 2;

  // Gathers pixels from the top left circular ROI.
  std::vector<uint16_t> top_left_pixels(support);
  uint16_t* tlp = top_left_pixels.data();
  for (int y = 0; y <= 2 * radius; ++y) {
    const int inset_v = inset[std::abs(radius - y)];
    for (int x = inset_v; x <= 2 * radius - inset_v; ++x) {
      *tlp++ = ordinal_input(x + col, y);
    }
  }
  // Finds the 'rank'th value in the top left circular ROI, which is the
  // result for this output pixel.
  std::nth_element(top_left_pixels.begin(), top_left_pixels.begin() + rank - 1,
                   top_left_pixels.end());

  cardinal_output(col, 0) = ordinal_to_cardinal_lut[top_left_pixels[rank - 1]];

  // Rounds down to multiple of 64 for SIMD-friendliness.
  int pivot_val = top_left_pixels[rank - 1] & 0xFFC0;
  // Counts the number of pixels in the top left circular ROI that are below
  // the 'rank'th value.
  int count_val = 0;
  for (int i = 0; i < rank - 1; ++i) {
    if (top_left_pixels[i] < pivot_val) ++count_val;
  }
  pivot[col] = pivot_val;
  count[col] = count_val;
}

template <typename ordinal_t, typename cardinal_t>
void SolveFirstRow(ReadViewGray<ordinal_t> ordinal_input, int rank, int support,
                   const uint16_t* omnigram, const uint8_t* inset,
                   absl::Span<const cardinal_t> ordinal_to_cardinal_lut,
                   ordinal_t* pivot, uint16_t* count,
                   WriteViewGray<cardinal_t> cardinal_output) {
  const int output_width = cardinal_output.width();
  if constexpr (std::is_same_v<ordinal_t, uint8_t>) {
    CHECK_EQ(output_width % 16, 0);
  } else {
    CHECK_EQ(output_width % 8, 0);
  }

  // Note: for reference, the entire first row could be solved "brute-force"
  // as follows:
  //
  //     for (int col = 0; col < output_width; ++col) {
  //       SolveFirstPixel(ordinal_input, rank, support, omnigram,
  //       inset,
  //                               ordinal_to_cardinal_lut, pivot,
  //                               count, cardinal_output, col);
  //     }

  SolveFirstPixel(ordinal_input, rank, support, omnigram, inset,
                  ordinal_to_cardinal_lut, pivot, count, cardinal_output);

  int pivot_val = pivot[0];
  int count_val = count[0];

  const int radius = (ordinal_input.width() - cardinal_output.width()) / 2;
  const int threshold = ThresholdForRadius(radius);
  const int kernel_width_ceil = (2 * radius + 16) & ~15;

  // Rearranges the pixels in the top 'kernel_width' rows.
  std::vector<ordinal_t> pixels_left(output_width * kernel_width_ceil);
  std::vector<ordinal_t> pixels_right(output_width * kernel_width_ceil);

  TransposeFirstRows(ordinal_input, radius, inset, pixels_left.data(),
                     pixels_right.data());

  // Walks across the row, updating pivot and count for each output pixel.
  for (int col = 1; col < output_width; ++col) {
    const ordinal_t* pixels_left_ptr =
        pixels_left.data() + col * kernel_width_ceil;
    const ordinal_t* pixels_right_ptr =
        pixels_right.data() + col * kernel_width_ceil;

    for (int row = 0; row <= 2 * radius; ++row) {
      if (pixels_left_ptr[row] < pivot_val) --count_val;
      if (pixels_right_ptr[row] < pivot_val) ++count_val;
    }

    pivot[col] = pivot_val;
    count[col] = count_val;

    uint16_t ordinal_output;
    ordinal_output = SearchUpDown(col + radius, /*row=*/0 + radius, threshold,
                                  rank, omnigram, &pivot[col], &count[col]);

    pivot_val = pivot[col];
    count_val = count[col];
    cardinal_output(col, 0) = ordinal_to_cardinal_lut[ordinal_output];
  }
}

template <typename ordinal_t, typename cardinal_t>
void SearchUpDownRow(int row, int radius, int output_width, int threshold,
                     int rank, const uint16_t* omnigram, ordinal_t* pivot,
                     uint16_t* count,
                     absl::Span<const cardinal_t> ordinal_to_cardinal_lut,
                     PixelOffset* last_row_offsets,
                     WriteViewGray<cardinal_t> cardinal_output) {
  for (int col = 0; col < output_width; ++col) {
    const uint16_t ordinal_output =
        SearchUpDown(col + radius, row + radius, threshold, rank, omnigram,
                     &pivot[col], &count[col]);
    // Inverts the ordinal transform for this pixel and writes to output.
    if (last_row_offsets != nullptr) {
      *reinterpret_cast<uint16_t*>(&last_row_offsets[col]) = ordinal_output;
    } else {
      cardinal_output(col, row) = ordinal_to_cardinal_lut[ordinal_output];
    }
  }
}

// Processes the percentile filter for a single ordinal tile.
template <typename ordinal_t, typename cardinal_t>
void FastIsotropicMedianFilterOrdinal(
    ReadViewGray<ordinal_t> ordinal_input, float percentile, int support,
    const uint8_t* inset, absl::Span<const cardinal_t> ordinal_to_cardinal_lut,
    const uint16_t* omnigram, const PixelOffset* first_row_offsets,
    PixelOffset* last_row_offsets, WriteViewGray<cardinal_t> cardinal_output) {
  constexpr int kOrdinalShift = std::is_same_v<ordinal_t, uint8_t> ? 6 : 0;
  const int input_width = ordinal_input.width();
  const int input_height = ordinal_input.height();
  const int output_width = cardinal_output.width();
  const int output_height = cardinal_output.height();
  const int radius = (input_width - output_width) / 2;

  // Consistency checks.
  CHECK_EQ(input_width - output_width, input_height - output_height);
  CHECK_GT(input_width, output_width);
  CHECK_GT(input_height, output_height);
  CHECK_EQ((input_width - output_width) % 2, 0);
  CHECK_EQ((input_height - output_height) % 2, 0);
  CHECK_GE(percentile, 0.f);
  CHECK_LE(percentile, 1.f);
  CHECK_LE(output_width, kMaxOutputTileCols);

  // Note: `rank` is 1-based. E.g. for a 3x3 kernel it will be in [1 .. 9].
  const int rank = static_cast<int>(percentile * (support - 1) + 1.5f);

  // Pivots and Counts. The "pivot" for each output column is a brightness
  // threshold close to the previous scanline's output for that column, and
  // the "count" is the number of input pixels within that output pixel's ROI
  // that are strictly below the corresponding 'pivot' value. These arrays are
  // updated as the ROI's sweep from top to bottom of the tile.
  // Note: the arrays are allocated with `alignas` to ensure 32-byte vector
  // alignment, required for certain AVX load and store instructions.
  alignas(32) ordinal_t pivot[kMaxOutputTileCols] = {0};
  alignas(32) uint16_t count[kMaxOutputTileCols] = {0};

  // If solutions have been forwarded from previous tile, incorporates them.
  if (first_row_offsets != nullptr) {
    const int threshold = ThresholdForRadius(radius);
    for (int col = 0; col < output_width; ++col) {
      const int offset_x = first_row_offsets[col].dx;
      const int offset_y = first_row_offsets[col].dy;

      int piv = ordinal_input(col + radius + offset_x, radius + offset_y)
                << kOrdinalShift;
      int ct = rank - 1;

      if constexpr (std::is_same_v<ordinal_t, uint8_t>) {
        pivot[col] = piv >> kOrdinalShift;
        // The pivot is approximate. We need to search the omnigram
        // to find a match for the exact offset, which will tell us the exact
        // pivot value. The exact value is somewhere in [piv .. piv + 63]. As
        // we search, we keep track of the number of non-solution pixels that
        // are within the circular window, and decrement the count for the
        // original quantized pivot to match.
        const int target_x = col + radius + offset_x;
        const int target_y = radius + offset_y;
        const uint16_t target_xy = (target_y << 8) | target_x;
        int ind;
        bool found = false;
        for (ind = 0; ind < (1 << kOrdinalShift); ++ind) {
          int coords_xy = omnigram[piv + ind];
          if (coords_xy == target_xy) {
            found = true;
            break;
          }
          const int coords_x = coords_xy & 0xFF;
          const int coords_y = coords_xy >> 8;
          const int dx = coords_x - (col + radius);
          const int dy = coords_y - radius;
          if (dx * dx + dy * dy <= threshold) --ct;
          DCHECK_GE(ct, 0);
        }
        DCHECK(found);
        cardinal_output(col, 0) = ordinal_to_cardinal_lut[piv + ind];
        count[col] = ct;
      } else {  // 16-bit ordinal image.
        // We've found the solution for this pixel, so we can write it to the
        // output immediately.
        cardinal_output(col, 0) = ordinal_to_cardinal_lut[piv];

        // Sets the pivot to an 64-element-aligned value.
        if (piv < 0x8000) {
          while (piv & 63) {
            const int hval = omnigram[piv++];
            const int dx = (hval & 0xFF) - col - radius;
            const int dy = (hval >> 8) - radius;
            if (dx * dx + dy * dy <= threshold) ct++;
          }
        } else {
          while (piv & 63) {
            const int hval = omnigram[--piv];
            const int dx = (hval & 0xFF) - col - radius;
            const int dy = (hval >> 8) - radius;
            if (dx * dx + dy * dy <= threshold) --ct;
          }
        }
        pivot[col] = piv;
        count[col] = ct;
      }
    }
  } else {
    SolveFirstRow(ordinal_input, rank, support, omnigram, inset,
                  ordinal_to_cardinal_lut, pivot, count, cardinal_output);
  }

  const int threshold = ThresholdForRadius(radius);
  for (int row = 1; row < output_height; ++row) {
    // Updates 'counts' array from previous row.
    UpdateCounts<ordinal_t>(ordinal_input, radius, row - 1, inset, pivot,
                            count);

    const bool encode = last_row_offsets != nullptr && row == output_height - 1;
    // Using pivots and counts as a starting point, searches the compound
    // histogram for solutions for this output column.
    SearchUpDownRow<ordinal_t>(
        row, radius, output_width, threshold, rank, omnigram, pivot, count,
        absl::MakeSpan(ordinal_to_cardinal_lut),
        encode ? last_row_offsets : nullptr, cardinal_output);
  }

  // If this is not the last tile in the column, finds the offset of the
  // solution for each output pixel in the last row.
  if (last_row_offsets != nullptr) {
    for (int col = 0; col < output_width; ++col) {
      const uint16_t ordv =
          *reinterpret_cast<uint16_t*>(&last_row_offsets[col]);
      const int hv = omnigram[ordv];
      const int x = hv & 0xFF;
      const int y = hv >> 8;

      const int offset_x = x - (col + radius);
      const int offset_y = y - (output_height - 1 + radius);

      // Writes encoded offsets to buffer, to be used by next tile.
      last_row_offsets[col].dx = offset_x;
      last_row_offsets[col].dy = offset_y;
    }
  }
}

// Returns a target output tile size for the given kernel width. For the special
// case of radii larger than 96, we shrink the output tile slightly to avoid
// the input tile becoming larger than 256x256. Otherwise, in this simplified
// implementation, we use a fixed output tile size of 64x64.
template <typename ordinal_t, typename cardinal_t>
int GetTargetOutputTileSize(int radius) {
  return radius <= 96 ? 64 : 56;
}

template <typename ordinal_t, typename cardinal_t>
void ProcessBlock(ReadViewGray<cardinal_t> input_block, int radius,
                  float percentile, int support, const uint8_t* inset,
                  int out_tile_size_y, WriteViewGray<cardinal_t> output_block) {
  // Multiple of 32 bytes for SIMD friendliness.
  constexpr int kTileWidthAlignment = 16 / sizeof(ordinal_t);
  const int output_block_height = output_block.height();
  const int in_tile_size_y = out_tile_size_y + 2 * radius;

  // If the blocks are not aligned, we need to operate on padded copies.
  WriteViewGray<cardinal_t> output_block_orig = output_block;
  ImageGray<cardinal_t> input_block_padded;
  ImageGray<cardinal_t> output_block_padded;
  bool needs_padding = false;
  if (output_block.width() % kTileWidthAlignment != 0) {
    needs_padding = true;
    const int pad =
        kTileWidthAlignment - (output_block.width() % kTileWidthAlignment);
    input_block_padded = ImageGray<cardinal_t>(input_block.width() + pad,
                                               input_block.height(), 0);
    output_block_padded = ImageGray<cardinal_t>(
        {output_block.width() + pad, output_block_height});
    CopyContents(input_block, input_block_padded.WriteView());
    input_block = input_block_padded.ReadView();
    output_block = output_block_padded.WriteView();
  }
  const int in_tile_size_x = input_block.width();

  // Allocates buffers for this thread.
  ImageGray<ordinal_t> input_ordinal({in_tile_size_x, in_tile_size_y});

  // Lookup table for inverse ordinal transform.
  std::vector<cardinal_t> ordinal_to_cardinal_lut(input_ordinal.width() *
                                                  input_ordinal.height());
  std::vector<PixelOffset> pixel_offsets(input_block.width());

  // Steps down the column of tiles, processing them sequentially.
  for (int row = 0; row < output_block_height; row += out_tile_size_y) {
    const int out_tile_height =
        std::min(row + out_tile_size_y, output_block_height) - row;
    const int in_tile_height = out_tile_height + 2 * radius;
    auto input_tile = FastCrop(input_block, 0, row, input_block.width(),
                               row + in_tile_height);
    auto output_tile = FastCrop(output_block, 0, row, output_block.width(),
                                row + out_tile_height);
    CHECK_EQ(input_tile.width(), output_tile.width() + 2 * radius);
    CHECK_EQ(input_tile.height(), output_tile.height() + 2 * radius);
    auto input_ordinal_cropped =
        FastCrop(input_ordinal.WriteView(), 0, 0, input_tile.width(),
                 input_tile.height());
    // The sorted_coords array can be thought of as an "omnigram"; any
    // element of the binary histogram of any region of the ordinal image
    // can be instantly extracted from it. Note that the span is padded by
    // kHistPad on either end to allow slight overshoot by the vector
    // inner loop which scans the array with a large stride. We allocate an
    // extra 16 elements (32 bytes) to allow for SIMD alignment.
    std::vector<uint16_t> sorted_coords(
        input_tile.width() * input_tile.height() + 2 * kHistPad + 16);

    // For the base pointer of the omnigram, we want to inset by kHistPad, but
    // also align to a multiple of 32 bytes for optimal SIMD performance.
    intptr_t base = reinterpret_cast<intptr_t>(&sorted_coords[kHistPad]);
    base = (base + 31) & ~31;
    absl::Span<uint16_t> omnigram(reinterpret_cast<uint16_t*>(base),
                                  input_tile.width() * input_tile.height());
    OrdinalTransformSimple(
        input_tile, input_ordinal_cropped,
        absl::Span<cardinal_t>(ordinal_to_cardinal_lut.data(),
                               input_tile.width() * input_tile.height()),
        omnigram);

    if (kFeedForward) {
      FastIsotropicMedianFilterOrdinal<ordinal_t, cardinal_t>(
          input_ordinal_cropped.ReadView(), percentile, support, inset,
          ordinal_to_cardinal_lut, omnigram.data(),
          /*first_row_offsets=*/row > 0 ? pixel_offsets.data() : nullptr,
          /*last_row_offsets=*/row + out_tile_size_y < output_block_height
              ? pixel_offsets.data()
              : nullptr,
          output_tile);
      // If not last tile in row, sets up next tile to overlap one row to
      // process encoded solutions.
      if (row + out_tile_size_y < output_block_height) {
        row--;
      }
    } else {
      FastIsotropicMedianFilterOrdinal<ordinal_t, cardinal_t>(
          input_ordinal_cropped.ReadView(), percentile, support, inset,
          ordinal_to_cardinal_lut, omnigram.data(),
          /*first_row_offsets=*/nullptr,
          /*last_row_offsets=*/nullptr, output_tile);
    }
  }
  if (needs_padding) {
    CopyContents(output_block.ReadView(), output_block_orig);
  }
}

template <typename ordinal_t, typename cardinal_t>
void FastIsotropicMedianFilterSimpleImpl(ReadViewGray<cardinal_t> input,
                                         float percentile,
                                         WriteViewGray<cardinal_t> output) {
  // Multiple of 32 bytes for SIMD friendliness.
  constexpr int kTileWidthAlignment = 32 / sizeof(ordinal_t);
  const int input_width = input.width();
  const int input_height = input.height();
  const int output_width = output.width();
  const int output_height = output.height();
  const int radius = (input_width - output_width) / 2;
  const int square_support = (2 * radius + 1) * (2 * radius + 1);

  // Consistency checks.
  CHECK_EQ(input_width - output_width, input_height - output_height);
  CHECK_GT(input_width, output_width);
  CHECK_EQ((input_width - output_width) % 2, 0);
  CHECK_LE(radius, kMaximumFilterRadius);
  CHECK_GE(percentile, 0.f);
  CHECK_LE(percentile, 1.f);
  CHECK_NE(thread_pool, nullptr);

  uint8_t inset_data[kMaximumFilterRadius + 1] = {0};
  int support = square_support;

  // Initializes the 'inset' array to a circular shape.
  const int threshold = ThresholdForRadius(radius);
  for (int rad = 0; rad <= radius; ++rad) {
    const int chord = sqrt(threshold - rad * rad);
    inset_data[rad] = radius - chord;
    support -= 4 * inset_data[rad];
  }
  CHECK_GT(inset_data[radius], 0);  // The kernel shouldn't be a perfect square!
  const uint8_t* inset = inset_data;  // For threading, avoids copying array.

  // Determines the optimal output tile size.
  const int target_output_tile_size =
      GetTargetOutputTileSize<ordinal_t, cardinal_t>(radius);
  int out_tile_size_x = std::clamp(target_output_tile_size, 32, 256);
  int out_tile_size_y = std::clamp(target_output_tile_size, 32, 256);

  // If the ordinal image is quantized to 8-bit, we can have a maximum of 16384
  // pixels in each input tile.
  if constexpr (std::is_same_v<ordinal_t, uint8_t>) {
    out_tile_size_y = 16384 / (out_tile_size_x + 2 * radius) - 2 * radius;

    // Empirical restriction to near-square.
    out_tile_size_y = std::min(out_tile_size_y, out_tile_size_x + 16);
  }

  out_tile_size_x =
      std::min(out_tile_size_x, 256 - 2 * radius) & ~(kTileWidthAlignment - 1);
  const int in_tile_size_x = out_tile_size_x + 2 * radius;

  // Loops through columns of tiles.
  for (int col = 0; col < output_width; col += out_tile_size_x) {
    // Constructs cropped column views.
    ReadViewGray<cardinal_t> input_column =
        FastCrop(input, col, 0, col + in_tile_size_x, input_height);
    WriteViewGray<cardinal_t> output_column =
        FastCrop(output, col, 0, col + out_tile_size_x, output_height);
    // Tile widths, padded to ensure alignment.
    const int out_tile_width =
        output_column.width() + (kTileWidthAlignment - 1) &
        ~(kTileWidthAlignment - 1);
    const int in_tile_width = out_tile_width + 2 * radius;

    // Loops through vertical blocks of tiles, rather than entire columns. (The
    // finer granularity helps load-balance the work across threads.)
    constexpr int kTargetOutputPixelsPerBlock = 65536;
    const int tiles_per_block = std::max(
        1, kTargetOutputPixelsPerBlock / (out_tile_size_x * out_tile_size_y));
    const int rows_per_block = out_tile_size_y * tiles_per_block -
                               (kFeedForward ? tiles_per_block - 1 : 0);
    for (int block_top = 0; block_top < output_height;
         block_top += rows_per_block) {
      ReadViewGray<cardinal_t> input_block =
          FastCrop(input_column, 0, block_top, in_tile_width,
                   block_top + rows_per_block + 2 * radius);
      WriteViewGray<cardinal_t> output_block =
          FastCrop(output_column, 0, block_top, out_tile_width,
                   block_top + rows_per_block);
      thread_pool->Enqueue([=] {
        ProcessBlock<ordinal_t, cardinal_t>(input_block, radius, percentile,
                                            support, inset, out_tile_size_y,
                                            output_block);
      });
    }
  }
  thread_pool->Wait();
}

void FastIsotropicMedianFilterSimple(ReadViewGrayU8 input, float percentile,
                                     WriteViewGrayU8 output) {
  const int radius = (input.width() - output.width()) / 2;
  if (radius <= kMaximumQuantizedRadius) {
    FastIsotropicMedianFilterSimpleImpl</*ordinal_t=*/uint8_t>(
        input, percentile, output);
  } else {
    FastIsotropicMedianFilterSimpleImpl</*ordinal_t=*/uint16_t>(
        input, percentile, output);
  }
}

void FastIsotropicMedianFilterSimple(ReadViewGrayU16 input, float percentile,
                                     WriteViewGrayU16 output) {
  const int radius = (input.width() - output.width()) / 2;
  if (radius <= kMaximumQuantizedRadius) {
    FastIsotropicMedianFilterSimpleImpl</*ordinal_t=*/uint8_t>(
        input, percentile, output);
  } else {
    FastIsotropicMedianFilterSimpleImpl</*ordinal_t=*/uint16_t>(
        input, percentile, output);
  }
}

void FastIsotropicMedianFilterSimple(ReadViewGrayF input, float percentile,
                                     WriteViewGrayF output) {
  const int radius = (input.width() - output.width()) / 2;
  if (radius <= kMaximumQuantizedRadius) {
    FastIsotropicMedianFilterSimpleImpl</*ordinal_t=*/uint8_t>(
        input, percentile, output);
  } else {
    FastIsotropicMedianFilterSimpleImpl</*ordinal_t=*/uint16_t>(
        input, percentile, output);
  }
}

template <typename T>
absl::Status FastIsotropicMedianFilterSimpleImpl(
    ReadViewGray<T> input,
    const FastIsotropicMedianFilterSimpleOptions& options,
    WriteViewGray<T> output) {
  if (options.radius < 1 || options.radius > kMaximumFilterRadius) {
    return absl::InvalidArgumentError("Radius must be in range [1 .. " +
                                      std::to_string(kMaximumFilterRadius) +
                                      "].");
  }
  if (options.percentile < 0.0f || options.percentile > 1.0f) {
    return absl::InvalidArgumentError(
        "Percentile must be in range [0.0 .. 1.0].");
  }

  if (input.width() != output.width() + 2 * options.radius ||
      input.height() != output.height() + 2 * options.radius) {
    return absl::InvalidArgumentError(
        "Input must be padded by radius on each side with respect to output.");
  }

  if (thread_pool == nullptr) {
    return absl::FailedPreconditionError("Thread pool must be set.");
  }

  FastIsotropicMedianFilterSimple(input, options.percentile, output);

  return absl::OkStatus();
}
}  // namespace

// Public API.
void SetThreadPoolSimple(ThreadPool* pool) { thread_pool = pool; }

absl::Status FastIsotropicMedianFilterSimple(
    ReadViewGrayU8 input, const FastIsotropicMedianFilterSimpleOptions& options,
    WriteViewGrayU8 output) {
  return FastIsotropicMedianFilterSimpleImpl(input, options, output);
}
absl::Status FastIsotropicMedianFilterSimple(
    ReadViewGrayU16 input,
    const FastIsotropicMedianFilterSimpleOptions& options,
    WriteViewGrayU16 output) {
  return FastIsotropicMedianFilterSimpleImpl(input, options, output);
}
absl::Status FastIsotropicMedianFilterSimple(
    ReadViewGrayF input, const FastIsotropicMedianFilterSimpleOptions& options,
    WriteViewGrayF output) {
  return FastIsotropicMedianFilterSimpleImpl(input, options, output);
}
}  // namespace fast_isotropic_median_filter
