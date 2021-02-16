/*
 * Copyright (c) 2020 ETH Zurich, Simon Frasch
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef SPLA_BLOCK_SIZE_SELECTION_SSB_HPP
#define SPLA_BLOCK_SIZE_SELECTION_SSB_HPP

#include <algorithm>
#include <cmath>
#include <utility>

#include "spla/config.h"
#include "spla/types.h"
#include "util/common_types.hpp"

namespace spla {

inline auto find_optimal_proc_grid(IntType commSize, IntType lowerDeviation, IntType upperDeviation)
    -> std::pair<IntType, IntType> {

  const IntType sqrtCommSize = std::sqrt(commSize);

  for(IntType rows = sqrtCommSize; rows <= commSize; ++rows) {
    for (IntType cols = sqrtCommSize; cols > 0; --cols) {
      if(rows*cols <= commSize + upperDeviation && rows*cols >= commSize - lowerDeviation) {
        return {rows, cols};
      }
    }
  }

  return {commSize, 1};
}

inline auto block_size_selection_ssb(bool isDisjointDistribution, double deviationFactor,
                                     IntType commSize, IntType m, IntType n,
                                     IntType targetBlockSize, IntType minBlockSize)
    -> std::pair<IntType, IntType> {
  if (m * n <= minBlockSize * minBlockSize) return {m, n};  // single block if too small

  if (!isDisjointDistribution) { // No ring can be formed for non-disjoint
                                 // distributions -> use target size
    return {std::min<IntType>(targetBlockSize, m), std::min<IntType>(targetBlockSize, n)};
  }

  // Try to find grid, such that the number of blocks is devisable by the comm
  // size, allowing for a given deviation
  auto grid = find_optimal_proc_grid(commSize, deviationFactor * commSize, 0);
  if(m > n && grid.first < grid.second) std::swap(grid.first, grid.second);

  IntType rowsInBlock = (m + grid.first - 1) / grid.first;
  IntType colsInBlock = (n + grid.second - 1) / grid.second;

  // If the required block size to have enough blocks is too small, use the minimum block size
  if (rowsInBlock * colsInBlock < minBlockSize * minBlockSize) {
    return {std::min<IntType>(minBlockSize, m),
            std::min<IntType>(minBlockSize, n)};
  }

  double factor = static_cast<double>(rowsInBlock * colsInBlock) /
                  static_cast<double>(targetBlockSize * targetBlockSize);

  if (factor >= 1.5) {
    // If current block sizes are too large relative to the target block size,
    // reduce size by multiplying the number of blocks with an integer
    IntType rowFactor = std::sqrt(factor);
    IntType colFactor = std::ceil(factor / rowFactor);

    if (m > n && rowFactor < colFactor)
      std::swap(rowFactor, colFactor);

    grid.first *= rowFactor;
    grid.second *= colFactor;

    rowsInBlock = (m + grid.first - 1) / grid.first;
    colsInBlock = (n + grid.second - 1) / grid.second;
  }

  // It's possible that the number of blocks can still be closer to a multiple
  // of the comm size
  IntType excessBlocks = (grid.first * grid.second) % commSize;
  if (excessBlocks > 0) {
    IntType missingBlocks = commSize - (excessBlocks);
    if (grid.first > grid.second) {
      if (grid.first <= missingBlocks)
        grid.second += 1;
      else if (grid.second <= missingBlocks)
        grid.first += 1;
    } else {
      if (grid.second <= missingBlocks)
        grid.first += 1;
      else if (grid.first <= missingBlocks)
        grid.second += 1;
    }

    rowsInBlock = (m + grid.first - 1) / grid.first;
    colsInBlock = (n + grid.second - 1) / grid.second;
  }

  return {rowsInBlock, colsInBlock};
}

}  // namespace spla
#endif
