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

#include <utility>
#include <cmath>
#include <algorithm>
#include "spla/config.h"
#include "spla/types.h"
#include "util/common_types.hpp"

namespace spla {
inline auto block_size_selection_ssb(
    bool isDisjointDistribution, IntType commSize, IntType m, IntType n,
    IntType rowsInMatBlock, IntType colsInMatBlock, IntType targetBlockSize,
    double deviationFactor, IntType minBlockSize)
    -> std::pair<IntType, IntType> {

  // Create initial sizes matching shape of matrix distribution block
  const double blockSkinnyFactor =
      static_cast<double>(rowsInMatBlock) /
      static_cast<double>(colsInMatBlock);
  IntType rowsInBlock = targetBlockSize * blockSkinnyFactor;
  IntType colsInBlock = targetBlockSize / blockSkinnyFactor;

  // Use distribution block size if within given deviation of target size
  if ((1.0 - deviationFactor) * rowsInBlock * colsInBlock <
          rowsInMatBlock * colsInMatBlock &&
      (1.0 + deviationFactor)* rowsInBlock * colsInBlock >
          rowsInMatBlock * colsInMatBlock) {
    rowsInBlock = rowsInMatBlock;
    colsInBlock = colsInMatBlock;
  }

  // Decrease block size to form ring if neccessary
  if (isDisjointDistribution) {
    const double minBlockRows = (m / static_cast<double>(rowsInBlock));
    const double minBlockCols = (n / static_cast<double>(colsInBlock));
    if (minBlockRows * minBlockCols < commSize) {
      const double factor =
          std::sqrt(static_cast<double>(minBlockRows * minBlockCols) /
                    static_cast<double>(commSize));
      rowsInBlock = factor * rowsInBlock;
      rowsInBlock += (m % rowsInBlock != 0); // Increase size by one to avoid small overhang
      colsInBlock = factor * colsInBlock;
      colsInBlock += (n % colsInBlock != 0);
    }
  }

  // Make sure block is at least given size
  rowsInBlock = std::max<IntType>(rowsInBlock, minBlockSize);
  colsInBlock = std::max<IntType>(colsInBlock, minBlockSize);

  return {rowsInBlock, colsInBlock};
}

}  // namespace spla
#endif
