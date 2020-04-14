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
#ifndef SPLA_BLOCK_CYCLIC_GENERATOR_HPP
#define SPLA_BLOCK_CYCLIC_GENERATOR_HPP

#include <cassert>
#include <algorithm>
#include "spla/config.h"
#include "util/common_types.hpp"
#include "block_generation/matrix_block_generator.hpp"

namespace spla {

class BlockCyclicGenerator : public MatrixBlockGenerator {
public:
  BlockCyclicGenerator(IntType rowsInBlock, IntType colsInBlock, IntType gridRows, IntType gridCols,
                          IntType globalNumRows, IntType globalNumCols, IntType globalRowOffset,
                          IntType globalColOffset)
      : rowsInBlock_(rowsInBlock),
        colsInBlock_(colsInBlock),
        gridRows_(gridRows),
        gridCols_(gridCols),
        globalNumRows_(globalNumRows),
        globalNumCols_(globalNumCols),
        globalRowOffset_(globalRowOffset),
        globalColOffset_(globalColOffset) {
    const IntType firstBlockRowIdx= (globalRowOffset_ / rowsInBlock_) * rowsInBlock_;
    const IntType firstBlockColIdx= (globalColOffset_ / colsInBlock_) * colsInBlock_;

    numBlockRows_ = (globalRowOffset_ - firstBlockRowIdx + globalNumRows_ + rowsInBlock_ - 1) / rowsInBlock_;
    numBlockCols_ = (globalColOffset_ - firstBlockColIdx + globalNumCols_ + colsInBlock_ - 1) / colsInBlock_;
  }

  auto get_block_info(IntType blockIdx) -> BlockInfo override {
    assert(blockIdx < num_blocks());
    assert(blockIdx >= 0);
    const IntType blockRowIdx = blockIdx % numBlockRows_;
    const IntType globalBlockRowIdx = blockRowIdx + (globalRowOffset_ / rowsInBlock_);
    const IntType blockColIdx = blockIdx / numBlockRows_;
    const IntType globalBlockColIdx = blockColIdx + (globalColOffset_ / colsInBlock_);

    const IntType firstBlockRowIdx = (globalRowOffset_ / rowsInBlock_) * rowsInBlock_;
    const IntType firstBlockColIdx = (globalColOffset_ / colsInBlock_) * colsInBlock_;

    const IntType globalRowIdx = std::max(globalRowOffset_, firstBlockRowIdx + blockRowIdx * rowsInBlock_); 
    const IntType globalColIdx = std::max(globalColOffset_, firstBlockColIdx + blockColIdx * colsInBlock_); 

    const IntType numRows = std::min((globalBlockRowIdx + 1) * rowsInBlock_, globalRowOffset_ + globalNumRows_) - globalRowIdx;
    const IntType numCols = std::min((globalBlockColIdx + 1) * colsInBlock_, globalColOffset_ + globalNumCols_) - globalColIdx;

    const IntType mpiRank =
        (globalBlockRowIdx % gridRows_) + (globalBlockColIdx % gridCols_) * gridRows_;

    const IntType localRowIdx =
        (globalBlockRowIdx / gridRows_) * rowsInBlock_ + globalRowIdx % rowsInBlock_;
    const IntType localColIdx =
        (globalBlockColIdx / gridCols_) * colsInBlock_ + globalColIdx % colsInBlock_;

    return BlockInfo{globalRowIdx,
                     globalColIdx,
                     globalRowIdx - globalRowOffset_,
                     globalColIdx - globalColOffset_,
                     localRowIdx,
                     localColIdx,
                     numRows,
                     numCols,
                     mpiRank};
  }

  auto get_block_info(IntType blockRowIdx, IntType blockColIdx) -> BlockInfo override {
    assert(blockRowIdx < numBlockRows_);
    assert(blockColIdx < numBlockCols_);
    return this->get_block_info(blockRowIdx + blockColIdx * numBlockRows_);

  }

  auto num_blocks() -> IntType override {
    return numBlockRows_ * numBlockCols_;
  }

  auto num_block_rows() -> IntType override { return numBlockRows_; }

  auto num_block_cols() -> IntType override { return numBlockCols_; }

  auto max_rows_in_block() -> IntType override { return rowsInBlock_; }

  auto max_cols_in_block() -> IntType override { return colsInBlock_; }

private:
  IntType rowsInBlock_, colsInBlock_;
  IntType gridRows_, gridCols_;
  IntType globalNumRows_, globalNumCols_;
  IntType globalRowOffset_, globalColOffset_;

  IntType numBlockRows_, numBlockCols_;
};
}

#endif
