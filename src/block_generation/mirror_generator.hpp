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

#ifndef SPLA_MIRROR_DISTRIBUTION_HPP
#define SPLA_MIRROR_DISTRIBUTION_HPP

#include <algorithm>
#include <cassert>
#include "spla/config.h"
#include "util/common_types.hpp"
#include "block_generation/matrix_block_generator.hpp"

namespace spla {
class MirrorGenerator {
public:
  MirrorGenerator(IntType rowsInBlock, IntType colsInBlock,
                  IntType globalNumRows, IntType globalNumCols,
                  IntType globalRowOffset, IntType globalColOffset)
      : rowsInBlock_(rowsInBlock), colsInBlock_(colsInBlock),
        globalNumRows_(globalNumRows), globalNumCols_(globalNumCols),
        globalRowOffset_(globalRowOffset), globalColOffset_(globalColOffset),
        numBlockRows_((globalNumRows + rowsInBlock - 1) / rowsInBlock),
        numBlockCols_((globalNumCols + colsInBlock - 1) / colsInBlock) {}

  auto create_sub_generator(BlockCoord block) -> MirrorGenerator {
    return MirrorGenerator(rowsInBlock_, colsInBlock_, block.numRows,
                           block.numCols, block.row + globalRowOffset_,
                           block.col + globalColOffset_);
  }

  auto get_block_info(IntType blockIdx) -> BlockInfo {
    const IntType blockRowIdx = blockIdx % numBlockRows_;
    const IntType blockColIdx = blockIdx / numBlockRows_;
    return this->get_block_info(blockRowIdx, blockColIdx);
  }

  auto get_block_info(IntType blockRowIdx, IntType blockColIdx) -> BlockInfo {
    assert(blockRowIdx < rowsInBlock_);
    assert(blockColIdx < colsInBlock_);

    const IntType globalRowIdx = blockRowIdx * rowsInBlock_ + globalRowOffset_;
    const IntType globalColIdx = blockColIdx * colsInBlock_ + globalColOffset_;

    const IntType numRows =
        std::min(globalNumRows_ - blockRowIdx * rowsInBlock_, rowsInBlock_);
    const IntType numCols =
        std::min(globalNumCols_ - blockColIdx * colsInBlock_, colsInBlock_);

    const IntType mpiRank = -1;

    BlockInfo info{globalRowIdx,
                   globalColIdx,
                   globalRowIdx - globalRowOffset_,
                   globalColIdx - globalColOffset_,
                   globalRowIdx,
                   globalColIdx,
                   numRows,
                   numCols,
                   mpiRank};
    return info;
  }

  auto get_mpi_rank(IntType blockIdx) -> IntType { return -1; }

  auto get_mpi_rank(IntType blockRowIdx, IntType blockColIdx) -> IntType {
    return -1;
  }

  auto num_blocks() -> IntType { return numBlockRows_ * numBlockCols_; }

  auto num_block_rows() -> IntType { return numBlockRows_; }

  auto num_block_cols() -> IntType { return numBlockCols_; }

  auto max_rows_in_block() -> IntType { return rowsInBlock_; }

  auto max_cols_in_block() -> IntType { return colsInBlock_; }

  auto local_rows(IntType rank) -> IntType { return globalNumRows_; }

  auto local_cols(IntType rank) -> IntType { return globalNumCols_; }

private:
  IntType rowsInBlock_, colsInBlock_;
  IntType globalNumRows_, globalNumCols_;
  IntType globalRowOffset_, globalColOffset_;

  IntType numBlockRows_, numBlockCols_;
};
} // namespace spla

#endif
