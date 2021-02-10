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

#include <algorithm>
#include <cassert>

#include "block_generation/block.hpp"
#include "spla/config.h"
#include "util/common_types.hpp"

namespace spla {

class BlockCyclicGenerator {
public:
  BlockCyclicGenerator(IntType rowsInBlock, IntType colsInBlock, IntType gridRows, IntType gridCols,
                       IntType globalNumRows, IntType globalNumCols, IntType globalRowOffset,
                       IntType globalColOffset);

  auto create_sub_generator(Block block) -> BlockCyclicGenerator {
    return BlockCyclicGenerator(rowsInBlock_, colsInBlock_, gridRows_, gridCols_, block.numRows,
                                block.numCols, block.row + globalRowOffset_,
                                block.col + globalColOffset_);
  }

  auto get_block_info(IntType blockIdx) -> BlockInfo;

  auto get_block_info(IntType blockRowIdx, IntType blockColIdx) -> BlockInfo {
    assert(blockRowIdx < numBlockRows_);
    assert(blockColIdx < numBlockCols_);
    return this->get_block_info(blockRowIdx + blockColIdx * numBlockRows_);
  }

  auto get_mpi_rank(IntType blockIdx) -> IntType;

  auto get_mpi_rank(IntType blockRowIdx, IntType blockColIdx) -> IntType {
    assert(blockRowIdx < numBlockRows_);
    assert(blockColIdx < numBlockCols_);
    return this->get_mpi_rank(blockRowIdx + blockColIdx * numBlockRows_);
  }

  auto num_blocks() -> IntType { return numBlockRows_ * numBlockCols_; }

  auto num_block_rows() -> IntType { return numBlockRows_; }

  auto num_block_cols() -> IntType { return numBlockCols_; }

  auto max_rows_in_block() -> IntType { return rowsInBlock_; }

  auto max_cols_in_block() -> IntType { return colsInBlock_; }

  auto local_rows(IntType rank) -> IntType;

  auto local_cols(IntType rank) -> IntType;

private:
  IntType rowsInBlock_, colsInBlock_;
  IntType gridRows_, gridCols_;
  IntType globalNumRows_, globalNumCols_;
  IntType globalRowOffset_, globalColOffset_;

  IntType numBlockRows_, numBlockCols_;
};

template <>
struct IsDisjointGenerator<BlockCyclicGenerator> {
  static constexpr bool value = true;
};

}  // namespace spla

#endif
