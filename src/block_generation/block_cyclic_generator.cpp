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
#include "block_generation/block_cyclic_generator.hpp"

#include <algorithm>
#include <cassert>

#include "spla/config.h"
#include "util/common_types.hpp"

namespace spla {

BlockCyclicGenerator::BlockCyclicGenerator(IntType rowsInBlock, IntType colsInBlock,
                                           IntType gridRows, IntType gridCols,
                                           IntType globalNumRows, IntType globalNumCols,
                                           IntType globalRowOffset, IntType globalColOffset)
    : rowsInBlock_(rowsInBlock),
      colsInBlock_(colsInBlock),
      gridRows_(gridRows),
      gridCols_(gridCols),
      globalNumRows_(globalNumRows),
      globalNumCols_(globalNumCols),
      globalRowOffset_(globalRowOffset),
      globalColOffset_(globalColOffset) {
  const IntType firstBlockRowIdx = (globalRowOffset_ / rowsInBlock_) * rowsInBlock_;
  const IntType firstBlockColIdx = (globalColOffset_ / colsInBlock_) * colsInBlock_;

  numBlockRows_ =
      (globalRowOffset_ - firstBlockRowIdx + globalNumRows_ + rowsInBlock_ - 1) / rowsInBlock_;
  numBlockCols_ =
      (globalColOffset_ - firstBlockColIdx + globalNumCols_ + colsInBlock_ - 1) / colsInBlock_;
}

auto BlockCyclicGenerator::get_block_info(IntType blockIdx) -> BlockInfo {
  assert(blockIdx < num_blocks());
  assert(blockIdx >= 0);
  const IntType blockRowIdx = blockIdx % numBlockRows_;
  const IntType globalBlockRowIdx = blockRowIdx + (globalRowOffset_ / rowsInBlock_);
  const IntType blockColIdx = blockIdx / numBlockRows_;
  const IntType globalBlockColIdx = blockColIdx + (globalColOffset_ / colsInBlock_);

  const IntType firstBlockRowIdx = (globalRowOffset_ / rowsInBlock_) * rowsInBlock_;
  const IntType firstBlockColIdx = (globalColOffset_ / colsInBlock_) * colsInBlock_;

  const IntType globalRowIdx =
      std::max(globalRowOffset_, firstBlockRowIdx + blockRowIdx * rowsInBlock_);
  const IntType globalColIdx =
      std::max(globalColOffset_, firstBlockColIdx + blockColIdx * colsInBlock_);

  const IntType numRows =
      std::min((globalBlockRowIdx + 1) * rowsInBlock_, globalRowOffset_ + globalNumRows_) -
      globalRowIdx;
  const IntType numCols =
      std::min((globalBlockColIdx + 1) * colsInBlock_, globalColOffset_ + globalNumCols_) -
      globalColIdx;

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

auto BlockCyclicGenerator::get_mpi_rank(IntType blockIdx) -> IntType {
  assert(blockIdx < num_blocks());
  assert(blockIdx >= 0);
  const IntType blockRowIdx = blockIdx % numBlockRows_;
  const IntType globalBlockRowIdx = blockRowIdx + (globalRowOffset_ / rowsInBlock_);
  const IntType blockColIdx = blockIdx / numBlockRows_;
  const IntType globalBlockColIdx = blockColIdx + (globalColOffset_ / colsInBlock_);

  const IntType mpiRank =
      (globalBlockRowIdx % gridRows_) + (globalBlockColIdx % gridCols_) * gridRows_;

  return mpiRank;
}

static auto local_size(IntType globalSize, IntType blockSize, IntType procIdx, IntType numProcs)
    -> IntType {
  const IntType numBlocks = globalSize / blockSize;  // number of full blocks eually distributed
  const IntType numOverhangBlocks = numBlocks % numProcs;  // number of full blocks left
  IntType localSize = (numBlocks / numProcs) * blockSize;

  if (procIdx + 1 < numOverhangBlocks) localSize += blockSize;
  // add partial block if required
  if (procIdx + 1 == numOverhangBlocks) localSize += globalSize % blockSize;
  return localSize;
}

auto BlockCyclicGenerator::local_rows(IntType rank) -> IntType {
  if (rank < gridRows_ * gridCols_)
    return local_size(globalNumRows_ + globalRowOffset_, rowsInBlock_, rank % gridRows_,
                      gridRows_) -
           local_size(globalRowOffset_, rowsInBlock_, rank % gridRows_, gridRows_);
  return 0;
}

auto BlockCyclicGenerator::local_cols(IntType rank) -> IntType {
  if (rank < gridRows_ * gridCols_)
    return local_size(globalNumCols_ + globalColOffset_, colsInBlock_, rank / gridRows_,
                      gridCols_) -
           local_size(globalColOffset_, colsInBlock_, rank / gridRows_, gridCols_);
  return 0;
}

}  // namespace spla
