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
#include "pgemm_ssb/ring_reduce_tile_host.hpp"
#include "pgemm_ssb/add_kernel.hpp"
#include "gemm/gemm_host.hpp"
#include "mpi_util/mpi_check_status.hpp"
#include "mpi_util/mpi_match_elementary_type.hpp"
#include "mpi_util/mpi_datatype_handle.hpp"
#include "util/blas_interface.hpp"
#include "util/common_types.hpp"
#include <algorithm>
#include <cassert>
#include <complex>
#include <cstring>
#include <vector>
#include <iostream>

namespace spla {

static constexpr int resultTag = 1;
static constexpr int ringTag = 2;

template <typename T>
RingReduceTileHost<T>::RingReduceTileHost(
    IntType maxBlockSize, IntType numThreads, MPICommunicatorHandle comm,
    std::shared_ptr<Buffer<MPIAllocator>> buffer,
    std::shared_ptr<Buffer<MPIAllocator>> resultBuffer,
    BlockCyclicGenerator matrixDist, SplaOperation opA, ValueType alpha,
    const HostArrayConstView2D<ValueType> &A,
    const HostArrayConstView2D<ValueType> &B, ValueType beta,
    HostArrayView2D<ValueType> C)
    : state_(TileState::Empty), matrixDist_(std::move(matrixDist)),
      buffer_(std::move(buffer)), resultBuffer_(std::move(resultBuffer)),
      comm_(std::move(comm)), A_(A), B_(B), C_(C), alpha_(alpha), beta_(beta),
      opA_(opA), numThreads_(numThreads), maxBlockSize_(maxBlockSize) {
  assert(A_.dim_inner() == B_.dim_inner());
  assert(buffer_);
  assert(opA_ == SplaOperation::SPLA_OP_CONJ_TRANSPOSE ||
         opA_ == SplaOperation::SPLA_OP_TRANSPOSE);
  buffer_->resize<ValueType>(2 * maxBlockSize_);
  sendView_ = HostArrayView1D<T>(buffer_->data<T>(), maxBlockSize_);
  recvView_ = HostArrayView1D<T>(buffer_->data<T>() + maxBlockSize_, maxBlockSize_);
}

template <typename T>
auto RingReduceTileHost<T>::prepare(
    std::vector<BlockCoord>::const_iterator begin,
    std::vector<BlockCoord>::const_iterator end) -> void {
  assert(state_ == TileState::Empty);
  assert(begin != end);

  blockInfos_.assign(begin, end);

  currentBlockIdx = 0;
  const IntType rankOffset =
      matrixDist_.create_sub_generator(blockInfos_.front()).get_mpi_rank(0);
  myStartIdx_ = (rankOffset + comm_.rank()) % blockInfos_.size();
  sendRank_ = comm_.rank() == 0 ? comm_.size() - 1 : comm_.rank() - 1;
  recvRank_ = (comm_.rank() + 1) % comm_.size();

  const bool accumulateRequired = blockInfos_.size() != comm_.size();
  // const bool accumulateRequired = true;

  myBlockInfos_.resize(0);
  for (IntType i = 0; i < blockInfos_.size(); ++i) {
    auto gen = matrixDist_.create_sub_generator(blockInfos_[i]);
    // Determine rank to receive result from by computing the rank, which
    // holds the block initially and substracting the number of steps in the
    // ring (blocks are send backwards)
    const auto originRank =
        (i + 2 * comm_.size() - rankOffset - (blockInfos_.size() - 1)) %
        comm_.size();
    for(IntType j = 0; j < gen.num_blocks(); ++j ) {
      if(gen.get_mpi_rank(j) == comm_.rank()) {
        myBlockInfos_.emplace_back(originRank, gen.get_block_info(j));
      }
    }
  }

  std::memset(recvView_.data(), 0, recvView_.size() * sizeof(T));
  resultBuffer_->resize<T>(std::max<std::size_t>(myBlockInfos_.size(), 1) *
                           maxBlockSize_);
  resultRecvs_.resize(myBlockInfos_.size());

  if (!accumulateRequired) {
    for(IntType i = 0; i < myBlockInfos_.size(); ++i){
      const auto& pair = myBlockInfos_[i];
      MPI_Irecv(resultBuffer_->data<T>() + i * maxBlockSize_,
                pair.second.numCols * pair.second.numRows,
                MPIMatchElementaryType<T>::get(), pair.first, resultTag,
                comm_.get(), resultRecvs_[i].get_and_activate());
    }
  } else {
    std::memset(resultBuffer_->data<T>(), 0,
                resultBuffer_->size<T>() * sizeof(T));
  }

  state_ = TileState::Prepared;
}


template <typename T> auto RingReduceTileHost<T>::process_step_ring() -> void {
  const IntType numBlocks = blockInfos_.size();

  const auto &block =
      blockInfos_[(myStartIdx_ + currentBlockIdx) % blockInfos_.size()];
  const auto &nextBlock =
      blockInfos_[(myStartIdx_ + currentBlockIdx + 1) % blockInfos_.size()];

  sendReq_.wait_if_active();
  recvReq_.wait_if_active();
  std::swap(sendView_, recvView_);

  if (currentBlockIdx < numBlocks - 1) {
    MPI_Irecv(recvView_.data(), nextBlock.numCols * nextBlock.numRows,
              MPIMatchElementaryType<T>::get(), recvRank_, ringTag, comm_.get(),
              recvReq_.get_and_activate());
  }
  if (A_.dim_inner() != 0) {
    gemm_host<T>(numThreads_, opA_, SplaOperation::SPLA_OP_NONE, block.numRows,
                 block.numCols, A_.dim_inner(), alpha_,
                 &A_(block.row, 0), A_.ld_inner(),
                 &B_(block.col, 0), B_.ld_inner(), 1.0,
                 sendView_.data(), block.numRows);
  }
  if (currentBlockIdx < numBlocks - 1) { // continue sending around in ring
    MPI_Isend(sendView_.data(), block.numRows * block.numCols,
              MPIMatchElementaryType<T>::get(), sendRank_, ringTag, comm_.get(),
              sendReq_.get_and_activate());
  } else { // send final result to target rank
    auto gen = matrixDist_.create_sub_generator(block);
    for (IntType i = 0; i < gen.num_blocks(); ++i) {
      auto info = gen.get_block_info(i);
      auto datatType = MPIDatatypeHandle::create_vector(
          info.numCols, info.numRows, block.numRows,
          MPIMatchElementaryType<T>::get());
      HostArrayConstView2D<T> resultView(
          sendView_.data(), block.numCols, block.numRows);
      MPI_Send(&resultView(info.globalSubColIdx, info.globalSubRowIdx), 1,
               datatType.get(), info.mpiRank, resultTag, comm_.get());
    }
  }
  state_ = TileState::PartiallyProcessed;
}


template <typename T> auto RingReduceTileHost<T>::process_step_reduction() -> void {
  const auto &block = blockInfos_[currentBlockIdx];

  sendReq_.wait_if_active();

  if(currentBlockIdx) {
    const auto& previousBlock = blockInfos_[currentBlockIdx - 1];
    auto gen = matrixDist_.create_sub_generator(previousBlock);
    HostArrayConstView2D<T> resultView(sendView_.data(), previousBlock.numCols,
                                       previousBlock.numRows);
    for (IntType i = 0; i < gen.num_blocks(); ++i) {
      if (gen.get_mpi_rank(i) == comm_.rank()) {
        const auto info = gen.get_block_info(i);

        add_kernel(info.numRows, info.numCols,
                   &resultView(info.globalSubColIdx, info.globalSubRowIdx),
                   resultView.ld_inner(), beta_,
                   &C_(info.localColIdx, info.localRowIdx), C_.ld_inner());
      }
    }
  }

  if (A_.dim_inner() == 0) {
    std::memset(sendView_.data(), 0, sendView_.size() * sizeof(T));
  } else {
    gemm_host<T>(numThreads_, opA_, SplaOperation::SPLA_OP_NONE, block.numRows,
                 block.numCols, A_.dim_inner(), alpha_,
                 &A_(block.row, 0), A_.ld_inner(),
                 &B_(block.col, 0), B_.ld_inner(), 0.0,
                 sendView_.data(), block.numRows);
  }

  mpi_check_status(MPI_Iallreduce(
      MPI_IN_PLACE, sendView_.data(), block.numCols * block.numRows,
      MPIMatchElementaryType<ValueType>::get(), MPI_SUM, comm_.get(),
      sendReq_.get_and_activate()));

  state_ = TileState::PartiallyProcessed;
}

template <typename T>
auto RingReduceTileHost<T>::process_step_finalize() -> void {
  // add tile to result as final step
  sendReq_.wait_if_active();
  recvReq_.wait_if_active();

  const bool accumulateRequired = blockInfos_.size() != comm_.size();
  // const bool accumulateRequired = true;
  if (accumulateRequired) {
    const auto &previousBlock = blockInfos_.back();
    auto gen = matrixDist_.create_sub_generator(previousBlock);
    HostArrayConstView2D<T> resultView(sendView_.data(), previousBlock.numCols,
                                       previousBlock.numRows);
    for (IntType i = 0; i < gen.num_blocks(); ++i) {
      if (gen.get_mpi_rank(i) == comm_.rank()) {
        const auto info = gen.get_block_info(i);

        add_kernel(info.numRows, info.numCols,
                   &resultView(info.globalSubColIdx, info.globalSubRowIdx),
                   resultView.ld_inner(), beta_,
                   &C_(info.localColIdx, info.localRowIdx), C_.ld_inner());
      }
    }
  } else {
    for (IntType i = 0; i < myBlockInfos_.size(); ++i) {
      resultRecvs_[i].wait_if_active();
      const auto &info = myBlockInfos_[i].second;

      add_kernel(info.numRows, info.numCols,
                 resultBuffer_->data<T>() + i * maxBlockSize_, info.numRows,
                 beta_, &C_(info.localColIdx, info.localRowIdx), C_.ld_inner());
    }
  }

  state_ = TileState::Empty;
}

template <typename T> auto RingReduceTileHost<T>::process_step() -> bool {
  const bool accumulateRequired = blockInfos_.size() != comm_.size();
  // const bool accumulateRequired = true;
  const IntType numBlocks = blockInfos_.size();

  if (currentBlockIdx < numBlocks) {
    if (!accumulateRequired) {
      this->process_step_ring();
    } else {
      this->process_step_reduction();
    }
  } else if (currentBlockIdx == numBlocks && currentBlockIdx > 0) {
    this->process_step_finalize();
  }

  ++currentBlockIdx;
  return currentBlockIdx <= numBlocks;
}

template class RingReduceTileHost<double>;
template class RingReduceTileHost<float>;
template class RingReduceTileHost<std::complex<double>>;
template class RingReduceTileHost<std::complex<float>>;

} // namespace spla
