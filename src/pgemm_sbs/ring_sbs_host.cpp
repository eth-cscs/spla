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
#include "pgemm_sbs/ring_sbs_host.hpp"

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstring>
#include <vector>

#include "block_generation/block_cyclic_generator.hpp"
#include "block_generation/mirror_generator.hpp"
#include "gemm/gemm_host.hpp"
#include "mpi_util/mpi_check_status.hpp"
#include "mpi_util/mpi_datatype_handle.hpp"
#include "mpi_util/mpi_match_elementary_type.hpp"
#include "pgemm_ssb/add_kernel.hpp"
#include "timing/timing.hpp"
#include "util/blas_interface.hpp"
#include "util/common_types.hpp"

namespace spla {

static constexpr int collectTag = 1;
static constexpr int ringTag = 2;

template <typename T, typename BLOCK_GEN>
RingSBSHost<T, BLOCK_GEN>::RingSBSHost(
    double ringThreshold, IntType maxBlockSize, IntType numThreads, MPICommunicatorHandle comm,
    const std::shared_ptr<Allocator<MemLoc::Host>> allocator, BLOCK_GEN baseMatGen, ValueType alpha,
    const HostArrayConstView2D<ValueType> &A, const HostArrayConstView2D<ValueType> &B,
    IntType bRowOffset, IntType bColOffset, ValueType beta, HostArrayView2D<ValueType> C)
    : state_(TileState::Empty),
      baseMatGen_(std::move(baseMatGen)),
      buffer_(allocator, 0),
      comm_(std::move(comm)),
      A_(A),
      B_(B),
      C_(C),
      bRowOffset_(bRowOffset),
      bColOffset_(bColOffset),
      alpha_(alpha),
      beta_(beta),
      numThreads_(numThreads),
      maxBlockSize_(maxBlockSize),
      ringThreshold_(ringThreshold) {
  assert(A_.dim_inner() == C_.dim_inner());
  buffer_.resize(2 * maxBlockSize_);
  sendView_ = HostArrayView1D<T>(buffer_.data(), maxBlockSize_);
  recvView_ = HostArrayView1D<T>(buffer_.data() + maxBlockSize_, maxBlockSize_);

  sendRank_ = comm_.rank() == 0 ? comm_.size() - 1 : comm_.rank() - 1;
  recvRank_ = (comm_.rank() + 1) % comm_.size();
}

template <typename T, typename BLOCK_GEN>
auto RingSBSHost<T, BLOCK_GEN>::prepare(std::vector<Block>::const_iterator begin,
                                        std::vector<Block>::const_iterator end) -> void {
  SCOPED_TIMING("prepare")
  assert(state_ == TileState::Empty);
  assert(begin != end);

  blocks_.assign(begin, end);

  stepIdx_ = 0;
  rankOffset_ = baseMatGen_.create_sub_generator(blocks_.front()).get_mpi_rank(0);
  myStartIdx_ = (rankOffset_ + comm_.rank()) % comm_.size();
  useRing_ =
      IsDisjointGenerator<BLOCK_GEN>::value &&
      static_cast<double>(blocks_.size()) >= static_cast<double>(comm_.size()) * ringThreshold_;

  // Issue receives if this rank holds initial block
  collectRecvs_.resize(0);
  if (myStartIdx_ < blocks_.size()) {
    SCOPED_TIMING("irecv")
    auto myStartBlock = blocks_[myStartIdx_];
    HostArrayView2D<T> startBlockView(recvView_.data(), myStartBlock.numCols, myStartBlock.numRows);
    auto gen = baseMatGen_.create_sub_generator(myStartBlock);
    for (IntType j = 0; j < gen.num_blocks(); ++j) {
      auto info = gen.get_block_info(j);
      if (info.mpiRank == comm_.rank()) {
        // If data is local, copy directly instead of using MPI
        for (IntType c = 0; c < info.numCols; ++c) {
          std::memcpy(&startBlockView(info.globalColIdx - myStartBlock.col - bColOffset_ + c,
                                      info.globalRowIdx - myStartBlock.row - bRowOffset_),
                      &B_(info.localColIdx + c, info.localRowIdx), info.numRows * sizeof(T));
        }
      } else {
        auto mpiVec =
            MPIDatatypeHandle::create_vector(info.numCols, info.numRows, startBlockView.ld_inner(),
                                             MPIMatchElementaryType<T>::get());
        collectRecvs_.emplace_back();
        MPI_Irecv(&startBlockView(info.globalColIdx - myStartBlock.col - bColOffset_,
                                  info.globalRowIdx - myStartBlock.row - bRowOffset_),
                  1, mpiVec.get(), info.mpiRank, collectTag, comm_.get(),
                  collectRecvs_.back().get_and_activate());
      }
    }
  }

  START_TIMING("send")
  // Send data required for blocks in ring
  for (IntType i = 0; i < blocks_.size(); ++i) {
    auto gen = baseMatGen_.create_sub_generator(blocks_[i]);
    for (IntType j = 0; j < gen.num_blocks(); ++j) {
      if (gen.get_mpi_rank(j) == comm_.rank()) {
        auto info = gen.get_block_info(j);
        const auto targetRank = (i + comm_.size() - rankOffset_) % comm_.size();
        if (targetRank != comm_.rank()) {
          // Copy into send buffer. Only send to other ranks, since local data was already copied.
          // Testing showed copying is faster than using custom mpi vec type directly.
          for (IntType c = 0; c < info.numCols; ++c) {
            std::memcpy(sendView_.data() + c * info.numRows,
                        &B_(info.localColIdx + c, info.localRowIdx), info.numRows * sizeof(T));
          }
          MPI_Send(sendView_.data(), info.numRows * info.numCols, MPIMatchElementaryType<T>::get(),
                   targetRank, collectTag, comm_.get());
        }
      }
    }
  }
  STOP_TIMING("send")

  START_TIMING("wait_recv")
  // Wait for all receives
  for (auto &r : collectRecvs_) {
    r.wait_if_active();
  }
  STOP_TIMING("wait_recv")

  state_ = TileState::Prepared;
}

template <typename T, typename BLOCK_GEN>
auto RingSBSHost<T, BLOCK_GEN>::process_step_ring(std::unordered_set<IntType> &betaColIndeces)
    -> void {
  SCOPED_TIMING("ring_step")
  const IntType numBlocks = blocks_.size();

  const IntType blockIdx = (myStartIdx_ + stepIdx_) % comm_.size();
  const IntType nextBlockIdx = (myStartIdx_ + stepIdx_ + 1) % comm_.size();

  START_TIMING("mpi_wait")
  sendReq_.wait_if_active();
  recvReq_.wait_if_active();
  STOP_TIMING("mpi_wait")
  std::swap(sendView_, recvView_);

  if (stepIdx_ < comm_.size() - 1 && nextBlockIdx < numBlocks) {
    SCOPED_TIMING("irecv")
    const auto &nextBlock = blocks_[nextBlockIdx];
    MPI_Irecv(recvView_.data(), nextBlock.numCols * nextBlock.numRows,
              MPIMatchElementaryType<T>::get(), recvRank_, ringTag, comm_.get(),
              recvReq_.get_and_activate());
  }

  if (blockIdx < numBlocks) {
    const auto &block = blocks_[blockIdx];
    if (stepIdx_ < comm_.size() - 1) {
      SCOPED_TIMING("isend")
      MPI_Isend(sendView_.data(), block.numRows * block.numCols, MPIMatchElementaryType<T>::get(),
                sendRank_, ringTag, comm_.get(), sendReq_.get_and_activate());
    }

    if (A_.dim_inner() != 0) {
      SCOPED_TIMING("gemm")
      T beta = 1.0;
      if (!betaColIndeces.count(block.col)) {
        betaColIndeces.emplace(block.col);
        beta = beta_;
      }
      gemm_host<T>(numThreads_, SplaOperation::SPLA_OP_NONE, SplaOperation::SPLA_OP_NONE,
                   A_.dim_inner(), block.numCols, block.numRows, alpha_, &A_(block.row, 0),
                   A_.ld_inner(), sendView_.data(), block.numRows, beta, &C_(block.col, 0),
                   C_.ld_inner());
    }
  }
  state_ = stepIdx_ >= comm_.size() - 1 ? TileState::Empty : TileState::PartiallyProcessed;
}

template <typename T, typename BLOCK_GEN>
auto RingSBSHost<T, BLOCK_GEN>::process_step_broadcast(std::unordered_set<IntType> &betaColIndeces)
    -> void {
  SCOPED_TIMING("broadcast_step")
  IntType numBlocks = blocks_.size();

  if (stepIdx_ < numBlocks) {
    auto blockView = myStartIdx_ == stepIdx_ ? recvView_ : sendView_;
    auto block = blocks_[stepIdx_];
    const auto sourceRank = (stepIdx_ + comm_.size() - rankOffset_) % comm_.size();
    START_TIMING("bcast")
    MPI_Bcast(blockView.data(), block.numCols * block.numRows, MPIMatchElementaryType<T>::get(),
              sourceRank, comm_.get());
    STOP_TIMING("bcast")
    if (A_.dim_inner() != 0) {
      SCOPED_TIMING("gemm")
      T beta = 1.0;
      if (!betaColIndeces.count(block.col)) {
        betaColIndeces.emplace(block.col);
        beta = beta_;
      }
      gemm_host<T>(numThreads_, SplaOperation::SPLA_OP_NONE, SplaOperation::SPLA_OP_NONE,
                   A_.dim_inner(), block.numCols, block.numRows, alpha_, &A_(block.row, 0),
                   A_.ld_inner(), blockView.data(), block.numRows, beta, &C_(block.col, 0),
                   C_.ld_inner());
    }
  }

  state_ = stepIdx_ >= numBlocks - 1 ? TileState::Empty : TileState::PartiallyProcessed;
}

template <typename T, typename BLOCK_GEN>
auto RingSBSHost<T, BLOCK_GEN>::process_step(std::unordered_set<IntType> &betaColIndeces) -> bool {
  if (blocks_.empty()) return false;

  if (stepIdx_ < comm_.size()) {
    if (useRing_) {
      this->process_step_ring(betaColIndeces);
    } else {
      this->process_step_broadcast(betaColIndeces);
    }
  }

  ++stepIdx_;
  return stepIdx_ < comm_.size();
}

template class RingSBSHost<double, BlockCyclicGenerator>;
template class RingSBSHost<float, BlockCyclicGenerator>;
template class RingSBSHost<std::complex<double>, BlockCyclicGenerator>;
template class RingSBSHost<std::complex<float>, BlockCyclicGenerator>;

template class RingSBSHost<double, MirrorGenerator>;
template class RingSBSHost<float, MirrorGenerator>;
template class RingSBSHost<std::complex<double>, MirrorGenerator>;
template class RingSBSHost<std::complex<float>, MirrorGenerator>;

}  // namespace spla
