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
#include "pgemm_sbs/ring_sbs_gpu.hpp"

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstring>
#include <memory>

#include "block_generation/block_cyclic_generator.hpp"
#include "block_generation/mirror_generator.hpp"
#include "gpu_util/gpu_blas_api.hpp"
#include "gpu_util/gpu_helper.hpp"
#include "gpu_util/gpu_runtime_api.hpp"
#include "gpu_util/gpu_transfer.hpp"
#include "gpu_util/multiply_gpu.hpp"
#include "memory/gpu_array_const_view.hpp"
#include "memory/host_array_view.hpp"
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

template <typename T>
static auto sbs_gemm_gpu_async(GPUBlasHandle& blasHandle, T alpha, GPUConstMatrixAccessor<T> matA,
                               GPUConstMatrixAccessor<T> matB, T beta, GPUMatrixAccessor<T> matC)
    -> void {
  const IntType m = matC.rows();
  const IntType n = matC.cols();
  const IntType k = matA.cols();
  IntType rowBlockSize = m;
  if (matC.max_tile_size() < matC.size()) {
    // if not fully on GPU, try square size
    rowBlockSize =
        std::min<IntType>(std::sqrt(matC.max_tile_size()), rowBlockSize);
  }

  const IntType colBlockSize = std::min(matC.max_tile_size() / rowBlockSize, n);
  rowBlockSize = std::min(matC.max_tile_size() / colBlockSize, m);

  IntType counter = 0;
  for (IntType col = 0; col < n; col += colBlockSize) {
    const IntType currentCols = std::min(n - col, colBlockSize);

    for (IntType row = 0; row < m; row += rowBlockSize, ++counter) {
      const IntType currentRows = std::min(m - row, rowBlockSize);

      auto viewC =
          matC.get_tile(row, col, currentRows, currentCols, blasHandle.stream_handle().get());
      multiply_gpu<T>(blasHandle.get(), gpu::blas::operation::None, gpu::blas::operation::None,
                      alpha, matA.sub_accessor(row, 0, currentRows, k),
                      matB.sub_accessor(0, col, k, currentCols), beta, viewC);
      matC.copy_back(viewC, row, col, blasHandle.stream_handle().get());
    }
  }
}

template <typename T, typename BLOCK_GEN>
RingSBSGPU<T, BLOCK_GEN>::RingSBSGPU(double ringThreshold, IntType maxBlockSize,
                                     MPICommunicatorHandle comm,
                                     std::vector<RingProcessorSBS<T>> ringProcs,
                                     BLOCK_GEN baseMatGen, ValueType alpha, ValueType beta,
                                     HostArrayConstView2D<ValueType> HostMatB,
                                     GPUArrayConstView2D<ValueType> GPUMatB, IntType bRowOffset,
                                     IntType bColOffset)
    : state_(TileState::Empty),
      comm_(std::move(comm)),
      baseMatGen_(std::move(baseMatGen)),
      ringProcs_(std::move(ringProcs)),
      HostMatB_(HostMatB),
      GPUMatB_(GPUMatB),
      bRowOffset_(bRowOffset),
      bColOffset_(bColOffset),
      alpha_(alpha),
      beta_(beta),
      maxBlockSize_(maxBlockSize),
      ringThreshold_(ringThreshold) {
  sendRank_ = comm_.rank() == 0 ? comm_.size() - 1 : comm_.rank() - 1;
  recvRank_ = (comm_.rank() + 1) % comm_.size();
}

template <typename T, typename BLOCK_GEN>
auto RingSBSGPU<T, BLOCK_GEN>::prepare(std::vector<Block>::const_iterator begin,
                                              std::vector<Block>::const_iterator end) -> void {
  SCOPED_TIMING("prepare")
  assert(state_ == TileState::Empty);
  assert(begin != end);

  blocks_.assign(begin, end);

  stepIdx_ = 0;
  rankOffset_ = baseMatGen_.create_sub_generator(blocks_.front()).get_mpi_rank(0);
  myStartIdx_ = (rankOffset_ + comm_.rank()) % comm_.size();
  // useRing_ =
  //     IsDisjointGenerator<BLOCK_GEN>::value &&
  //     static_cast<double>(blocks_.size()) >= static_cast<double>(comm_.size()) * ringThreshold_;
  useRing_ = false;

  auto& firstProc = ringProcs_.front();
  // Make sure no memory transfer is still running before receiving
  gpu::check_status(gpu::stream_synchronize(firstProc.blasHandle.stream_handle().get()));

  // Issue receives if this rank holds initial block
  collectRecvs_.resize(0);
  if (myStartIdx_ < blocks_.size()) {
    SCOPED_TIMING("irecv")
    auto myStartBlock = blocks_[myStartIdx_];
    HostArrayView2D<T> startBlockView(firstProc.recvView.data(), myStartBlock.numCols,
                                      myStartBlock.numRows);
    assert(firstProc.recvView.size() >= myStartBlock.numCols * myStartBlock.numRows);
    auto gen = baseMatGen_.create_sub_generator(myStartBlock);
    for (IntType j = 0; j < gen.num_blocks(); ++j) {
      auto info = gen.get_block_info(j);
      if (info.mpiRank == comm_.rank()) {
        // If data is local, copy directly instead of using MPI
        if (HostMatB_.empty()) {
          copy_from_gpu_async(
              firstProc.blasHandle.stream_handle().get(),
              GPUArrayConstView2D<T>(
                  GPUMatB_.data() + GPUMatB_.index(info.localColIdx, info.localRowIdx),
                  info.numCols, info.numRows, GPUMatB_.ld_inner()),
              HostArrayView2D<T>(
                  &startBlockView(info.globalColIdx - myStartBlock.col - bColOffset_,
                                  info.globalRowIdx - myStartBlock.row - bRowOffset_),
                  info.numCols, info.numRows, startBlockView.ld_inner()));
        } else {
          for (IntType c = 0; c < info.numCols; ++c) {
            std::memcpy(&startBlockView(info.globalColIdx - myStartBlock.col - bColOffset_ + c,
                                        info.globalRowIdx - myStartBlock.row - bRowOffset_),
                        &HostMatB_(info.localColIdx + c, info.localRowIdx),
                        info.numRows * sizeof(T));
          }
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
          if (HostMatB_.empty()) {
            copy_from_gpu(GPUArrayConstView2D<T>(
                              GPUMatB_.data() + GPUMatB_.index(info.localColIdx, info.localRowIdx),
                              info.numCols, info.numRows, GPUMatB_.ld_inner()),
                          HostArrayView2D<T>(firstProc.sendView.data(), info.numCols, info.numRows,
                                             info.numRows));
          } else {
            assert(firstProc.sendView.size() >= info.numCols * info.numRows);
            for (IntType c = 0; c < info.numCols; ++c) {
              std::memcpy(firstProc.sendView.data() + c * info.numRows,
                          &HostMatB_(info.localColIdx + c, info.localRowIdx),
                          info.numRows * sizeof(T));
            }
          }
          MPI_Send(firstProc.sendView.data(), info.numRows * info.numCols, MPIMatchElementaryType<T>::get(),
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
auto RingSBSGPU<T, BLOCK_GEN>::process_step_ring(std::unordered_set<IntType>& betaColIndeces,
                                                 std::deque<GPUEventHandle>& colEvents) -> void {
  SCOPED_TIMING("ring_step")
  const IntType numBlocks = blocks_.size();

  const IntType blockIdx = (myStartIdx_ + stepIdx_) % comm_.size();
  const IntType nextBlockIdx = (myStartIdx_ + stepIdx_ + 1) % comm_.size();

  auto& proc = ringProcs_[stepIdx_ % ringProcs_.size()];
  auto& nextProc = ringProcs_[(stepIdx_ + 1) % ringProcs_.size()];

  if (stepIdx_ == 0) {
    // Make sure memory transfers for assembling the first block are done.
    gpu::check_status(gpu::stream_synchronize(proc.blasHandle.stream_handle().get()));
  }

  sendReq_.wait_if_active();
  recvReq_.wait_if_active();
  std::swap(proc.sendView, proc.recvView);


  if (stepIdx_ < comm_.size() - 1 && nextBlockIdx < numBlocks) {
    // Make sure data is on GPU before receiving again
    gpu::check_status(gpu::stream_synchronize(nextProc.blasHandle.stream_handle().get()));
    const auto &nextBlock = blocks_[nextBlockIdx];
    MPI_Irecv(nextProc.recvView.data(), nextBlock.numCols * nextBlock.numRows,
              MPIMatchElementaryType<T>::get(), recvRank_, ringTag, comm_.get(),
              recvReq_.get_and_activate());
  }

  if (blockIdx < numBlocks) {
    const auto &block = blocks_[blockIdx];
    HostArrayConstView2D<T> blockView(proc.sendView.data(), block.numCols, block.numRows);
    GPUArrayView2D<T> blockViewGPU(proc.tileViewGPU.data(), block.numCols, block.numRows);

    if (proc.matA.size() != 0) {
      copy_to_gpu_async(proc.blasHandle.stream_handle().get(), blockView, blockViewGPU);
    }

    if (stepIdx_ < comm_.size() - 1) {
      MPI_Isend(proc.sendView.data(), block.numRows * block.numCols, MPIMatchElementaryType<T>::get(),
                sendRank_, ringTag, comm_.get(), sendReq_.get_and_activate());
    }

    if (proc.matA.size() != 0) {
      SCOPED_TIMING("gemm")
      T beta = RealValueGPU<T>::create(1.0);
      if(!betaColIndeces.count(block.col)) {
        betaColIndeces.emplace(block.col);
        beta = beta_;
      }

      // Make sure no other stream is writing to the same location. Select event based on coloumn
      // index, which determines the write location.
      auto& event = colEvents[(block.col / block.numCols) % colEvents.size()];
      gpu::stream_wait_event(proc.blasHandle.stream_handle().get(), event.get(), 0);

      sbs_gemm_gpu_async<T>(proc.blasHandle, alpha_,
                            proc.matA.sub_accessor(0, block.row, proc.matA.rows(), block.numRows),
                            GPUConstMatrixAccessor<T>(blockViewGPU), beta,
                            proc.matC.sub_accessor(0, block.col, proc.matC.rows(), block.numCols));
      gpu::event_record(event.get(), proc.blasHandle.stream_handle().get());
    }
  }
  state_ = stepIdx_ >= comm_.size() - 1 ? TileState::Empty : TileState::PartiallyProcessed;
}

template <typename T, typename BLOCK_GEN>
auto RingSBSGPU<T, BLOCK_GEN>::process_step_broadcast(std::unordered_set<IntType>& betaColIndeces,
                                                      std::deque<GPUEventHandle>& colEvents)
    -> void {
  SCOPED_TIMING("broadcast_step")
  IntType numBlocks = blocks_.size();

  if (stepIdx_ < numBlocks) {
    const auto sourceRank = (stepIdx_ + comm_.size() - rankOffset_) % comm_.size();
    auto block = blocks_[stepIdx_];
    auto& proc = ringProcs_[stepIdx_ % ringProcs_.size()];
    auto& firstProc = ringProcs_.front();
    auto blockView = HostArrayView2D<T>(
        sourceRank == comm_.rank() ? firstProc.recvView.data() : proc.sendView.data(),
        block.numCols, block.numRows);
    auto blockViewGPU = GPUArrayView2D<T>(proc.tileViewGPU.data(), block.numCols, block.numRows);

    // Make sure memory transfers are done before overwriting data through broadcast
    gpu::check_status(gpu::stream_synchronize(proc.blasHandle.stream_handle().get()));

    MPI_Bcast(blockView.data(), block.numCols * block.numRows, MPIMatchElementaryType<T>::get(),
              sourceRank, comm_.get());

    if (proc.matA.size() != 0) {
      SCOPED_TIMING("gemm")
      copy_to_gpu_async(proc.blasHandle.stream_handle().get(), HostArrayConstView2D<T>(blockView),
                        blockViewGPU);

      T beta = RealValueGPU<T>::create(1.0);
      if(!betaColIndeces.count(block.col)) {
        betaColIndeces.emplace(block.col);
        beta = beta_;
      }

      // Make sure no other stream is writing to the same location. Select event based on coloumn
      // index, which determines the write location.
      auto& event = colEvents[(block.col / block.numCols) % colEvents.size()];
      gpu::stream_wait_event(proc.blasHandle.stream_handle().get(), event.get(), 0);

      sbs_gemm_gpu_async<T>(proc.blasHandle, alpha_,
                            proc.matA.sub_accessor(0, block.row, proc.matA.rows(), block.numRows),
                            GPUConstMatrixAccessor<T>(blockViewGPU), beta,
                            proc.matC.sub_accessor(0, block.col, proc.matC.rows(), block.numCols));
      gpu::event_record(event.get(), proc.blasHandle.stream_handle().get());
    }
  }

  state_ = stepIdx_ >= numBlocks - 1 ? TileState::Empty : TileState::PartiallyProcessed;
}

template <typename T, typename BLOCK_GEN>
auto RingSBSGPU<T, BLOCK_GEN>::process_step(std::unordered_set<IntType>& betaColIndeces,
                                            std::deque<GPUEventHandle>& colEvents) -> bool {
  if (blocks_.empty()) return false;

  if(stepIdx_ < comm_.size()) {
    if (useRing_) {
      this->process_step_ring(betaColIndeces, colEvents);
    } else {
      this->process_step_broadcast(betaColIndeces, colEvents);
    }
  }

  ++stepIdx_;
  return stepIdx_ < comm_.size();
}

template class RingSBSGPU<double, BlockCyclicGenerator>;
template class RingSBSGPU<float, BlockCyclicGenerator>;
template class RingSBSGPU<gpu::blas::ComplexFloatType, BlockCyclicGenerator>;
template class RingSBSGPU<gpu::blas::ComplexDoubleType, BlockCyclicGenerator>;

template class RingSBSGPU<double, MirrorGenerator>;
template class RingSBSGPU<float, MirrorGenerator>;
template class RingSBSGPU<gpu::blas::ComplexFloatType, MirrorGenerator>;
template class RingSBSGPU<gpu::blas::ComplexDoubleType, MirrorGenerator>;

}  // namespace spla
