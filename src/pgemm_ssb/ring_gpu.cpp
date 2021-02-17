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
#include "pgemm_ssb/ring_gpu.hpp"

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
#include "util/blas_interface.hpp"
#include "util/common_types.hpp"

namespace spla {

static constexpr int resultTag = 1;
static constexpr int ringTag = 2;

static auto call_gpu_geam(const gpu::blas::HandleType &handle,
                          const gpu::blas::OperationType &transa,
                          const gpu::blas::OperationType &transb, int m, int n, float alpha,
                          const float *A, int lda, float beta, const float *B, int ldb, float *C,
                          int ldc) -> void {
  gpu::blas::sgeam(handle, transa, transb, m, n, &alpha, A, lda, &beta, B, ldb, C, ldc);
}

static auto call_gpu_geam(const gpu::blas::HandleType &handle,
                          const gpu::blas::OperationType &transa,
                          const gpu::blas::OperationType &transb, int m, int n, double alpha,
                          const double *A, int lda, double beta, const double *B, int ldb,
                          double *C, int ldc) -> void {
  gpu::blas::dgeam(handle, transa, transb, m, n, &alpha, A, lda, &beta, B, ldb, C, ldc);
}

static auto call_gpu_geam(const gpu::blas::HandleType &handle,
                          const gpu::blas::OperationType &transa,
                          const gpu::blas::OperationType &transb, int m, int n,
                          gpu::blas::ComplexDoubleType alpha, const gpu::blas::ComplexDoubleType *A,
                          int lda, gpu::blas::ComplexDoubleType beta,
                          const gpu::blas::ComplexDoubleType *B, int ldb,
                          gpu::blas::ComplexDoubleType *C, int ldc) -> void {
  gpu::blas::zgeam(handle, transa, transb, m, n, &alpha, A, lda, &beta, B, ldb, C, ldc);
}

static auto call_gpu_geam(const gpu::blas::HandleType &handle,
                          const gpu::blas::OperationType &transa,
                          const gpu::blas::OperationType &transb, int m, int n,
                          gpu::blas::ComplexFloatType alpha, const gpu::blas::ComplexFloatType *A,
                          int lda, gpu::blas::ComplexFloatType beta,
                          const gpu::blas::ComplexFloatType *B, int ldb,
                          gpu::blas::ComplexFloatType *C, int ldc) -> void {
  gpu::blas::cgeam(handle, transa, transb, m, n, &alpha, A, lda, &beta, B, ldb, C, ldc);
}

template <typename T, typename BLOCK_GEN>
RingGPU<T, BLOCK_GEN>::RingGPU(
    double ringThreshold, IntType maxBlockSize, MPICommunicatorHandle comm,
    std::vector<RingProcessor<T>> ringProcs,
    std::shared_ptr<Buffer<PinnedAllocator>> resultBufferHost, BLOCK_GEN baseMatGen,
    SplaOperation opA, ValueType alpha, ValueType beta, HostArrayView2D<ValueType> HostMatC,
    GPUArrayView2D<ValueType> GPUMatC)
    : state_(TileState::Empty),
      comm_(std::move(comm)),
      baseMatGen_(std::move(baseMatGen)),
      ringProcs_(std::move(ringProcs)),
      resultBufferHost_(std::move(resultBufferHost)),
      HostMatC_(HostMatC),
      GPUMatC_(GPUMatC),
      alpha_(alpha),
      beta_(beta),
      opA_(opA),
      maxBlockSize_(maxBlockSize),
      ringThreshold_(ringThreshold) {
  assert(ringProcs_.size() >= 2);  // Ring algorithm relies on at least 2
  assert(resultBufferHost_);
  assert(opA_ == SplaOperation::SPLA_OP_CONJ_TRANSPOSE || opA_ == SplaOperation::SPLA_OP_TRANSPOSE);
}

template <typename T, typename BLOCK_GEN>
auto RingGPU<T, BLOCK_GEN>::prepare(std::vector<Block>::const_iterator begin,
                                              std::vector<Block>::const_iterator end) -> void {
  assert(state_ == TileState::Empty);

  blocks_.assign(begin, end);

  const IntType rankOffset = baseMatGen_.create_sub_generator(blocks_.front()).get_mpi_rank(0) + 1;
  myStartIdx_ = (rankOffset + comm_.rank()) % comm_.size();
  sendRank_ = comm_.rank() == 0 ? comm_.size() - 1 : comm_.rank() - 1;
  recvRank_ = (comm_.rank() + 1) % comm_.size();
  stepIdx_ = 0;
  procIdx_ = 0;
  numMultipliedBlocks_ = 0;

  useRing_ =
      IsDisjointGenerator<BLOCK_GEN>::value &&
      static_cast<double>(blocks_.size()) >= static_cast<double>(comm_.size()) * ringThreshold_;

  myBlockInfos_.resize(0);
  std::size_t requiredBufferSize = 0;
  for (IntType i = 0; i < blocks_.size(); ++i) {
    auto gen = baseMatGen_.create_sub_generator(blocks_[i]);
    // Determine rank to receive result from by computing the rank, which
    // holds the proc initially and substracting the number of steps in the
    // ring (blocks are send backwards)
    const auto originRank = (i + comm_.size() - rankOffset + 1) % comm_.size();
    for (IntType j = 0; j < gen.num_blocks(); ++j) {
      if (gen.get_mpi_rank(j) == comm_.rank()) {
        auto info = gen.get_block_info(j);
        requiredBufferSize += info.numCols * info.numRows;
        myBlockInfos_.emplace_back(originRank, info);
      }
    }
  }

  for (auto &proc : ringProcs_) {
    gpu::check_status(gpu::stream_synchronize(proc.blasHandle.stream_handle().get()));
  }

  resultBufferHost_->resize<T>(std::max<std::size_t>(requiredBufferSize, 1));

  resultRecvs_.resize(myBlockInfos_.size());

  if (useRing_) {
    // Post receive for final result of all blocks this rank requires
    IntType offset = 0;
    for (IntType i = 0; i < myBlockInfos_.size(); ++i) {
      const auto &pair = myBlockInfos_[i];
      MPI_Irecv(resultBufferHost_->data<T>() + offset, pair.second.numCols * pair.second.numRows,
                MPIMatchElementaryType<T>::get(), pair.first, resultTag, comm_.get(),
                resultRecvs_[i].get_and_activate());
      offset += pair.second.numCols * pair.second.numRows;
    }
  }

  // Start processing of first blocks
  if (ringProcs_.front().matA.rows() != 0) {
    // If ring is used, start with first actual block to be proccessed. If start index is greater
    // than number of blocks, the first block to process will always be 0
    const IntType myFirstBlockIdx = (!useRing_ || myStartIdx_ >= blocks_.size()) ? 0 : myStartIdx_;
    // Use offset if the first block being processed requires result from other rank before being send on
    const IntType procOffset =
        useRing_ && myStartIdx_ >= blocks_.size() ? ringProcs_.size() - 1 : 0;
    for (IntType i = 0; i < std::min<IntType>(ringProcs_.size(), blocks_.size()); ++i) {
      const IntType blockIdx = (myFirstBlockIdx + i) % blocks_.size();
      const auto &block = blocks_[blockIdx];
      auto &proc = ringProcs_[(i + procOffset) % ringProcs_.size()];
      ValueType beta = RealValueGPU<T>::create(0.0);

      auto opAGPU = opA_ == SplaOperation::SPLA_OP_TRANSPOSE
                        ? gpu::blas::operation::Transpose
                        : gpu::blas::operation::ConjugateTranspose;
      multiply_gpu<ValueType>(
          proc.blasHandle.get(), opAGPU, gpu::blas::operation::None, alpha_,
          proc.matA.sub_accessor(0, block.row, proc.matA.rows(), block.numRows),
          proc.matB.sub_accessor(0, block.col, proc.matB.rows(), block.numCols), beta,
          GPUArrayView2D<T>(proc.tileViewGPU.data(), block.numCols, block.numRows));

      // No result from other rank has to be added to first proc or for standard reduce case ->
      // start copying to host
      if ((i == 0 && myStartIdx_ < blocks_.size()) || !useRing_) {
        copy_from_gpu_async(
            proc.blasHandle.stream_handle().get(),
            GPUArrayConstView1D<T>(proc.tileViewGPU.data(), block.numCols * block.numRows),
            HostArrayView1D<T>(proc.tileViewHost.data(), block.numCols * block.numRows));
      }
      ++numMultipliedBlocks_;
    }
  } else {
    for (auto &proc : ringProcs_) {
      std::memset(proc.tileViewHost.data(), 0, maxBlockSize_ * sizeof(T));
    }
  }

  state_ = TileState::Prepared;
}

template <typename T, typename BLOCK_GEN>
auto RingGPU<T, BLOCK_GEN>::process_step_ring() -> void {
  const IntType numBlocks = blocks_.size();

  assert(ringProcs_.size() >= 2);

  IntType sendBlockIdx = (myStartIdx_ + stepIdx_) % comm_.size();
  IntType recvBlockIdx = (myStartIdx_ + stepIdx_ + 1) % comm_.size();
  if (stepIdx_ < comm_.size() - 1) {
    auto &proc = ringProcs_[procIdx_];
    auto &nextProc = ringProcs_[(procIdx_ + 1) % ringProcs_.size()];

    if(recvBlockIdx < numBlocks) {
      const auto &recvBlock = blocks_[recvBlockIdx];
      MPI_Irecv(nextProc.tileViewHost.data(), recvBlock.numCols * recvBlock.numRows,
                MPIMatchElementaryType<T>::get(), recvRank_, ringTag, comm_.get(),
                recvReq_.get_and_activate());
    }

    if (sendBlockIdx < numBlocks) {
      gpu::check_status(gpu::stream_synchronize(proc.blasHandle.stream_handle().get()));
      const auto &sendBlock = blocks_[sendBlockIdx];
      MPI_Send(proc.tileViewHost.data(), sendBlock.numRows * sendBlock.numCols,
               MPIMatchElementaryType<T>::get(), sendRank_, ringTag, comm_.get());
    }


    if(recvBlockIdx < numBlocks) {
      recvReq_.wait_if_active();
      const auto &recvBlock = blocks_[recvBlockIdx];

      if (nextProc.matA.rows() != 0) {
        copy_to_gpu_async(
            nextProc.recvStream.get(),
            HostArrayConstView1D<T>(nextProc.tileViewHost.data(),
                                    recvBlock.numCols * recvBlock.numRows),
            GPUArrayView1D<T>(nextProc.recvViewGPU.data(), recvBlock.numCols * recvBlock.numRows));
        nextProc.event.record(nextProc.recvStream.get());
        // make sure transfer of received result is done first to avoid scheduling
        // performance issues
        nextProc.event.stream_wait(nextProc.blasHandle.stream_handle().get());
        call_gpu_geam(nextProc.blasHandle.get(), gpu::blas::operation::None,
                      gpu::blas::operation::None, recvBlock.numRows, recvBlock.numCols,
                      RealValueGPU<T>::create(1.0), nextProc.recvViewGPU.data(), recvBlock.numRows,
                      RealValueGPU<T>::create(1.0), nextProc.tileViewGPU.data(), recvBlock.numRows,
                      nextProc.tileViewGPU.data(), recvBlock.numRows);
        copy_from_gpu_async(nextProc.blasHandle.stream_handle().get(),
                            GPUArrayConstView1D<T>(nextProc.tileViewGPU.data(),
                                                   recvBlock.numCols * recvBlock.numRows),
                            HostArrayView1D<T>(nextProc.tileViewHost.data(),
                                               recvBlock.numCols * recvBlock.numRows));
      }
    }

    if (sendBlockIdx < numBlocks) {
      if (proc.matA.rows() != 0 && numMultipliedBlocks_ < numBlocks) {
        const auto &block = blocks_[(sendBlockIdx + ringProcs_.size()) % blocks_.size()];
        const ValueType beta = RealValueGPU<T>::create(0.0);

        auto opAGPU = opA_ == SplaOperation::SPLA_OP_TRANSPOSE
                          ? gpu::blas::operation::Transpose
                          : gpu::blas::operation::ConjugateTranspose;
        // make sure transfer of received result is done first to avoid scheduling
        // performance issues
        nextProc.event.stream_wait(proc.blasHandle.stream_handle().get());
        // compute gemm
        multiply_gpu<ValueType>(
            proc.blasHandle.get(), opAGPU, gpu::blas::operation::None, alpha_,
            proc.matA.sub_accessor(0, block.row, proc.matA.rows(), block.numRows),
            proc.matB.sub_accessor(0, block.col, proc.matB.rows(), block.numCols), beta,
            GPUArrayView2D<T>(proc.tileViewGPU.data(), block.numCols, block.numRows));

        ++numMultipliedBlocks_;
      }
    }

    if(recvBlockIdx < numBlocks) {
      // Advance proc index
      procIdx_ = (procIdx_ + 1) % ringProcs_.size();
    }
  }

  if (stepIdx_ < comm_.size() - 1)
    state_ = TileState::PartiallyProcessed;
  else
    state_ = TileState::Processed;
}

template <typename T, typename BLOCK_GEN>
auto RingGPU<T, BLOCK_GEN>::process_step_reduction() -> void {
  const auto &block = blocks_[stepIdx_];
  auto &proc = ringProcs_[stepIdx_ % ringProcs_.size()];

  gpu::check_status(gpu::stream_synchronize(proc.blasHandle.stream_handle().get()));

  if (proc.matA.rows() == 0) {
    // If no local contribution, make sure to overwrite previous received results
    std::memset(proc.tileViewHost.data(), 0, block.numCols * block.numRows * sizeof(T));
  }
  mpi_check_status(MPI_Allreduce(MPI_IN_PLACE, proc.tileViewHost.data(),
                                 block.numCols * block.numRows,
                                 MPIMatchElementaryType<ValueType>::get(), MPI_SUM, comm_.get()));

  const bool resultOnHost = GPUMatC_.empty();

  if (resultOnHost) {
    using hostType = typename TypeTranslationHost<T>::type;
    auto matCHostConverted = HostArrayView2D<hostType>(
        reinterpret_cast<hostType *>(HostMatC_.data()), HostMatC_.dim_outer(),
        HostMatC_.dim_inner(), HostMatC_.ld_inner());

    auto betaHost = TypeTranslationHost<T>::convert(this->beta_);

    auto gen = baseMatGen_.create_sub_generator(block);

    HostArrayConstView2D<hostType> resultView(
        reinterpret_cast<hostType *>(proc.tileViewHost.data()), block.numCols, block.numRows);
    for (IntType i = 0; i < gen.num_blocks(); ++i) {
      const auto targetRank = gen.get_mpi_rank(i);
      if (targetRank == comm_.rank() || targetRank < 0) {
        const auto info = gen.get_block_info(i);

        add_kernel(info.numRows, info.numCols,
                   &resultView(info.globalSubColIdx, info.globalSubRowIdx), resultView.ld_inner(),
                   betaHost, &matCHostConverted(info.localColIdx, info.localRowIdx),
                   matCHostConverted.ld_inner());
      }
    }

  } else {
    // result should be placed in gpu memory

    auto gen = baseMatGen_.create_sub_generator(block);
    HostArrayConstView2D<T> resultViewHost(proc.tileViewHost.data(), block.numCols, block.numRows);
    GPUArrayView2D<T> resultView(proc.tileViewGPU.data(), block.numCols, block.numRows);
    bool resultCopied = false;
    for (IntType i = 0; i < gen.num_blocks(); ++i) {
      const auto targetRank = gen.get_mpi_rank(i);
      if (targetRank == comm_.rank() || targetRank < 0) {
        const auto info = gen.get_block_info(i);
        if (!resultCopied) {
          copy_to_gpu_async(proc.blasHandle.stream_handle().get(), resultViewHost, resultView);
          resultCopied = true;
        }
        T *subMatPtr = GPUMatC_.data() + GPUMatC_.index(info.localColIdx, info.localRowIdx);
        call_gpu_geam(
            proc.blasHandle.get(), gpu::blas::operation::None, gpu::blas::operation::None,
            info.numRows, info.numCols, RealValueGPU<T>::create(1.0),
            resultView.data() + resultView.index(info.globalSubColIdx, info.globalSubRowIdx),
            resultView.ld_inner(), beta_, subMatPtr, GPUMatC_.ld_inner(), subMatPtr,
            GPUMatC_.ld_inner());
      }
    }
  }

  if (proc.matA.rows() != 0 && stepIdx_ + ringProcs_.size() < blocks_.size()) {
    const auto &nextBlock = blocks_[stepIdx_ + ringProcs_.size()];
    auto opAGPU = opA_ == SplaOperation::SPLA_OP_TRANSPOSE
                      ? gpu::blas::operation::Transpose
                      : gpu::blas::operation::ConjugateTranspose;
    multiply_gpu<ValueType>(
        proc.blasHandle.get(), opAGPU, gpu::blas::operation::None, alpha_,
        proc.matA.sub_accessor(0, nextBlock.row, proc.matA.rows(), nextBlock.numRows),
        proc.matB.sub_accessor(0, nextBlock.col, proc.matB.rows(), nextBlock.numCols),
        RealValueGPU<T>::create(0.0),
        GPUArrayView2D<T>(proc.tileViewGPU.data(), nextBlock.numCols, nextBlock.numRows));
    copy_from_gpu_async(
        proc.blasHandle.stream_handle().get(),
        GPUArrayConstView1D<T>(proc.tileViewGPU.data(), nextBlock.numCols * nextBlock.numRows),
        HostArrayView1D<T>(proc.tileViewHost.data(), nextBlock.numCols * nextBlock.numRows));
  }

  if (stepIdx_ < blocks_.size() - 1)
    state_ = TileState::PartiallyProcessed;
  else
    state_ = TileState::Processed;
}

template <typename T, typename BLOCK_GEN>
auto RingGPU<T, BLOCK_GEN>::finalize() -> void {
  assert(state_ == TileState::Processed);

  // add tile to result as final step
  const bool resultOnHost = GPUMatC_.empty();
  const IntType numBlocks = blocks_.size();

  for (auto &b : ringProcs_) {
    gpu::check_status(gpu::stream_synchronize(b.blasHandle.stream_handle().get()));
  }

  if (useRing_) {
    const IntType lastRingBlockIdx = (myStartIdx_ + comm_.size() - 1) % comm_.size();

    // send final result to target rank
    if (lastRingBlockIdx < blocks_.size()) {
      const auto &block = blocks_[lastRingBlockIdx];
      auto &proc = ringProcs_[procIdx_];
      auto gen = baseMatGen_.create_sub_generator(block);
      HostArrayConstView2D<T> resultView(proc.tileViewHost.data(), block.numCols, block.numRows);

      for (IntType i = 0; i < gen.num_blocks(); ++i) {
        auto info = gen.get_block_info(i);
        auto datatType = MPIDatatypeHandle::create_vector(info.numCols, info.numRows, block.numRows,
                                                          MPIMatchElementaryType<T>::get());
        MPI_Send(&resultView(info.globalSubColIdx, info.globalSubRowIdx), 1, datatType.get(),
                 info.mpiRank, resultTag, comm_.get());
      }
    }

    if (!myBlockInfos_.empty()) {
      if (resultOnHost) {
        using hostType = typename TypeTranslationHost<T>::type;
        auto matCHostConverted = HostArrayView2D<hostType>(
            reinterpret_cast<hostType *>(HostMatC_.data()), HostMatC_.dim_outer(),
            HostMatC_.dim_inner(), HostMatC_.ld_inner());

        auto betaHost = TypeTranslationHost<T>::convert(this->beta_);

        IntType offset = 0;
        for (IntType i = 0; i < myBlockInfos_.size(); ++i) {
          resultRecvs_[i].wait_if_active();
          const auto &info = myBlockInfos_[i].second;

          add_kernel(info.numRows, info.numCols, resultBufferHost_->data<hostType>() + offset,
                     info.numRows, betaHost, &matCHostConverted(info.localColIdx, info.localRowIdx),
                     matCHostConverted.ld_inner());
          offset += info.numCols * info.numRows;
        }

      } else {
        // result should be placed in gpu memory

        IntType offset = 0;
        for (IntType i = 0; i < myBlockInfos_.size(); ++i) {
          resultRecvs_[i].wait_if_active();
          const auto &info = myBlockInfos_[i].second;

          copy_to_gpu_async<T, T>(
              ringProcs_.back().blasHandle.stream_handle().get(),
              HostArrayConstView1D<T>(resultBufferHost_->data<T>() + offset,
                                      info.numCols * info.numRows),
              GPUArrayView1D<T>(ringProcs_.back().tileViewGPU.data(), info.numCols * info.numRows));

          T *subMatPtr = GPUMatC_.data() + GPUMatC_.index(info.localColIdx, info.localRowIdx);
          call_gpu_geam(ringProcs_.back().blasHandle.get(), gpu::blas::operation::None,
                        gpu::blas::operation::None, info.numRows, info.numCols,
                        RealValueGPU<T>::create(1.0), ringProcs_.back().tileViewGPU.data(),
                        info.numRows, beta_, subMatPtr, GPUMatC_.ld_inner(), subMatPtr,
                        GPUMatC_.ld_inner());
          offset += info.numCols * info.numRows;
        }
      }
    }
  }

  state_ = TileState::Empty;
}

template <typename T, typename BLOCK_GEN>
auto RingGPU<T, BLOCK_GEN>::process_step() -> bool {
  const IntType numSteps = useRing_ ? comm_.size() : blocks_.size();

  if (stepIdx_ < numSteps) {
    if (useRing_) {
      this->process_step_ring();
    } else {
      this->process_step_reduction();
    }
  }

  ++stepIdx_;
  return stepIdx_ < numSteps;
}

template class RingGPU<double, BlockCyclicGenerator>;
template class RingGPU<float, BlockCyclicGenerator>;
template class RingGPU<gpu::blas::ComplexFloatType, BlockCyclicGenerator>;
template class RingGPU<gpu::blas::ComplexDoubleType, BlockCyclicGenerator>;

template class RingGPU<double, MirrorGenerator>;
template class RingGPU<float, MirrorGenerator>;
template class RingGPU<gpu::blas::ComplexFloatType, MirrorGenerator>;
template class RingGPU<gpu::blas::ComplexDoubleType, MirrorGenerator>;

}  // namespace spla
