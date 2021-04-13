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
#ifndef SPLA_RING_SBS_GPU_HPP
#define SPLA_RING_SBS_GPU_HPP

#include <array>
#include <memory>
#include <vector>
#include <deque>
#include <unordered_set>

#include "block_generation/block.hpp"
#include "gpu_util/gpu_blas_handle.hpp"
#include "gpu_util/gpu_event_handle.hpp"
#include "gpu_util/gpu_matrix_accessor.hpp"
#include "gpu_util/gpu_runtime_api.hpp"
#include "memory/buffer.hpp"
#include "memory/gpu_allocator.hpp"
#include "memory/gpu_array_const_view.hpp"
#include "memory/gpu_array_view.hpp"
#include "memory/host_array_const_view.hpp"
#include "memory/host_array_view.hpp"
#include "memory/pinned_allocator.hpp"
#include "mpi_util/mpi_communicator_handle.hpp"
#include "mpi_util/mpi_request_handle.hpp"
#include "mpi_util/mpi_window_handle.hpp"
#include "spla/config.h"
#include "spla/spla.hpp"
#include "util/common_types.hpp"
#include "util/tile_state.hpp"

namespace spla {

// Provides resources to proccess a block
template <typename T>
struct RingProcessorSBS {
  RingProcessorSBS(IntType blockSize_, GPUBlasHandle blasHandle_,
                   std::shared_ptr<Buffer<PinnedAllocator>> bufferHost_,
                   std::shared_ptr<Buffer<GPUAllocator>> bufferGPU_,
                   GPUMatrixAccessor<GPUArrayConstView2D<T>> matA_,
                   GPUMatrixAccessor<GPUArrayView2D<T>> matC_)
      : blockSize(blockSize_),
        blasHandle(std::move(blasHandle_)),
        bufferHost(std::move(bufferHost_)),
        bufferGPU(std::move(bufferGPU_)),
        matA(std::move(matA_)),
        matC(std::move(matC_)) {
    assert(bufferHost);
    assert(bufferGPU);
    bufferHost->resize<T>(2 * blockSize);
    bufferGPU->resize<T>(blockSize);

    tileViewGPU = GPUArrayView1D<T>(bufferGPU->data<T>(), blockSize);
    sendView = HostArrayView1D<T>(bufferHost->data<T>(), blockSize);
    recvView = HostArrayView1D<T>(bufferHost->data<T>() + blockSize, blockSize);
  }

  IntType blockSize;
  GPUBlasHandle blasHandle;
  std::shared_ptr<Buffer<PinnedAllocator>> bufferHost;
  std::shared_ptr<Buffer<GPUAllocator>> bufferGPU;
  GPUMatrixAccessor<GPUArrayConstView2D<T>> matA;
  GPUMatrixAccessor<GPUArrayView2D<T>> matC;
  HostArrayView1D<T> sendView;
  HostArrayView1D<T> recvView;
  GPUArrayView1D<T> tileViewGPU;
};

// Compute and reduce for pgemm_ssb. If number of input blocks is equal to comm size, a ring
// communication pattern is used. Otherwise, each block is processed individually.
template <typename T, typename BLOCK_GEN>
class RingSBSGPU {
public:
  using ValueType = T;

  RingSBSGPU(double ringThreshold, IntType maxBlockSize, MPICommunicatorHandle comm,
             std::vector<RingProcessorSBS<T>> ringProcs, BLOCK_GEN baseMatGen, ValueType alpha,
             ValueType beta, HostArrayConstView2D<ValueType> HostMatB,
             GPUArrayConstView2D<ValueType> GPUMatB, IntType bRowOffset, IntType bColOffset);

  // Prepare to process input blocks
  auto prepare(std::vector<Block>::const_iterator begin,
               std::vector<Block>::const_iterator end) -> void;

  // Do one step within ring, prcosseing blocks. Returns true if more steps required, false
  // otherwise.
  auto process_step(std::unordered_set<IntType>& betaColIndeces,
                    std::deque<GPUEventHandle>& colEvents) -> bool;

  // Must be called after all processing steps are done and before preparing for more blocks.
  auto finalize() -> void;

  inline auto state() -> TileState { return state_; }

  inline auto synchronize() -> void {
    for (auto& b : ringProcs_) {
      gpu::check_status(gpu::stream_synchronize(b.blasHandle.stream_handle().get()));
    }
  }

private:
  auto process_step_ring(std::unordered_set<IntType>& betaColIndeces,
                         std::deque<GPUEventHandle>& colEvents) -> void;

  auto process_step_broadcast(std::unordered_set<IntType>& betaColIndeces,
                              std::deque<GPUEventHandle>& colEvents) -> void;

  // state dependend
  bool useRing_ = false;
  IntType sendRank_ = 0;
  IntType recvRank_ = 0;
  IntType myStartIdx_ = 0;
  IntType rankOffset_ = 0;
  IntType stepIdx_ = 0;
  IntType procIdx_ = 0;
  IntType numMultipliedBlocks_ = 0;
  MPIRequestHandle sendReq_;
  MPIRequestHandle recvReq_;
  std::vector<Block> blocks_;
  std::vector<MPIRequestHandle> collectRecvs_;
  TileState state_;

  // fixed
  MPICommunicatorHandle comm_;
  BLOCK_GEN baseMatGen_;
  std::vector<RingProcessorSBS<T>> ringProcs_;
  HostArrayConstView2D<ValueType> HostMatB_;
  GPUArrayConstView2D<ValueType> GPUMatB_;
  const IntType bRowOffset_, bColOffset_;
  const ValueType alpha_, beta_;
  const IntType maxBlockSize_;
  const double ringThreshold_;
};

}  // namespace spla
#endif
