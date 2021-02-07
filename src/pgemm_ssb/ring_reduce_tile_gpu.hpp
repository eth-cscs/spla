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
#ifndef SPLA_RING_REDUCE_TILE_GPU_HPP
#define SPLA_RING_REDUCE_TILE_GPU_HPP

#include <memory>
#include <array>
#include <vector>
#include "gpu_util/gpu_runtime_api.hpp"
#include "memory/host_array_const_view.hpp"
#include "memory/host_array_view.hpp"
#include "memory/gpu_array_view.hpp"
#include "memory/gpu_array_const_view.hpp"
#include "memory/buffer.hpp"
#include "memory/gpu_allocator.hpp"
#include "memory/pinned_allocator.hpp"
#include "mpi_util/mpi_communicator_handle.hpp"
#include "mpi_util/mpi_request_handle.hpp"
#include "mpi_util/mpi_window_handle.hpp"
#include "spla/config.h"
#include "spla/spla.hpp"
#include "util/common_types.hpp"
#include "block_generation/matrix_block_generator.hpp"
#include "gpu_util/gpu_matrix_accessor.hpp"
#include "gpu_util/gpu_blas_handle.hpp"
#include "gpu_util/gpu_event_handle.hpp"
#include "util/tile_state.hpp"

namespace spla {

template <typename T> struct RingProcessor {
  RingProcessor(IntType blockSize_, GPUBlasHandle blasHandle_,
            GPUEventHandle event_, GPUStreamHandle recvStream_,
            std::shared_ptr<Buffer<PinnedAllocator>> bufferHost_,
            std::shared_ptr<Buffer<GPUAllocator>> bufferGPU_,
            GPUMatrixAccessor<GPUArrayConstView2D<T>> matA_,
            GPUMatrixAccessor<GPUArrayConstView2D<T>> matB_)
      : blockSize(blockSize_), blasHandle(std::move(blasHandle_)),
        event(std::move(event_)), recvStream(std::move(recvStream_)),
        bufferHost(std::move(bufferHost_)), bufferGPU(std::move(bufferGPU_)),
        matA(std::move(matA_)), matB(std::move(matB_)) {
    assert(bufferHost);
    assert(bufferGPU);
    bufferHost->resize<T>(blockSize);
    bufferGPU->resize<T>(2 * blockSize);

    tileViewHost = HostArrayView1D<T>(bufferHost->data<T>(), blockSize);
    tileViewGPU = GPUArrayView1D<T>(bufferGPU->data<T>(), blockSize);
    recvViewGPU =
        GPUArrayView1D<T>(bufferGPU->data<T>() + blockSize, blockSize);
  }

  IntType blockSize;
  GPUBlasHandle blasHandle;
  GPUEventHandle event;
  GPUStreamHandle recvStream;
  std::shared_ptr<Buffer<PinnedAllocator>> bufferHost;
  std::shared_ptr<Buffer<GPUAllocator>> bufferGPU;
  GPUMatrixAccessor<GPUArrayConstView2D<T>> matA;
  GPUMatrixAccessor<GPUArrayConstView2D<T>> matB;
  HostArrayView1D<T> tileViewHost;
  GPUArrayView1D<T> tileViewGPU;
  GPUArrayView1D<T> recvViewGPU;
};

template <typename T, typename BLOCK_GEN>
class RingReduceTileGPU {
public:
  using ValueType = T;

  RingReduceTileGPU(IntType maxBlockSize, MPICommunicatorHandle comm,
                    std::vector<RingProcessor<T>> ringProcs,
                    std::shared_ptr<Buffer<PinnedAllocator>> resultBufferHost,
                    BLOCK_GEN baseMatGen, SplaOperation opA, ValueType alpha,
                    ValueType beta, HostArrayView2D<ValueType> HostMatC,
                    GPUArrayView2D<ValueType> GPUMatC);

  auto prepare(std::vector<BlockCoord>::const_iterator begin,
               std::vector<BlockCoord>::const_iterator end) -> void;

  auto process_step() -> bool;

  auto finalize() -> void;

  inline auto state() -> TileState { return state_; }

  inline auto synchronize() -> void {
    for(auto& b : ringProcs_) {
      gpu::check_status(
          gpu::stream_synchronize(b.blasHandle.stream_handle().get()));
    }
  }

private:

  auto process_step_ring() -> void;

  auto process_step_reduction() -> void;

  // state dependend
  bool useRing_ = false;
  IntType sendRank_ = 0;
  IntType recvRank_ = 0;
  IntType myStartIdx_ = 0;
  IntType currentBlockIdx = 0;
  MPIRequestHandle sendReq_;
  MPIRequestHandle recvReq_;
  std::vector<BlockCoord> blocks_;
  std::vector<std::pair<IntType, BlockInfo>> myBlockInfos_;
  std::vector<MPIRequestHandle> resultRecvs_;
  TileState state_;

  // fixed
  MPICommunicatorHandle comm_;
  BLOCK_GEN baseMatGen_;
  std::vector<RingProcessor<T>> ringProcs_;
  std::shared_ptr<Buffer<PinnedAllocator>> resultBufferHost_;
  HostArrayView2D<ValueType> HostMatC_;
  GPUArrayView2D<ValueType> GPUMatC_;
  const ValueType alpha_, beta_;
  const SplaOperation opA_;
  const IntType maxBlockSize_;
};

} // namespace spla
#endif
