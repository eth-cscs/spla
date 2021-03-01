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
#ifndef SPLA_STRIPE_GPU_HPP
#define SPLA_STRIPE_GPU_HPP

#include <atomic>
#include <memory>
#include <vector>

#include "block_generation/block.hpp"
#include "gpu_util/gpu_blas_handle.hpp"
#include "gpu_util/gpu_matrix_accessor.hpp"
#include "memory/buffer.hpp"
#include "memory/gpu_allocator.hpp"
#include "memory/host_array_const_view.hpp"
#include "memory/host_array_view.hpp"
#include "memory/pinned_allocator.hpp"
#include "mpi_util/mpi_communicator_handle.hpp"
#include "mpi_util/mpi_request_handle.hpp"
#include "spla/config.h"
#include "spla/spla.hpp"
#include "util/common_types.hpp"
#include "util/stripe_state.hpp"

namespace spla {
template <typename T, typename BLOCK_GEN>
class StripeGPU {
public:
  using ValueType = T;

  StripeGPU(MPICommunicatorHandle comm, GPUBlasHandle blasHandle,
            std::shared_ptr<Buffer<PinnedAllocator>> buffer,
            std::shared_ptr<Buffer<PinnedAllocator>> recvBuffer,
            std::shared_ptr<Buffer<GPUAllocator>> bufferGPU, IntType maxGPUStripeSize,
            BLOCK_GEN baseMatGen, ValueType alpha, GPUMatrixAccessor<GPUArrayConstView2D<T>> A,
            HostArrayConstView2D<ValueType> matBViewHost,
            GPUArrayConstView2D<ValueType> matBViewGPU, ValueType beta,
            GPUMatrixAccessor<GPUArrayView2D<T>> C, HostArrayView2D<T> viewCHost,
            IntType numBlockCols);

  auto collect(IntType blockColIdx) -> void;

  auto start_exchange() -> void;

  auto finalize_exchange() -> void;

  auto multiply() -> void;

  inline auto state() -> StripeState { return state_.get(); }

  inline auto exchange_is_ready_and_active() -> bool { return request_.is_ready_and_active(); }

  inline auto synchronize() -> void {
    gpu::check_status(gpu::stream_synchronize(blasHandle_.stream_handle().get()));
  }

protected:
  // state dependent
  AtomicStripeState state_;
  HostArrayView2D<ValueType> tile_;
  MPIRequestHandle request_;
  std::vector<BlockInfo> blockInfos_;
  std::vector<int> localCounts_;
  std::vector<int> recvDispls_;
  std::vector<IntType> localRows_;        // number of rows of B each rank has stored
  std::vector<IntType> localCols_;        // number of cols of B each rank has stored
  std::vector<IntType> localRowOffsets_;  // Row offset of sub-matrix of B on each rank
  std::vector<IntType> localColOffsets_;  // Col offset of sub-matrix of B on each rank

  // fixed
  BLOCK_GEN baseMatGen_;
  std::shared_ptr<Buffer<PinnedAllocator>> buffer_;
  std::shared_ptr<Buffer<PinnedAllocator>> recvBuffer_;
  std::shared_ptr<Buffer<GPUAllocator>> bufferGPU_;
  MPICommunicatorHandle comm_;
  GPUBlasHandle blasHandle_;
  const IntType maxGPUStripeSize_;
  const IntType numBlockCols_;
  GPUMatrixAccessor<GPUArrayConstView2D<T>> matA_;
  GPUMatrixAccessor<GPUArrayView2D<T>> matC_;
  HostArrayView2D<T> viewCHost_;
  HostArrayConstView2D<ValueType> matBViewHost_;
  GPUArrayConstView2D<ValueType> matBViewGPU_;
  const ValueType alpha_, beta_;
};

}  // namespace spla
#endif
