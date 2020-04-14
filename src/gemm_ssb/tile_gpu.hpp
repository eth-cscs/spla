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
#ifndef SPLA_TILE_GPU_HPP
#define SPLA_TILE_GPU_HPP

#include <memory>
#include "memory/host_array_const_view.hpp"
#include "memory/host_array_view.hpp"
#include "memory/gpu_array_view.hpp"
#include "memory/gpu_array_const_view.hpp"
#include "memory/buffer.hpp"
#include "memory/gpu_allocator.hpp"
#include "memory/pinned_allocator.hpp"
#include "mpi_util/mpi_communicator_handle.hpp"
#include "mpi_util/mpi_request_handle.hpp"
#include "spla/config.h"
#include "spla/spla.hpp"
#include "util/common_types.hpp"
#include "block_generation/matrix_block_generator.hpp"
#include "util/tile_state.hpp"
#include "gpu_util/gpu_matrix_accessor.hpp"
#include "gpu_util/gpu_blas_handle.hpp"

namespace spla {
template <typename T>
class TileGPU {
public:
  using ValueType = T;

  TileGPU(MPICommunicatorHandle comm, GPUBlasHandle blasHandle,
          std::shared_ptr<Buffer<PinnedAllocator>> bufferHost,
          std::shared_ptr<Buffer<GPUAllocator>> bufferGPU,
          std::shared_ptr<MatrixBlockGenerator> matrixDist, ValueType alpha,
          GPUMatrixAccessor<GPUArrayConstView2D<ValueType>> matA,
          GPUMatrixAccessor<GPUArrayConstView2D<ValueType>> matB, ValueType beta,
          HostArrayView2D<ValueType> HostMatC, GPUArrayView2D<ValueType> GPUMatC,
          IntType numBlockRows, IntType numBlockCols);

  auto multiply(IntType blockRowIdx, IntType blockColIdx) -> void;

  auto exchange() -> void;

  auto extract() -> void;

  inline auto state() -> TileState { return state_.get(); }

  auto synchronize() -> void {
    gpu::check_status(gpu::stream_synchronize(blasHandle_.stream_handle().get()));
  }

protected:
  // state dependent
  AtomicTileState state_;
  HostArrayView2D<ValueType> tileHost_;
  GPUArrayView2D<ValueType> tileGPU_;
  std::vector<BlockInfo> blockInfos_;

  // fixed
  MPICommunicatorHandle comm_;
  std::shared_ptr<MatrixBlockGenerator> matrixDist_;
  std::shared_ptr<Buffer<PinnedAllocator>> bufferHost_;
  std::shared_ptr<Buffer<GPUAllocator>> bufferGPU_;
  GPUBlasHandle blasHandle_;
  const IntType numBlockRows_, numBlockCols_;
  GPUMatrixAccessor<GPUArrayConstView2D<ValueType>> matA_;
  GPUMatrixAccessor<GPUArrayConstView2D<ValueType>> matB_;
  HostArrayView2D<ValueType> HostMatC_;
  GPUArrayView2D<ValueType> GPUMatC_;
  const ValueType alpha_, beta_;
};

}  // namespace spla
#endif
