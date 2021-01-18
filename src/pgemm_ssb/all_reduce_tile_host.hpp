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
#ifndef SPLA_ALL_REDUCE_TILE_HOST_HPP
#define SPLA_ALL_REDUCE_TILE_HOST_HPP

#include <atomic>
#include <memory>
#include <vector>
#include "block_generation/matrix_block_generator.hpp"
#include "memory/buffer.hpp"
#include "memory/host_array_const_view.hpp"
#include "memory/host_array_view.hpp"
#include "memory/mpi_allocator.hpp"
#include "mpi_util/mpi_communicator_handle.hpp"
#include "mpi_util/mpi_request_handle.hpp"
#include "spla/config.h"
#include "spla/spla.hpp"
#include "spla/types.h"
#include "util/common_types.hpp"
#include "util/tile_state.hpp"

namespace spla {
template <typename T>
class AllReduceTileHost {
public:
  using ValueType = T;

  AllReduceTileHost(IntType numThreads, MPICommunicatorHandle comm,
           std::shared_ptr<Buffer<MPIAllocator>> buffer,
           std::shared_ptr<MatrixBlockGenerator> matrixDist, SplaOperation opA,
           ValueType alpha, const HostArrayConstView2D<ValueType> &A,
           const HostArrayConstView2D<ValueType> &B, ValueType beta,
           HostArrayView2D<ValueType> C, IntType numBlockRows,
           IntType numBlockCols);

  // Multiply tile starting from given indices.
  auto multiply(IntType blockRowIdx, IntType blockColIdx) -> void;

  // Start exchange data with MPI.
  auto start_exchange() -> void;

  // Finalize exchange data with MPI.
  auto finalize_exchange() -> void;

  // Add tile to C.
  auto extract() -> void;

  inline auto state() -> TileState { return state_.get(); }

protected:
  // state dependent
  AtomicTileState state_;
  HostArrayView2D<ValueType> tile_;
  std::vector<BlockInfo> blockInfos_;
  MPIRequestHandle mpiRequest_;

  // fixed
  std::shared_ptr<MatrixBlockGenerator> matrixDist_;
  std::shared_ptr<Buffer<MPIAllocator>> buffer_;
  MPICommunicatorHandle comm_;
  const IntType numBlockRows_, numBlockCols_;
  HostArrayConstView2D<ValueType> A_;
  HostArrayConstView2D<ValueType> B_;
  HostArrayView2D<ValueType> C_;
  const ValueType alpha_, beta_;
  const SplaOperation opA_;
  const IntType numThreads_;
};

}  // namespace spla
#endif
