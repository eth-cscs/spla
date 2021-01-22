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
#ifndef SPLA_RING_REDUCE_TILE_HOST_HPP
#define SPLA_RING_REDUCE_TILE_HOST_HPP

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
#include "mpi_util/mpi_window_handle.hpp"
#include "spla/config.h"
#include "spla/spla.hpp"
#include "spla/types.h"
#include "util/common_types.hpp"
#include "util/tile_state.hpp"

namespace spla {
template <typename T>
class RingReduceTileHost {
public:
  using ValueType = T;

  RingReduceTileHost(IntType numThreads, MPICommunicatorHandle comm,
           std::shared_ptr<Buffer<MPIAllocator>> buffer,
           std::shared_ptr<MatrixBlockGenerator> matrixDist, SplaOperation opA,
           ValueType alpha, const HostArrayConstView2D<ValueType> &A,
           const HostArrayConstView2D<ValueType> &B, ValueType beta,
           HostArrayView2D<ValueType> C);

  auto prepare(IntType blockRowIdx, IntType blockColIdx, IntType numBlockRows,
               IntType numBlockCols) -> void;

  auto process_step() -> bool;

protected:
  // state dependend
  IntType numBlockRows_ = 0;
  IntType numBlockCols_ = 0;
  IntType myBlockIdx_ = -1;
  IntType currentBlockIdx = 0;
  MPIRequestHandle sendReq_;
  MPIRequestHandle recvReq_;
  std::vector<BlockInfo> blockInfos_;

  // fixed
  HostArrayView2D<ValueType> recvView_;
  HostArrayView2D<ValueType> sendView_;
  HostArrayView2D<ValueType> resultView_;
  std::shared_ptr<MatrixBlockGenerator> matrixDist_;
  std::shared_ptr<Buffer<MPIAllocator>> buffer_;
  MPICommunicatorHandle comm_;
  HostArrayConstView2D<ValueType> A_;
  HostArrayConstView2D<ValueType> B_;
  HostArrayView2D<ValueType> C_;
  const ValueType alpha_, beta_;
  const SplaOperation opA_;
  const IntType numThreads_;
};

}  // namespace spla
#endif
