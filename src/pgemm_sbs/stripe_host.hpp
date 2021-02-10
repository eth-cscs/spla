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
#ifndef SPLA_STRIPE_HOST_HPP
#define SPLA_STRIPE_HOST_HPP

#include <atomic>
#include <memory>
#include <vector>

#include "block_generation/block.hpp"
#include "memory/buffer.hpp"
#include "memory/host_array_const_view.hpp"
#include "memory/host_array_view.hpp"
#include "memory/mpi_allocator.hpp"
#include "mpi_util/mpi_communicator_handle.hpp"
#include "mpi_util/mpi_request_handle.hpp"
#include "spla/config.h"
#include "spla/spla.hpp"
#include "util/common_types.hpp"
#include "util/stripe_state.hpp"

namespace spla {
template <typename T, typename BLOCK_GEN>
class StripeHost {
public:
  using ValueType = T;

  StripeHost(IntType numThreads, MPICommunicatorHandle comm,
             std::shared_ptr<Buffer<MPIAllocator>> buffer,
             std::shared_ptr<Buffer<MPIAllocator>> recvBuffer, BLOCK_GEN baseMatGen,
             ValueType alpha, const HostArrayConstView2D<ValueType> &A,
             const HostArrayConstView2D<ValueType> &B, ValueType beta, HostArrayView2D<ValueType> C,
             IntType numBlockCols);

  // Assemble send buffer for MPI exchange.
  auto collect(IntType blockColIdx) -> void;

  // Start exchange data with MPI.
  auto start_exchange() -> void;

  // Finalize exchange data with MPI.
  auto finalize_exchange() -> void;

  // Multiply stripe.
  auto multiply() -> void;

  inline auto state() -> StripeState { return state_.get(); }

protected:
  // state dependent
  AtomicStripeState state_;
  HostArrayView2D<ValueType> tile_;
  std::vector<BlockInfo> blockInfos_;
  std::vector<int> localCounts_;
  std::vector<int> recvDispls_;
  std::vector<IntType> localRows_;        // number of rows of B each rank has stored
  std::vector<IntType> localCols_;        // number of cols of B each rank has stored
  std::vector<IntType> localRowOffsets_;  // Row offset of sub-matrix of B on each rank
  std::vector<IntType> localColOffsets_;  // Col offset of sub-matrix of B on each rank
  MPIRequestHandle mpiRequest_;

  // fixed
  BLOCK_GEN baseMatGen_;
  std::shared_ptr<Buffer<MPIAllocator>> buffer_;
  std::shared_ptr<Buffer<MPIAllocator>> recvBuffer_;
  MPICommunicatorHandle comm_;
  const IntType numBlockCols_;
  HostArrayConstView2D<ValueType> A_;
  HostArrayConstView2D<ValueType> B_;
  HostArrayView2D<ValueType> C_;
  const ValueType alpha_, beta_;
  const IntType numThreads_;
};

}  // namespace spla
#endif
