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
#ifndef SPLA_RING_HOST_HPP
#define SPLA_RING_HOST_HPP

#include <atomic>
#include <memory>
#include <utility>
#include <vector>

#include "block_generation/block.hpp"
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

// Compute and reduce for pgemm_ssb. If number of input blocks is equal to comm size, a ring
// communication pattern is used. Otherwise, each block is processed individually.
template <typename T, typename BLOCK_GEN>
class RingHost {
public:
  using ValueType = T;

  RingHost(double ringThreshold, IntType maxBlockSize, IntType numThreads,
                     MPICommunicatorHandle comm, std::shared_ptr<Buffer<MPIAllocator>> buffer,
                     std::shared_ptr<Buffer<MPIAllocator>> resultBuffer, BLOCK_GEN baseMatGen,
                     SplaOperation opA, ValueType alpha, const HostArrayConstView2D<ValueType> &A,
                     const HostArrayConstView2D<ValueType> &B, ValueType beta,
                     HostArrayView2D<ValueType> C);

  // Prepare to process input blocks
  auto prepare(std::vector<Block>::const_iterator begin,
               std::vector<Block>::const_iterator end) -> void;

  // Do one step within ring, prcosseing blocks. Returns true if more steps required, false
  // otherwise.
  auto process_step() -> bool;

  // Must be called after all processing steps are done and before preparing for more blocks.
  inline auto state() -> TileState { return state_; }

private:
  auto process_step_ring() -> void;

  auto process_step_reduction() -> void;

  auto process_step_ring_finalize() -> void;

  auto process_step_reduction_finalize() -> void;

  // state dependend
  bool useRing_ = false;
  IntType sendRank_ = 0;
  IntType recvRank_ = 0;
  IntType myStartIdx_ = 0;
  IntType stepIdx_ = 0;
  MPIRequestHandle sendReq_;
  MPIRequestHandle recvReq_;
  std::vector<Block> blocks_;
  std::vector<std::pair<IntType, BlockInfo>> myBlockInfos_;
  std::vector<MPIRequestHandle> resultRecvs_;
  TileState state_;

  // fixed
  HostArrayView1D<ValueType> recvView_;
  HostArrayView1D<ValueType> sendView_;
  BLOCK_GEN baseMatGen_;
  std::shared_ptr<Buffer<MPIAllocator>> buffer_;
  std::shared_ptr<Buffer<MPIAllocator>> resultBuffer_;
  MPICommunicatorHandle comm_;
  HostArrayConstView2D<ValueType> A_;
  HostArrayConstView2D<ValueType> B_;
  HostArrayView2D<ValueType> C_;
  const ValueType alpha_, beta_;
  const SplaOperation opA_;
  const IntType numThreads_;
  const IntType maxBlockSize_;
  const double ringThreshold_;
};

}  // namespace spla
#endif
