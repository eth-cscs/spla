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
#ifndef SPLA_CONTEXT_INTERNAL_HPP
#define SPLA_CONTEXT_INTERNAL_HPP

#include "memory/buffer.hpp"
#include "memory/mpi_allocator.hpp"
#include "mpi_util/mpi_check_status.hpp"
#include "spla/config.h"
#include "spla/context.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <deque>
#include <memory>
#include <cstddef>
#include <mpi.h>
#include <type_traits>

#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
#include "gpu_util/gpu_blas_handle.hpp"
#include "memory/gpu_allocator.hpp"
#include "memory/pinned_allocator.hpp"
#endif

namespace spla {

class ContextInternal {
public:
  explicit ContextInternal(SplaProcessingUnit pu)
      : pu_(pu),
        numThreads_(omp_get_max_threads()),
        numTilesPerThread_(2),
        numGPUStreams_(4),
        tileLengthTarget_(pu == SplaProcessingUnit::SPLA_PU_HOST ? 256 : 256), gpuMemoryLimit_(1024 * 1024 * 1024) {}

  inline auto mpi_buffers(IntType numBuffers)
      -> std::deque<std::shared_ptr<Buffer<MPIAllocator>>> & {
    const IntType numMissing = numBuffers - static_cast<IntType>(mpiBuffers_.size());
    for(IntType i = 0; i < numMissing; ++i) {
      mpiBuffers_.emplace_back(new Buffer<MPIAllocator>());
    }
    return mpiBuffers_;
  }

#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
  inline auto pinned_buffers(IntType numBuffers)
      -> std::deque<std::shared_ptr<Buffer<PinnedAllocator>>> & {
    const IntType numMissing = numBuffers - static_cast<IntType>(pinnedBuffers_.size());
    for(IntType i = 0; i < numMissing; ++i) {
      pinnedBuffers_.emplace_back(new Buffer<PinnedAllocator>());
    }
    return pinnedBuffers_;
  }

  inline auto gpu_buffers(IntType numBuffers)
      -> std::deque<std::shared_ptr<Buffer<GPUAllocator>>> & {
    const IntType numMissing = numBuffers - static_cast<IntType>(gpuBuffers_.size());
    for(IntType i = 0; i < numMissing; ++i) {
      gpuBuffers_.emplace_back(new Buffer<GPUAllocator>());
    }
    return gpuBuffers_;
  }

  inline auto gpu_blas_handles(IntType numHandles) -> std::deque<GPUBlasHandle>& {
    const IntType numMissing = numHandles - static_cast<IntType>(gpuBlasHandles_.size());
    if(static_cast<IntType>(gpuBlasHandles_.size()) < numHandles) {
      gpuBlasHandles_.resize(numHandles);
    }
    return gpuBlasHandles_;
  }
#endif

  // Get methods

  inline auto processing_unit() const -> SplaProcessingUnit { return pu_; }

  inline auto num_threads() const -> IntType { return numThreads_; }

  inline auto num_tiles() const -> IntType { return numTilesPerThread_; }

  inline auto num_gpu_streams() const -> IntType { return numGPUStreams_; }

  inline auto tile_length_target() const -> IntType { return tileLengthTarget_; }

  inline auto gpu_memory_limit() const -> std::size_t { return gpuMemoryLimit_; }

  // Set methods

  inline auto set_num_threads(IntType numThreads) -> void {
#ifdef SPLA_OMP
    if (numThreads > 0) numThreads_ = numThreads;
    else numThreads_ = omp_get_max_threads();
#endif
  }

  inline auto set_num_tiles(IntType numTilesPerThread) -> void {
    numTilesPerThread_ = numTilesPerThread;
  }

  inline auto set_num_gpu_streams(IntType numGPUStreams) -> void {
    numGPUStreams_ = numGPUStreams;
  }

  inline auto set_tile_length_target(IntType tileLengthTarget) -> void {
    tileLengthTarget_ = tileLengthTarget;
  }

  inline auto set_gpu_memory_limit(std::size_t gpuMemoryLimit) -> void {
    gpuMemoryLimit_ = gpuMemoryLimit;
  }

private:
  SplaProcessingUnit pu_;
  IntType numThreads_;
  IntType numTilesPerThread_;
  IntType numGPUStreams_;
  IntType tileLengthTarget_;
  std::size_t gpuMemoryLimit_;

  std::deque<std::shared_ptr<Buffer<MPIAllocator>>> mpiBuffers_;
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
  std::deque<std::shared_ptr<Buffer<GPUAllocator>>> gpuBuffers_;
  std::deque<std::shared_ptr<Buffer<PinnedAllocator>>> pinnedBuffers_;
  std::deque<GPUBlasHandle> gpuBlasHandles_;
#endif
};
} // namespace spla

#endif
