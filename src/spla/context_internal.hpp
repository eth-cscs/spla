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

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <deque>
#include <memory>
#include <type_traits>

#include "memory/buffer.hpp"
#include "memory/mpi_allocator.hpp"
#include "mpi_util/mpi_check_status.hpp"
#include "spla/config.h"
#include "spla/context.hpp"
#include "spla/exceptions.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"

#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
#include "gpu_util/gpu_blas_handle.hpp"
#include "gpu_util/gpu_event_handle.hpp"
#include "gpu_util/gpu_stream_handle.hpp"
#include "memory/gpu_allocator.hpp"
#include "memory/pinned_allocator.hpp"
#endif

namespace spla {

class ContextInternal {
public:
  explicit ContextInternal(SplaProcessingUnit pu)
      : pu_(pu),
        numThreads_(omp_get_max_threads()),
        numTiles_(4),
        tileSizeHost_(pu == SplaProcessingUnit::SPLA_PU_HOST ? 256 : 1024),
        tileSizeGPU_(2048),
        opThresholdGPU_(2000000),
        gpuDeviceId_(0) {
    if (pu == SplaProcessingUnit::SPLA_PU_GPU) {
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
      gpu::check_status(gpu::get_device(&gpuDeviceId_));
#else
      throw GPUSupportError();
#endif
    } else if (pu != SplaProcessingUnit::SPLA_PU_HOST) {
      throw InvalidParameterError();
    }
  }

  inline auto mpi_buffers(IntType numBuffers)
      -> std::deque<std::shared_ptr<Buffer<MPIAllocator>>>& {
    const IntType numMissing = numBuffers - static_cast<IntType>(mpiBuffers_.size());
    for (IntType i = 0; i < numMissing; ++i) {
      mpiBuffers_.emplace_back(new Buffer<MPIAllocator>());
    }
    return mpiBuffers_;
  }

#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
  inline auto pinned_buffers(IntType numBuffers)
      -> std::deque<std::shared_ptr<Buffer<PinnedAllocator>>>& {
    const IntType numMissing = numBuffers - static_cast<IntType>(pinnedBuffers_.size());
    for (IntType i = 0; i < numMissing; ++i) {
      pinnedBuffers_.emplace_back(new Buffer<PinnedAllocator>());
    }
    return pinnedBuffers_;
  }

  inline auto gpu_buffers(IntType numBuffers)
      -> std::deque<std::shared_ptr<Buffer<GPUAllocator>>>& {
    const IntType numMissing = numBuffers - static_cast<IntType>(gpuBuffers_.size());
    for (IntType i = 0; i < numMissing; ++i) {
      gpuBuffers_.emplace_back(new Buffer<GPUAllocator>());
    }
    return gpuBuffers_;
  }

  inline auto gpu_blas_handles(IntType numHandles) -> std::deque<GPUBlasHandle>& {
    const IntType numMissing = numHandles - static_cast<IntType>(gpuBlasHandles_.size());
    if (static_cast<IntType>(gpuBlasHandles_.size()) < numHandles) {
      gpuBlasHandles_.resize(numHandles);
    }
    return gpuBlasHandles_;
  }

  inline auto gpu_event_handles(IntType numHandles) -> std::deque<GPUEventHandle>& {
    const IntType numMissing = numHandles - static_cast<IntType>(gpuEventHandles_.size());
    if (static_cast<IntType>(gpuEventHandles_.size()) < numHandles) {
      gpuEventHandles_.resize(numHandles);
    }
    return gpuEventHandles_;
  }

  inline auto gpu_stream_handles(IntType numHandles) -> std::deque<GPUStreamHandle>& {
    const IntType numMissing = numHandles - static_cast<IntType>(gpuStreamHandles_.size());
    if (static_cast<IntType>(gpuStreamHandles_.size()) < numHandles) {
      gpuStreamHandles_.resize(numHandles);
    }
    return gpuStreamHandles_;
  }
#endif

  // Get methods

  inline auto processing_unit() const -> SplaProcessingUnit { return pu_; }

  inline auto num_threads() const -> IntType { return numThreads_; }

  inline auto num_tiles() const -> IntType { return numTiles_; }

  inline auto tile_size_host() const -> IntType { return tileSizeHost_; }

  inline auto tile_size_gpu() const -> IntType { return tileSizeGPU_; }

  inline auto op_threshold_gpu() const -> IntType { return opThresholdGPU_; }

  inline auto gpu_device_id() const -> int { return gpuDeviceId_; }

  // Set methods

  inline auto set_num_threads(IntType numThreads) -> void {
#ifdef SPLA_OMP
    if (numThreads > 0)
      numThreads_ = numThreads;
    else
      numThreads_ = omp_get_max_threads();
#endif
  }

  inline auto set_num_tiles(IntType numTilesPerThread) -> void {
    if (numTilesPerThread < 1) throw InvalidParameterError();
    numTiles_ = numTilesPerThread;
  }

  inline auto set_tile_size_host(IntType tileSizeHost) -> void {
    if (tileSizeHost < 1) throw InvalidParameterError();
    tileSizeHost_ = tileSizeHost;
  }

  inline auto set_tile_size_gpu(IntType tileSizeGPU) -> void {
    if (tileSizeGPU < 1) throw InvalidParameterError();
    tileSizeGPU_ = tileSizeGPU;
  }

  inline auto set_op_threshold_gpu(IntType opThresholdGPU) -> void {
    if (opThresholdGPU < 0) throw InvalidParameterError();
    opThresholdGPU_ = opThresholdGPU;
  }

private:
  SplaProcessingUnit pu_;
  IntType numThreads_;
  IntType numTiles_;
  IntType tileSizeHost_;
  IntType tileSizeGPU_;
  IntType opThresholdGPU_;
  int gpuDeviceId_;

  std::deque<std::shared_ptr<Buffer<MPIAllocator>>> mpiBuffers_;
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
  std::deque<std::shared_ptr<Buffer<GPUAllocator>>> gpuBuffers_;
  std::deque<std::shared_ptr<Buffer<PinnedAllocator>>> pinnedBuffers_;
  std::deque<GPUBlasHandle> gpuBlasHandles_;
  std::deque<GPUEventHandle> gpuEventHandles_;
  std::deque<GPUStreamHandle> gpuStreamHandles_;
#endif
};
}  // namespace spla

#endif
