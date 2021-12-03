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

#ifndef SPLA_ALLOCATOR_COLLECTION_HPP
#define SPLA_ALLOCATOR_COLLECTION_HPP

#include <mpi.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <memory>

#include "memory/allocator.hpp"
#include "memory/pool_allocator.hpp"
#include "spla/config.h"
#include "spla/exceptions.hpp"

#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
#include "gpu_util/gpu_runtime_api.hpp"
#endif

namespace spla {
class AllocatorCollection {
public:
  AllocatorCollection()
      : allocHost_(new PoolAllocator<MemLoc::Host>(
            [](std::size_t size) -> void* {
              void* ptr = nullptr;
              if (MPI_Alloc_mem(size, MPI_INFO_NULL, &ptr) != MPI_SUCCESS) throw MPIAllocError();
              return ptr;
            },
            [](void* ptr) -> void {
              int finalized = 0;
              MPI_Finalized(&finalized);
              if (!finalized) MPI_Free_mem(ptr);
            }))
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
        ,
        allocPinned_(new PoolAllocator<MemLoc::Host>(
            [](std::size_t size) -> void* {
              void* ptr = nullptr;
              gpu::check_status(gpu::malloc_host(&ptr, size));
              return ptr;
            },
            [](void* ptr) -> void { gpu::free_host(ptr); })),
        allocGPU_(new PoolAllocator<MemLoc::GPU>(
            [](std::size_t size) -> void* {
              void* ptr = nullptr;
              gpu::check_status(gpu::malloc(&ptr, size));
              return ptr;
            },
            [](void* ptr) -> void { gpu::free(ptr); }))
#endif
  {
    int mpiInitialized = 0;
    MPI_Initialized(&mpiInitialized);

    // if MPI has not been initialized, use malloc as fallback
    if(!mpiInitialized) {
      allocHost_.reset(new PoolAllocator<MemLoc::Host>(std::malloc, std::free));
    }
  }

  auto host() const -> const std::shared_ptr<Allocator<MemLoc::Host>>& { return allocHost_; }

  auto set_host(std::shared_ptr<Allocator<MemLoc::Host>> allocator) -> void {
    assert(allocator);
    allocHost_ = std::move(allocator);
  }

#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
  auto pinned() const -> const std::shared_ptr<Allocator<MemLoc::Host>>& { return allocPinned_; }

  auto set_pinned(std::shared_ptr<Allocator<MemLoc::Host>> allocator) -> void {
    assert(allocator);
    allocPinned_ = std::move(allocator);
  }

  auto gpu() const -> const std::shared_ptr<Allocator<MemLoc::GPU>>& { return allocGPU_; }

  auto set_gpu(std::shared_ptr<Allocator<MemLoc::GPU>> allocator) -> void {
    assert(allocator);
    allocGPU_ = std::move(allocator);
  }
#endif

private:
    std::shared_ptr<Allocator<MemLoc::Host>> allocHost_;
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
    std::shared_ptr<Allocator<MemLoc::Host>> allocPinned_;
    std::shared_ptr<Allocator<MemLoc::GPU>> allocGPU_;
#endif

};
}  // namespace spla

#endif
