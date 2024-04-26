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
#include "spla/context.hpp"
#include <memory>
#include <cstddef>

#include "spla/context.h"
#include "spla/context_internal.hpp"
#include "spla/errors.h"
#include "spla/exceptions.hpp"
#include "memory/simple_allocator.hpp"

namespace spla {
Context::Context(SplaProcessingUnit pu) : ctxInternal_(new ContextInternal(pu)) {}

SplaProcessingUnit Context::processing_unit() const { return ctxInternal_->processing_unit(); }

int Context::num_tiles() const { return ctxInternal_->num_tiles(); }

int Context::tile_size_host() const { return ctxInternal_->tile_size_host(); }

int Context::tile_size_gpu() const { return ctxInternal_->tile_size_gpu(); }

int Context::op_threshold_gpu() const { return ctxInternal_->op_threshold_gpu(); }

int Context::gpu_device_id() const { return ctxInternal_->gpu_device_id(); }

std::uint_least64_t Context::allocated_memory_host() const {
  return ctxInternal_->allocators().host()->size();
}

std::uint_least64_t Context::allocated_memory_pinned() const {
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
  return ctxInternal_->allocators().pinned()->size();
#else
  return 0;
#endif
}

std::uint_least64_t Context::allocated_memory_gpu() const {
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
  return ctxInternal_->allocators().gpu()->size();
#else
  return 0;
#endif
}

void Context::set_num_tiles(int numTiles) {
  ctxInternal_->set_num_tiles(numTiles);
}

void Context::set_tile_size_host(int tileSizeHost) {
  ctxInternal_->set_tile_size_host(tileSizeHost);
}

void Context::set_tile_size_gpu(int tileSizeGPU) { ctxInternal_->set_tile_size_gpu(tileSizeGPU); }

void Context::set_op_threshold_gpu(int opThresholdGPU) {
  ctxInternal_->set_op_threshold_gpu(opThresholdGPU);
}

void Context::set_alloc_host(std::function<void*(std::size_t)> allocateFunc,
                             std::function<void(void*)> deallocateFunc) {
  if(!allocateFunc || !deallocateFunc) {
    throw InvalidAllocatorFunctionError();
  }
  ctxInternal_->allocators().set_host(std::shared_ptr<Allocator<MemLoc::Host>>(
      new SimpleAllocator<MemLoc::Host>(allocateFunc, deallocateFunc)));
}

void Context::set_alloc_pinned(std::function<void*(std::size_t)> allocateFunc,
                               std::function<void(void*)> deallocateFunc) {
  if(!allocateFunc || !deallocateFunc) {
    throw InvalidAllocatorFunctionError();
  }
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
  ctxInternal_->allocators().set_pinned(std::shared_ptr<Allocator<MemLoc::Host>>(
      new SimpleAllocator<MemLoc::Host>(allocateFunc, deallocateFunc)));
#endif
}

void Context::set_alloc_gpu(std::function<void*(std::size_t)> allocateFunc,
                            std::function<void(void*)> deallocateFunc) {
  if(!allocateFunc || !deallocateFunc) {
    throw InvalidAllocatorFunctionError();
  }
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
  ctxInternal_->allocators().set_gpu(std::shared_ptr<Allocator<MemLoc::GPU>>(
      new SimpleAllocator<MemLoc::GPU>(allocateFunc, deallocateFunc)));
#endif
}

}  // namespace spla

extern "C" {

SplaError spla_ctx_create(SplaContext* ctx, SplaProcessingUnit pu) {
  try {
    *ctx = reinterpret_cast<void*>(new spla::Context(pu));
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_ctx_destroy(SplaContext* ctx) {
  if (!ctx) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    delete reinterpret_cast<spla::Context*>(*ctx);
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  *ctx = nullptr;
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_ctx_num_tiles(SplaContext ctx, int* numTiles) {
  if (!ctx) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    *numTiles = reinterpret_cast<spla::Context*>(ctx)->num_tiles();
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_ctx_tile_size_host(SplaContext ctx, int* tileSizeHost) {
  if (!ctx) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    *tileSizeHost = reinterpret_cast<spla::Context*>(ctx)->tile_size_host();
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_ctx_tile_size_gpu(SplaContext ctx, int* tileSizeGPU) {
  if (!ctx) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    *tileSizeGPU = reinterpret_cast<spla::Context*>(ctx)->tile_size_gpu();
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_ctx_op_threshold_gpu(SplaContext ctx, int* opThresholdGPU) {
  if (!ctx) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    *opThresholdGPU = reinterpret_cast<spla::Context*>(ctx)->op_threshold_gpu();
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_ctx_gpu_device_id(SplaContext ctx, int* deviceId) {
  if (!ctx) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    *deviceId = reinterpret_cast<spla::Context*>(ctx)->gpu_device_id();
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_ctx_allocated_memory_host(SplaContext ctx, uint_least64_t* size) {
  if (!ctx) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    *size = reinterpret_cast<spla::Context*>(ctx)->allocated_memory_host();
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_ctx_allocated_memory_pinned(SplaContext ctx, uint_least64_t* size) {
  if (!ctx) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    *size = reinterpret_cast<spla::Context*>(ctx)->allocated_memory_host();
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_ctx_allocated_memory_gpu(SplaContext ctx, uint_least64_t* size) {
  if (!ctx) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    *size = reinterpret_cast<spla::Context*>(ctx)->allocated_memory_host();
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_ctx_set_num_tiles(SplaContext ctx, int numTiles) {
  if (!ctx) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spla::Context*>(ctx)->set_num_tiles(numTiles);
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_ctx_set_tile_size_host(SplaContext ctx, int tileSizeHost) {
  if (!ctx) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spla::Context*>(ctx)->set_tile_size_host(tileSizeHost);
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_ctx_set_tile_size_gpu(SplaContext ctx, int tileSizeGPU) {
  if (!ctx) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spla::Context*>(ctx)->set_tile_size_gpu(tileSizeGPU);
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_ctx_set_op_threshold_gpu(SplaContext ctx, int opThresholdGPU) {
  if (!ctx) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spla::Context*>(ctx)->set_op_threshold_gpu(opThresholdGPU);
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_ctx_set_alloc_host(SplaContext ctx, void* (*allocateFunc)(std::size_t),
                                  void (*deallocateFunc)(void*)) {
  if (!ctx) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spla::Context*>(ctx)->set_alloc_host(
        std::function<void*(std::size_t)>(allocateFunc),
        std::function<void(void*)>(deallocateFunc));
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_ctx_set_alloc_pinned(SplaContext ctx, void* (*allocateFunc)(std::size_t),
                                    void (*deallocateFunc)(void*)) {
  if (!ctx) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spla::Context*>(ctx)->set_alloc_pinned(
        std::function<void*(std::size_t)>(allocateFunc),
        std::function<void(void*)>(deallocateFunc));
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_ctx_set_alloc_gpu(SplaContext ctx, void* (*allocateFunc)(std::size_t),
                                 void (*deallocateFunc)(void*)) {
  if (!ctx) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spla::Context*>(ctx)->set_alloc_gpu(
        std::function<void*(std::size_t)>(allocateFunc),
        std::function<void(void*)>(deallocateFunc));
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}
}
