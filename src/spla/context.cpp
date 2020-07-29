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
#include "spla/context.h"
#include "spla/context_internal.hpp"
#include "spla/errors.h"
#include "spla/exceptions.hpp"

namespace spla {
Context::Context(SplaProcessingUnit pu) : ctxInternal_(new ContextInternal(pu)) {}

SplaProcessingUnit Context::processing_unit() const { return ctxInternal_->processing_unit(); }

int Context::num_threads() const { return ctxInternal_->num_threads(); }

int Context::num_tiles() const { return ctxInternal_->num_tiles(); }

int Context::tile_length_target() const { return ctxInternal_->tile_length_target(); }

std::size_t Context::gpu_memory_limit() const { return ctxInternal_->gpu_memory_limit(); }

int Context::gpu_device_id() const { return ctxInternal_->gpu_device_id(); }

void Context::set_num_threads(int numThreads) { ctxInternal_->set_num_threads(numThreads); }

void Context::set_num_tiles(int numTilesPerThread) {
  ctxInternal_->set_num_tiles(numTilesPerThread);
}

void Context::set_tile_length_target(int tileLengthTarget) {
  ctxInternal_->set_tile_length_target(tileLengthTarget);
}

void Context::set_gpu_memory_limit(std::size_t gpuMemoryLimit) {
  ctxInternal_->set_gpu_memory_limit(gpuMemoryLimit);
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

SplaError spla_ctx_num_threads(SplaContext ctx, int* numThreads) {
  if (!ctx) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    *numThreads = reinterpret_cast<spla::Context*>(ctx)->num_threads();
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
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

SplaError spla_ctx_tile_length_target(SplaContext ctx, int* tileLengthTarget) {
  if (!ctx) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    *tileLengthTarget = reinterpret_cast<spla::Context*>(ctx)->tile_length_target();
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_ctx_gpu_memory_limit(SplaContext ctx, size_t* limit) {
  if (!ctx) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    *limit = reinterpret_cast<spla::Context*>(ctx)->gpu_memory_limit();
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

SplaError spla_ctx_set_num_threads(SplaContext ctx, int numThreads) {
  if (!ctx) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spla::Context*>(ctx)->set_num_threads(numThreads);
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

SplaError spla_ctx_set_tile_length_target(SplaContext ctx, int tileLengthTarget) {
  if (!ctx) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spla::Context*>(ctx)->set_tile_length_target(tileLengthTarget);
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}

SplaError spla_ctx_set_gpu_memory_limit(SplaContext ctx, size_t gpuMemoryLimit) {
  if (!ctx) {
    return SplaError::SPLA_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spla::Context*>(ctx)->set_gpu_memory_limit(gpuMemoryLimit);
  } catch (const spla::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SplaError::SPLA_UNKNOWN_ERROR;
  }
  return SplaError::SPLA_SUCCESS;
}
}

