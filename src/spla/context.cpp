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
#include "spla/context_internal.hpp"

namespace spla {
Context::Context(SplaProcessingUnit pu) : ctxInternal_(new ContextInternal(pu)) {}

SplaProcessingUnit Context::processing_unit() const { return ctxInternal_->processing_unit(); }

int Context::num_threads() const {
 return ctxInternal_->num_threads();
}

int Context::num_tiles_per_thread() const {
 return ctxInternal_->num_tiles_per_thread();
}

int Context::num_gpu_streams() const {
 return ctxInternal_->num_gpu_streams();
}

int Context::tile_length_target() const {
 return ctxInternal_->tile_length_target();
}

std::size_t Context::gpu_memory_limit() const {
 return ctxInternal_->gpu_memory_limit();
}

void Context::set_num_threads(int numThreads) {
 ctxInternal_->set_num_threads(numThreads);
}

void Context::set_num_tiles_per_thread(int numTilesPerThread) {
 ctxInternal_->set_num_tiles_per_thread(numTilesPerThread);
}

void Context::set_num_gpu_streams(int numGPUStreams) {
 ctxInternal_->set_num_gpu_streams(numGPUStreams);
}

void Context::set_tile_length_target(int tileLengthTarget) {
 ctxInternal_->set_tile_length_target(tileLengthTarget);
}

void Context::set_gpu_memory_limit(std::size_t gpuMemoryLimit) {
 ctxInternal_->set_gpu_memory_limit(gpuMemoryLimit);
}

}  // namespace spla
