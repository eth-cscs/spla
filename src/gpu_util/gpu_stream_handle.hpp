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
#ifndef SPLA_GPU_STREAM_HANDLE_HPP
#define SPLA_GPU_STREAM_HANDLE_HPP

#include "spla/config.h"
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
#include <memory>
#include <tuple>

#include "gpu_util/gpu_runtime_api.hpp"
#include "spla/exceptions.hpp"
namespace spla {
class GPUStreamHandle {
public:
  GPUStreamHandle() : GPUStreamHandle(false) {}

  explicit GPUStreamHandle(const bool blockedByDefaultStream) : deviceId_(0) {
    gpu::check_status(gpu::get_device(&deviceId_));
    gpu::StreamType rawStream;
    if (blockedByDefaultStream)
      gpu::check_status(gpu::stream_create_with_flags(&rawStream, gpu::flag::StreamDefault));
    else
      gpu::check_status(gpu::stream_create_with_flags(&rawStream, gpu::flag::StreamNonBlocking));

    stream_ =
        std::shared_ptr<gpu::StreamType>(new gpu::StreamType(rawStream), [](gpu::StreamType* ptr) {
          std::ignore = gpu::stream_destroy(*ptr);
          delete ptr;
        });
  };

  inline auto get() const -> gpu::StreamType { return *stream_; }

  inline auto device_id() const noexcept -> int { return deviceId_; }

private:
  std::shared_ptr<gpu::StreamType> stream_;
  int deviceId_ = 0;
};
}  // namespace spla

#endif
#endif
