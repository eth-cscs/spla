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
#ifndef SPLA_GPU_BLAS_HANDLE_HPP
#define SPLA_GPU_BLAS_HANDLE_HPP

#include "spla/config.h"
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
#include <memory>
#include <cassert>
#include "gpu_util/gpu_blas_api.hpp"
#include "gpu_util/gpu_runtime_api.hpp"
#include "gpu_util/gpu_stream_handle.hpp"
#include "spla/exceptions.hpp"
namespace spla {
class GPUBlasHandle {
public:
  explicit GPUBlasHandle(GPUStreamHandle stream): stream_(std::move(stream)) {
    gpu::check_status(gpu::get_device(&deviceId_));
    gpu::blas::HandleType rawHandle;
    gpu::blas::check_status(gpu::blas::create(&rawHandle));

    handle_ =
        std::shared_ptr<gpu::blas::HandleType>(new gpu::blas::HandleType(rawHandle), [](gpu::blas::HandleType* ptr) {
          gpu::blas::destroy(*ptr);
          delete ptr;
        });

    gpu::blas::set_stream(*handle_, stream_.get());

  }

  GPUBlasHandle() : GPUBlasHandle(GPUStreamHandle(false)) {}

  inline auto get() const -> gpu::blas::HandleType {
    assert(handle_);
    return *handle_;
  }

  inline auto device_id() const noexcept -> int { return deviceId_; }

  inline auto stream_handle() const noexcept -> const GPUStreamHandle& { return stream_; }

private:
  GPUStreamHandle stream_;
  std::shared_ptr<gpu::blas::HandleType> handle_;
  int deviceId_ = 0;
};
}  // namespace spla

#endif
#endif
