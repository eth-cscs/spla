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

#ifndef SPLA_GPU_BLAS_API_HPP
#define SPLA_GPU_BLAS_API_HPP

#include <stdexcept>
#include <utility>

#include "spla/config.h"

#if defined(SPLA_CUDA)
#include <cublas_v2.h>

#elif defined(SPLA_ROCM)
#include <rocblas.h>

#else
#error Either SPLA_CUDA or SPLA_ROCM must be defined!
#endif

#include "spla/exceptions.hpp"

namespace spla {
namespace gpu {
namespace blas {

#if defined(SPLA_CUDA)
using HandleType = cublasHandle_t;
using StatusType = cublasStatus_t;
using OperationType = cublasOperation_t;
using ComplexFloatType = cuComplex;
using ComplexDoubleType = cuDoubleComplex;
#endif

#if defined(SPLA_ROCM)
using HandleType = rocblas_handle;
using StatusType = rocblas_status;
using OperationType = rocblas_operation;
using ComplexFloatType = rocblas_float_complex;
using ComplexDoubleType = rocblas_double_complex;
#endif

namespace operation {
#if defined(SPLA_CUDA)
constexpr auto None = CUBLAS_OP_N;
constexpr auto Transpose = CUBLAS_OP_T;
constexpr auto ConjugateTranspose = CUBLAS_OP_C;
#endif

#if defined(SPLA_ROCM)
constexpr auto None = rocblas_operation_none;
constexpr auto Transpose = rocblas_operation_transpose;
constexpr auto ConjugateTranspose = rocblas_operation_conjugate_transpose;
#endif
}  // namespace operation

namespace status {
#if defined(SPLA_CUDA)
constexpr auto Success = CUBLAS_STATUS_SUCCESS;
#endif

#if defined(SPLA_ROCM)
constexpr auto Success = rocblas_status_success;
#endif

static const char* get_string(StatusType error) {
#if defined(SPLA_CUDA)
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
#endif

#if defined(SPLA_ROCM)
  switch (error) {
    case rocblas_status_success:
      return "rocblas_status_success";

    case rocblas_status_invalid_handle:
      return "rocblas_status_invalid_handle";

    case rocblas_status_not_implemented:
      return "rocblas_status_not_implemented";

    case rocblas_status_invalid_pointer:
      return "rocblas_status_invalid_pointer";

    case rocblas_status_invalid_size:
      return "rocblas_status_invalid_size";

    case rocblas_status_memory_error:
      return "rocblas_status_memory_error";

    case rocblas_status_internal_error:
      return "rocblas_status_internal_error";

    case rocblas_status_perf_degraded:
      return "rocblas_status_perf_degraded";

    case rocblas_status_size_query_mismatch:
      return "rocblas_status_size_query_mismatch";

    case rocblas_status_size_increased:
      return "rocblas_status_size_increased";

    case rocblas_status_size_unchanged:
      return "rocblas_status_size_unchanged";
  }
#endif

  return "<unknown>";
}
}  // namespace status

inline auto check_status(StatusType error) -> void {
  if (error != status::Success) {
    throw GPUBlasError();
  }
}

// =======================================
// Forwarding functions of to GPU BLAS API
// =======================================
template <typename... ARGS>
inline auto create(ARGS&&... args) -> StatusType {
#if defined(SPLA_CUDA)
  return cublasCreate(std::forward<ARGS>(args)...);
#else
  return rocblas_create_handle(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
inline auto destroy(ARGS&&... args) -> StatusType {
#if defined(SPLA_CUDA)
  return cublasDestroy(std::forward<ARGS>(args)...);
#else
  return rocblas_destroy_handle(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
inline auto set_stream(ARGS&&... args) -> StatusType {
#if defined(SPLA_CUDA)
  return cublasSetStream(std::forward<ARGS>(args)...);
#else
  return rocblas_set_stream(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
inline auto get_stream(ARGS&&... args) -> StatusType {
#if defined(SPLA_CUDA)
  return cublasGetStream(std::forward<ARGS>(args)...);
#else
  return rocblas_get_stream(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
inline auto sgemm(ARGS&&... args) -> StatusType {
#if defined(SPLA_CUDA)
  return cublasSgemm(std::forward<ARGS>(args)...);
#else
  return rocblas_sgemm(std::forward<ARGS>(args)...);
#endif  // SPLA_CUDA
}

template <typename... ARGS>
inline auto dgemm(ARGS&&... args) -> StatusType {
#if defined(SPLA_CUDA)
  return cublasDgemm(std::forward<ARGS>(args)...);
#else
  return rocblas_dgemm(std::forward<ARGS>(args)...);
#endif  // SPLA_CUDA
}

template <typename... ARGS>
inline auto cgemm(ARGS&&... args) -> StatusType {
#if defined(SPLA_CUDA)
  return cublasCgemm(std::forward<ARGS>(args)...);
#else
  return rocblas_cgemm(std::forward<ARGS>(args)...);
#endif  // SPLA_CUDA
}

template <typename... ARGS>
inline auto zgemm(ARGS&&... args) -> StatusType {
#if defined(SPLA_CUDA)
  return cublasZgemm(std::forward<ARGS>(args)...);
#else
  return rocblas_zgemm(std::forward<ARGS>(args)...);
#endif  // SPLA_CUDA
}

template <typename... ARGS>
inline auto sgeam(ARGS&&... args) -> StatusType {
#if defined(SPLA_CUDA)
  return cublasSgeam(std::forward<ARGS>(args)...);
#else
  return rocblas_sgeam(std::forward<ARGS>(args)...);
#endif  // SPLA_CUDA
}

template <typename... ARGS>
inline auto dgeam(ARGS&&... args) -> StatusType {
#if defined(SPLA_CUDA)
  return cublasDgeam(std::forward<ARGS>(args)...);
#else
  return rocblas_dgeam(std::forward<ARGS>(args)...);
#endif  // SPLA_CUDA
}

template <typename... ARGS>
inline auto cgeam(ARGS&&... args) -> StatusType {
#if defined(SPLA_CUDA)
  return cublasCgeam(std::forward<ARGS>(args)...);
#else
  return rocblas_cgeam(std::forward<ARGS>(args)...);
#endif  // SPLA_CUDA
}

template <typename... ARGS>
inline auto zgeam(ARGS&&... args) -> StatusType {
#if defined(SPLA_CUDA)
  return cublasZgeam(std::forward<ARGS>(args)...);
#else
  return rocblas_zgeam(std::forward<ARGS>(args)...);
#endif  // SPLA_CUDA
}

}  // namespace blas
}  // namespace gpu
}  // namespace spla

#endif
