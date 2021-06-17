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
#ifndef SPLA_ERRORS_H
#define SPLA_ERRORS_H

#include "spla/config.h"

enum SplaError {
  /**
   * Success. No error.
   */
  SPLA_SUCCESS,
  /**
   * Unknown error.
   */
  SPLA_UNKNOWN_ERROR,
  /**
   * Internal error.
   */
  SPLA_INTERNAL_ERROR,
  /**
   * Invalid parameter error.
   */
  SPLA_INVALID_PARAMETER_ERROR,
  /**
   * Invalid pointer error.
   */
  SPLA_INVALID_POINTER_ERROR,
  /**
   * Invalid handle error.
   */
  SPLA_INVALID_HANDLE_ERROR,
  /**
   * MPI error.
   */
  SPLA_MPI_ERROR,
  /**
   * MPI allocation error.
   */
  SPLA_MPI_ALLOCATION_ERROR,
  /**
   * MPI thread suppport error.
   */
  SPLA_MPI_THREAD_SUPPORT_ERROR,
  /**
   * GPU error.
   */
  SPLA_GPU_ERROR,
  /**
   * GPU support error.
   */
  SPLA_GPU_SUPPORT_ERROR,
  /**
   * GPU allocation error.
   */
  SPLA_GPU_ALLOCATION_ERROR,
  /**
   * GPU launch error.
   */
  SPLA_GPU_LAUNCH_ERROR,
  /**
   * GPU no device error.
   */
  SPLA_GPU_NO_DEVICE_ERROR,
  /**
   * GPU invalid value error.
   */
  SPLA_GPU_INVALID_VALUE_ERROR,
  /**
   * Invalid device pointer error.
   */
  SPLA_GPU_INVALID_DEVICE_POINTER_ERROR,
  /**
   * GPU blas error.
   */
  SPLA_GPU_BLAS_ERROR,
  /**
   * Invalid allocator function error.
   */
  SPLA_INVALID_ALLOCATOR_FUNCTION
};

#ifndef __cplusplus
/*! \cond PRIVATE */
// C only
typedef enum SplaError SplaError;
/*! \endcond */
#endif  // cpp

#endif

