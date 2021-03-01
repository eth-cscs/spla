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
#ifndef SPLA_GEMM_HPP
#define SPLA_GEMM_HPP

/*! \file gemm.hpp
   \brief General matrix multiplication functions for locally computing \f$ C
   \leftarrow \alpha OP(A) OP(B) + \beta C \f$.
*/

#include <complex>

#include "spla/config.h"
#include "spla/context.hpp"

/*! \cond PRIVATE */
namespace spla {
/*! \endcond */
/**
 * Computes a local general matrix multiplication of the form \f$ C
 * \leftarrow \alpha op(A) op(B) + \beta C \f$ in single precision. If context
 * with processing unit set to GPU, pointers to matrices can be any combination
 * of host and device pointers.
 *
 * @param[in] opA Operation applied when reading matrix \f$A\f$.
 * @param[in] opB Operation applied when reading matrix \f$B\f$.
 * @param[in] m Number of rows of \f$OP(A)\f$
 * @param[in] n Number of columns of \f$OP(B)\f$
 * @param[in] k Number rows of \f$OP(B)\f$ and number of columns of \f$OP(A)\f$
 * @param[in] alpha Scaling of multiplication of \f$A^H\f$ and \f$B\f$
 * @param[in] A Pointer to matrix \f$A\f$.
 * @param[in] lda Leading dimension of \f$A\f$.
 * @param[in] B Pointer to matrix \f$B\f$.
 * @param[in] ldb Leading dimension of \f$B\f$.
 * @param[in] beta Scaling of \f$C\f$ before summation.
 * @param[out] C Pointer to matrix \f$C\f$.
 * @param[in] ldc Leading dimension of \f$C\f$.
 * @param[in] ctx Context, which provides configuration settings and reusable
 * resources.
 */
SPLA_EXPORT void gemm(SplaOperation opA, SplaOperation opB, int m, int n, int k, float alpha,
                      const float *A, int lda, const float *B, int ldb, float beta, float *C,
                      int ldc, Context &ctx);

/**
 * Computes a local general matrix multiplication of the form \f$ C
 * \leftarrow \alpha OP(A) OP(B) + \beta C \f$ in double precision. See
 * documentation above.
 */
SPLA_EXPORT void gemm(SplaOperation opA, SplaOperation opB, int m, int n, int k, double alpha,
                      const double *A, int lda, const double *B, int ldb, double beta, double *C,
                      int ldc, Context &ctx);

/**
 * Computes a local general matrix multiplication of the form \f$ C
 * \leftarrow \alpha OP(A) OP(B) + \beta C \f$ in single precision complex
 * types. See documentation above.
 */
SPLA_EXPORT void gemm(SplaOperation opA, SplaOperation opB, int m, int n, int k,
                      std::complex<float> alpha, const std::complex<float> *A, int lda,
                      const std::complex<float> *B, int ldb, std::complex<float> beta,
                      std::complex<float> *C, int ldc, Context &ctx);

/**
 * Computes a local general matrix multiplication of the form \f$ C
 * \leftarrow \alpha OP(A) OP(B) + \beta C \f$ in double precision complex
 * types. See documentation above.
 */
SPLA_EXPORT void gemm(SplaOperation opA, SplaOperation opB, int m, int n, int k,
                      std::complex<double> alpha, const std::complex<double> *A, int lda,
                      const std::complex<double> *B, int ldb, std::complex<double> beta,
                      std::complex<double> *C, int ldc, Context &ctx);

/*! \cond PRIVATE */
}  // namespace spla
/*! \endcond */

#endif
