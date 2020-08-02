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
#ifndef SPLA_PGEMM_SSB_HPP
#define SPLA_PGEMM_SSB_HPP

/*! \file pgemm_ssb.hpp
   \brief General matrix multiplication functions for computing \f$ C \leftarrow \alpha A^H B +
   \beta C \f$ with stripe-stripe-block distribution.
   \verbatim
     ------ T     ------
     |    |       |    |
     |    |       |    |
     ------       ------
     |    |       |    |        -------         -------
     |    |       |    |        |  |  |         |  |  |
     ------   *   ------    +   -------   -->   -------
     |    |       |    |        |  |  |         |  |  |
     |    |       |    |        -------         -------
     ------       ------           C               C
     |    |       |    |
     |    |       |    |
     ------       ------
       A            B
    \endverbatim
*/

#include <complex>

#include "spla/config.h"
#include "spla/context.hpp"
#include "spla/matrix_distribution.hpp"

/*! \cond PRIVATE */
namespace spla {
/*! \endcond */
/**
 * Computes a distributed general matrix multiplication of the form \f$ C \leftarrow \alpha A^H B +
 * \beta C \f$ in single precision. \f$A\f$ and \f$B\f$ are only split along the row dimension
 * (stripes), while \f$C\f$ can be distributed as any supported MatrixDistribution type.
 *
 * @param[in] m Number of rows of \f$A^H\f$
 * @param[in] n Number of columns of \f$B\f$
 * @param[in] kLocal Number rows of \f$B\f$ and number of columns of \f$A^H\f$ stored at calling MPI
 * rank. This number may differ for each rank.
 * @param[in] opA Operation applied when reading matrix A. Must be SPLA_OP_TRANSPOSE or
 * SPLA_OP_CONJ_TRANSPOSE.
 * @param[in] alpha Scaling of multiplication of \f$A^H\f$ and \f$B\f$
 * @param[in] A Pointer to matrix \f$A\f$.
 * @param[in] lda Leading dimension of \f$A\f$ with lda \f$\geq\f$ kLocal.
 * @param[in] B Pointer to matrix \f$B\f$.
 * @param[in] ldb Leading dimension of \f$B\f$ with ldb \f$\geq\f$ kLocal.
 * @param[in] beta Scaling of \f$C\f$ before summation.
 * @param[out] C Pointer to global matrix \f$C\f$.
 * @param[in] ldc Leading dimension of \f$C\f$ with ldc \f$\geq\f$ loc(m), where loc(m) is the
 * number of locally stored rows of \f$C\f$.
 * @param[in] cRowOffset Row offset in the global matrix \f$C\f$, identifying the first row of the
 * submatrix \f$C\f$.
 * @param[in] cColOffset Column offset in the global matrix \f$C\f$, identifying the first coloumn
 * of the submatrix \f$C\f$.
 * @param[in] distC Matrix distribution of global matrix \f$C\f$.
 * @param[in] ctx Context, which provides configuration settings and reusable resources.
 */
SPLA_EXPORT void pgemm_ssb(int m, int n, int kLocal, SplaOperation opA, float alpha, const float *A,
                           int lda, const float *B, int ldb, float beta, float *C, int ldc,
                           int cRowOffset, int cColOffset, MatrixDistribution &distC, Context &ctx);

/**
 * Computes a distributed general matrix multiplication of the form \f$ C \leftarrow \alpha A^H B +
 * \beta C \f$ in double precision. See documentation above.
 */
SPLA_EXPORT void pgemm_ssb(int m, int n, int kLocal, SplaOperation opA, double alpha,
                           const double *A, int lda, const double *B, int ldb, double beta,
                           double *C, int ldc, int cRowOffset, int cColOffset,
                           MatrixDistribution &distC, Context &ctx);

/**
 * Computes a distributed general matrix multiplication of the form \f$ C \leftarrow \alpha A^H B +
 * \beta C \f$ in single precision for complex types. See documentation above.
 */
SPLA_EXPORT void pgemm_ssb(int m, int n, int kLocal, SplaOperation opA, std::complex<float> alpha,
                           const std::complex<float> *A, int lda, const std::complex<float> *B,
                           int ldb, std::complex<float> beta, std::complex<float> *C, int ldc,
                           int cRowOffset, int cColOffset, MatrixDistribution &distC, Context &ctx);

/**
 * Computes a distributed general matrix multiplication of the form \f$ C \leftarrow \alpha A^H B +
 * \beta C \f$ in double precision for complex types. See documentation above.
 */
SPLA_EXPORT void pgemm_ssb(int m, int n, int kLocal, SplaOperation opA, std::complex<double> alpha,
                           const std::complex<double> *A, int lda, const std::complex<double> *B,
                           int ldb, std::complex<double> beta, std::complex<double> *C, int ldc,
                           int cRowOffset, int cColOffset, MatrixDistribution &distC, Context &ctx);

#ifndef SPLA_DOXYGEN_SKIP
}  // namespace spla
#endif

#endif
