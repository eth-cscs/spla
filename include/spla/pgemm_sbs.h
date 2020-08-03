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
#ifndef SPLA_PGEMM_SBS_H
#define SPLA_PGEMM_SBS_H
/*! \file pgemm_sbs.h
   \brief General matrix multiplication functions for computing \f$ C \leftarrow \alpha A B +
   \beta C \f$ with stripe-block-stipe distribution.
   \verbatim
     ------                     ------         ------
     |    |                     |    |         |    |
     |    |                     |    |         |    |
     ------                     ------         ------
     |    |       -------       |    |         |    |
     |    |       |  |  |       |    |         |    |
     ------   *   -------   +   ------   -->   ------
     |    |       |  |  |       |    |         |    |
     |    |       -------       |    |         |    |
     ------          B          ------         ------
     |    |                     |    |         |    |
     |    |                     |    |         |    |
     ------                     ------         ------
       A                          C              C
    \endverbatim
*/

#include "spla/config.h"
#include "spla/context.h"
#include "spla/errors.h"
#include "spla/matrix_distribution.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Computes a distributed general matrix multiplication of the form \f$ C \leftarrow \alpha A B +
 * \beta C \f$ in single precision. \f$A\f$ and \f$C\f$ are only split along the row dimension
 * (stripes), while \f$B\f$ can be distributed as any supported MatrixDistribution type.
 *
 * @param[in] mLocal Number rows of \f$A\f$ and \f$C\f$ stored at calling MPI rank. This number may
 * differ for each rank.
 * @param[in] n Number of columns of \f$B\f$.
 * @param[in] k Number of columns of \f$C\f$ and rows of \f$B\f$.
 * @param[in] alpha Scaling of multiplication of \f$A^H\f$ and \f$B\f$
 * @param[in] A Pointer to matrix \f$A\f$.
 * @param[in] lda Leading dimension of \f$A\f$ with lda \f$\geq\f$ kLocal.
 * @param[in] B Pointer to matrix \f$B\f$.
 * @param[in] ldb Leading dimension of \f$B\f$ with ldb \f$\geq\f$ loc(k), where loc(k) is the
 * number of locally stored rows of \f$B\f$.
 * @param[in] bRowOffset Row offset in the global matrix \f$B\f$, identifying the first row of the
 * submatrix \f$B\f$.
 * @param[in] bColOffset Column offset in the global matrix \f$B\f$, identifying the first coloumn
 * of the submatrix \f$B\f$.
 * @param[in] distB Matrix distribution of global matrix \f$B\f$.
 * @param[in] beta Scaling of \f$C\f$ before summation.
 * @param[out] C Pointer to matrix \f$C\f$.
 * @param[in] ldc Leading dimension of \f$C\f$ with ldC \f$\geq\f$ mLocal.
 * @param[in] ctx Context, which provides configuration settings and reusable resources.
 */
SPLA_EXPORT SplaError spla_psgemm_sbs(int mLocal, int n, int k, float alpha, const float *A,
                                      int lda, const float *B, int ldb, int bRowOffset,
                                      int bColOffset, SplaMatrixDistribution distB, float beta,
                                      float *C, int ldc, SplaContext ctx);

/**
 * Computes a distributed general matrix multiplication of the form \f$ C \leftarrow \alpha A B +
 * \beta C \f$ in double precision. See documentation above.
 */
SPLA_EXPORT SplaError spla_pdgemm_sbs(int mLocal, int n, int k, double alpha, const double *A,
                                      int lda, const double *B, int ldb, int bRowOffset,
                                      int bColOffset, SplaMatrixDistribution distB, double beta,
                                      double *C, int ldc, SplaContext ctx);

/**
 * Computes a distributed general matrix multiplication of the form \f$ C \leftarrow \alpha A B +
 * \beta C \f$ in double precision. See documentation above.
 */
SPLA_EXPORT SplaError spla_pcgemm_sbs(int mLocal, int n, int k, const void *alpha, const void *A,
                                      int lda, const void *B, int ldb, int bRowOffset,
                                      int bColOffset, SplaMatrixDistribution distB,
                                      const void *beta, void *C, int ldc, SplaContext ctx);

/**
 * Computes a distributed general matrix multiplication of the form \f$ C \leftarrow \alpha A B +
 * \beta C \f$ in double precision. See documentation above.
 */
SPLA_EXPORT SplaError spla_pzgemm_sbs(int mLocal, int n, int k, const void *alpha, const void *A,
                                      int lda, const void *B, int ldb, int bRowOffset,
                                      int bColOffset, SplaMatrixDistribution distB,
                                      const void *beta, void *C, int ldc, SplaContext ctx);
#ifdef __cplusplus
}
#endif

#endif
