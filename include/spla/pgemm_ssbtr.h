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
#ifndef SPLA_PGEMM_SSBTR_H
#define SPLA_PGEMM_SSBTR_H

/*! \file pgemm_ssb.h
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

#include "spla/config.h"
#include "spla/context.h"
#include "spla/errors.h"
#include "spla/context.h"
#include "spla/matrix_distribution.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Computes a distributed general matrix multiplication of the form \f$ C \leftarrow \alpha A^H B +
 * \beta C \f$ in single precision. \f$A\f$ and \f$B\f$ are only split along the row dimension
 * (stripes), while \f$C\f$ can be distributed as any supported SplaMatrixDistribution type. The
 * fill mode of \f$C\f$ indicates the part of the matrix which must be computed, while any other
 * part may or may not be computed. It is therefore not a strict limitation. For example, given
 * SPLA_FILL_MODE_UPPER, a small matrix may still be fully computed, while a large matrix will be
 * computed block wise, such that the computed blocks cover the upper triangle. The fill mode is
 * always in refenrence to the full matrix, so offsets are taken into account.
 *
 * @param[in] m Number of rows of \f$A^H\f$
 * @param[in] n Number of columns of \f$B\f$
 * @param[in] kLocal Number rows of \f$B\f$ and number of columns of \f$A^H\f$ stored at calling MPI
 * rank. This number may differ for each rank.
 * @param[in] opA Operation applied when reading matrix A. Must be SPLA_OP_TRANSPOSE or
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
 * @param[in] cFillMode Fill mode of matrix C.
 * @param[in] distC Matrix distribution of global matrix \f$C\f$.
 * @param[in] ctx Context, which provides configuration settings and reusable resources.
 */
SPLA_EXPORT SplaError spla_psgemm_ssbtr(int m, int n, int kLocal, SplaOperation opA, float alpha,
                                        const float *A, int lda, const float *B, int ldb,
                                        float beta, float *C, int ldc, int cRowOffset,
                                        int cColOffset, SplaFillMode cFillMode,
                                        SplaMatrixDistribution distC, SplaContext ctx);

/**
 * Computes a distributed general matrix multiplication of the form \f$ C \leftarrow \alpha A^H B +
 * \beta C \f$ in double precision. See documentation above.
 */
SPLA_EXPORT SplaError spla_pdgemm_ssbtr(int m, int n, int kLocal, SplaOperation opA, double alpha,
                                        const double *A, int lda, const double *B, int ldb,
                                        double beta, double *C, int ldc, int cRowOffset,
                                        int cColOffset, SplaFillMode cFillMode,
                                        SplaMatrixDistribution distC, SplaContext ctx);

/**
 * Computes a distributed general matrix multiplication of the form \f$ C \leftarrow \alpha A^H B +
 * \beta C \f$ in single precision for complex types. See documentation above.
 */
SPLA_EXPORT SplaError spla_pcgemm_ssbtr(int m, int n, int kLocal, SplaOperation opA,
                                        const void *alpha, const void *A, int lda, const void *B,
                                        int ldb, const void *beta, void *C, int ldc, int cRowOffset,
                                        int cColOffset, SplaFillMode cFillMode,
                                        SplaMatrixDistribution distC, SplaContext ctx);

/**
 * Computes a distributed general matrix multiplication of the form \f$ C \leftarrow \alpha A^H B +
 * \beta C \f$ in double precision for complex types. See documentation above.
 */
SPLA_EXPORT SplaError spla_pzgemm_ssbtr(int m, int n, int kLocal, SplaOperation opA,
                                        const void *alpha, const void *A, int lda, const void *B,
                                        int ldb, const void *beta, void *C, int ldc, int cRowOffset,
                                        int cColOffset, SplaFillMode cFillMode,
                                        SplaMatrixDistribution distC, SplaContext ctx);
#ifdef __cplusplus
}
#endif
#endif
