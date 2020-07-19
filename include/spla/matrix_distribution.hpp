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
#ifndef SPLA_MATRIX_DISTRIBUTION_HPP
#define SPLA_MATRIX_DISTRIBUTION_HPP

#include <mpi.h>

#include <complex>
#include <memory>

#include "spla/config.h"
#include "spla/types.h"

/*! \cond PRIVATE */
namespace spla {
/*! \endcond */

class MatrixDistributionInternal;

class Context;

class SPLA_EXPORT MatrixDistribution {
public:
  /**
   * Create a blacs block cyclic matrix distribution with row major or coloumn major ordering of MPI
   * ranks.
   *
   * @param[in] comm MPI communicator to be used.
   * @param[in] order Either 'R' for row major ordering or 'C' for coloumn major ordering.
   * @param[in] procGridRows Number of rows in process grid.
   * @param[in] procGridCols Number of coloumns in process grid.
   * @param[in] rowBlockSize Row block size for matrix partitioning.
   * @param[in] colBlockSize Coloumn block size for matrix partitioning.
   */
  static MatrixDistribution create_blacs_block_cyclic(MPI_Comm comm, char order, int procGridRows,
                                                      int procGridCols, int rowBlockSize,
                                                      int colBlockSize);

  /**
   * Create a blacs block cyclic matrix distribution with given process grid mapping.
   *
   * @param[in] comm MPI communicator to be used.
   * @param[in] mapping Pointer to array of size procGridRows * procGridCols mapping MPI ranks onto
   * a coloumn major process grid.
   * @param[in] procGridRows Number of rows in process grid.
   * @param[in] procGridCols Number of coloumns in process grid.
   * @param[in] rowBlockSize Row block size for matrix partitioning.
   * @param[in] colBlockSize Coloumn block size for matrix partitioning.
   */
  static MatrixDistribution create_blacs_block_cyclic_from_mapping(
      MPI_Comm comm, const int *mapping, int procGridRows, int procGridCols, int rowBlockSize,
      int colBlockSize);

  /**
   * Create a mirror distribution, where the full matrix is stored on each MPI rank.
   *
   * @param[in] comm MPI communicator to be used.
   */
  static MatrixDistribution create_mirror(MPI_Comm comm);

  /**
   * Default move constructor.
   */
  MatrixDistribution(MatrixDistribution &&) = default;

  /**
   * Default copy constructor.
   */
  MatrixDistribution(const MatrixDistribution &) = default;

  /**
   * Default move assignment operator.
   */
  MatrixDistribution &operator=(MatrixDistribution &&) = default;

  /**
   * Default copy assignment operator.
   */
  MatrixDistribution &operator=(const MatrixDistribution &) = default;

  /**
   * Access a distribution parameter.
   * @return Number of rows in process grid.
   */
  int proc_grid_rows() const;

  /**
   * Access a distribution parameter.
   * @return Number of coloumns in process grid.
   */
  int proc_grid_cols() const;

  /**
   * Access a distribution parameter.
   * @return Row block size used for matrix partitioning.
   */
  int row_block_size() const;

  /**
   * Access a distribution parameter.
   * @return Coloumn block size used for matrix partitioning.
   */
  int col_block_size() const;

  /**
   * Access a distribution parameter.
   * @return Distribution type
   */
  SplaDistributionType type() const;

  /**
   * Access a distribution parameter.
   * @return Communicator used internally. Order of ranks may differ from communicator provided for
   * creation of distribution.
   */
  MPI_Comm comm();

private:
  /*! \cond PRIVATE */
  explicit MatrixDistribution(std::shared_ptr<MatrixDistributionInternal> descInternal);

  friend void pgemm_ssb(int m, int n, int kLocal, float alpha, const float *A, int lda,
                        const float *B, int ldb, float beta, float *C, int ldc, int cRowStart,
                        int cColStart, MatrixDistribution &descC, Context &ctx);

  friend void pgemm_ssb(int m, int n, int kLocal, double alpha, const double *A, int lda,
                        const double *B, int ldb, double beta, double *C, int ldc, int cRowStart,
                        int cColStart, MatrixDistribution &descC, Context &ctx);

  friend void pgemm_ssb(int m, int n, int kLocal, std::complex<float> alpha,
                        const std::complex<float> *A, int lda, const std::complex<float> *B,
                        int ldb, std::complex<float> beta, std::complex<float> *C, int ldc,
                        int cRowStart, int cColStart, MatrixDistribution &descC, Context &ctx);

  friend void pgemm_ssb(int m, int n, int kLocal, std::complex<double> alpha,
                        const std::complex<double> *A, int lda, const std::complex<double> *B,
                        int ldb, std::complex<double> beta, std::complex<double> *C, int ldc,
                        int cRowStart, int cColStart, MatrixDistribution &descC, Context &ctx);

  friend void pgemm_sbs(int mLocal, int n, int k, float alpha, const float *A, int lda,
                        const float *B, int ldb, int bRowOffset, int bColOffset,
                        MatrixDistribution &descB, float beta, float *C, int ldc, Context &ctx);

  friend void pgemm_sbs(int mLocal, int n, int k, double alpha, const double *A, int lda,
                        const double *B, int ldb, int bRowOffset, int bColOffset,
                        MatrixDistribution &descB, double beta, double *C, int ldc, Context &ctx);

  friend void pgemm_sbs(int mLocal, int n, int k, std::complex<float> alpha,
                        const std::complex<float> *A, int lda, const std::complex<float> *B,
                        int ldb, int bRowOffset, int bColOffset, MatrixDistribution &descB,
                        std::complex<float> beta, std::complex<float> *C, int ldc, Context &ctx);

  friend void pgemm_sbs(int mLocal, int n, int k, std::complex<double> alpha,
                        const std::complex<double> *A, int lda, const std::complex<double> *B,
                        int ldb, int bRowOffset, int bColOffset, MatrixDistribution &descB,
                        std::complex<double> beta, std::complex<double> *C, int ldc, Context &ctx);

  std::shared_ptr<MatrixDistributionInternal> descInternal_;
  /*! \endcond */
};

}  // namespace spla

#endif
