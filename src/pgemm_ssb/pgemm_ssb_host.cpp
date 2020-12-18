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
#include <algorithm>
#include <atomic>
#include <memory>
#include <vector>

#include "block_generation/block_cyclic_generator.hpp"
#include "block_generation/matrix_block_generator.hpp"
#include "block_generation/mirror_generator.hpp"
#include "gemm/gemm_host.hpp"
#include "mpi_util/mpi_check_status.hpp"
#include "spla/context_internal.hpp"
#include "spla/exceptions.hpp"
#include "spla/matrix_distribution_internal.hpp"
#include "spla/spla.hpp"
#include "spla/types.h"
#include "tile_host.hpp"
#include "timing/timing.hpp"
#include "util/blas_interface.hpp"
#include "util/blas_threads_guard.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"
#include "util/check_gemm_param.hpp"
#include "mpi_util/mpi_match_elementary_type.hpp"
#include "mpi_util/mpi_request_handle.hpp"

namespace spla {

template <typename T>
void pgemm_ssb_host(int m, int n, int kLocal, SplaOperation opA, T alpha, const T *A, int lda,
                    const T *B, int ldb, T beta, T *C, int ldc, int cRowStart, int cColStart,
                    MatrixDistributionInternal &descC, ContextInternal &ctx) {
  SCOPED_TIMING("inner_host");
  if (m == 0 || n == 0) {
    return;
  }
  if (opA != SplaOperation::SPLA_OP_TRANSPOSE && opA != SplaOperation::SPLA_OP_CONJ_TRANSPOSE) {
    throw InvalidParameterError();
  }

  if (m < 0 || n < 0 || cRowStart < 0 || cColStart < 0) {
    throw InvalidParameterError();
  }

  // if (descC.comm().size() == 1) {
  //   return gemm_host<T>(ctx.num_threads(), opA, SPLA_OP_NONE, m, n, kLocal, alpha, A, lda, B, ldb,
  //                       beta, C + cRowStart + cColStart * ldc, ldc);
  // }

  std::shared_ptr<MatrixBlockGenerator> matrixDist;
  if (descC.type() == SplaDistributionType::SPLA_DIST_BLACS_BLOCK_CYCLIC) {
    matrixDist.reset(new BlockCyclicGenerator(descC.row_block_size(), descC.col_block_size(),
                                              descC.proc_grid_rows(), descC.proc_grid_cols(), m, n,
                                              cRowStart, cColStart));
  } else {
    return; // TODO:remove
    matrixDist.reset(new MirrorGenerator(ctx.tile_size_host(), ctx.tile_size_host(), m, n,
                                         cRowStart, cColStart));
  }

  check_gemm_param(opA, SplaOperation::SPLA_OP_NONE, matrixDist->local_rows(descC.comm().rank()),
                   matrixDist->local_cols(descC.comm().rank()), kLocal, A, lda, B, ldb, C, ldc);

  HostArrayConstView2D<T> viewA(A, m, kLocal, lda);
  HostArrayConstView2D<T> viewB(B, n, kLocal, ldb);
  HostArrayView2D<T> viewC(C, n + cColStart, ldc, ldc);

  auto buffers = ctx.mpi_buffers(3);
  for(auto& b : buffers) {
    b->resize<T>(matrixDist->max_cols_in_block() *
                 matrixDist->max_rows_in_block());
  }
  HostArrayView2D<T> recvView(buffers[0]->data<T>(), matrixDist->max_cols_in_block(), matrixDist->max_rows_in_block());
  HostArrayView2D<T> sendView(buffers[1]->data<T>(), matrixDist->max_cols_in_block(), matrixDist->max_rows_in_block());
  HostArrayView2D<T> resultView(buffers[2]->data<T>(), matrixDist->max_cols_in_block(), matrixDist->max_rows_in_block());

  const IntType gridSize = descC.proc_grid_cols() * descC.proc_grid_rows();
  const IntType numBlocks = matrixDist->num_blocks();

  std::vector<BlockInfo> blockInfos(gridSize);

  MPIRequestHandle sendReq;
  MPIRequestHandle recvReq;
  auto& comm = descC.comm();
  MPI_Win resultWindow;
  MPI_Win_create(resultView.data(), resultView.size() * sizeof(T), sizeof(T),
                 MPI_INFO_NULL, comm.get(), &resultWindow);

  for (IntType blockColIdx = 0; blockColIdx < matrixDist->num_block_cols();
       blockColIdx += descC.proc_grid_cols()) {
    for (IntType blockRowIdx = 0; blockRowIdx < matrixDist->num_block_rows();
         blockRowIdx += descC.proc_grid_rows()) {
      const IntType numCurrentBlockRows =
          std::min<IntType>(matrixDist->num_block_rows() - blockRowIdx, descC.proc_grid_rows());
      const IntType numCurrentBlockCols =
          std::min<IntType>(matrixDist->num_block_cols() - blockColIdx, descC.proc_grid_cols());
      const IntType numCurrentBlocks = numCurrentBlockRows * numCurrentBlockCols;
      // compute block infos
      IntType myBlockIdx = -1;
      for (IntType ic = 0; ic < numCurrentBlockCols; ++ic) {
        for (IntType ir = 0; ir < numCurrentBlockRows; ++ir) {
          blockInfos[ic * numCurrentBlockRows + ir] = matrixDist->get_block_info(blockRowIdx + ir, blockColIdx + ic);
          if (blockInfos[ic * numCurrentBlockRows + ir].mpiRank == comm.rank()) {
            assert(myBlockIdx < 0); // each rank must receive at most one block within grid
            myBlockIdx = ic * numCurrentBlockRows + ir;
          }
        }
      }
      const bool accRequired = numCurrentBlocks != comm.size();

      if (accRequired) {
        // make sure result is 0 for accumulation
        std::memset(resultView.data(), 0, resultView.size() * sizeof(T));
        // Remote access required after this fence
        MPI_Win_fence(0, resultWindow);
      }

      if (myBlockIdx >= 0) {
        const IntType startGridIdx = (myBlockIdx + 1) % numCurrentBlocks;

        const int sendRank =
            blockInfos[myBlockIdx == 0 ? numCurrentBlocks - 1 : myBlockIdx - 1].mpiRank;
        const int recvRank = blockInfos[(myBlockIdx + 1) % numCurrentBlocks].mpiRank;

        std::memset(recvView.data(), 0, recvView.size() * sizeof(T));

        for (IntType i = 0; i < numCurrentBlocks; ++i) {
          const IntType gridBlockIdx = (startGridIdx + i) % numCurrentBlocks;
          const auto &info = blockInfos[gridBlockIdx];

          sendReq.wait_if_active();
          recvReq.wait_if_active();
          std::swap(sendView, recvView);

          if (i < numCurrentBlocks - 1) {
            MPI_Irecv(recvView.data(), recvView.size(),
                      MPIMatchElementaryType<T>::get(), recvRank, 0, comm.get(),
                      recvReq.get_and_activate());
          }
          if (viewA.dim_inner() != 0) {
            gemm_host<T>(ctx.num_threads(), opA, SplaOperation::SPLA_OP_NONE,
                         info.numRows, info.numCols, kLocal, alpha,
                         &viewA(info.globalSubRowIdx, 0), lda,
                         &viewB(info.globalSubColIdx, 0), ldb, 1.0,
                         sendView.data(), sendView.ld_inner());
          }
          if (i < numCurrentBlocks - 1)
            MPI_Send(sendView.data(), sendView.size(),
                     MPIMatchElementaryType<T>::get(), sendRank, 0, comm.get());
          else {
            if(accRequired)
              mpi_check_status(MPI_Raccumulate(
                  sendView.data(), sendView.size(),
                  MPIMatchElementaryType<T>::get(), info.mpiRank, 0,
                  sendView.size(), MPIMatchElementaryType<T>::get(), MPI_SUM,
                  resultWindow, sendReq.get_and_activate()));
          }
        }

      } else {
        // More ranks than blocks -> extra ranks simply do accumulate on target
        assert(accRequired);
        for (IntType gridBlockIdx = 0; gridBlockIdx < numCurrentBlocks;
             ++gridBlockIdx) {
          BlockInfo info =
              blockInfos[(gridBlockIdx + comm.rank()) % numCurrentBlocks];
          assert(info.numCols <= sendView.dim_outer());
          assert(info.numRows <= sendView.dim_inner());
          sendReq.wait_if_active();
          if (viewA.dim_inner() == 0) {
            std::memset(sendView.data(), 0, sendView.size() * sizeof(T));
          } else {
            gemm_host<T>(ctx.num_threads(), opA, SplaOperation::SPLA_OP_NONE,
                         info.numRows, info.numCols, kLocal, alpha,
                         &viewA(info.globalSubRowIdx, 0), lda,
                         &viewB(info.globalSubColIdx, 0), ldb, 0.0,
                         sendView.data(), sendView.ld_inner());
          }
          if (info.mpiRank < 0) {
            mpi_check_status(MPI_Allreduce(
                sendView.data(), resultView.data(), sendView.size(),
                MPIMatchElementaryType<T>::get(), MPI_SUM, comm.get()));

          } else {
            mpi_check_status(MPI_Raccumulate(
                sendView.data(), sendView.size(),
                MPIMatchElementaryType<T>::get(), info.mpiRank, 0,
                sendView.size(), MPIMatchElementaryType<T>::get(), MPI_SUM,
                resultWindow, sendReq.get_and_activate()));
          }
        }
      }

      sendReq.wait_if_active();
      recvReq.wait_if_active();

      // local access only after this fence
      if (accRequired) {
        MPI_Win_fence(0, resultWindow);
      }

      if (myBlockIdx >= 0) {
        const auto &myInfo = blockInfos[myBlockIdx];
        auto TileView = accRequired ? resultView : sendView;
        for (IntType col = 0; col < myInfo.numCols; ++col) {
          for (IntType row = 0; row < myInfo.numRows; ++row) {
            viewC(myInfo.localColIdx + col, myInfo.localRowIdx + row) =
                beta *
                    viewC(myInfo.localColIdx + col, myInfo.localRowIdx + row) +
                TileView(col, row);
          }
        }
      }
    }
  }

  MPI_Win_free(&resultWindow);
}

template void pgemm_ssb_host<float>(int m, int n, int kLocal, SplaOperation opA, float alpha,
                                    const float *A, int lda, const float *B, int ldb, float beta,
                                    float *C, int ldc, int cRowStart, int cColStart,
                                    MatrixDistributionInternal &descC, ContextInternal &ctx);

template void pgemm_ssb_host<double>(int m, int n, int kLocal, SplaOperation opA, double alpha,
                                     const double *A, int lda, const double *B, int ldb,
                                     double beta, double *C, int ldc, int cRowStart, int cColStart,
                                     MatrixDistributionInternal &descC, ContextInternal &ctx);

template void pgemm_ssb_host<std::complex<float>>(
    int m, int n, int kLocal, SplaOperation opA, std::complex<float> alpha,
    const std::complex<float> *A, int lda, const std::complex<float> *B, int ldb,
    std::complex<float> beta, std::complex<float> *C, int ldc, int cRowStart, int cColStart,
    MatrixDistributionInternal &descC, ContextInternal &ctx);

template void pgemm_ssb_host<std::complex<double>>(
    int m, int n, int kLocal, SplaOperation opA, std::complex<double> alpha,
    const std::complex<double> *A, int lda, const std::complex<double> *B, int ldb,
    std::complex<double> beta, std::complex<double> *C, int ldc, int cRowStart, int cColStart,
    MatrixDistributionInternal &descC, ContextInternal &ctx);
}  // namespace spla
