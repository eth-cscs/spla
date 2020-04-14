#include <mpi.h>
#include <vector>
#include <random>
#include <array>
#include <sstream>
#include <cmath>
#include "gtest/gtest.h"
#include "mpi_util/mpi_communicator_handle.hpp"
#include "mpi_util/mpi_match_elementary_type.hpp"
#include "spla/spla.hpp"


// #include <sstream>
// #include <iostream>
// #include <iomanip>

// template <typename T>
// static auto print_matrix(const T* A, const int rows, const int cols, const int ld, const std::string& label)
//     -> void {
//   int rank, size;
//   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//   MPI_Comm_size(MPI_COMM_WORLD, &size);
//   for (int r = 0; r < size; ++r) {
//     MPI_Barrier(MPI_COMM_WORLD);
//     if (r != rank) continue;
//     std::stringstream stream;
//     // stream << std::scientific;
//     stream << std::fixed;
//     stream << std::setprecision(1);
//     stream << " -------------------- " << std::endl;
//     stream << "Rank = " << rank << ", " << label << ":" << std::endl;
//     for (int r = 0; r < rows; ++r) {
//       for (int c = 0; c < cols; ++c) {
//         stream << std::setw(8) << std::right << std::real(A[r + c * ld]);
//       }
//       stream << std::endl;
//     }
//     stream << " -------------------- " << std::endl;
//     std::cout << stream.str();
//   }
//   MPI_Barrier(MPI_COMM_WORLD);
// }

extern "C" {

int Csys2blacs_handle(MPI_Comm SysCtxt);

MPI_Comm Cblacs2sys_handle(int BlacsCtxt);

void Cblacs_gridinit(int* ConTxt, const char* order, int nprow, int npcol);

void Cblacs_gridinfo(int, int*, int*, int*, int*);

void Cblacs_gridmap(int* ConTxt, int* usermap, int ldup, int nprow0, int npcol0);

void Cblacs_gridinfo(int ConTxt, int* nprow, int* npcol, int* myrow, int* mycol);

void Cfree_blacs_system_handle(int ISysCtxt);

void Cblacs_barrier(int ConTxt, const char* scope);

void Cblacs_gridexit(int ConTxt);

void descinit_(int* desc, const int* m, const int* n, const int* mb, const int* nb, const int* irsrc,
              const int* icsrc, const int* ictxt, const int* lld, int* info);

void pdgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA, double* A,
             int* IA, int* JA, int* DESCA, double* B, int* IB, int* JB, int* DESCB, double* BETA,
             double* C, int* IC, int* JC, int* DESCC);

void pzgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, void* ALPHA, void* A,
             int* IA, int* JA, int* DESCA, void* B, int* IB, int* JB, int* DESCB, void* BETA,
             void* C, int* IC, int* JC, int* DESCC);

void pdgemr2d_(int* m, int* n, double* a, int* ia, int* ja, int* desca, double* b,
              int* ib, int* jb, int* descb, int* ictxt);

void pzgemr2d_(int* m, int* n, void* a, int* ia, int* ja, int* desca, void* b,
              int* ib, int* jb, int* descb, int* ictxt);

int numroc_(int const& n, int const& nb, int const& iproc, int const& isproc, int const& nprocs);

} // extern C

static auto call_descinit(int* desc, int m, int n, int mb, int nb, int irsrc, int icsrc, int ictxt,
                          int lld, int* info) -> void {
  descinit_(desc, &m, &n, &mb, &nb, &irsrc, &icsrc, &ictxt, &lld, info);
}

static auto call_pgemm(char TRANSA, char TRANSB, int M, int N, int K, double ALPHA, double* A,
                        int IA, int JA, int* DESCA, double* B, int IB, int JB, int* DESCB,
                        double BETA, double* C, int IC, int JC, int* DESCC) -> void {
  pdgemm_(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &IA, &JA, DESCA, B, &IB, &JB, DESCB, &BETA, C,
          &IC, &JC, DESCC);
}

static auto call_pgemm(char TRANSA, char TRANSB, int M, int N, int K, std::complex<double> ALPHA,
                       std::complex<double>* A, int IA, int JA, int* DESCA, std::complex<double>* B,
                       int IB, int JB, int* DESCB, std::complex<double> BETA,
                       std::complex<double>* C, int IC, int JC, int* DESCC) -> void {
  pzgemm_(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &IA, &JA, DESCA, B, &IB, &JB, DESCB, &BETA, C,
          &IC, &JC, DESCC);
}

static void call_pgemr2d(int m, int n, double* a, int ia, int ja, int* desca, double* b,
              int ib, int jb, int* descb, int ictxt) {
  pdgemr2d_(&m, &n, a, &ia, &ja, desca, b, &ib, &jb, descb, &ictxt);
}

static void call_pgemr2d(int m, int n, std::complex<double>* a, int ia, int ja, int* desca, std::complex<double>* b,
              int ib, int jb, int* descb, int ictxt) {
  pzgemr2d_(&m, &n, a, &ia, &ja, desca, b, &ib, &jb, descb, &ictxt);
}

using namespace spla;

static auto mpi_world_size() -> int {
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  return worldSize;

}

static auto mpi_world_rank() -> int {
  int worldRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  return worldRank;

}



// numThreads, rowBlockSize, colBlockSize, colsA, colsB, numLocalRows
template <typename T>
class GemmSBSTest : public ::testing::TestWithParam<
                        std::tuple<SplaProcessingUnit, int, int, int, int, int, int>> {
protected:
  GemmSBSTest()
      : numThreads_(std::get<1>(GetParam())),
        rowBlockSize_(std::get<2>(GetParam())),
        colBlockSize_(std::get<3>(GetParam())),
        maxMLocal_(std::get<4>(GetParam())),
        n_(std::get<5>(GetParam())),
        k_(std::get<6>(GetParam())),
        vecA_(maxMLocal_ * k_),
        vecB_(n_ * k_),
        vecC_(maxMLocal_ * n_),
        vecCRef_(vecC_.size()),
        ctx_(std::get<0>(GetParam())) {
    ctx_.set_num_threads(numThreads_);
    std::mt19937 randGen(mpi_world_rank() * 42);
    std::uniform_real_distribution<double> uniformRandDis(0.0, 100.0);

    for(auto& val : vecA_) {
      val = uniformRandDis(randGen);
    }
    for(auto& val : vecB_) {
      val = uniformRandDis(randGen);
    }
    for(auto& val : vecC_) {
      val = uniformRandDis(randGen);
    }
    vecCRef_.assign(vecC_.begin(), vecC_.end());
  }

  // void SetUp() override {
  // }

  auto multiply(MatrixDistribution& desc, int rowOffset, int colOffset) -> void {

    // init ScaLAPACK handles
    int blacsCtx = Csys2blacs_handle(MPI_COMM_WORLD);
    int info;
    int grid = 0;
    Cblacs_gridinit(&grid, "r", desc.proc_grid_rows(), desc.proc_grid_cols());
    int gridAC = 0;
    Cblacs_gridinit(&gridAC, "r", mpi_world_size(), 1);

    const auto numLocalRows =
        mpi_world_rank() < desc.proc_grid_cols() * desc.proc_grid_rows() ? maxMLocal_ : 0;

    const int subMatrixRows = std::max<int>(1, k_ - rowOffset);
    const int subMatrixCols = std::max<int>(1, n_ - colOffset);
    const int subMatrixRowOffset = std::min<int>(k_ - 1, rowOffset);
    const int subMatrixColOffset = std::min<int>(n_ - 1, colOffset);

    const auto globalRows = mpi_world_size() * maxMLocal_;

    // use same data on all ranks if mirror distribution
    if (desc.type() == SplaDistributionType::SPLA_DIST_MIRROR) {
      rowBlockSize_ = k_;
      colBlockSize_ = n_;
      MPI_Bcast(vecB_.data(), vecB_.size(), MPIMatchElementaryType<T>::get(), 0, MPI_COMM_WORLD);
    }

    spla::gemm_sbs(numLocalRows, subMatrixCols, subMatrixRows, T(1.0), vecA_.data(), maxMLocal_, vecB_.data(), k_,
                subMatrixRowOffset, subMatrixColOffset, desc, T(0.0), vecC_.data(), maxMLocal_,
                ctx_);

    std::array<int, 9> descA;
    std::array<int, 9> descRedistA;
    std::array<int, 9> descB;
    std::array<int, 9> descC;
    std::array<int, 9> descRedistC;

    // original A, C description on 1D grid
    call_descinit(descA.data(), globalRows, k_, maxMLocal_, k_, 0, 0, gridAC, maxMLocal_, &info);
    call_descinit(descC.data(), globalRows, n_, maxMLocal_, n_, 0, 0, gridAC, maxMLocal_, &info);

    // A, B, C description on 2D grid of C
    call_descinit(descRedistA.data(), globalRows, k_, maxMLocal_, k_, 0, 0, grid, globalRows, &info);
    call_descinit(descB.data(), k_, n_, rowBlockSize_, colBlockSize_, 0, 0, grid,
                  k_, &info);
    call_descinit(descRedistC.data(), globalRows, n_, maxMLocal_, n_, 0, 0, grid, globalRows, &info);

    // remap A and C
    std::vector<T> vecRedistA(globalRows * k_);
    call_pgemr2d(globalRows, k_, vecA_.data(), 1, 1, descA.data(), vecRedistA.data(), 1, 1,
                  descRedistA.data(), blacsCtx);

    std::vector<T> vecRedistC(globalRows * n_);
    call_pgemr2d(globalRows, n_, vecC_.data(), 1, 1, descC.data(), vecRedistC.data(), 1, 1,
                  descRedistC.data(), blacsCtx);

    // multiply with pdgemm
    call_pgemm('N', 'N', globalRows, subMatrixCols, subMatrixRows, 1.0, vecRedistA.data(), 1, 1,
               descRedistA.data(), vecB_.data(), subMatrixRowOffset + 1, subMatrixColOffset + 1,
               descB.data(), 0.0, vecRedistC.data(), 1, 1, descRedistC.data());

    // map back C
    call_pgemr2d(globalRows, n_, vecRedistC.data(), 1, 1, descRedistC.data(), vecCRef_.data(), 1, 1,
                  descC.data(), blacsCtx);


    // // free handler
    Cblacs_gridexit(gridAC);
    Cblacs_gridexit(grid);
    Cfree_blacs_system_handle(blacsCtx);
    // NOTE: free must happen before ASSERT, because ASSERT may exit early

    // compare results
    for (std::size_t i = 0; i < vecC_.size(); ++i) {
      ASSERT_NEAR(std::real(vecC_[i]), std::real(vecCRef_[i]), 1e-6);
      ASSERT_NEAR(std::imag(vecC_[i]), std::imag(vecCRef_[i]), 1e-6);
    }

  }

  int numThreads_, rowBlockSize_, colBlockSize_, maxMLocal_, n_, k_;
  std::vector<T> vecA_, vecB_, vecC_, vecCRef_;
  spla::Context ctx_;
};

typedef GemmSBSTest<double> GemmSBSScalar;

typedef GemmSBSTest<std::complex<double>> GemmSBSComplex;

TEST_P(GemmSBSScalar, FlatGrid) {
  try {
    auto desc = MatrixDistribution::create_blacs_block_cyclic(MPI_COMM_WORLD, 'R', mpi_world_size(),
                                                              1, rowBlockSize_, colBlockSize_);
    multiply(desc, 0, 0);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << mpi_world_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(GemmSBSComplex, FlatGrid) {
  try {
    auto desc = MatrixDistribution::create_blacs_block_cyclic(MPI_COMM_WORLD, 'R', mpi_world_size(),
                                                              1, rowBlockSize_, colBlockSize_);
    multiply(desc, 0, 0);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << mpi_world_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(GemmSBSScalar, SquareGrid) {
  try {
    const int gridRows = std::max<int>(1, std::sqrt(mpi_world_size()));
    const int gridCols = std::max<int>(1, mpi_world_size() / gridRows);

    auto desc = MatrixDistribution::create_blacs_block_cyclic(
        MPI_COMM_WORLD, 'R', gridRows, gridCols, rowBlockSize_, colBlockSize_);
    multiply(desc, 0, 0);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << mpi_world_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(GemmSBSComplex, SquareGrid) {
  try {
    const int gridRows = std::max<int>(1, std::sqrt(mpi_world_size()));
    const int gridCols = std::max<int>(1, mpi_world_size() / gridRows);

    auto desc = MatrixDistribution::create_blacs_block_cyclic(
        MPI_COMM_WORLD, 'R', gridRows, gridCols, rowBlockSize_, colBlockSize_);
    multiply(desc, 0, 0);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << mpi_world_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(GemmSBSScalar, SquareGridOffset) {
  try {
    const int gridRows = std::max<int>(1, std::sqrt(mpi_world_size()));
    const int gridCols = std::max<int>(1, mpi_world_size() / gridRows);

    auto desc = MatrixDistribution::create_blacs_block_cyclic(
        MPI_COMM_WORLD, 'R', gridRows, gridCols, rowBlockSize_, colBlockSize_);
    multiply(desc, 2, 3);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << mpi_world_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(GemmSBSComplex, SquareGridOffset) {
  try {
    const int gridRows = std::max<int>(1, std::sqrt(mpi_world_size()));
    const int gridCols = std::max<int>(1, mpi_world_size() / gridRows);

    auto desc = MatrixDistribution::create_blacs_block_cyclic(
        MPI_COMM_WORLD, 'R', gridRows, gridCols, rowBlockSize_, colBlockSize_);
    multiply(desc, 2, 3);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << mpi_world_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(GemmSBSScalar, Mirror) {
  try {
    auto desc = MatrixDistribution::create_mirror(MPI_COMM_WORLD);
    multiply(desc, 0, 0);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << mpi_world_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(GemmSBSComplex, Mirror) {
  try {
    auto desc = MatrixDistribution::create_mirror(MPI_COMM_WORLD);
    multiply(desc, 0, 0);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << mpi_world_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(GemmSBSScalar, MirrorOffset) {
  try {
    auto desc = MatrixDistribution::create_mirror(MPI_COMM_WORLD);
    multiply(desc, 2, 3);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << mpi_world_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(GemmSBSComplex, MirrorOffset) {
  try {
    auto desc = MatrixDistribution::create_mirror(MPI_COMM_WORLD);
    multiply(desc, 2, 3);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << mpi_world_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}



static auto param_type_names(
    const ::testing::TestParamInfo<std::tuple<SplaProcessingUnit, int, int, int, int, int, int>>&
        info) -> std::string {
  std::stringstream stream;
  if (std::get<0>(info.param) == SplaProcessingUnit::SPLA_PU_HOST) {
    stream << "Host_";
  } else {
    stream << "GPU_";
  }
  stream << "t_" << std::to_string(std::get<1>(info.param)) << "_";
  stream << "mb_" << std::to_string(std::get<2>(info.param)) << "_";
  stream << "nb_" << std::get<3>(info.param) << "_";
  stream << "m_" << std::get<4>(info.param) << "_";
  stream << "n_" << std::get<5>(info.param) << "_";
  stream << "kLocal_" << std::get<6>(info.param);

  return stream.str();
}

INSTANTIATE_TEST_CASE_P(FullGemmSBSTest, GemmSBSScalar,
                        ::testing::Combine(::testing::Values(SplaProcessingUnit::SPLA_PU_HOST
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
                                                             ,
                                                             SplaProcessingUnit::SPLA_PU_GPU
#endif
                                                             ),
                                           ::testing::Values(1, 4),            // number of threads
                                           ::testing::Values(1, 64),           // coloumn block size
                                           ::testing::Values(1, 64),           // row block size
                                           ::testing::Values(1, 13, 32, 263),  // m
                                           ::testing::Values(1, 13, 32, 263),  // n
                                           ::testing::Values(1, 13, 32, 263)),  // k
                        param_type_names);

INSTANTIATE_TEST_CASE_P(FullGemmSBSTest, GemmSBSComplex,
                        ::testing::Combine(::testing::Values(SplaProcessingUnit::SPLA_PU_HOST
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
                                                             ,
                                                             SplaProcessingUnit::SPLA_PU_GPU
#endif
                                                             ),
                                           ::testing::Values(1, 4),            // number of threads
                                           ::testing::Values(1, 64),           // coloumn block size
                                           ::testing::Values(1, 64),           // row block size
                                           ::testing::Values(1, 13, 32, 263),  // m
                                           ::testing::Values(1, 13, 32, 263),  // n
                                           ::testing::Values(1, 13, 32, 263)),  // k
                        param_type_names);


