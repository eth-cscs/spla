#include <mpi.h>
#include <vector>
#include <random>
#include <array>
#include <sstream>
#include <cmath>
#include <utility>
#include <tuple>
#include <algorithm>
#include "spla/spla.hpp"
#include "gtest/gtest.h"
#include "mpi_util/mpi_communicator_handle.hpp"
#include "mpi_util/mpi_match_elementary_type.hpp"
#include "memory/host_array_const_view.hpp"
#include "util/common_types.hpp"
#include "memory/host_array_view.hpp"
#include "memory/host_array_const_view.hpp"
#include "memory/buffer.hpp"

#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
#include "memory/gpu_allocator.hpp"
#include "gpu_util/gpu_runtime_api.hpp"
#endif

// template<typename T>
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
//         stream << std::setw(12) << std::right << std::real(A[r + c * ld]);
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

static auto find_rectangle(int n) -> std::pair<int, int> {
  const int idealSize = std::sqrt(n);
  for (int range = 0; range < idealSize; ++range) {
    for (int i = idealSize; i <= idealSize + range; ++i) {
      for (int j = idealSize - range; j <= idealSize; ++j) {
        if (i * j == n) {
          return {i, j};
        }
      }
    }
  }
  return {n, 1};
}



// numThreads, rowBlockSize, colBlockSize, colsA, colsB, numLocalRows
template <typename T>
class GemmSSBTest : public ::testing::TestWithParam<
                        std::tuple<SplaProcessingUnit, int, int, int, int, int, std::pair<int, int>>> {
protected:
  GemmSSBTest()
      : rowBlockSize_(std::get<2>(GetParam())),
        colBlockSize_(std::get<3>(GetParam())),
        m_(std::get<4>(GetParam())),
        n_(std::get<5>(GetParam())),
        k_(0),
        ctx_(std::get<0>(GetParam())) {
    ctx_.set_num_threads(std::get<1>(GetParam()));

    const std::pair<int, int> kRange = std::get<6>(GetParam());
    std::uniform_int_distribution<int> kLocalDistribution(kRange.first, kRange.second);

    // generate local k size within range
    kLocalPerRank_.resize(mpi_world_size());
    for(auto& k : kLocalPerRank_) {
      k = kLocalDistribution(staticRandGen_);
      k_ += k;
    }
  }

  // void SetUp() override {
  // }

  auto multiply(MatrixDistribution& desc, int rowOffset, int colOffset) -> void {

    std::uniform_real_distribution<double> valueDistribution(0.0, 100.0);

    // initialize values in matrices
    std::vector<T> vecA(m_ * k_);
    for(auto& val : vecA) {
      val = valueDistribution(staticRandGen_);
    }
    std::vector<T> vecB(n_ * k_);
    for(auto& val : vecB) {
      val = valueDistribution(staticRandGen_);
    }
    std::vector<T> vecC(m_ * n_);
    for(auto& val : vecC) {
      val = valueDistribution(staticRandGen_);
    }
    std::vector<T> vecCRef = vecC; // copy C to compare with ScaLAPACK

    int localRowOffset = 0;
    for(int r =0; r < mpi_world_rank(); ++r) {
      localRowOffset += kLocalPerRank_[r];
    }

    spla::HostArrayConstView2D<T> localViewA(vecA.data() + localRowOffset, m_, kLocalPerRank_[mpi_world_rank()], k_);
    spla::HostArrayConstView2D<T> localViewB(vecB.data() + localRowOffset, n_, kLocalPerRank_[mpi_world_rank()], k_);

    int blacsCtx = Csys2blacs_handle(MPI_COMM_WORLD);
    int info;

    int grid = 0;
    Cblacs_gridinit(&grid, "r", desc.proc_grid_rows(), desc.proc_grid_cols());


    const int subMatrixRows = std::max<int>(1, m_ - rowOffset);
    const int subMatrixCols = std::max<int>(1, n_ - colOffset);
    const int subMatrixRowOffset = std::min<int>(m_ - 1, rowOffset);
    const int subMatrixColOffset = std::min<int>(n_ - 1, colOffset);

    // use same data on all ranks if mirror distribution
    if (desc.type() == SplaDistributionType::SPLA_DIST_MIRROR) {
      rowBlockSize_ = m_;
      colBlockSize_ = n_;
    }


    std::array<int, 9> descA;
    std::array<int, 9> descB;
    std::array<int, 9> descC;


    call_descinit(descA.data(), k_, m_, std::max<int>(k_, 1), m_, 0, 0, grid, std::max<int>(k_, 1), &info);
    call_descinit(descB.data(), k_, n_, std::max<int>(k_, 1), n_, 0, 0, grid, std::max<int>(k_, 1), &info);
    call_descinit(descC.data(), m_, n_, rowBlockSize_, colBlockSize_, 0, 0, grid, m_, &info);

    // multiply with pdgemm
    call_pgemm('C', 'N', subMatrixRows, subMatrixCols, k_, 1.0, vecA.data(), 1, 1, descA.data(),
               vecB.data(), 1, 1, descB.data(), 0.0, vecCRef.data(), subMatrixRowOffset + 1,
               subMatrixColOffset + 1, descC.data());

    // // free handler
    Cblacs_gridexit(grid);
    Cfree_blacs_system_handle(blacsCtx);
    // NOTE: free must happen before ASSERT, because ASSERT may exit early

    // if mirror distribution, broadcast reference result to all ranks
    if (desc.type() == SplaDistributionType::SPLA_DIST_MIRROR) {
      MPI_Bcast(vecCRef.data(), vecCRef.size(), MPIMatchElementaryType<T>::get(), 0, MPI_COMM_WORLD);
    }


#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
    std::vector<T> vecCFromGPU;
    // compare starting from device buffers if GPU enabled
    if(ctx_.processing_unit() == SPLA_PU_GPU) {
      Buffer<GPUAllocator> gpuBufferA;
      gpuBufferA.resize<T>(vecA.size());
      Buffer<GPUAllocator> gpuBufferB;
      gpuBufferB.resize<T>(vecB.size());
      Buffer<GPUAllocator> gpuBufferC;
      gpuBufferC.resize<T>(vecC.size());

      if (vecA.size())
        gpu::check_status(gpu::memcpy(static_cast<void*>(gpuBufferA.data<T>()),
                                      static_cast<const void*>(vecA.data()),
                                      vecA.size() * sizeof(T), gpu::flag::MemcpyHostToDevice));
      if (vecB.size())
      gpu::check_status(gpu::memcpy(static_cast<void*>(gpuBufferB.data<T>()),
                                    static_cast<const void*>(vecB.data()), vecB.size() * sizeof(T),
                                    gpu::flag::MemcpyHostToDevice));
      if (vecC.size())
      gpu::check_status(gpu::memcpy(static_cast<void*>(gpuBufferC.data<T>()),
                                    static_cast<const void*>(vecC.data()), vecC.size() * sizeof(T),
                                    gpu::flag::MemcpyHostToDevice));

      spla::pgemm_ssb(
          subMatrixRows, subMatrixCols, kLocalPerRank_[mpi_world_rank()], SPLA_OP_CONJ_TRANSPOSE,
          T(1.0), gpuBufferA.empty() ? nullptr : gpuBufferA.data<T>() + localRowOffset,
          localViewA.ld_inner(),
          gpuBufferB.empty() ? nullptr : gpuBufferB.data<T>() + localRowOffset,
          localViewB.ld_inner(), T(0.0), gpuBufferC.empty() ? nullptr : gpuBufferC.data<T>(), m_,
          subMatrixRowOffset, subMatrixColOffset, desc, ctx_);

      vecCFromGPU.resize(vecC.size());


      if (vecC.size())
        gpu::check_status(gpu::memcpy(
            static_cast<void*>(vecCFromGPU.data()), static_cast<const void*>(gpuBufferC.data<T>()),
            vecCFromGPU.size() * sizeof(T), gpu::flag::MemcpyDeviceToHost));
    }
#endif


    // compute from host memory
    spla::pgemm_ssb(subMatrixRows, subMatrixCols, kLocalPerRank_[mpi_world_rank()],
                    SPLA_OP_CONJ_TRANSPOSE, T(1.0), localViewA.data(), localViewA.ld_inner(),
                    localViewB.data(), localViewB.ld_inner(), T(0.0), vecC.data(), m_,
                    subMatrixRowOffset, subMatrixColOffset, desc, ctx_);
    // print_matrix(vecC.data(), m_, n_, m_, "vecC");
    // print_matrix(vecCRef.data(), m_, n_, m_, "vecCRef");

    // Assertions must only be used after all MPI calls, deadlock otherwise

    // compare results
    for (std::size_t i = 0; i < vecC.size(); ++i) {
      ASSERT_NEAR(std::real(vecC[i]), std::real(vecCRef[i]), 1e-6);
      ASSERT_NEAR(std::imag(vecC[i]), std::imag(vecCRef[i]), 1e-6);
    }

#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
    for (std::size_t i = 0; i < vecCFromGPU.size(); ++i) {
      ASSERT_NEAR(std::real(vecCFromGPU[i]), std::real(vecCRef[i]), 1e-6);
      ASSERT_NEAR(std::imag(vecCFromGPU[i]), std::imag(vecCRef[i]), 1e-6);
    }
#endif
  }

  int rowBlockSize_, colBlockSize_, m_, n_, k_;
  std::vector<int> kLocalPerRank_;
  spla::Context ctx_;

  static std::mt19937 staticRandGen_; // must produce same numbers on each rank
};

template<typename T>
std::mt19937 GemmSSBTest<T>::staticRandGen_(42);

typedef GemmSSBTest<double> GemmSSBScalar;

typedef GemmSSBTest<std::complex<double>> GemmSSBComplex;

TEST_P(GemmSSBScalar, FlatGrid) {
  try {
    auto desc = MatrixDistribution::create_blacs_block_cyclic(MPI_COMM_WORLD, 'R', mpi_world_size(),
                                                              1, rowBlockSize_, colBlockSize_);
    multiply(desc, 0, 0);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << mpi_world_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(GemmSSBComplex, FlatGrid) {
  try {
    auto desc = MatrixDistribution::create_blacs_block_cyclic(MPI_COMM_WORLD, 'R', mpi_world_size(),
                                                              1, rowBlockSize_, colBlockSize_);
    multiply(desc, 0, 0);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << mpi_world_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(GemmSSBScalar, SquareGrid) {
  try {
    int gridRows, gridCols;
    std::tie(gridRows, gridCols) = find_rectangle(mpi_world_size());

    auto desc = MatrixDistribution::create_blacs_block_cyclic(
        MPI_COMM_WORLD, 'R', gridRows, gridCols, rowBlockSize_, colBlockSize_);
    multiply(desc, 0, 0);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << mpi_world_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(GemmSSBComplex, SquareGrid) {
  try {
    int gridRows, gridCols;
    std::tie(gridRows, gridCols) = find_rectangle(mpi_world_size());

    auto desc = MatrixDistribution::create_blacs_block_cyclic(
        MPI_COMM_WORLD, 'R', gridRows, gridCols, rowBlockSize_, colBlockSize_);
    multiply(desc, 0, 0);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << mpi_world_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(GemmSSBScalar, SquareGridOffset) {
  try {
    int gridRows, gridCols;
    std::tie(gridRows, gridCols) = find_rectangle(mpi_world_size());

    auto desc = MatrixDistribution::create_blacs_block_cyclic(
        MPI_COMM_WORLD, 'R', gridRows, gridCols, rowBlockSize_, colBlockSize_);
    multiply(desc, 2, 3);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << mpi_world_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(GemmSSBComplex, SquareGridOffset) {
  try {
    int gridRows, gridCols;
    std::tie(gridRows, gridCols) = find_rectangle(mpi_world_size());

    auto desc = MatrixDistribution::create_blacs_block_cyclic(
        MPI_COMM_WORLD, 'R', gridRows, gridCols, rowBlockSize_, colBlockSize_);
    multiply(desc, 2, 3);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << mpi_world_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(GemmSSBScalar, Mirror) {
  try {
    auto desc = MatrixDistribution::create_mirror(MPI_COMM_WORLD);
    multiply(desc, 0, 0);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << mpi_world_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(GemmSSBComplex, Mirror) {
  try {
    auto desc = MatrixDistribution::create_mirror(MPI_COMM_WORLD);
    multiply(desc, 0, 0);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << mpi_world_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(GemmSSBScalar, MirrorOffset) {
  try {
    auto desc = MatrixDistribution::create_mirror(MPI_COMM_WORLD);
    multiply(desc, 2, 3);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << mpi_world_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

TEST_P(GemmSSBComplex, MirrorOffset) {
  try {
    auto desc = MatrixDistribution::create_mirror(MPI_COMM_WORLD);
    multiply(desc, 2, 3);
  } catch (const std::exception& e) {
    std::cout << "ERROR: Rank " << mpi_world_rank() << ", " << e.what() << std::endl;
    ASSERT_TRUE(false);
  }
}

static auto param_type_names(
    const ::testing::TestParamInfo<std::tuple<SplaProcessingUnit, int, int, int, int, int, std::pair<int,int>>>&
        info) -> std::string {
  std::stringstream stream;
  if(std::get<0>(info.param) == SplaProcessingUnit::SPLA_PU_HOST) {
    stream << "Host_";
  } else {
    stream << "GPU_";
  }
  stream << "t_" << std::to_string(std::get<1>(info.param)) << "_";
  stream << "mb_" << std::to_string(std::get<2>(info.param)) << "_";
  stream << "nb_" << std::get<3>(info.param) << "_";
  stream << "m_" << std::get<4>(info.param) << "_";
  stream << "n_" << std::get<5>(info.param) << "_";
  stream << "kMin_" << std::get<6>(info.param).first << "_";
  stream << "kMax_" << std::get<6>(info.param).second;

  return stream.str();
}

INSTANTIATE_TEST_CASE_P(FullGemmSSBTest, GemmSSBScalar,
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
                                           ::testing::Values(std::pair<int, int>(0, 1), std::pair<int, int>(50, 400))),  // k range
                        param_type_names);

INSTANTIATE_TEST_CASE_P(FullGemmSSBTest, GemmSSBComplex,
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
                                           ::testing::Values(std::pair<int, int>(0, 1), std::pair<int, int>(50, 400))),  // k range
                        param_type_names);
