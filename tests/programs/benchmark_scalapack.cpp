#include <mpi.h>

#include <array>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "CLI/CLI.hpp"
#include "memory/allocator_collection.hpp"
#include "memory/buffer.hpp"
#include "memory/pool_allocator.hpp"
#include "mpi_util/mpi_init_handle.hpp"
#include "spla/matrix_distribution.hpp"
#include "spla/spla.hpp"
#include "timing/rt_graph.hpp"
#include "timing/timing.hpp"

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

void descinit_(int* desc, const int* m, const int* n, const int* mb, const int* nb,
               const int* irsrc, const int* icsrc, const int* ictxt, const int* lld, int* info);

void pdgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA, double* A, int* IA,
             int* JA, int* DESCA, double* B, int* IB, int* JB, int* DESCB, double* BETA, double* C,
             int* IC, int* JC, int* DESCC);

void pzgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, void* ALPHA, void* A, int* IA,
             int* JA, int* DESCA, void* B, int* IB, int* JB, int* DESCB, void* BETA, void* C,
             int* IC, int* JC, int* DESCC);

void pdgemr2d_(int* m, int* n, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb,
               int* descb, int* ictxt);

int numroc_(int const& n, int const& nb, int const& iproc, int const& isproc, int const& nprocs);
}

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

static void call_pdgemr2d(int m, int n, double* a, int ia, int ja, int* desca, double* b, int ib,
                          int jb, int* descb, int ictxt) {
  pdgemr2d_(&m, &n, a, &ia, &ja, desca, b, &ib, &jb, descb, &ictxt);
}

template <typename T>
void run_gemm(const std::shared_ptr<spla::Allocator<spla::MemLoc::Host>>& allocator,
              spla::Context& ctx, int globalRows, int colsA, int colsB, int numThreads,
              int blacsBlockSize, int numRepeats) {
  int worldRank, worldSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  const int maxRowsPerRank = (globalRows + worldSize - 1) / worldSize;
  const int localNumRows =
      std::max(std::min(globalRows - worldRank * maxRowsPerRank, maxRowsPerRank), 0);
  const int maxRowsC = (colsA / (blacsBlockSize * worldSize) + 1) * blacsBlockSize;

  spla::Buffer<T, spla::MemLoc::Host> A(allocator, maxRowsPerRank * colsA);
  spla::Buffer<T, spla::MemLoc::Host> B(allocator, maxRowsPerRank * colsB);
  spla::Buffer<T, spla::MemLoc::Host> C(allocator, maxRowsC * colsB);

  rt_graph::Timer timer;

  auto arrayDesc = spla::MatrixDistribution::create_blacs_block_cyclic(
      MPI_COMM_WORLD, 'R', worldSize, 1, blacsBlockSize, blacsBlockSize);
  ctx.set_num_threads(numThreads);

  // run once to warm up
  spla::pgemm_ssb(colsA, colsB, localNumRows, SPLA_OP_CONJ_TRANSPOSE, 1.0, A.data(), localNumRows,
                  B.data(), localNumRows, 0.0, C.data(), maxRowsC, 0, 0, arrayDesc, ctx);

  START_TIMING("spla - host memory");
  for (int r = 0; r < numRepeats; ++r) {
    SCOPED_TIMING("multiply");
    spla::pgemm_ssb(colsA, colsB, localNumRows, SPLA_OP_CONJ_TRANSPOSE, 1.0, A.data(), localNumRows,
                    B.data(), localNumRows, 0.0, C.data(), maxRowsC, 0, 0, arrayDesc, ctx);
  }
  STOP_TIMING("spla - host memory");

  std::array<int, 9> descA{0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::array<int, 9> descB{0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::array<int, 9> descC{0, 0, 0, 0, 0, 0, 0, 0, 0};
  int info;
  int grid = 0;
  int blacsCtx = Csys2blacs_handle(MPI_COMM_WORLD);
  Cblacs_gridinit(&grid, "r", worldSize, 1);
  call_descinit(descA.data(), globalRows, colsA, maxRowsPerRank, colsA, 0, 0, grid, maxRowsPerRank,
                &info);
  call_descinit(descB.data(), globalRows, colsB, maxRowsPerRank, colsB, 0, 0, grid, maxRowsPerRank,
                &info);
  call_descinit(descC.data(), colsA, colsB, blacsBlockSize, blacsBlockSize, 0, 0, grid, maxRowsC,
                &info);

  call_pgemm('C', 'N', colsA, colsB, globalRows, 1.0, A.data(), 1, 1, descA.data(), B.data(), 1, 1,
             descB.data(), 0.0, C.data(), 1, 1, descC.data());

  START_TIMING("ScaLAPACK");
  for (int r = 0; r < numRepeats; ++r) {
    SCOPED_TIMING("multiply");
    call_pgemm('C', 'N', colsA, colsB, globalRows, 1.0, A.data(), 1, 1, descA.data(), B.data(), 1,
               1, descB.data(), 0.0, C.data(), 1, 1, descC.data());
  }
  STOP_TIMING("ScaLAPACK");

  Cblacs_gridexit(grid);
  Cfree_blacs_system_handle(blacsCtx);
}

int main(int argc, char** argv) {
  spla::MPIInitHandle initHandle(argc, argv, true);

  int repeats = 100;
  int colsA = 5;
  int colsB = 5;
  int rows = 5;
  int numThreads = 6;
  int blacsBlockSize = 256;
  std::string procName;
  std::string typeName;
  std::string outputFileName;
  int lengthTarget = 256;

  CLI::App app{"spla benchmark"};
  app.add_option("-r", repeats, "Number of repeats")->default_val("100");
  app.add_option("-o", outputFileName, "Output file name")->default_val("timers.json");
  app.add_option("-n", colsB, "Number of columns in C")->required();
  app.add_option("-m", colsA, "Number of rows in C")->required();
  app.add_option("-k", rows, "Number of rows in A and B")->required();
  app.add_option("-t,--threads", numThreads, "Number of threads")->required();
  app.add_option("-b,--blocksize", blacsBlockSize, "ScaLAPACK block size of C")->required();
  app.add_set("-p", procName, std::set<std::string>{"cpu", "gpu"}, "Processing unit")->required();
  app.add_option("-l", lengthTarget, "Length target")->default_val("1024");
  app.add_set("--type", typeName, std::set<std::string>{"scalar", "complex"}, "Data type")
      ->default_val("complex");
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  SplaProcessingUnit pu =
      procName == "cpu" ? SplaProcessingUnit::SPLA_PU_HOST : SplaProcessingUnit::SPLA_PU_GPU;
  spla::Context ctx(pu);
  ctx.set_tile_size_host(lengthTarget);
  ctx.set_num_threads(numThreads);
  ctx.set_tile_size_gpu(4096);

  spla::AllocatorCollection allocators;

  if (ctx.processing_unit() == SPLA_PU_GPU) {
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
    if (typeName == "scalar")
      run_gemm<double>(allocators.pinned(), ctx, rows, colsA, colsB, numThreads, blacsBlockSize,
                       repeats);
    else
      run_gemm<std::complex<double>>(allocators.pinned(), ctx, rows, colsA, colsB, numThreads,
                                     blacsBlockSize, repeats);
#else
    throw spla::GPUSupportError();
#endif
  } else {
    if (typeName == "scalar")
      run_gemm<double>(allocators.host(), ctx, rows, colsA, colsB, numThreads, blacsBlockSize,
                       repeats);
    else
      run_gemm<std::complex<double>>(allocators.host(), ctx, rows, colsA, colsB, numThreads,
                                     blacsBlockSize, repeats);
  }

  int worldRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  if (worldRank == 0) {
    std::cout << "Context memory usage:" << std::endl;
    std::cout << ctx.allocated_memory_host() / 1000000 << " MB host memory" << std::endl;
    std::cout << ctx.allocated_memory_pinned() / 1000000 << " MB pinned host memory" << std::endl;
    std::cout << ctx.allocated_memory_gpu() / 1000000 << " MB gpu memory" << std::endl;
    auto result = spla::timing::GlobalTimer.process();
    std::cout << result.print() << std::endl;
    std::ofstream file(outputFileName);
    file << result.json();
  }

  return 0;
}
