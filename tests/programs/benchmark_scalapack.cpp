#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <complex>
#include <array>
#include <mpi.h>
#include "spla/matrix_distribution.hpp"
#include "spla/spla.hpp"
#include "mpi_util/mpi_init_handle.hpp"
#include "CLI/CLI.hpp"
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

void descinit_(int* desc, const int* m, const int* n, const int* mb, const int* nb, const int* irsrc,
              const int* icsrc, const int* ictxt, const int* lld, int* info);

void pdgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA, double* A,
             int* IA, int* JA, int* DESCA, double* B, int* IB, int* JB, int* DESCB, double* BETA,
             double* C, int* IC, int* JC, int* DESCC);

void pdgemr2d_(int* m, int* n, double* a, int* ia, int* ja, int* desca, double* b,
              int* ib, int* jb, int* descb, int* ictxt);

int numroc_(int const& n, int const& nb, int const& iproc, int const& isproc, int const& nprocs);

}

static auto call_descinit(int* desc, int m, int n, int mb, int nb, int irsrc, int icsrc, int ictxt,
                          int lld, int* info) -> void {
  descinit_(desc, &m, &n, &mb, &nb, &irsrc, &icsrc, &ictxt, &lld, info);
}

static auto call_pdgemm(char TRANSA, char TRANSB, int M, int N, int K, double ALPHA, double* A,
                        int IA, int JA, int* DESCA, double* B, int IB, int JB, int* DESCB,
                        double BETA, double* C, int IC, int JC, int* DESCC) -> void {
  pdgemm_(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &IA, &JA, DESCA, B, &IB, &JB, DESCB, &BETA, C,
          &IC, &JC, DESCC);
}

static void call_pdgemr2d(int m, int n, double* a, int ia, int ja, int* desca, double* b,
              int ib, int jb, int* descb, int ictxt) {
  pdgemr2d_(&m, &n, a, &ia, &ja, desca, b, &ib, &jb, descb, &ictxt);
}

void run_gemm(int globalRows, int colsA, int colsB, int numThreads, int blacsBlockSize, int numRepeats) {

  int worldRank, worldSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  const int maxRowsPerRank = (globalRows + worldSize - 1) / worldSize;
  // const int globalRows = worldSize * localNumRows;
  const int localNumRows = std::min(globalRows - worldRank * maxRowsPerRank, maxRowsPerRank);

  std::vector<double> A(maxRowsPerRank * colsA);
  std::vector<double> B(maxRowsPerRank * colsB);
  std::vector<double> C(colsA * colsB);

  rt_graph::Timer timer;

  auto arrayDesc = spla::MatrixDistribution::create_blacs_block_cyclic(
      MPI_COMM_WORLD, 'R', worldSize, 1, blacsBlockSize, blacsBlockSize);
  spla::Context ctx(SPLA_PU_HOST);
  ctx.set_num_threads(numThreads);

  // run once to warm up
  spla::gemm_ssb(colsA, colsB, localNumRows, decltype(A)::value_type(1.0), A.data(), localNumRows,
                 B.data(), localNumRows, decltype(C)::value_type(0.0), C.data(), colsA, 0, 0,
                 arrayDesc, ctx);

  START_TIMING("spla");
  for (int r = 0; r < numRepeats; ++r) {
    SCOPED_TIMING("multiply");
    spla::gemm_ssb(colsA, colsB, localNumRows, decltype(A)::value_type(1.0), A.data(), localNumRows,
                   B.data(), localNumRows, decltype(C)::value_type(0.0), C.data(), colsA, 0, 0,
                   arrayDesc, ctx);
  }
  STOP_TIMING("spla");

  std::array<int, 9> descA{0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::array<int, 9> descB{0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::array<int, 9> descC{0, 0, 0, 0, 0, 0, 0, 0, 0};
  int info;
  int grid = 0;
  int blacsCtx = Csys2blacs_handle(MPI_COMM_WORLD);
  Cblacs_gridinit(&grid, "r", worldSize, 1);
  call_descinit(descA.data(), globalRows, colsA, maxRowsPerRank, colsA, 0, 0, grid,
                maxRowsPerRank, &info);
  call_descinit(descB.data(), globalRows, colsB, maxRowsPerRank, colsB, 0, 0, grid,
                maxRowsPerRank, &info);
  call_descinit(descC.data(), colsA, colsB, blacsBlockSize, blacsBlockSize, 0, 0, grid, colsA,
                &info);

  call_pdgemm('C', 'N', colsA, colsB, globalRows, 1.0, A.data(), 1, 1, descA.data(), B.data(), 1, 1,
              descB.data(), 0.0, C.data(), 1, 1, descC.data());

  START_TIMING("ScaLAPACK");
  for (int r = 0; r < numRepeats; ++r) {
    SCOPED_TIMING("multiply");
    call_pdgemm('C', 'N', colsA, colsB, globalRows, 1.0, A.data(), 1, 1, descA.data(), B.data(), 1,
                1, descB.data(), 0.0, C.data(), 1, 1, descC.data());
  }
  STOP_TIMING("ScaLAPACK");

  Cblacs_gridexit(grid);
  Cfree_blacs_system_handle(blacsCtx);

  if (worldRank == 0) std::cout << spla::timing::GlobalTimer.process().print() << std::endl;
}

int main(int argc, char** argv) {
  spla::MPIInitHandle initHandle(argc, argv, true);


  int repeats = 100;
  int colsA = 5;
  int colsB = 5;
  int rows = 5;
  int numThreads = 6;
  int blacsBlockSize = 64;

  CLI::App app{"spla benchmark"};
  app.add_option("-r", repeats, "Number of repeats")->default_val("100");
  app.add_option("-n", colsB, "Number of columns in C")->required();
  app.add_option("-m", colsA, "Number of rows in C")->required();
  app.add_option("-k", rows, "Number of rows in A and B")->required();
  app.add_option("-t,--threads", numThreads, "Number of threads")->required();
  app.add_option("-b,--blocksize", blacsBlockSize, "ScaLAPACK block size of C")->required();
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }
  run_gemm(rows, colsA, colsB, numThreads, blacsBlockSize, repeats);

  return 0;
}
