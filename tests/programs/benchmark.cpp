#include <mpi.h>
#include <complex>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "CLI/CLI.hpp"
#include "mpi_util/mpi_init_handle.hpp"
#include "spla/context.hpp"
#include "spla/matrix_distribution.hpp"
#include "spla/spla.hpp"
#include "timing/rt_graph.hpp"
#include "timing/timing.hpp"

void run_gemm(spla::Context& ctx, int globalRows, int colsA, int colsB, int blacsBlockSize,
              int numRepeats) {
  int worldRank, worldSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  const int maxRowsPerRank = (globalRows + worldSize - 1) / worldSize;
  const int localNumRows = std::min(globalRows - worldRank * maxRowsPerRank, maxRowsPerRank);
  const int maxRowsC = (colsA / (blacsBlockSize * worldSize) + 1) * blacsBlockSize;

  std::vector<double> A(maxRowsPerRank * colsA);
  std::vector<double> B(maxRowsPerRank * colsB);
  std::vector<double> C(maxRowsC * colsB);

  for(std::size_t i = 0; i < A.size(); ++i) {
    A[i] = i;
  }
  for(std::size_t i = 0; i < B.size(); ++i) {
    B[i] = i;
  }

  rt_graph::Timer timer;

  auto arrayDesc = spla::MatrixDistribution::create_blacs_block_cyclic(
      MPI_COMM_WORLD, 'R', worldSize, 1, blacsBlockSize, blacsBlockSize);

  // run once to warm up
  spla::pgemm_ssb(colsA, colsB, localNumRows, SPLA_OP_CONJ_TRANSPOSE, decltype(A)::value_type(1.0),
                  A.data(), localNumRows, B.data(), localNumRows, decltype(C)::value_type(0.0),
                  C.data(), maxRowsC, 0, 0, arrayDesc, ctx);

  START_TIMING("spla");
  for (int r = 0; r < numRepeats; ++r) {
    SCOPED_TIMING("multiply");
    spla::pgemm_ssb(colsA, colsB, localNumRows, SPLA_OP_CONJ_TRANSPOSE,
                    decltype(A)::value_type(1.0), A.data(), localNumRows, B.data(), localNumRows,
                    decltype(C)::value_type(0.0), C.data(), maxRowsC, 0, 0, arrayDesc, ctx);
  }
  STOP_TIMING("spla");
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
  int lengthTarget = 256;
  std::string procName;

  CLI::App app{"spla benchmark"};
  app.add_option("-r", repeats, "Number of repeats")->default_val("100");
  app.add_option("-l", lengthTarget, "Length target")->default_val("256");
  app.add_option("-n", colsB, "Number of columns in C")->required();
  app.add_option("-m", colsA, "Number of rows in C")->required();
  app.add_option("-k", rows, "Number of rows in A and B")->required();
  app.add_option("-t,--threads", numThreads, "Number of threads")->required();
  app.add_option("-b,--blocksize", blacsBlockSize, "ScaLAPACK block size of C")->required();
  app.add_set("-p", procName, std::set<std::string>{"cpu", "gpu"}, "Processing unit")
      ->required();
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }
  SplaProcessingUnit pu = procName == "cpu" ? SplaProcessingUnit::SPLA_PU_HOST : SplaProcessingUnit::SPLA_PU_GPU;

  spla::Context ctx(pu);
  ctx.set_tile_length_target(lengthTarget);
  ctx.set_num_threads(numThreads);

  run_gemm(ctx, rows, colsA, colsB, blacsBlockSize, repeats);

  return 0;
}
