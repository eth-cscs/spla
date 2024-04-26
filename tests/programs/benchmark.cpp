#include <mpi.h>

#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "CLI/CLI.hpp"
#include "memory/buffer.hpp"
#include "memory/allocator_collection.hpp"
#include "memory/pool_allocator.hpp"
#include "mpi_util/mpi_init_handle.hpp"
#include "spla/context.hpp"
#include "spla/exceptions.hpp"
#include "spla/matrix_distribution.hpp"
#include "spla/spla.hpp"
#include "spla/types.h"
#include "timing/rt_graph.hpp"
#include "timing/timing.hpp"

template <typename T, spla::MemLoc LOCATION>
void run_pgemm_ssb(const std::shared_ptr<spla::Allocator<LOCATION>>& allocator, spla::Context& ctx,
                   int m, int n, int k, int blacsBlockSize, int numRepeats) {
  int worldRank, worldSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  const int blockRows = std::sqrt(worldSize);
  const int blockCols = worldSize / blockRows;

  const int maxRowsPerRank = (k + worldSize - 1) / worldSize;
  const int localNumRows = std::max(std::min(k - worldRank * maxRowsPerRank, maxRowsPerRank), 0);
  const int maxRowsC = (m / (blacsBlockSize * blockRows) + 1) * blacsBlockSize;
  const int maxColsC = (n / (blacsBlockSize * blockCols) + 1) * blacsBlockSize;

  spla::Buffer<T, LOCATION> A(allocator, maxRowsPerRank * m);
  spla::Buffer<T, LOCATION> B(allocator, maxRowsPerRank * n);
  spla::Buffer<T, LOCATION> C(allocator, maxRowsC * maxColsC);

  const T alpha = 2.0;
  const T beta = 0.0;

  rt_graph::Timer timer;

  auto arrayDesc = spla::MatrixDistribution::create_blacs_block_cyclic(
      MPI_COMM_WORLD, 'R', blockRows, blockCols, blacsBlockSize, blacsBlockSize);

  // run once to warm up
  spla::pgemm_ssb(m, n, localNumRows, SPLA_OP_CONJ_TRANSPOSE, alpha, A.size() ? A.data() : nullptr,
                  localNumRows, B.size() ? B.data() : nullptr, localNumRows, beta,
                  C.size() ? C.data() : nullptr, maxRowsC, 0, 0, arrayDesc, ctx);

  START_TIMING("spla");
  for (int r = 0; r < numRepeats; ++r) {
    SCOPED_TIMING("pgemm_ssb");
    spla::pgemm_ssb(m, n, localNumRows, SPLA_OP_CONJ_TRANSPOSE, alpha,
                    A.size() ? A.data() : nullptr, localNumRows, B.size() ? B.data() : nullptr,
                    localNumRows, beta, C.size() ? C.data() : nullptr, maxRowsC, 0, 0, arrayDesc,
                    ctx);
  }
  STOP_TIMING("spla");
}

template <typename T, spla::MemLoc LOCATION>
void run_pgemm_sbs(const std::shared_ptr<spla::Allocator<LOCATION>>& allocator, spla::Context& ctx,
                   int m, int n, int k, int blacsBlockSize, int numRepeats) {
  int worldRank, worldSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  const int blockRows = std::sqrt(worldSize);
  const int blockCols = worldSize / blockRows;

  const int maxRowsPerRank = (m + worldSize - 1) / worldSize;
  const int localNumRows = std::max(std::min(m - worldRank * maxRowsPerRank, maxRowsPerRank), 0);
  const int maxRowsB = (k / (blacsBlockSize * blockRows) + 1) * blacsBlockSize;
  const int maxColsB = (n / (blacsBlockSize * blockCols) + 1) * blacsBlockSize;

  spla::Buffer<T, LOCATION> A(allocator, maxRowsPerRank * k);
  spla::Buffer<T, LOCATION> B(allocator, maxRowsB * maxColsB);
  spla::Buffer<T, LOCATION> C(allocator, maxRowsPerRank * n);

  const T alpha = 2.0;
  const T beta = 0.0;

  rt_graph::Timer timer;

  auto arrayDesc = spla::MatrixDistribution::create_blacs_block_cyclic(
      MPI_COMM_WORLD, 'R', blockRows, blockCols, blacsBlockSize, blacsBlockSize);

  // run once to warm up
  spla::pgemm_sbs(localNumRows, n, k, alpha, A.size() ? A.data() : nullptr, localNumRows,
                  B.size() ? B.data() : nullptr, maxRowsB, 0, 0, arrayDesc, beta,
                  C.size() ? C.data() : nullptr, localNumRows, ctx);

  START_TIMING("spla");
  for (int r = 0; r < numRepeats; ++r) {
    SCOPED_TIMING("pgemm_sbs");
    spla::pgemm_sbs(localNumRows, n, k, alpha, A.size() ? A.data() : nullptr, localNumRows,
                    B.size() ? B.data() : nullptr, maxRowsB, 0, 0, arrayDesc, beta,
                    C.size() ? C.data() : nullptr, localNumRows, ctx);
  }
  STOP_TIMING("spla");
}

int main(int argc, char** argv) {
  spla::MPIInitHandle initHandle(argc, argv, true);

  int worldRank, worldSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  int repeats = 100;
  int m = 5;
  int n = 5;
  int k = 5;
  int blacsBlockSize = 64;
  int lengthTarget = 256;
  std::string procName;
  std::string outputFileName;
  std::string typeName;
  std::string funcName;

  CLI::App app{"spla benchmark"};
  app.add_option("-r", repeats, "Number of repeats")->default_val("100");
  app.add_option("-l", lengthTarget, "Length target")->default_val("1024");
  app.add_option("-n", n, "Number of columns in C")->required();
  app.add_option("-m", m, "Number of rows in C")->required();
  app.add_option("-k", k, "Number of rows in A and B")->required();
  app.add_option("-o", outputFileName, "Output file name")->default_val("timers.json");
  app.add_option("--type", typeName, "Data type")
      ->check(CLI::IsMember({"scalar", "complex"}))
      ->default_val("complex");
  app.add_option("-f, --func", funcName, "Function to benchmark")
      ->check(CLI::IsMember({"ssb", "sbs"}))
      ->default_val("ssb");
  app.add_option("-b,--blocksize", blacsBlockSize, "ScaLAPACK block size of C")->required();
  app.add_option("-p", procName, "Processing unit")
      ->check(CLI::IsMember({"cpu", "gpu", "gpu-gpu"}))
      ->required();

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  SplaProcessingUnit pu =
      procName == "cpu" ? SplaProcessingUnit::SPLA_PU_HOST : SplaProcessingUnit::SPLA_PU_GPU;
  spla::Context ctx(pu);
  ctx.set_tile_size_host(lengthTarget);
  ctx.set_tile_size_gpu(4096);

  if (worldRank == 0) {
    std::cout << "function = " << funcName << std::endl;
    std::cout << "m = " << m << std::endl;
    std::cout << "n = " << n << std::endl;
    std::cout << "k = " << k << std::endl;
    std::cout << "tile length = " << lengthTarget << std::endl;
    std::cout << "block size = " << blacsBlockSize << std::endl;
    std::cout << "repeats = " << repeats << std::endl;
    std::cout << "proc = " << procName << std::endl;
    std::cout << "type = " << typeName << std::endl;
  }

  spla::AllocatorCollection allocators;

  if (funcName == "ssb") {
    if (procName == "cpu") {
      if (typeName == "scalar")
        run_pgemm_ssb<double>(allocators.host(), ctx, m, n, k, blacsBlockSize, repeats);
      else
        run_pgemm_ssb<std::complex<double>>(allocators.host(), ctx, m, n, k, blacsBlockSize,
                                            repeats);
    }
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
    else if (procName == "gpu") {
      if (typeName == "scalar")
        run_pgemm_ssb<double>(allocators.pinned(),ctx, m, n, k, blacsBlockSize, repeats);
      else
        run_pgemm_ssb<std::complex<double>>(allocators.pinned(),ctx, m, n, k, blacsBlockSize,
                                                                   repeats);
    } else if (procName == "gpu-gpu") {
      if (typeName == "scalar")
        run_pgemm_ssb<double>(allocators.gpu(),ctx, m, n, k, blacsBlockSize, repeats);
      else
        run_pgemm_ssb<std::complex<double>>(allocators.gpu(),ctx, m, n, k, blacsBlockSize,
                                                                repeats);
    }
#else
    else {
      throw spla::GPUSupportError();
    }
#endif
  } else {
    if (procName == "cpu") {
      if (typeName == "scalar")
        run_pgemm_sbs<double>(allocators.host(), ctx, m, n, k, blacsBlockSize, repeats);
      else
        run_pgemm_sbs<std::complex<double>>(allocators.host(), ctx, m, n, k, blacsBlockSize,
                                            repeats);
    }
#if defined(SPLA_CUDA) || defined(SPLA_ROCM)
    else if (procName == "gpu") {
      if (typeName == "scalar")
        run_pgemm_sbs<double>(allocators.pinned(), ctx, m, n, k, blacsBlockSize, repeats);
      else
        run_pgemm_sbs<std::complex<double>>(allocators.pinned(), ctx, m, n, k, blacsBlockSize,
                                            repeats);
    } else if (procName == "gpu-gpu") {
      if (typeName == "scalar")
        run_pgemm_sbs<double>(allocators.gpu(), ctx, m, n, k, blacsBlockSize, repeats);
      else
        run_pgemm_sbs<std::complex<double>>(allocators.gpu(), ctx, m, n, k, blacsBlockSize,
                                            repeats);
    }
#else
    else {
      throw spla::GPUSupportError();
    }
#endif
  }

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
