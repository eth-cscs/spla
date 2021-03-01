#include <mpi.h>
#include <unistd.h>  // for MPI debugging

#include "gtest/gtest.h"
#include "gtest_mpi/gtest_mpi.hpp"

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

int main(int argc, char* argv[]) {
  // Initialize MPI before any call to gtest_mpi
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  // if (mpi_world_rank() == 0) {
  //   std::cout << "PID = " << getpid() << std::endl;
  //   bool wait = true;
  //   while (wait) {
  //     sleep(5);
  //   }
  // }

  // Intialize google test
  ::testing::InitGoogleTest(&argc, argv);

  // Add a test envirnment, which will initialize a test communicator
  // (a duplicate of MPI_COMM_WORLD)
  ::testing::AddGlobalTestEnvironment(new gtest_mpi::MPITestEnvironment());

  auto& test_listeners = ::testing::UnitTest::GetInstance()->listeners();

  // Remove default listener and replace with the custom MPI listener
  delete test_listeners.Release(test_listeners.default_result_printer());
  test_listeners.Append(new gtest_mpi::PrettyMPIUnitTestResultPrinter());

  // run tests
  auto exit_code = RUN_ALL_TESTS();

  // Finalize MPI before exiting
  MPI_Finalize();

  return exit_code;
}
