#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <complex>
#include <mpi.h>
#include "spla/spla.hpp"
#include "mpi_util/mpi_init_handle.hpp"

// void print_matrix(const double* A, const int rows, const int cols) {
//   int rank, size;
//   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//   MPI_Comm_size(MPI_COMM_WORLD, &size);
//   for (int r = 0; r < rows; ++r) {
//     for (int c = 0; c < cols; ++c) {
//       std::cout << A[r + c * rows] << ", ";
//     }
//     std::cout << std::endl;
//   }
// }

auto print_matrix(const double* A, const int rows, const int cols, const std::string& label)
    -> void {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  for (int r = 0; r < size; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (r != rank) continue;
    std::stringstream stream;
    // stream << std::scientific;
    stream << std::fixed;
    stream << std::setprecision(1);
    stream << " -------------------- " << std::endl;
    stream << "Rank = " << rank << ", " << label << ":" << std::endl;
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        stream << std::setw(8) << std::right << A[r + c * rows];
      }
      stream << std::endl;
    }
    stream << " -------------------- " << std::endl;
    std::cout << stream.str();
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

template<typename T>
auto print_matrix(const std::complex<T>* A, const int rows, const int cols, const std::string& label)
    -> void {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  for (int r = 0; r < size; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (r != rank) continue;
    std::stringstream stream;
    // stream << std::scientific;
    stream << std::fixed;
    stream << std::setprecision(1);
    stream << " -------------------- " << std::endl;
    stream << "Rank = " << rank << ", " << label << ":" << std::endl;
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        stream << std::setw(8) << std::right << A[r + c * rows].real();
        if (std::signbit(A[r + c * rows].imag())) {
          stream << " - ";
        } else {
          stream << " + ";
        }
        stream << std::left << std::setw(6) << std::abs(A[r + c * rows].imag());
      }
      stream << std::endl;
    }
    stream << " -------------------- " << std::endl;
    std::cout << stream.str();
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

void run_gemm() {
  const int colsA = 4;
  const int colsB = 5;

  int worldRank, worldSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  // const int maxLocalRows = rows / worldSize;
  // const int localRows =
  //     worldRank == worldSize - 1 ? rows - (worldSize - 1) * maxLocalRows : maxLocalRows;
  const int localRows = 5;
  const int maxLocalRows = localRows;
  // int rows = worldSize * localRows;

  std::vector<double> A(localRows * colsA);
  std::vector<double> B(localRows * colsB);
  std::vector<double> C(colsA * colsB);
  // for(auto& val: C) {
  //   val = 1.0;
  // }

  double count = 1.0 + worldRank * maxLocalRows * colsA;
  for (auto& val : A) {
    val = count;
    count += 1.0;
  }
  count = 1.0 + worldRank * maxLocalRows * colsB;
  for (auto& val : B) {
    val = count;
    count += 1.0;
  }
  for (auto& val : C) {
    val = 1;
  }
  print_matrix(A.data(), localRows, colsA, "A");

  print_matrix(B.data(), localRows, colsB, "B");

  auto descC = spla::MatrixDistribution::create_blacs_block_cyclic(MPI_COMM_WORLD, 'R', worldSize, 1, 1, 1);
  // spla::MatrixDistribution descC{MPI_COMM_WORLD};
  spla::Context ctx(SPLA_PU_HOST);
  spla::pgemm_ssb(colsA - 1, colsB, localRows, SPLA_OP_CONJ_TRANSPOSE, decltype(A)::value_type(1.0), A.data(), localRows,
                  B.data(), localRows, decltype(C)::value_type(2.0), C.data(), colsA, 1, 0, descC,
                  ctx);

  print_matrix(C.data(), colsA, colsB, "C");
}

int main(int argc, char** argv) {
  spla::MPIInitHandle initHandle(argc, argv, true);
  run_gemm();

  return 0;
}
