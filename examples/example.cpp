#include <vector>
#include <cmath>
#include <mpi.h>

#include "spla/spla.hpp"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int m = 100;
  int n = 100;
  int k_local = 100;

  int block_size = 256;
  int proc_grid_rows = std::sqrt(world_size);
  int proc_grid_cols = world_size / proc_grid_rows;

  std::vector<double> A(m * k_local);
  std::vector<double> B(n * k_local);
  std::vector<double> C(m * n); // Allocate full C for simplicity

  int lda = k_local;
  int ldb = k_local;
  int ldc = m;

  // Create context, which holds any resources SPLA will require, allowing reuse between functions
  // calls. The given processing unit will be used for any computations.
  spla::Context ctx(SPLA_PU_HOST);

  // Create matrix distribution for C
  auto c_dist = spla::MatrixDistribution::create_blacs_block_cyclic(
      MPI_COMM_WORLD, 'R', proc_grid_rows, proc_grid_cols, block_size, block_size);
  // This is mostly equivalent to the following ScaLAPACK calls combined:
  /*
  int info = 0;
  int rsrc = 0;
  int csrc = 0;
  int blacs_ctx = Csys2blacs_handle(MPI_COMM_WORLD);
  Cblacs_gridinit(&blacs_ctx, 'R', proc_grid_rows, proc_grid_cols);
  int desc[9];
  descinit_(desc.data(), &m, &n, &block_size, &block_size, &rsrc, &csrc, &blacs_ctx, &ldc,
                &info);
  */

  double alpha = 1.0;
  double beta = 0.0;

  // Compute parallel stripe-stripe-block matrix multiplication. To describe the stripe distribution
  // of matrices A and B, only the local k dimension is required.
  spla::pgemm_ssb(m, n, k_local, SPLA_OP_TRANSPOSE, alpha, A.data(), lda, B.data(), ldb, beta,
                  C.data(), ldc, 0, 0, c_dist, ctx);

  MPI_Finalize();
  return 0;
}
