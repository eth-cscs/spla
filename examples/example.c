#include <mpi.h>
#include <math.h>
#include <stdlib.h>

#include "spla/spla.h"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int m = 100;
  int n = 100;
  int k_local = 100;

  int block_size = 256;
  int proc_grid_rows = sqrt(world_size);
  int proc_grid_cols = world_size / proc_grid_rows;

  double* A = (double*) malloc(sizeof(double) * m * k_local);
  double* B = (double*) malloc(sizeof(double) * n * k_local);
  double* C = (double*) malloc(sizeof(double) * m * n); // Allocate full C for simplicity

  int lda = k_local;
  int ldb = k_local;
  int ldc = m;

  // Create context, which holds any resources SPLA will require, allowing reuse between functions
  // calls. The given processing unit will be used for any computations.
  SplaContext ctx;
  spla_ctx_create(&ctx, SPLA_PU_HOST);

  // Create matrix distribution for C
  SplaMatrixDistribution c_dist;
  spla_mat_dis_create_block_cyclic(&c_dist, MPI_COMM_WORLD, 'R', proc_grid_rows, proc_grid_cols,
                                   block_size, block_size);
  // This is mostly equivalent to the following ScaLAPACK calls combined:
  /*
  int info = 0;
  int rsrc = 0;
  int csrc = 0;
  int blacs_ctx = Csys2blacs_handle(MPI_COMM_WORLD);
  Cblacs_gridinit(&blacs_ctx, 'R', proc_grid_rows, proc_grid_cols);
  int desc[9];
  descinit_(desc, &m, &n, &block_size, &block_size, &rsrc, &csrc, &blacs_ctx, &ldc,
            &info);
  */

  double alpha = 1.0;
  double beta = 0.0;

  // Compute parallel stripe-stripe-block matrix multiplication. To describe the stripe distribution
  // of matrices A and B, only the local k dimension is required.
  spla_pdgemm_ssb(m, n, k_local, SPLA_OP_TRANSPOSE, alpha, A, lda, B, ldb, beta, C, ldc, 0, 0,
                  c_dist, ctx);

  spla_ctx_destroy(&ctx);
  spla_mat_dis_destroy(&c_dist);
  free(A);
  free(B);
  free(C);
  MPI_Finalize();
  return 0;
}
