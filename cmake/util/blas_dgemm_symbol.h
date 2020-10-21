#ifndef SPLA_BLAS_GEMM_SYMBOL_H
extern "C" {

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

void cblas_dgemm(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
                 int M, int N, int K, double alpha, const double *A, int lda, const double *B,
                 int ldb, double beta, double *C, int ldc);
}

#endif
