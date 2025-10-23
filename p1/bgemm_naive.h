#pragma once
#include <cstddef>

// Public GEMM entry (row-major):
// C[M x N] += A[M x K] * B[K x N]
// lda/ldb/ldc are leading dimensions (for row-major they are N, N, N typically)
void gemm_blocked(
    int M, int N, int K,
    const double* A, int lda,
    const double* B, int ldb,
    double* C, int ldc);

// Optional: set/tune block sizes at runtime (falls back to defaults if not called)
void set_block_sizes(int NC, int KC, int MC, int NR, int MR);
