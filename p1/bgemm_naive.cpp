#include <algorithm>
#include "bgemm_naive.h"

// Default block sizes (tune to your machine)
// Outer (cache) blocks:
static int sNC = 4096; // big N slab to reuse B in LLC (36 MiB L3 ⇒ kc*nc*8 fits easily)
static int sKC = 256;  // K chunk sized for L2 when paired with mc
static int sMC = 192;  // M chunk so A(mc×kc) + B(kc×nc) don't thrash L2 per-core

// Micro (register) blocks (AVX-512, FP64 has 8 lanes)
static int sNR = 32;   // along N; keeps B-row and C tile small & contiguous
static int sMR = 8;    // along M; matches 8 FP64 lanes

void set_block_sizes(int NC, int KC, int MC, int NR, int MR) {
    if (NC > 0) sNC = NC;
    if (KC > 0) sKC = KC;
    if (MC > 0) sMC = MC;
    if (NR > 0) sNR = NR;
    if (MR > 0) sMR = MR;
}

// Naive micro-kernel: computes
// C_sub[ mr x nr ] += A_panel[ mr x kc ] * B_panel[ kc x nr ]
// All pointers are row-major views into the big matrices (no packing).
static inline void microkernel_naive(
    int mr, int nr, int kc,
    const double* A_base, int lda,   // points to A[ic+ir, pc]
    const double* B_base, int ldb,   // points to B[pc, jc+jr]
    double*       C_base, int ldc)   // points to C[ic+ir, jc+jr]
{
    // Order: p (kc) → ii (mr) → jj (nr)
    for (int p = 0; p < kc; ++p) {
        const double* Brow = B_base + p*ldb;  // B[p, :]
        for (int ii = 0; ii < mr; ++ii) {
            const double a_val = A_base[ii*lda + p];
            double* Crow = C_base + ii*ldc;
            for (int jj = 0; jj < nr; ++jj) {
                Crow[jj] += a_val * Brow[jj];
            }
        }
    }
}

void gemm_blocked(
    int M, int N, int K,
    const double* A, int lda,
    const double* B, int ldb,
    double* C, int ldc)
{
    // Six nested loops (OpenBLAS/BLIS-like structure):
    // 1) jc (N big block)
    for (int jc = 0; jc < N; jc += sNC) {
        const int nc = std::min(sNC, N - jc);

        // 2) pc (K big block)
        for (int pc = 0; pc < K; pc += sKC) {
            const int kc = std::min(sKC, K - pc);

            // 3) ic (M big block)
            for (int ic = 0; ic < M; ic += sMC) {
                const int mc = std::min(sMC, M - ic);

                // 4) jr (N micro/register tile)
                for (int jr = 0; jr < nc; jr += sNR) {
                    const int nr = std::min(sNR, nc - jr);

                    // 5) ir (M micro/register tile)
                    for (int ir = 0; ir < mc; ir += sMR) {
                        const int mr = std::min(sMR, mc - ir);

                        // 6) (micro-kernel iterates over kc internally)
                        const double* A_block = A + (ic + ir) * lda + pc;   // A[ic+ir, pc]
                        const double* B_block = B + (pc) * ldb + (jc + jr); // B[pc, jc+jr]
                        double*       C_block = C + (ic + ir) * ldc + (jc + jr); // C[ic+ir, jc+jr]

                        microkernel_naive(mr, nr, kc, A_block, lda, B_block, ldb, C_block, ldc);
                    }
                }
            }
        }
    }
}
