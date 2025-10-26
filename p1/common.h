#pragma once
#include <vector>
#include <immintrin.h>


template <typename T>
inline void microkernel_packing(
    int MR, int KC, int NR, // the padded size, for the consistence of microkernel
    int mr, int nr, // the actual size for the result matrix
    const T* A_base, 
    const T* B_base, 
    T*       C_base, int ldc)
{
    std::vector<T> C_tmp_vec(MR * NR, T{});
    T* C_tmp_base = C_tmp_vec.data();

    for (int k = 0; k < KC; ++k) {
        const T* A_col = A_base + k * MR;
        const T* B_row = B_base + k * NR;
        for (int i = 0; i < MR; ++i) {
            T A_val = A_col[i];
            T* C_tmp_base_row = C_tmp_base + i * NR;
            for (int j = 0; j < NR; ++j) {
                C_tmp_base_row[j] += A_val * B_row[j];
            }
        }
    }

    for (int i = 0; i < mr; ++i) {
        T* C_base_row = C_base + i * ldc;
        T* C_tmp_base_row = C_tmp_base + i * NR;
        for (int j = 0; j < nr; ++j) {
            C_base_row[j] += C_tmp_base_row[j];
        }
    }
}

//TODO: padding KC dimension as well if kc<sKC
template <typename T>
void packing_A(const T* A, int lda, int mc, int kc, int KC, int MR, T* Ap){
    // pack A [MC, KC] into Ap, padding is needed
    for (int im = 0; im < mc; im += MR) {
        int MR_eff = std::min(mc - im, MR);
        const T* A_mr = A + im * lda; // address of a block
        for (int ik = 0; ik < kc; ++ik) {
            for (int imr = 0; imr < MR_eff; ++imr) {
                Ap[imr] = A_mr[imr * lda + ik];
            }
            if (MR_eff < MR) {
                for (int imr = MR_eff; imr < MR; ++imr) {
                    Ap[imr] = T{};
                }
            }
            Ap += MR;
        }
        if (kc < KC) {
            for (int ik = kc; ik < KC; ++ik) {
                for (int imr = 0; imr < MR; ++imr) {
                    Ap[imr] = T{};
                }
                Ap += MR;
            }            
        }

    }
}

//TODO: padding KC dimension as well if kc<sKC
template <typename T>
void packing_B(const T* B, int ldb, int kc, int KC, int nc, int NR, T* Bp){
    // pack B [KC, NC] into Bp, padding is needed
    for (int in = 0; in < nc; in += NR) {
        int NR_eff = std::min(nc - in, NR);
        const T* B_nr = B + in; // address of a block
        for (int ik = 0; ik < kc; ++ik) {
            for (int inr = 0; inr < NR_eff; ++inr) {
                Bp[inr] = B_nr[inr + ik * ldb];
            }
            if (NR_eff < NR) {
                for (int inr = NR_eff; inr < NR; ++inr) {
                    Bp[inr] = T{};
                }
            }
            Bp += NR;
        }
        if (kc < KC) {
            for (int ik = kc; ik < KC; ++ik) {
                for (int inr = 0; inr < NR; ++inr) {
                    Bp[inr] = T{};
                }
                Bp += NR;
            }            
        }
    }
}


template <typename T>
inline void microkernel_simd(
    int mr, int nr, int kc,
    const T* A_base, int lda,
    const T* B_base, int ldb,
    T*       C_base, int ldc)
{
    for (int p = 0; p < kc; ++p) {
        const T* Brow = B_base + p * ldb;
        for (int ii = 0; ii < mr; ++ii) {
            const T a_val = A_base[ii * lda + p];
            T* Crow = C_base + ii * ldc;
            for (int jj = 0; jj < nr; ++jj) {
                Crow[jj] += a_val * Brow[jj];
            }
        }
    }
}