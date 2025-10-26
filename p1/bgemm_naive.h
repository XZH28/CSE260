#pragma once
#include <string>
#include <algorithm>
#include "bgemm.h"
#include "common.h"

template <typename T>
class GemmKernelNaive : public GemmKernel<T> {
public:
    GemmKernelNaive() = default;
    void run(const T* A, const T* B, T* C) const override;
    std::string display() const override;
    ~GemmKernelNaive() override = default;
};


template <typename T>
void GemmKernelNaive<T>::run(const T* A, const T* B, T* C) const
{
    const int sNC = this->NC;
    const int sKC = this->KC;
    const int sMC = this->MC;
    const int sNR = this->NR;
    const int sMR = this->MR;

    for (int ic = 0; ic < this->M; ic += sMC) {
        const int mc = std::min(sMC, this->M - ic);
        for (int pc = 0; pc < this->K; pc += sKC) {
            const int kc = std::min(sKC, this->K - pc);
            for (int jc = 0; jc < this->N; jc += sNC) {
                const int nc = std::min(sNC, this->N - jc);
                for (int ir = 0; ir < mc; ir += sMR) {
                    const int mr = std::min(sMR, mc - ir);
                    for (int jr = 0; jr < nc; jr += sNR) {
                        const int nr = std::min(sNR, nc - jr);
                        const T* A_block = A + (ic + ir) * this->lda + pc;
                        const T* B_block = B + pc * this->ldb + (jc + jr);
                        T*       C_block = C + (ic + ir) * this->ldc + (jc + jr);
                        microkernel_naive(mr, nr, kc, A_block, this->lda, B_block, this->ldb, C_block, this->ldc);
                    }
                }
            }
        }
    }
}

template <typename T>
std::string GemmKernelNaive<T>::display() const
{
    return "GemmKernelNaive";
}
