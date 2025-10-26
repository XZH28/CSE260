#pragma once
#include <string>
#include <algorithm>
#include "bgemm.h"
#include "common.h"

template <typename T>
class GemmKernelPack : public GemmKernel<T> {
public:
    GemmKernelPack() = default;
    void run(const T* A, const T* B, T* C) const override;
    std::string display() const override;
    ~GemmKernelPack() override = default;
};

template <typename T>
void GemmKernelPack<T>::run(const T* A, const T* B, T* C) const
{
    const int sNC = this->NC;
    const int sKC = this->KC;
    const int sMC = this->MC;
    const int sNR = this->NR;
    const int sMR = this->MR;
    const int sM = this->M;
    const int sN = this->N;
    const int sK = this->K;

    T* Ap = new T[(sMC + sMR - 1) * sKC];
    T* Bp = new T[(sNC + sNR - 1) * sKC];

    for (int ic = 0; ic < sM; ic += sMC) {
        const int mc = std::min(sMC, sM - ic);
        for (int pc = 0; pc < sK; pc += sKC) {
            const int kc = std::min(sKC, this->K - pc);
            const T* A_topack = A + ic * this->lda + pc;
            packing_A(A_topack, this->lda, mc, kc, sKC, sMR, Ap);
            for (int jc = 0; jc < this->N; jc += sNC) {
                const int nc = std::min(sNC, this->N - jc);
                const T* B_topack = B + pc * this->ldb + jc;
                packing_B(B_topack, this->ldb, kc, sKC, nc, sNR, Bp);
                for (int ir = 0; ir < mc; ir += sMR) {
                    const int mr = std::min(sMR, mc - ir);
                    for (int jr = 0; jr < nc; jr += sNR) {
                        const int nr = std::min(sNR, nc - jr);
                        const T* Ap_block = Ap + ir * sKC;
                        const T* Bp_block = Bp + jr * sKC;
                        T*       C_block = C + (ic + ir) * this->ldc + (jc + jr);
                        microkernel_packing(sMR, sKC, sNR, mr, nr, Ap_block, Bp_block, C_block, this->ldc);
                    }
                }
            }
        }
    }

    delete[] Ap;
    Ap = nullptr;
    delete[] Bp;
    Bp = nullptr;
}

template <typename T>
std::string GemmKernelPack<T>::display() const
{
    return "GemmKernel with Packing";
}
