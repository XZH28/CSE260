// gemm_kernel.h
#pragma once
#include <string>

template <typename T>
class GemmKernel {
public:
    static int M, K, N;
    static int lda, ldb, ldc;

    static int NC; // columns of C (N)
    static int KC; // reduction dim chunk (K)
    static int MC; // rows of C (M)
    static int NR; // micro-tile along N
    static int MR; // micro-tile along M

    virtual void run(const T* A, const T* B, T* C) const = 0;
    virtual std::string display() const = 0;

    virtual ~GemmKernel() = default;
};

// Static member definitions
template <typename T> int GemmKernel<T>::M = 0;
template <typename T> int GemmKernel<T>::K = 0;
template <typename T> int GemmKernel<T>::N = 0;
template <typename T> int GemmKernel<T>::lda = 0;
template <typename T> int GemmKernel<T>::ldb = 0;
template <typename T> int GemmKernel<T>::ldc = 0;
template <typename T> int GemmKernel<T>::NC = 0;
template <typename T> int GemmKernel<T>::KC = 0;
template <typename T> int GemmKernel<T>::MC = 0;
template <typename T> int GemmKernel<T>::NR = 0;
template <typename T> int GemmKernel<T>::MR = 0;

