#include "bgemm_naive.h"
#include "bgemm_packing.h"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <cstring>


void fill_rand(std::vector<double>& v, double scale = 1.0) {
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-scale, scale);
    for (auto& x : v) x = dist(rng);
}

double gflops(double secs, long long M, long long N, long long K) {
    // GEMM does 2*M*N*K flops (mul+add)
    double ops = 2.0 * (double)M * (double)N * (double)K;
    return (ops / secs) / 1e9;
}

template <typename T>
void gemm_golden_result(
    int M, int K, int N, 
    const T* A, int lda, 
    const T* B, int ldb,
    T* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int loc_c = m * ldc + n;
            for (int k = 0; k < K; ++k) {
                int loc_a = m * lda + k;
                int loc_b = k * ldb + n;
                C[loc_c] += A[loc_a] * B[loc_b];
            }
        }
    }
}

#include <cmath>

template <typename T>
bool vector_equal_eps(const std::vector<T>& a, const std::vector<T>& b, T eps) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > eps) return false;
    return true;
}


int main(int argc, char** argv) {
    using dtype = double;

    // arguments
    // Usage: ./app [M K N reps] [NC KC MC NR MR]  (all optional)
    int M = 1000, K = 2000, N = 500;
    int reps = 5;

    int NC = 512;
    int KC = 256;
    int MC = 512;
    int NR = 8;
    int MR = 8;

    if (argc >= 5) {
        M = std::atoi(argv[1]);
        K = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
        reps = std::max(1, std::atoi(argv[4]));
    }

    if (argc >= 10) {
        int NC = std::atoi(argv[5]);
        int KC = std::atoi(argv[6]);
        int MC = std::atoi(argv[7]);
        int NR = std::atoi(argv[8]);
        int MR = std::atoi(argv[9]);
    }

    const int lda = K; // row-major: A[M x K]
    const int ldb = N; // row-major: B[K x N]
    const int ldc = N; // row-major: C[M x N]
    // test data
    std::vector<dtype> A(M * K);
    std::vector<dtype> B(K * N);
    std::vector<dtype> C(M * N);
    std::vector<dtype> C_ref(M * N);

    fill_rand(A, 1.0);
    fill_rand(B, 1.0);
    std::fill(C.begin(), C.end(), 0.0);
    std::fill(C_ref.begin(), C_ref.end(), 0.0);
    // set static variables
    GemmKernel<dtype>::M = M;
    GemmKernel<dtype>::K = K;
    GemmKernel<dtype>::N = N;
    GemmKernel<dtype>::NC = NC;
    GemmKernel<dtype>::KC = KC;
    GemmKernel<dtype>::MC = MC;
    GemmKernel<dtype>::NR = NR;
    GemmKernel<dtype>::MR = MR;
    GemmKernel<dtype>::lda = lda;
    GemmKernel<dtype>::ldb = ldb;
    GemmKernel<dtype>::ldc = ldc;
    // compute golden result
    gemm_golden_result(M, K, N, A.data(), lda, B.data(), ldb, C_ref.data(), ldc);

    // create workers
    std::vector<GemmKernel<dtype>*> workers;
    GemmKernelNaive<dtype> gemm_naive;
    workers.push_back(&gemm_naive);
    GemmKernelPack<dtype> gemm_pack;
    workers.push_back(&gemm_pack);

    // run all workers once and check the arithemtic correctness
    for (const auto* worker : workers) {
        std::fill(C.begin(), C.end(), 0.0);
        worker->run(A.data(), B.data(), C.data());
        bool correctness = vector_equal_eps(C, C_ref, 1e-5);
        std::cout << "Arithmetic Correctness Verification: " 
        << worker->display() << "RESULT: " << std::boolalpha << correctness << std::endl;
    }
    // Measure
    double best = 1e30, total = 0.0;
    double runtime[reps];
    for (const auto* worker : workers) {
        best = 1e10; total = 0.0;
        // warm up
        worker->run(A.data(), B.data(), C.data());
        for (int r = 0; r < reps; ++r) {
            std::fill(C.begin(), C.end(), 0.0);

            auto t0 = std::chrono::high_resolution_clock::now();
            worker->run(A.data(), B.data(), C.data());
            auto t1 = std::chrono::high_resolution_clock::now();

            double secs = std::chrono::duration<double>(t1 - t0).count();
            best = std::min(best, secs);
            total += secs;
            runtime[r] = secs;
        }
        std::cout << "===========" << worker->display() << "===========" << std::endl;
        std::cout << "Execution time: ";
        for (int i = 0; i < reps; ++i) {
            std::cout << runtime[i] << " s | ";
        }
        std::cout << "\n";
        std::cout << "Best time: " << best << " s  → "
                << gflops(best, M, N, K) << " GFLOP/s\n";
        std::cout << "Avg  time: " << (total / reps) << " s  → "
                << gflops(total / reps, M, N, K) << " GFLOP/s\n";

    }

    return 0;
}
