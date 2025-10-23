#include "bgemm_naive.h"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <cstring>

static void fill_rand(std::vector<double>& v, double scale = 1.0) {
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-scale, scale);
    for (auto& x : v) x = dist(rng);
}

static double gflops(double secs, long long M, long long N, long long K) {
    // GEMM does 2*M*N*K flops (mul+add)
    double ops = 2.0 * (double)M * (double)N * (double)K;
    return (ops / secs) / 1e9;
}

int main(int argc, char** argv) {
    // Usage: ./app [M N K reps] [NC KC MC NR MR]  (all optional)
    int M = 1024, N = 1024, K = 1024;
    int reps = 5;

    if (argc >= 5) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
        reps = std::max(1, std::atoi(argv[4]));
    }

    if (argc >= 10) {
        int NC = std::atoi(argv[5]);
        int KC = std::atoi(argv[6]);
        int MC = std::atoi(argv[7]);
        int NR = std::atoi(argv[8]);
        int MR = std::atoi(argv[9]);
        set_block_sizes(NC, KC, MC, NR, MR);
    }

    const int lda = K; // row-major: A[M x K]
    const int ldb = N; // row-major: B[K x N]
    const int ldc = N; // row-major: C[M x N]

    std::vector<double> A((size_t)M * K);
    std::vector<double> B((size_t)K * N);
    std::vector<double> C((size_t)M * N);

    fill_rand(A, 1.0);
    fill_rand(B, 1.0);
    std::fill(C.begin(), C.end(), 0.0);

    // Warm-up
    gemm_blocked(M, N, K, A.data(), lda, B.data(), ldb, C.data(), ldc);

    // Measure
    double best = 1e30, total = 0.0;
    for (int r = 0; r < reps; ++r) {
        // reset C to avoid already-accumulated values affecting cache behavior
        std::fill(C.begin(), C.end(), 0.0);

        auto t0 = std::chrono::high_resolution_clock::now();
        gemm_blocked(M, N, K, A.data(), lda, B.data(), ldb, C.data(), ldc);
        auto t1 = std::chrono::high_resolution_clock::now();

        double secs = std::chrono::duration<double>(t1 - t0).count();
        best = std::min(best, secs);
        total += secs;

        std::cout << "Run " << (r+1) << "/" << reps
                  << ": " << secs << " s, "
                  << gflops(secs, M, N, K) << " GFLOP/s\n";
    }

    std::cout << "Best time: " << best << " s  → "
              << gflops(best, M, N, K) << " GFLOP/s\n";
    std::cout << "Avg  time: " << (total / reps) << " s  → "
              << gflops(total / reps, M, N, K) << " GFLOP/s\n";

    // Quick sanity check on a few entries (not strict correctness test)
    double checksum = 0.0;
    for (int i = 0; i < std::min(M, 3); ++i)
        for (int j = 0; j < std::min(N, 3); ++j)
            checksum += C[(size_t)i * ldc + j];
    std::cout << "Checksum (first 3x3 of C summed): " << checksum << "\n";
    return 0;
}
