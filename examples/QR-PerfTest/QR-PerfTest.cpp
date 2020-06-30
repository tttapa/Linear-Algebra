/**
 * @example QR-PerfTest.cpp
 * Compares the performance of the naive included QR algorithm with
 * Eigen's implementation.
 * 
 * ---
 * 
 * The results on an i7-7700HQ were:
 * @image   html qr-perf-test-linear.svg
 * @image   html qr-perf-test-loglog.svg
 * 
 * For small matrices, the naive implementation is slightly faster than Eigen.
 * For matrices larger than 100×100, Eigen is significantly faster because it 
 * uses a blocked algorithm that makes more efficient use of the CPU's cache.
 * 
 * The Frobenius norm of the error ‖A - QR‖ is around twice as big for the naive
 * implementation than for Eigen's implementation.  
 * For example, for a 100×100 matrix, the naive implementation has an absolute
 * error of 5.58e-14, while Eigen's implementation has an error of 3.58e-14.
 * 
 * It's a small difference, and I don't have an exact explanation for it, but
 * the reason could be that Eigen uses a different normalization factor for the
 * Householder reflectors:  
 * The naive implementation normalizes them as  wₖ = √2·vₖ / ‖vₖ‖, such that
 * ‖wₖ‖ = √2, while Eigen follows the same convention used by LAPACK, where
 * wₖ = vₖ / vₖ[0], such that  wₖ[0] = 1.
 */

#include <chrono>
#include <iomanip>
#include <iostream>

#include <linalg/HouseholderQR.hpp>
#include <Eigen/QR>
using EigenMat =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

int main(int argc, const char *argv[]) {
    // First (optional) command line argument is the matrix size
    size_t size = 500;
    if (argc > 1) {
        size = std::stoi(argv[1]);
    }

    // Generate a random matrix of the given size
    Matrix A = Matrix::random(size, size, -1, +1);
    // Convert it to an Eigen matrix
    EigenMat A_eigen = EigenMat::Zero(size, size);
    std::copy(A.begin(), A.end(), A_eigen.data());

    {
        // Compute the QR factorization using our naive implementation:
        HouseholderQR qr;
        Matrix A_cpy = A; // make a copy here to prevent allocation while timing
        auto start   = std::chrono::high_resolution_clock::now();
        qr.compute(std::move(A_cpy));
        auto finish = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = finish - start;
        Matrix QR      = qr.apply_Q(qr.get_R()); // Multiply Q×R
        double err_fro = (A - QR).normFro();
        std::cout << "Elapsed time HouseholderQR: " << elapsed.count() << " s\n"
                  << "Error QR - A in Frobenius norm: " << std::scientific
                  << err_fro << std::defaultfloat << std::endl;
    }
    {
        // Now perform the factorization using Eigen:
        Eigen::HouseholderQR<EigenMat> qr_eigen;
        EigenMat A_eigen_cpy = A_eigen;
        auto start           = std::chrono::high_resolution_clock::now();
        qr_eigen.compute(std::move(A_eigen_cpy));
        auto finish = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = finish - start;
        EigenMat QR = qr_eigen.matrixQR().triangularView<Eigen::Upper>();
        qr_eigen.householderQ().applyThisOnTheLeft(QR); // Multiply Q×R
        auto err_fro = (A_eigen - QR).norm();
        std::cout << "Elapsed time Eigen: " << elapsed.count() << " s\n"
                  << "Error QR - A in Frobenius norm: " << std::scientific
                  << err_fro << std::defaultfloat << std::endl;
    }
}