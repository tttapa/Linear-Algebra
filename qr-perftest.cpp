#include <algorithm>
#include <cassert>
#include <cmath> // std::sqrt, std::copysign
#include <limits>
#include <random>
#include <vector>

#include <iomanip>
#include <iostream>

#include "HouseholderQR.hpp"

#include <chrono>

int main(int argc, const char *argv[]) {
    size_t size = 500;
    if (argc > 1) {
        size = std::stoi(argv[1]);
    }

    Matrix m = Matrix::random(size, size, -1e2, +1e2);
    HouseholderQR qr;

    auto start = std::chrono::high_resolution_clock::now();
    qr.compute(std::move(m));
    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";

    Vector x = qr.solve(Vector::ones(size));
    std::cout << "A \\ b = " << std::endl;
    std::cout << x(0) << std::endl;
}