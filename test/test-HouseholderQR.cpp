#include <gtest/gtest.h>

#include <HouseholderQR.hpp>
#include <Matrix.hpp>

#include "CountAllocationsTests.hpp"

TEST(HouseholderQR, QR) {
    Matrix A = {
        {1, 2, 1},
        {3, 4, 3},
        {1, 2, 3},
        {6, 5, 4},
    };
    HouseholderQR qr(A);

    Matrix QR = qr.get_R();
    qr.apply_Q_inplace(QR);

    for (size_t r = 0; r < A.rows(); ++r)
        for (size_t c = 0; c < A.cols(); ++c)
            EXPECT_FLOAT_EQ(A(r, c), QR(r, c)) << "(" << r << ", " << c << ")";
}

TEST(HouseholderQR, solveLeastSquares) {
    Matrix A = {
        {1, 2, 1},
        {3, 4, 3},
        {1, 2, 3},
        {6, 5, 4},
    };
    HouseholderQR qr(A);
    Vector x = {7, 11, 13};
    Vector b = A * x;

    Vector solution = qr.solve(b);

    ASSERT_EQ(x.size(), solution.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_FLOAT_EQ(solution(c), x(c)) << "(" << c << ")";
}

TEST(HouseholderQR, solveLeastSquaresInplace) {
    RESET_ALLOC_COUNT();
    Matrix A = {
        {1, 2, 1},
        {3, 4, 3},
        {1, 2, 3},
        {6, 5, 4},
    };
    HouseholderQR qr(A);
    Vector x = {7, 11, 13};
    Vector b = A * x;
    EXPECT_ALLOC_COUNT(5);
    EXPECT_ALLOC_ALIVE(5); // A, qr(2), x, b
    qr.solve_inplace(b);
    EXPECT_ALLOC_COUNT(6); // intermediate solution is allocated because
                           // A is not square
    EXPECT_ALLOC_ALIVE(5); // A, qr(2), x, b

    ASSERT_EQ(x.size(), b.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_FLOAT_EQ(b(c), x(c)) << "(" << c << ")";
}

TEST(HouseholderQR, solveSquare) {
    Matrix A = {
        {3, 4, 3},
        {1, 2, 3},
        {6, 5, 4},
    };
    HouseholderQR qr(A);
    Vector x = {7, 11, 13};
    Vector b = A * x;

    Vector solution = qr.solve(b);

    ASSERT_EQ(x.size(), solution.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_FLOAT_EQ(solution(c), x(c)) << "(" << c << ")";
}

TEST(HouseholderQR, solveSquareInplace) {
    RESET_ALLOC_COUNT();
    Matrix A = {
        {1, 2, 1},
        {3, 4, 3},
        {1, 2, 3},
    };
    HouseholderQR qr(A);
    Vector x = {7, 11, 13};
    Vector b = A * x;
    EXPECT_ALLOC_COUNT(5);
    EXPECT_ALLOC_ALIVE(5); // A, qr(2), x, b
    qr.solve_inplace(b);
    EXPECT_ALLOC_COUNT(5); // No intermediate solution is allocated because
                           // A is square
    EXPECT_ALLOC_ALIVE(5); // A, qr(2), x, b

    ASSERT_EQ(x.size(), b.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_FLOAT_EQ(b(c), x(c)) << "(" << c << ")";
}
