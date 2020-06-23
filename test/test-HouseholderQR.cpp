#include <gtest/gtest.h>

#include <HouseholderQR.hpp>
#include <Matrix.hpp>

#include "CountAllocationsTests.hpp"

#include <algorithm> // std::max
#include <cmath>     // std::abs

#define EXPECT_CLOSE_ENOUGH(X, R)                                              \
    EXPECT_NEAR((X), (R), std::max(std::abs(X) * 1e-14, 1e-14))

TEST(HouseholderQR, QR) {
    Matrix A = {
        {1, 2, 1},
        {3, 4, 3},
        {1, 2, 3},
        {6, 5, 4},
    };
    HouseholderQR qr(A);

    Matrix R   = qr.get_R();
    Matrix QR1 = qr.apply_Q(R);
    Matrix QR2 = qr.apply_Q(std::move(R));

    for (size_t r = 0; r < A.rows(); ++r)
        for (size_t c = 0; c < A.cols(); ++c) {
            EXPECT_CLOSE_ENOUGH(A(r, c), QR1(r, c))
                << "(" << r << ", " << c << ")";
            EXPECT_CLOSE_ENOUGH(A(r, c), QR2(r, c))
                << "(" << r << ", " << c << ")";
        }
}

TEST(HouseholderQR, QRInplace) {
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
            EXPECT_CLOSE_ENOUGH(A(r, c), QR(r, c))
                << "(" << r << ", " << c << ")";
}

TEST(HouseholderQR, QTA) {
    Matrix A = {
        {1, 2, 1},
        {3, 4, 3},
        {1, 2, 3},
        {6, 5, 4},
    };
    HouseholderQR qr(A);

    Matrix R   = qr.get_R();
    Matrix QTA = qr.apply_QT(std::move(A));

    for (size_t r = 0; r < R.rows(); ++r)
        for (size_t c = 0; c < R.cols(); ++c)
            EXPECT_CLOSE_ENOUGH(R(r, c), QTA(r, c))
                << "(" << r << ", " << c << ")";
}

TEST(HouseholderQR, QRExplicit) {
    Matrix A = {
        {1, 2, 1},
        {3, 4, 3},
        {1, 2, 3},
        {6, 5, 4},
    };
    HouseholderQR qr(A);

    Matrix Q  = qr.get_Q();
    Matrix R  = qr.steal_R();
    Matrix QR = Q * R;

    for (size_t r = 0; r < A.rows(); ++r)
        for (size_t c = 0; c < A.cols(); ++c)
            EXPECT_CLOSE_ENOUGH(A(r, c), QR(r, c))
                << "(" << r << ", " << c << ")";
}

TEST(HouseholderQR, solveLeastSquares) {
    Matrix A = {
        {1, 2, 1},
        {3, 4, 3},
        {1, 2, 3},
        {6, 5, 4},
    };
    Vector x = {7, 11, 13};
    Vector b = A * x;
    HouseholderQR qr(A);

    Vector solution = qr.solve(b);

    ASSERT_EQ(x.size(), solution.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_CLOSE_ENOUGH(solution(c), x(c)) << "(" << c << ")";
}

TEST(HouseholderQR, solveLeastSquaresMoveA) {
    Matrix A = {
        {1, 2, 1},
        {3, 4, 3},
        {1, 2, 3},
        {6, 5, 4},
    };
    Vector x = {7, 11, 13};
    Vector b = A * x;
    HouseholderQR qr(std::move(A));

    Vector solution = qr.solve(b);

    ASSERT_EQ(x.size(), solution.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_CLOSE_ENOUGH(solution(c), x(c)) << "(" << c << ")";
}

TEST(HouseholderQR, solveLeastSquaresInplace) {
    RESET_ALLOC_COUNT();
    Matrix A = {
        {1, 2, 1},
        {3, 4, 3},
        {1, 2, 3},
        {6, 5, 4},
    };
    Vector x = {7, 11, 13};
    Vector b = A * x;
    HouseholderQR qr(A);
    EXPECT_ALLOC_COUNT(5);
    EXPECT_ALLOC_ALIVE(5); // A, qr(2), x, b
    qr.solve_inplace(b);
    EXPECT_ALLOC_COUNT(6); // intermediate solution is allocated because
                           // A is not square
    EXPECT_ALLOC_ALIVE(5); // A, qr(2), x, b

    ASSERT_EQ(x.size(), b.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_CLOSE_ENOUGH(b(c), x(c)) << "(" << c << ")";
}

TEST(HouseholderQR, solveSquare) {
    Matrix A = {
        {3, 4, 3},
        {1, 2, 3},
        {6, 5, 4},
    };
    Vector x = {7, 11, 13};
    Vector b = A * x;
    HouseholderQR qr(A);

    Vector solution = qr.solve(b);

    ASSERT_EQ(x.size(), solution.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_CLOSE_ENOUGH(solution(c), x(c)) << "(" << c << ")";
}

TEST(HouseholderQR, solveSquareInplace) {
    RESET_ALLOC_COUNT();
    Matrix A = {
        {1, 2, 1},
        {3, 4, 3},
        {1, 2, 3},
    };
    Vector x = {7, 11, 13};
    Vector b = A * x;
    HouseholderQR qr(A);
    EXPECT_ALLOC_COUNT(5);
    EXPECT_ALLOC_ALIVE(5); // A, qr(2), x, b
    qr.solve_inplace(b);
    EXPECT_ALLOC_COUNT(5); // No intermediate solution is allocated because
                           // A is square
    EXPECT_ALLOC_ALIVE(5); // A, qr(2), x, b

    ASSERT_EQ(x.size(), b.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_CLOSE_ENOUGH(b(c), x(c)) << "(" << c << ")";
}

TEST(HouseholderQR, solveSquareInvert) {
    Matrix A = {
        {3, 4, 3},
        {1, 2, 3},
        {6, 5, 4},
    };
    HouseholderQR qr(A);
    Matrix A_inv = qr.solve(Matrix::identity(3));

    Matrix expected = Matrix::identity(3);
    Matrix result   = A * A_inv;

    for (size_t r = 0; r < expected.rows(); ++r)
        for (size_t c = 0; c < expected.cols(); ++c)
            EXPECT_CLOSE_ENOUGH(expected(r, c), result(r, c))
                << "(" << r << ", " << c << ")";
}

TEST(HouseholderQR, solveSquareMove) {
    RESET_ALLOC_COUNT();
    Matrix A = {
        {1, 2, 1},
        {3, 4, 3},
        {1, 2, 3},
    };
    Vector x = {7, 11, 13};
    HouseholderQR qr(A);
    EXPECT_ALLOC_COUNT(4);
    EXPECT_ALLOC_ALIVE(4); // A, qr(2), x
    Vector solution = qr.solve(A * x);
    EXPECT_ALLOC_COUNT(5); // No intermediate solution is allocated because
                           // A is square, one intermediate vector is allocated
                           // for the matrix multiplication A * b
    EXPECT_ALLOC_ALIVE(5); // A, qr(2), x, solution

    ASSERT_EQ(x.size(), solution.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_CLOSE_ENOUGH(solution(c), x(c)) << "(" << c << ")";
}

TEST(HouseholderQR, zeroDiagonal) { // TODO: is Q too trivial?
    Matrix R = {
        {1, 2, 5, 7},
        {0, 4, 3, 9},
        {0, 0, 0, 2},
        {0, 0, 0, 1},
    };
    Matrix Q = Matrix::identity(4);
    Matrix A = Q * R;
    HouseholderQR qr(A);

    Matrix QR = qr.get_R();
    qr.apply_Q_inplace(QR);

    std::cout << qr << std::endl;

    for (size_t r = 0; r < A.rows(); ++r)
        for (size_t c = 0; c < A.cols(); ++c)
            EXPECT_CLOSE_ENOUGH(A(r, c), QR(r, c))
                << "(" << r << ", " << c << ")";
}
