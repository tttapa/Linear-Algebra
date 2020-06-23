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

TEST(HouseholderQR, solveSquareMove) {
    RESET_ALLOC_COUNT();
    Matrix A = {
        {1, 2, 1},
        {3, 4, 3},
        {1, 2, 3},
    };
    HouseholderQR qr(A);
    Vector x = {7, 11, 13};
    EXPECT_ALLOC_COUNT(4);
    EXPECT_ALLOC_ALIVE(4); // A, qr(2), x
    Vector solution = qr.solve(A * x);
    EXPECT_ALLOC_COUNT(5); // No intermediate solution is allocated because
                           // A is square, one intermediate vector is allocated
                           // for the matrix multiplication A * b
    EXPECT_ALLOC_ALIVE(5); // A, qr(2), x, solution

    ASSERT_EQ(x.size(), solution.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_FLOAT_EQ(solution(c), x(c)) << "(" << c << ")";
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
            EXPECT_FLOAT_EQ(A(r, c), QR(r, c)) << "(" << r << ", " << c << ")";
}
