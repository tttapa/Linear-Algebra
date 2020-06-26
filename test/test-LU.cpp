#include <gtest/gtest.h>

#include <LU.hpp>
#include <Matrix.hpp>

#include "CountAllocationsTests.hpp"

#include <algorithm> // std::max
#include <cmath>     // std::abs

#define EXPECT_CLOSE_ENOUGH(X, R)                                              \
    EXPECT_NEAR((X), (R), std::max(std::abs(X) * 1e-14, 1e-14))

TEST(LU, LU) {
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    LU lu(A);

    SquareMatrix LU_prod = lu.get_L() * lu.get_U();

    for (size_t r = 0; r < A.rows(); ++r)
        for (size_t c = 0; c < A.cols(); ++c) {
            EXPECT_CLOSE_ENOUGH(A(r, c), LU_prod(r, c))
                << "(" << r << ", " << c << ")";
            EXPECT_CLOSE_ENOUGH(A(r, c), LU_prod(r, c))
                << "(" << r << ", " << c << ")";
        }
}

TEST(LU, solve) {
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    Vector x = {7, 11, 13};
    Vector b = A * x;
    LU lu(A);

    Vector solution = lu.solve(b);

    ASSERT_EQ(x.size(), solution.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_CLOSE_ENOUGH(solution(c), x(c)) << "(" << c << ")";
}

TEST(LU, solveMoveA) {
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    Vector x = {7, 11, 13};
    Vector b = A * x;
    LU lu(std::move(A));

    Vector solution = lu.solve(b);

    ASSERT_EQ(x.size(), solution.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_CLOSE_ENOUGH(solution(c), x(c)) << "(" << c << ")";
}

TEST(LU, solveMove) {
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    Vector x = {7, 11, 13};
    LU lu(A);

    Vector solution = lu.solve(A * x);

    ASSERT_EQ(x.size(), solution.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_CLOSE_ENOUGH(solution(c), x(c)) << "(" << c << ")";
}

TEST(LU, solveSquareInplace) {
    RESET_ALLOC_COUNT();
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    Vector x = {7, 11, 13};
    Vector b = A * x;
    LU lu(A);
    EXPECT_ALLOC_COUNT(4);
    EXPECT_ALLOC_ALIVE(4); // A, lu, x, b
    lu.solve_inplace(b);
    EXPECT_ALLOC_COUNT(4); // No intermediate solution is allocated
    EXPECT_ALLOC_ALIVE(4); // A, lu, x, b

    ASSERT_EQ(x.size(), b.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_CLOSE_ENOUGH(b(c), x(c)) << "(" << c << ")";
}

TEST(LU, solveSquareInvert) {
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    LU lu(A);
    Matrix A_inv = lu.solve(Matrix::identity(3));

    Matrix expected = Matrix::identity(3);
    Matrix result = A * A_inv;

    for (size_t r = 0; r < expected.rows(); ++r)
        for (size_t c = 0; c < expected.cols(); ++c)
            EXPECT_CLOSE_ENOUGH(expected(r, c), result(r, c))
                << "(" << r << ", " << c << ")";
}

TEST(LU, LURepresentation) {
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    LU lu(A);

    Matrix result = lu.get_L() + lu.get_U() - lu.get_LU() - Matrix::identity(3);

    for (size_t r = 0; r < A.rows(); ++r)
        for (size_t c = 0; c < A.cols(); ++c)
            EXPECT_LE(std::abs(result(r, c)), 1e-14)
                << "(" << r << ", " << c << ")";
}

TEST(LU, LUSteal) {
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    SquareMatrix LU_prod = LU(A).get_L() * LU(A).get_U();

    for (size_t r = 0; r < A.rows(); ++r)
        for (size_t c = 0; c < A.cols(); ++c) {
            EXPECT_CLOSE_ENOUGH(A(r, c), LU_prod(r, c))
                << "(" << r << ", " << c << ")";
            EXPECT_CLOSE_ENOUGH(A(r, c), LU_prod(r, c))
                << "(" << r << ", " << c << ")";
        }
}