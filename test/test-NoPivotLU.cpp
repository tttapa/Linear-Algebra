#include <gtest/gtest.h>

#include <linalg/NoPivotLU.hpp>
#include <linalg/Matrix.hpp>

#include "CountAllocationsTests.hpp"

#include <algorithm> // std::max
#include <cmath>     // std::abs

#define EXPECT_CLOSE_ENOUGH(X, R)                                              \
    EXPECT_NEAR((X), (R), std::max(std::abs(X) * 1e-14, 1e-14))

TEST(NoPivotLU, NoPivotLU) {
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    NoPivotLU lu(A);

    SquareMatrix LU_prod = lu.get_L() * lu.get_U();

    for (size_t r = 0; r < A.rows(); ++r)
        for (size_t c = 0; c < A.cols(); ++c) {
            EXPECT_CLOSE_ENOUGH(A(r, c), LU_prod(r, c))
                << "(" << r << ", " << c << ")";
            EXPECT_CLOSE_ENOUGH(A(r, c), LU_prod(r, c))
                << "(" << r << ", " << c << ")";
        }
}

TEST(NoPivotLU, solve) {
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    Vector x = {7, 11, 13};
    Vector b = A * x;
    NoPivotLU lu(A);

    Vector solution = lu.solve(b);

    ASSERT_EQ(x.size(), solution.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_CLOSE_ENOUGH(solution(c), x(c)) << "(" << c << ")";
}

TEST(NoPivotLU, solveMoveA) {
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    Vector x = {7, 11, 13};
    Vector b = A * x;
    NoPivotLU lu(std::move(A));

    Vector solution = lu.solve(b);

    ASSERT_EQ(x.size(), solution.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_CLOSE_ENOUGH(solution(c), x(c)) << "(" << c << ")";
}

TEST(NoPivotLU, solveMove) {
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    Vector x = {7, 11, 13};
    NoPivotLU lu(A);

    Vector solution = lu.solve(A * x);

    ASSERT_EQ(x.size(), solution.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_CLOSE_ENOUGH(solution(c), x(c)) << "(" << c << ")";
}

TEST(NoPivotLU, solveSquareInplace) {
    RESET_ALLOC_COUNT();
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    Vector x = {7, 11, 13};
    Vector b = A * x;
    NoPivotLU lu(A);
    EXPECT_ALLOC_COUNT(4);
    EXPECT_ALLOC_ALIVE(4); // A, lu, x, b
    lu.solve_inplace(b);
    EXPECT_ALLOC_COUNT(4); // No intermediate solution is allocated
    EXPECT_ALLOC_ALIVE(4); // A, lu, x, b

    ASSERT_EQ(x.size(), b.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_CLOSE_ENOUGH(b(c), x(c)) << "(" << c << ")";
}

TEST(NoPivotLU, solveSquareInvert) {
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    NoPivotLU lu(A);
    Matrix A_inv = lu.solve(Matrix::identity(3));

    Matrix expected = Matrix::identity(3);
    Matrix result = A * A_inv;

    for (size_t r = 0; r < expected.rows(); ++r)
        for (size_t c = 0; c < expected.cols(); ++c)
            EXPECT_CLOSE_ENOUGH(expected(r, c), result(r, c))
                << "(" << r << ", " << c << ")";
}

TEST(NoPivotLU, LURepresentation) {
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    NoPivotLU lu(A);

    Matrix result = lu.get_L() + lu.get_U() - lu.get_LU() - Matrix::identity(3);

    for (size_t r = 0; r < A.rows(); ++r)
        for (size_t c = 0; c < A.cols(); ++c)
            EXPECT_LE(std::abs(result(r, c)), 1e-14)
                << "(" << r << ", " << c << ")";
}

TEST(NoPivotLU, LUSteal) {
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    SquareMatrix LU_prod = NoPivotLU(A).get_L() * NoPivotLU(A).get_U();

    for (size_t r = 0; r < A.rows(); ++r)
        for (size_t c = 0; c < A.cols(); ++c) {
            EXPECT_CLOSE_ENOUGH(A(r, c), LU_prod(r, c))
                << "(" << r << ", " << c << ")";
            EXPECT_CLOSE_ENOUGH(A(r, c), LU_prod(r, c))
                << "(" << r << ", " << c << ")";
        }

    Matrix result =
        NoPivotLU(A).get_L() + NoPivotLU(A).get_U() - NoPivotLU(A).get_LU() - Matrix::identity(3);

    for (size_t r = 0; r < A.rows(); ++r)
        for (size_t c = 0; c < A.cols(); ++c)
            EXPECT_LE(std::abs(result(r, c)), 1e-14)
                << "(" << r << ", " << c << ")";
}