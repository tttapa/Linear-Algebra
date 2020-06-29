#include <gtest/gtest.h>

#include <linalg/Matrix.hpp>
#include <linalg/RowPivotLU.hpp>

#include "CountAllocationsTests.hpp"

#include <algorithm> // std::max
#include <cmath>     // std::abs

#define EXPECT_CLOSE_ENOUGH(X, R)                                              \
    EXPECT_NEAR((X), (R), std::max(std::abs(X) * 1e-14, 1e-14))

TEST(RowPivotLU, PALU) {
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    RowPivotLU lu(A);

    SquareMatrix LU_prod = lu.get_L() * lu.get_U();
    SquareMatrix PA = lu.get_P() * A;

    for (size_t r = 0; r < A.rows(); ++r)
        for (size_t c = 0; c < A.cols(); ++c) {
            EXPECT_CLOSE_ENOUGH(PA(r, c), LU_prod(r, c))
                << "(" << r << ", " << c << ")";
            EXPECT_CLOSE_ENOUGH(PA(r, c), LU_prod(r, c))
                << "(" << r << ", " << c << ")";
        }
}

TEST(RowPivotLU, APTLU) {
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    RowPivotLU lu(A);

    SquareMatrix PTLU_prod = transpose(lu.get_P()) * lu.get_L() * lu.get_U();

    for (size_t r = 0; r < A.rows(); ++r)
        for (size_t c = 0; c < A.cols(); ++c) {
            EXPECT_CLOSE_ENOUGH(A(r, c), PTLU_prod(r, c))
                << "(" << r << ", " << c << ")";
            EXPECT_CLOSE_ENOUGH(A(r, c), PTLU_prod(r, c))
                << "(" << r << ", " << c << ")";
        }
}

TEST(RowPivotLU, solve) {
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    Vector x = {7, 11, 13};
    Vector b = A * x;
    RowPivotLU lu(A);

    Vector solution = lu.solve(b);

    ASSERT_EQ(x.size(), solution.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_CLOSE_ENOUGH(solution(c), x(c)) << "(" << c << ")";
}

TEST(RowPivotLU, solveMoveA) {
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    Vector x = {7, 11, 13};
    Vector b = A * x;
    RowPivotLU lu(std::move(A));

    Vector solution = lu.solve(b);

    ASSERT_EQ(x.size(), solution.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_CLOSE_ENOUGH(solution(c), x(c)) << "(" << c << ")";
}

TEST(RowPivotLU, solveMove) {
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    Vector x = {7, 11, 13};
    RowPivotLU lu(A);

    Vector solution = lu.solve(A * x);

    ASSERT_EQ(x.size(), solution.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_CLOSE_ENOUGH(solution(c), x(c)) << "(" << c << ")";
}

TEST(RowPivotLU, solveInplace) {
    RESET_ALLOC_COUNT();
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    Vector x = {7, 11, 13};
    Vector b = A * x;
    RowPivotLU lu(A);
    EXPECT_ALLOC_COUNT(4);
    EXPECT_ALLOC_ALIVE(4); // A, lu, x, b
    lu.solve_inplace(b);
    EXPECT_ALLOC_COUNT(4); // No intermediate solution is allocated
    EXPECT_ALLOC_ALIVE(4); // A, lu, x, b

    ASSERT_EQ(x.size(), b.size());
    for (size_t c = 0; c < x.cols(); ++c)
        EXPECT_CLOSE_ENOUGH(b(c), x(c)) << "(" << c << ")";
}

TEST(RowPivotLU, solveInvert) {
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    RowPivotLU lu(A);
    Matrix A_inv = lu.solve(Matrix::identity(3));

    Matrix expected = Matrix::identity(3);
    Matrix result = A * A_inv;

    for (size_t r = 0; r < expected.rows(); ++r)
        for (size_t c = 0; c < expected.cols(); ++c)
            EXPECT_CLOSE_ENOUGH(expected(r, c), result(r, c))
                << "(" << r << ", " << c << ")";
}

TEST(RowPivotLU, LURepresentation) {
    SquareMatrix A = {
        {7, 3, 4},
        {1, 2, 3},
        {6, 5, 4},
    };
    RowPivotLU lu(A);

    Matrix result = lu.get_L() + lu.get_U() - lu.get_LU() - Matrix::identity(3);

    for (size_t r = 0; r < A.rows(); ++r)
        for (size_t c = 0; c < A.cols(); ++c)
            EXPECT_LE(std::abs(result(r, c)), 1e-14)
                << "(" << r << ", " << c << ")";
}

TEST(RowPivotLU, LUSteal) {
    SquareMatrix A = {
        {35, 1, 6, 26, 19, 24},  //
        {3, 32, 7, 21, 23, 25},  //
        {31, 9, 2, 12, 27, 20},  //
        {8, 28, 33, 17, 10, 15}, //
        {30, 5, 34, 12, 14, 16}, //
        {4, 36, 29, 13, 18, 11}, //
    };

    SquareMatrix PLU_prod = transpose(RowPivotLU(A).get_P()) *
                            RowPivotLU(A).get_L() * RowPivotLU(A).get_U();

    for (size_t r = 0; r < A.rows(); ++r)
        for (size_t c = 0; c < A.cols(); ++c) {
            EXPECT_CLOSE_ENOUGH(A(r, c), PLU_prod(r, c))
                << "(" << r << ", " << c << ")";
            EXPECT_CLOSE_ENOUGH(A(r, c), PLU_prod(r, c))
                << "(" << r << ", " << c << ")";
        }

    Matrix result = RowPivotLU(A).get_L() + RowPivotLU(A).get_U() -
                    RowPivotLU(A).get_LU() - Matrix::identity(6);

    for (size_t r = 0; r < A.rows(); ++r)
        for (size_t c = 0; c < A.cols(); ++c)
            EXPECT_LE(std::abs(result(r, c)), 1e-14)
                << "(" << r << ", " << c << ")";
}