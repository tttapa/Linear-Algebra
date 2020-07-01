#include <gtest/gtest.h>

#include <linalg/Matrix.hpp>

#include "CountAllocationsTests.hpp"

// Matrix
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(Matrix, divideScalar) {
    Matrix a        = {{1, 2}, {3, 4}};
    double b        = 1. / 16;
    Matrix expected = {{16, 32}, {48, 64}};
    Matrix result   = a / b;
    EXPECT_EQ(result, expected);
}
TEST(Matrix, divideScalarMoveA) {
    RESET_ALLOC_COUNT();
    Matrix a        = {{1, 2}, {3, 4}};
    double b        = 1. / 16;
    Matrix expected = {{16, 32}, {48, 64}};
    EXPECT_ALLOC_COUNT(2);
    Matrix result = std::move(a) / b;
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

// SquareMatrix
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(SquareMatrix, divideScalar) {
    SquareMatrix a        = {{1, 2}, {3, 4}};
    double b              = 1. / 16;
    SquareMatrix expected = {{16, 32}, {48, 64}};
    SquareMatrix result   = a / b;
    EXPECT_EQ(result, expected);
}
TEST(SquareMatrix, divideScalarMoveA) {
    RESET_ALLOC_COUNT();
    SquareMatrix a        = {{1, 2}, {3, 4}};
    double b              = 1. / 16;
    SquareMatrix expected = {{16, 32}, {48, 64}};
    EXPECT_ALLOC_COUNT(2);
    SquareMatrix result = std::move(a) / b;
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

// Vector
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(Vector, divideScalar) {
    Vector a        = {1, 2, 3};
    double b        = 1. / 16;
    Vector expected = {16, 32, 48};
    Vector result   = a / b;
    EXPECT_EQ(result, expected);
}
TEST(Vector, divideScalarMoveA) {
    RESET_ALLOC_COUNT();
    Vector a        = {1, 2, 3};
    double b        = 1. / 16;
    Vector expected = {16, 32, 48};
    EXPECT_ALLOC_COUNT(2);
    Vector result = std::move(a) / b;
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

// RowVector
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(RowVector, divideScalar) {
    RowVector a        = {1, 2, 3};
    double b           = 1. / 16;
    RowVector expected = {16, 32, 48};
    RowVector result   = a / b;
    EXPECT_EQ(result, expected);
}
TEST(RowVector, divideScalarMoveA) {
    RESET_ALLOC_COUNT();
    RowVector a        = {1, 2, 3};
    double b           = 1. / 16;
    RowVector expected = {16, 32, 48};
    EXPECT_ALLOC_COUNT(2);
    RowVector result = std::move(a) / b;
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}
