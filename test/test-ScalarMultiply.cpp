#include <gtest/gtest.h>

#include <linalg/Matrix.hpp>

#include "CountAllocationsTests.hpp"

// Matrix
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(Matrix, multiplyScalar) {
    Matrix a        = {{1, 2}, {3, 4}};
    double b        = 16;
    Matrix expected = {{16, 32}, {48, 64}};
    Matrix result   = a * b;
    EXPECT_EQ(result, expected);
}
TEST(Matrix, multiplyScalarMoveA) {
    RESET_ALLOC_COUNT();
    Matrix a        = {{1, 2}, {3, 4}};
    double b        = 16;
    Matrix expected = {{16, 32}, {48, 64}};
    EXPECT_ALLOC_COUNT(2);
    Matrix result = std::move(a) * b;
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}
TEST(Matrix, multiplyScalar2) {
    Matrix a        = {{1, 2}, {3, 4}};
    double b        = 16;
    Matrix expected = {{16, 32}, {48, 64}};
    Matrix result   = b * a;
    EXPECT_EQ(result, expected);
}
TEST(Matrix, multiplyScalarMoveA2) {
    RESET_ALLOC_COUNT();
    Matrix a        = {{1, 2}, {3, 4}};
    double b        = 16;
    Matrix expected = {{16, 32}, {48, 64}};
    EXPECT_ALLOC_COUNT(2);
    Matrix result = b * std::move(a);
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

// SquareMatrix
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(SquareMatrix, multiplyScalar) {
    SquareMatrix a        = {{1, 2}, {3, 4}};
    double b              = 16;
    SquareMatrix expected = {{16, 32}, {48, 64}};
    SquareMatrix result   = a * b;
    EXPECT_EQ(result, expected);
}
TEST(SquareMatrix, multiplyScalarMoveA) {
    RESET_ALLOC_COUNT();
    SquareMatrix a        = {{1, 2}, {3, 4}};
    double b              = 16;
    SquareMatrix expected = {{16, 32}, {48, 64}};
    EXPECT_ALLOC_COUNT(2);
    SquareMatrix result = std::move(a) * b;
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}
TEST(SquareMatrix, multiplyScalar2) {
    SquareMatrix a        = {{1, 2}, {3, 4}};
    double b              = 16;
    SquareMatrix expected = {{16, 32}, {48, 64}};
    SquareMatrix result   = b * a;
    EXPECT_EQ(result, expected);
}
TEST(SquareMatrix, multiplyScalarMoveA2) {
    RESET_ALLOC_COUNT();
    SquareMatrix a        = {{1, 2}, {3, 4}};
    double b              = 16;
    SquareMatrix expected = {{16, 32}, {48, 64}};
    EXPECT_ALLOC_COUNT(2);
    SquareMatrix result = b * std::move(a);
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

// Vector
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(Vector, multiplyScalar) {
    Vector a        = {1, 2, 3};
    double b        = 16;
    Vector expected = {16, 32, 48};
    Vector result   = a * b;
    EXPECT_EQ(result, expected);
}
TEST(Vector, multiplyScalarMoveA) {
    RESET_ALLOC_COUNT();
    Vector a        = {1, 2, 3};
    double b        = 16;
    Vector expected = {16, 32, 48};
    EXPECT_ALLOC_COUNT(2);
    Vector result = std::move(a) * b;
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}
TEST(Vector, multiplyScalar2) {
    Vector a        = {1, 2, 3};
    double b        = 16;
    Vector expected = {16, 32, 48};
    Vector result   = b * a;
    EXPECT_EQ(result, expected);
}
TEST(Vector, multiplyScalarMoveA2) {
    RESET_ALLOC_COUNT();
    Vector a        = {1, 2, 3};
    double b        = 16;
    Vector expected = {16, 32, 48};
    EXPECT_ALLOC_COUNT(2);
    Vector result = b * std::move(a);
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

// RowVector
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(RowVector, multiplyScalar) {
    RowVector a        = {1, 2, 3};
    double b           = 16;
    RowVector expected = {16, 32, 48};
    RowVector result   = a * b;
    EXPECT_EQ(result, expected);
}
TEST(RowVector, multiplyScalarMoveA) {
    RESET_ALLOC_COUNT();
    RowVector a        = {1, 2, 3};
    double b           = 16;
    RowVector expected = {16, 32, 48};
    EXPECT_ALLOC_COUNT(2);
    RowVector result = std::move(a) * b;
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}
TEST(RowVector, multiplyScalar2) {
    RowVector a        = {1, 2, 3};
    double b           = 16;
    RowVector expected = {16, 32, 48};
    RowVector result   = b * a;
    EXPECT_EQ(result, expected);
}
TEST(RowVector, multiplyScalarMoveA2) {
    RESET_ALLOC_COUNT();
    RowVector a        = {1, 2, 3};
    double b           = 16;
    RowVector expected = {16, 32, 48};
    EXPECT_ALLOC_COUNT(2);
    RowVector result = b * std::move(a);
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}