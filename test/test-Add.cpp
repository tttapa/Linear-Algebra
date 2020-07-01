#include <gtest/gtest.h>

#include <linalg/Matrix.hpp>

#include "CountAllocationsTests.hpp"

// Matrix
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(Matrix, add) {
    Matrix a        = {{1, 2}, {3, 4}};
    Matrix b        = {{10, 11}, {12, 13}};
    Matrix expected = {{11, 13}, {15, 17}};
    Matrix result   = a + b;
    EXPECT_EQ(result, expected);
}
TEST(Matrix, addMoveA) {
    RESET_ALLOC_COUNT();
    Matrix a        = {{1, 2}, {3, 4}};
    Matrix b        = {{10, 11}, {12, 13}};
    Matrix expected = {{11, 13}, {15, 17}};
    EXPECT_ALLOC_COUNT(3);
    Matrix result = std::move(a) + b;
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // b, expected, result
    EXPECT_EQ(result, expected);
}
TEST(Matrix, addMoveB) {
    RESET_ALLOC_COUNT();
    Matrix a        = {{1, 2}, {3, 4}};
    Matrix b        = {{10, 11}, {12, 13}};
    Matrix expected = {{11, 13}, {15, 17}};
    EXPECT_ALLOC_COUNT(3);
    Matrix result = a + std::move(b);
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // a, expected, result
    EXPECT_EQ(result, expected);
}
TEST(Matrix, addMoveAB) {
    RESET_ALLOC_COUNT();
    Matrix a        = {{1, 2}, {3, 4}};
    Matrix b        = {{10, 11}, {12, 13}};
    Matrix expected = {{11, 13}, {15, 17}};
    EXPECT_ALLOC_COUNT(3);
    Matrix result = std::move(a) + std::move(b);
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

// SquareMatrix
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(SquareMatrix, add) {
    SquareMatrix a        = {{1, 2}, {3, 4}};
    SquareMatrix b        = {{10, 11}, {12, 13}};
    SquareMatrix expected = {{11, 13}, {15, 17}};
    SquareMatrix result   = a + b;
    EXPECT_EQ(result, expected);
}
TEST(SquareMatrix, addMoveA) {
    RESET_ALLOC_COUNT();
    SquareMatrix a        = {{1, 2}, {3, 4}};
    SquareMatrix b        = {{10, 11}, {12, 13}};
    SquareMatrix expected = {{11, 13}, {15, 17}};
    EXPECT_ALLOC_COUNT(3);
    SquareMatrix result = std::move(a) + b;
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // b, expected, result
    EXPECT_EQ(result, expected);
}
TEST(SquareMatrix, addMoveB) {
    RESET_ALLOC_COUNT();
    SquareMatrix a        = {{1, 2}, {3, 4}};
    SquareMatrix b        = {{10, 11}, {12, 13}};
    SquareMatrix expected = {{11, 13}, {15, 17}};
    EXPECT_ALLOC_COUNT(3);
    SquareMatrix result = a + std::move(b);
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // a, expected, result
    EXPECT_EQ(result, expected);
}
TEST(SquareMatrix, addMoveAB) {
    RESET_ALLOC_COUNT();
    SquareMatrix a        = {{1, 2}, {3, 4}};
    SquareMatrix b        = {{10, 11}, {12, 13}};
    SquareMatrix expected = {{11, 13}, {15, 17}};
    EXPECT_ALLOC_COUNT(3);
    SquareMatrix result = std::move(a) + std::move(b);
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

// Vector
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(Vector, add) {
    Vector a        = {1, 2, 3};
    Vector b        = {10, 11, 12};
    Vector expected = {11, 13, 15};
    Vector result   = a + b;
    EXPECT_EQ(result, expected);
}
TEST(Vector, addMoveA) {
    RESET_ALLOC_COUNT();
    Vector a        = {1, 2, 3};
    Vector b        = {10, 11, 12};
    Vector expected = {11, 13, 15};
    EXPECT_ALLOC_COUNT(3);
    Vector result = std::move(a) + b;
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // b, expected, result
    EXPECT_EQ(result, expected);
}
TEST(Vector, addMoveB) {
    RESET_ALLOC_COUNT();
    Vector a        = {1, 2, 3};
    Vector b        = {10, 11, 12};
    Vector expected = {11, 13, 15};
    EXPECT_ALLOC_COUNT(3);
    Vector result = a + std::move(b);
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // a, expected, result
    EXPECT_EQ(result, expected);
}
TEST(Vector, addMoveAB) {
    RESET_ALLOC_COUNT();
    Vector a        = {1, 2, 3};
    Vector b        = {10, 11, 12};
    Vector expected = {11, 13, 15};
    EXPECT_ALLOC_COUNT(3);
    Vector result = std::move(a) + std::move(b);
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

// RowVector
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(RowVector, add) {
    RowVector a        = {1, 2, 3};
    RowVector b        = {10, 11, 12};
    RowVector expected = {11, 13, 15};
    RowVector result   = a + b;
    EXPECT_EQ(result, expected);
}
TEST(RowVector, addMoveA) {
    RESET_ALLOC_COUNT();
    RowVector a        = {1, 2, 3};
    RowVector b        = {10, 11, 12};
    RowVector expected = {11, 13, 15};
    EXPECT_ALLOC_COUNT(3);
    RowVector result = std::move(a) + b;
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // b, expected, result
    EXPECT_EQ(result, expected);
}
TEST(RowVector, addMoveB) {
    RESET_ALLOC_COUNT();
    RowVector a        = {1, 2, 3};
    RowVector b        = {10, 11, 12};
    RowVector expected = {11, 13, 15};
    EXPECT_ALLOC_COUNT(3);
    RowVector result = a + std::move(b);
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // a, expected, result
    EXPECT_EQ(result, expected);
}
TEST(RowVector, addMoveAB) {
    RESET_ALLOC_COUNT();
    RowVector a        = {1, 2, 3};
    RowVector b        = {10, 11, 12};
    RowVector expected = {11, 13, 15};
    EXPECT_ALLOC_COUNT(3);
    RowVector result = std::move(a) + std::move(b);
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}