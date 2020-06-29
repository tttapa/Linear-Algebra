#include <gtest/gtest.h>

#include <linalg/Matrix.hpp>

#include "CountAllocationsTests.hpp"

// Matrix
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(Matrix, subtract) {
    Matrix a        = {{1, 2}, {3, 4}};
    Matrix b        = {{13, 12}, {11, 10}};
    Matrix expected = {{-12, -10}, {-8, -6}};
    Matrix result      = a - b;
    EXPECT_EQ(result, expected);
}
TEST(Matrix, subtractMoveA) {
    RESET_ALLOC_COUNT();
    Matrix a        = {{1, 2}, {3, 4}};
    Matrix b        = {{13, 12}, {11, 10}};
    Matrix expected = {{-12, -10}, {-8, -6}};
    EXPECT_ALLOC_COUNT(3);
    Matrix result      = std::move(a) - b;
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // b, expected, result
    EXPECT_EQ(result, expected);
}
TEST(Matrix, subtractMoveB) {
    RESET_ALLOC_COUNT();
    Matrix a        = {{1, 2}, {3, 4}};
    Matrix b        = {{13, 12}, {11, 10}};
    Matrix expected = {{-12, -10}, {-8, -6}};
    EXPECT_ALLOC_COUNT(3);
    Matrix result      = a - std::move(b);
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // a, expected, result
    EXPECT_EQ(result, expected);
}
TEST(Matrix, subtractMoveAB) {
    RESET_ALLOC_COUNT();
    Matrix a        = {{1, 2}, {3, 4}};
    Matrix b        = {{13, 12}, {11, 10}};
    Matrix expected = {{-12, -10}, {-8, -6}};
    EXPECT_ALLOC_COUNT(3);
    Matrix result      = std::move(a) - std::move(b);
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

// SquareMatrix
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(SquareMatrix, subtract) {
    SquareMatrix a        = {{1, 2}, {3, 4}};
    SquareMatrix b        = {{13, 12}, {11, 10}};
    SquareMatrix expected = {{-12, -10}, {-8, -6}};
    SquareMatrix result      = a - b;
    EXPECT_EQ(result, expected);
}
TEST(SquareMatrix, subtractMoveA) {
    RESET_ALLOC_COUNT();
    SquareMatrix a        = {{1, 2}, {3, 4}};
    SquareMatrix b        = {{13, 12}, {11, 10}};
    SquareMatrix expected = {{-12, -10}, {-8, -6}};
    EXPECT_ALLOC_COUNT(3);
    SquareMatrix result      = std::move(a) - b;
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // b, expected, result
    EXPECT_EQ(result, expected);
}
TEST(SquareMatrix, subtractMoveB) {
    RESET_ALLOC_COUNT();
    SquareMatrix a        = {{1, 2}, {3, 4}};
    SquareMatrix b        = {{13, 12}, {11, 10}};
    SquareMatrix expected = {{-12, -10}, {-8, -6}};
    EXPECT_ALLOC_COUNT(3);
    SquareMatrix result      = a - std::move(b);
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // a, expected, result
    EXPECT_EQ(result, expected);
}
TEST(SquareMatrix, subtractMoveAB) {
    RESET_ALLOC_COUNT();
    SquareMatrix a        = {{1, 2}, {3, 4}};
    SquareMatrix b        = {{13, 12}, {11, 10}};
    SquareMatrix expected = {{-12, -10}, {-8, -6}};
    EXPECT_ALLOC_COUNT(3);
    SquareMatrix result      = std::move(a) - std::move(b);
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

// Vector
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(Vector, subtract) {
    Vector a        = {1, 2, 3};
    Vector b        = {12, 11, 10};
    Vector expected = {-11, -9, -7};
    Vector result      = a - b;
    EXPECT_EQ(result, expected);
}
TEST(Vector, subtractMoveA) {
    RESET_ALLOC_COUNT();
    Vector a        = {1, 2, 3};
    Vector b        = {12, 11, 10};
    Vector expected = {-11, -9, -7};
    EXPECT_ALLOC_COUNT(3);
    Vector result      = std::move(a) - b;
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // b, expected, result
    EXPECT_EQ(result, expected);
}
TEST(Vector, subtractMoveB) {
    RESET_ALLOC_COUNT();
    Vector a        = {1, 2, 3};
    Vector b        = {12, 11, 10};
    Vector expected = {-11, -9, -7};
    EXPECT_ALLOC_COUNT(3);
    Vector result      = a - std::move(b);
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // a, expected, result
    EXPECT_EQ(result, expected);
}
TEST(Vector, subtractMoveAB) {
    RESET_ALLOC_COUNT();
    Vector a        = {1, 2, 3};
    Vector b        = {12, 11, 10};
    Vector expected = {-11, -9, -7};
    EXPECT_ALLOC_COUNT(3);
    Vector result      = std::move(a) - std::move(b);
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

// RowVector
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(RowVector, subtract) {
    RowVector a        = {1, 2, 3};
    RowVector b        = {12, 11, 10};
    RowVector expected = {-11, -9, -7};
    RowVector result      = a - b;
    EXPECT_EQ(result, expected);
}
TEST(RowVector, subtractMoveA) {
    RESET_ALLOC_COUNT();
    RowVector a        = {1, 2, 3};
    RowVector b        = {12, 11, 10};
    RowVector expected = {-11, -9, -7};
    EXPECT_ALLOC_COUNT(3);
    RowVector result      = std::move(a) - b;
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // b, expected, result
    EXPECT_EQ(result, expected);
}
TEST(RowVector, subtractMoveB) {
    RESET_ALLOC_COUNT();
    RowVector a        = {1, 2, 3};
    RowVector b        = {12, 11, 10};
    RowVector expected = {-11, -9, -7};
    EXPECT_ALLOC_COUNT(3);
    RowVector result      = a - std::move(b);
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // a, expected, result
    EXPECT_EQ(result, expected);
}
TEST(RowVector, subtractMoveAB) {
    RESET_ALLOC_COUNT();
    RowVector a        = {1, 2, 3};
    RowVector b        = {12, 11, 10};
    RowVector expected = {-11, -9, -7};
    EXPECT_ALLOC_COUNT(3);
    RowVector result      = std::move(a) - std::move(b);
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}