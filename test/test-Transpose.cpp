#include <gtest/gtest.h>

#include <HouseholderQR.hpp>
#include <Matrix.hpp>

#include "CountAllocationsTests.hpp"

// Matrix
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(Matrix, transposeRectangular) {
    Matrix a        = {{11, 12, 13}, {21, 22, 23}};
    Matrix expected = {{11, 21}, {12, 22}, {13, 23}};
    Matrix result   = transpose(a);
    EXPECT_EQ(result, expected);
}
TEST(Matrix, transposeRectangularMoveA) {
    RESET_ALLOC_COUNT();
    Matrix a        = {{11, 12, 13}, {21, 22, 23}};
    Matrix expected = {{11, 21}, {12, 22}, {13, 23}};
    EXPECT_ALLOC_COUNT(2);
    Matrix result = transpose(std::move(a));
    EXPECT_ALLOC_COUNT(3); // rectangular transposition requires intermediate
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

TEST(Matrix, transposeSquare) {
    Matrix a        = {{11, 12}, {21, 22}};
    Matrix expected = {{11, 21}, {12, 22}};
    Matrix result   = transpose(a);
    EXPECT_EQ(result, expected);
}
TEST(Matrix, transposeSquareMoveA) {
    RESET_ALLOC_COUNT();
    Matrix a        = {{11, 12}, {21, 22}};
    Matrix expected = {{11, 21}, {12, 22}};
    EXPECT_ALLOC_COUNT(2);
    Matrix result = transpose(std::move(a));
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

TEST(Matrix, transposeVector) {
    Matrix a        = Vector({1, 2, 3});
    Matrix expected = RowVector({1, 2, 3});
    Matrix result   = transpose(a);
    EXPECT_EQ(result, expected);
}
TEST(Matrix, transposeVectorMoveA) {
    RESET_ALLOC_COUNT();
    Matrix a        = Vector({1, 2, 3});
    Matrix expected = RowVector({1, 2, 3});
    EXPECT_ALLOC_COUNT(2);
    Matrix result = transpose(std::move(a));
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

TEST(Matrix, transposeRowVector) {
    Matrix a        = RowVector({1, 2, 3});
    Matrix expected = Vector({1, 2, 3});
    Matrix result   = transpose(a);
    EXPECT_EQ(result, expected);
}
TEST(Matrix, transposeRowVectorMoveA) {
    RESET_ALLOC_COUNT();
    Matrix a        = RowVector({1, 2, 3});
    Matrix expected = Vector({1, 2, 3});
    EXPECT_ALLOC_COUNT(2);
    Matrix result = transpose(std::move(a));
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

// SquareMatrix
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(SquareMatrix, transpose) {
    SquareMatrix a        = {{11, 12}, {21, 22}};
    SquareMatrix expected = {{11, 21}, {12, 22}};
    SquareMatrix result   = transpose(a);
    EXPECT_EQ(result, expected);
}
TEST(SquareMatrix, transposeMoveA) {
    RESET_ALLOC_COUNT();
    SquareMatrix a        = {{11, 12}, {21, 22}};
    SquareMatrix expected = {{11, 21}, {12, 22}};
    EXPECT_ALLOC_COUNT(2);
    SquareMatrix result = transpose(std::move(a));
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

// Vector
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(Vector, transpose) {
    Vector a           = {1, 2, 3};
    RowVector expected = {1, 2, 3};
    RowVector result   = transpose(a);
    EXPECT_EQ(result, expected);
}
TEST(Vector, transposeMoveA) {
    RESET_ALLOC_COUNT();
    Vector a           = {1, 2, 3};
    RowVector expected = {1, 2, 3};
    EXPECT_ALLOC_COUNT(2);
    RowVector result = transpose(std::move(a));
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

// RowVector
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(RowVector, transpose) {
    RowVector a     = {1, 2, 3};
    Vector expected = {1, 2, 3};
    Vector result   = transpose(a);
    EXPECT_EQ(result, expected);
}
TEST(RowVector, transposeMoveA) {
    RESET_ALLOC_COUNT();
    RowVector a     = {1, 2, 3};
    Vector expected = {1, 2, 3};
    EXPECT_ALLOC_COUNT(2);
    Vector result = transpose(std::move(a));
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}