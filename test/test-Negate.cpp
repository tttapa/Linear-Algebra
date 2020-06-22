#include <gtest/gtest.h>

#include <HouseholderQR.hpp>
#include <Matrix.hpp>

#include "CountAllocationsTests.hpp"

// Matrix
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(Matrix, negate) {
    Matrix a        = {{1, 2}, {3, 4}};
    Matrix expected = {{-1, -2}, {-3, -4}};
    Matrix result   = -a;
    EXPECT_EQ(result, expected);
}
TEST(Matrix, negateMoveA) {
    RESET_ALLOC_COUNT();
    Matrix a        = {{1, 2}, {3, 4}};
    Matrix expected = {{-1, -2}, {-3, -4}};
    EXPECT_ALLOC_COUNT(2);
    Matrix result = -std::move(a);
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

// SquareMatrix
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(SquareMatrix, negate) {
    SquareMatrix a        = {{1, 2}, {3, 4}};
    SquareMatrix expected = {{-1, -2}, {-3, -4}};
    SquareMatrix result   = -a;
    EXPECT_EQ(result, expected);
}
TEST(SquareMatrix, negateMoveA) {
    RESET_ALLOC_COUNT();
    SquareMatrix a        = {{1, 2}, {3, 4}};
    SquareMatrix expected = {{-1, -2}, {-3, -4}};
    EXPECT_ALLOC_COUNT(2);
    SquareMatrix result = -std::move(a);
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

// Vector
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(Vector, negate) {
    Vector a        = {1, 2, 3};
    Vector expected = {-1, -2, -3};
    Vector result   = -a;
    EXPECT_EQ(result, expected);
}
TEST(Vector, negateMoveA) {
    RESET_ALLOC_COUNT();
    Vector a        = {1, 2, 3};
    Vector expected = {-1, -2, -3};
    EXPECT_ALLOC_COUNT(2);
    Vector result = -std::move(a);
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

// RowVector
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(RowVector, negate) {
    RowVector a        = {1, 2, 3};
    RowVector expected = {-1, -2, -3};
    RowVector result   = -a;
    EXPECT_EQ(result, expected);
}
TEST(RowVector, negateMoveA) {
    RESET_ALLOC_COUNT();
    RowVector a        = {1, 2, 3};
    RowVector expected = {-1, -2, -3};
    EXPECT_ALLOC_COUNT(2);
    RowVector result = -std::move(a);
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}