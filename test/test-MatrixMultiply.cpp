#include <gtest/gtest.h>

#include <HouseholderQR.hpp>
#include <Matrix.hpp>

#include "CountAllocationsTests.hpp"

// Matrix
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(Matrix, matrixMultiply) {
    Matrix a        = {{23, 29, 31}, {37, 41, 43}};
    Matrix b        = {{3, 5}, {7, 11}, {13, 17}};
    Matrix expected = {{675, 961}, {957, 1367}};
    Matrix result   = a * b;
    EXPECT_EQ(result, expected);
}
TEST(Matrix, matrixMultiplyMoveA) {
    RESET_ALLOC_COUNT();
    Matrix a        = {{23, 29, 31}, {37, 41, 43}};
    Matrix b        = {{3, 5}, {7, 11}, {13, 17}};
    Matrix expected = {{675, 961}, {957, 1367}};
    EXPECT_ALLOC_COUNT(3);
    Matrix result = std::move(a) * b;
    EXPECT_ALLOC_COUNT(4); // matrix multiplication requires intermediate
    EXPECT_ALLOC_ALIVE(3); // b, expected, result
    EXPECT_EQ(result, expected);
}
TEST(Matrix, matrixMultiplyMoveB) {
    RESET_ALLOC_COUNT();
    Matrix a        = {{23, 29, 31}, {37, 41, 43}};
    Matrix b        = {{3, 5}, {7, 11}, {13, 17}};
    Matrix expected = {{675, 961}, {957, 1367}};
    EXPECT_ALLOC_COUNT(3);
    Matrix result = a * std::move(b);
    EXPECT_ALLOC_COUNT(4); // matrix multiplication requires intermediate
    EXPECT_ALLOC_ALIVE(3); // a, expected, result
    EXPECT_EQ(result, expected);
}
TEST(Matrix, matrixMultiplyMoveAB) {
    RESET_ALLOC_COUNT();
    Matrix a        = {{23, 29, 31}, {37, 41, 43}};
    Matrix b        = {{3, 5}, {7, 11}, {13, 17}};
    Matrix expected = {{675, 961}, {957, 1367}};
    EXPECT_ALLOC_COUNT(3);
    Matrix result = std::move(a) * std::move(b);
    EXPECT_ALLOC_COUNT(4); // matrix multiplication requires intermediate
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

// SquareMatrix
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(SquareMatrix, matrixMultiply) {
    SquareMatrix a        = {{23, 29}, {37, 41}};
    SquareMatrix b        = {{3, 5}, {7, 11}};
    SquareMatrix expected = {{272, 434}, {398, 636}};
    SquareMatrix result   = a * b;
    EXPECT_EQ(result, expected);
}
TEST(SquareMatrix, matrixMultiplyMoveA) {
    RESET_ALLOC_COUNT();
    SquareMatrix a        = {{23, 29}, {37, 41}};
    SquareMatrix b        = {{3, 5}, {7, 11}};
    SquareMatrix expected = {{272, 434}, {398, 636}};
    EXPECT_ALLOC_COUNT(3);
    SquareMatrix result = std::move(a) * b;
    EXPECT_ALLOC_COUNT(4); // matrix multiplication requires intermediate
    EXPECT_ALLOC_ALIVE(3); // b, expected, result
    EXPECT_EQ(result, expected);
}
TEST(SquareMatrix, matrixMultiplyMoveB) {
    RESET_ALLOC_COUNT();
    SquareMatrix a        = {{23, 29}, {37, 41}};
    SquareMatrix b        = {{3, 5}, {7, 11}};
    SquareMatrix expected = {{272, 434}, {398, 636}};
    EXPECT_ALLOC_COUNT(3);
    SquareMatrix result = a * std::move(b);
    EXPECT_ALLOC_COUNT(4); // matrix multiplication requires intermediate
    EXPECT_ALLOC_ALIVE(3); // a, expected, result
    EXPECT_EQ(result, expected);
}
TEST(SquareMatrix, matrixMultiplyMoveAB) {
    RESET_ALLOC_COUNT();
    SquareMatrix a        = {{23, 29}, {37, 41}};
    SquareMatrix b        = {{3, 5}, {7, 11}};
    SquareMatrix expected = {{272, 434}, {398, 636}};
    EXPECT_ALLOC_COUNT(3);
    SquareMatrix result = std::move(a) * std::move(b);
    EXPECT_ALLOC_COUNT(4); // matrix multiplication requires intermediate
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

// Vector
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(Vector, matrixMultiply) {
    Vector a        = {3, 5, 7};
    RowVector b     = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    double result   = a * b;
    EXPECT_EQ(result, expected);
}
TEST(Vector, matrixMultiplyMoveA) {
    RESET_ALLOC_COUNT();
    Vector a        = {3, 5, 7};
    RowVector b     = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    EXPECT_ALLOC_COUNT(2);
    double result = std::move(a) * b;
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(1); // b
    EXPECT_EQ(result, expected);
}
TEST(Vector, matrixMultiplyMoveB) {
    RESET_ALLOC_COUNT();
    Vector a        = {3, 5, 7};
    RowVector b     = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    EXPECT_ALLOC_COUNT(2);
    double result = a * std::move(b);
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(1); // a
    EXPECT_EQ(result, expected);
}
TEST(Vector, matrixMultiplyMoveAB) {
    RESET_ALLOC_COUNT();
    Vector a        = {3, 5, 7};
    RowVector b     = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    EXPECT_ALLOC_COUNT(2);
    double result = std::move(a) * std::move(b);
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(0);
    EXPECT_EQ(result, expected);
}

TEST(Vector, matrixVectorMultiply) {
    Matrix a        = {{11, 12, 13}, {21, 22, 23}};
    Vector b        = {11, 13, 17};
    Vector expected = {498, 908};
    Vector result   = a * b;
    EXPECT_EQ(result, expected);
}
TEST(Vector, matrixVectorMultiplyMoveA) {
    RESET_ALLOC_COUNT();
    Matrix a        = {{11, 12, 13}, {21, 22, 23}};
    Vector b        = {11, 13, 17};
    Vector expected = {498, 908};
    EXPECT_ALLOC_COUNT(3);
    Vector result = std::move(a) * b;
    EXPECT_ALLOC_COUNT(4);
    EXPECT_ALLOC_ALIVE(3); // b, expected, result
    EXPECT_EQ(result, expected);
}
TEST(Vector, matrixVectorMultiplyMoveB) {
    RESET_ALLOC_COUNT();
    Matrix a        = {{11, 12, 13}, {21, 22, 23}};
    Vector b        = {11, 13, 17};
    Vector expected = {498, 908};
    EXPECT_ALLOC_COUNT(3);
    Vector result = a * std::move(b);
    EXPECT_ALLOC_COUNT(4);
    EXPECT_ALLOC_ALIVE(3); // a, expected, result
    EXPECT_EQ(result, expected);
}
TEST(Vector, matrixVectorMultiplyMoveAB) {
    RESET_ALLOC_COUNT();
    Matrix a        = {{11, 12, 13}, {21, 22, 23}};
    Vector b        = {11, 13, 17};
    Vector expected = {498, 908};
    EXPECT_ALLOC_COUNT(3);
    Vector result = std::move(a) * std::move(b);
    EXPECT_ALLOC_COUNT(4);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

// RowVector
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(RowVector, matrixMultiply) {
    RowVector a     = {3, 5, 7};
    Vector b        = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    double result   = a * b;
    EXPECT_EQ(result, expected);
}
TEST(RowVector, matrixMultiplyMoveA) {
    RESET_ALLOC_COUNT();
    RowVector a     = {3, 5, 7};
    Vector b        = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    EXPECT_ALLOC_COUNT(2);
    double result = std::move(a) * b;
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(1); // b
    EXPECT_EQ(result, expected);
}
TEST(RowVector, matrixMultiplyMoveB) {
    RESET_ALLOC_COUNT();
    RowVector a     = {3, 5, 7};
    Vector b        = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    EXPECT_ALLOC_COUNT(2);
    double result = a * std::move(b);
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(1); // a
    EXPECT_EQ(result, expected);
}
TEST(RowVector, matrixMultiplyMoveAB) {
    RESET_ALLOC_COUNT();
    RowVector a     = {3, 5, 7};
    Vector b        = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    EXPECT_ALLOC_COUNT(2);
    double result = std::move(a) * std::move(b);
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(0);
    EXPECT_EQ(result, expected);
}

TEST(RowVector, matrixVectorMultiply) {
    RowVector a        = {11, 13, 17};
    Matrix b           = {{11, 21}, {12, 22}, {13, 23}};
    RowVector expected = {498, 908};
    RowVector result   = a * b;
    EXPECT_EQ(result, expected);
}
TEST(RowVector, matrixVectorMultiplyMoveA) {
    RESET_ALLOC_COUNT();
    RowVector a        = {11, 13, 17};
    Matrix b           = {{11, 21}, {12, 22}, {13, 23}};
    RowVector expected = {498, 908};
    EXPECT_ALLOC_COUNT(3);
    RowVector result = std::move(a) * b;
    EXPECT_ALLOC_COUNT(4);
    EXPECT_ALLOC_ALIVE(3); // b, expected, result
    EXPECT_EQ(result, expected);
}
TEST(RowVector, matrixVectorMultiplyMoveB) {
    RESET_ALLOC_COUNT();
    RowVector a        = {11, 13, 17};
    Matrix b           = {{11, 21}, {12, 22}, {13, 23}};
    RowVector expected = {498, 908};
    EXPECT_ALLOC_COUNT(3);
    RowVector result = a * std::move(b);
    EXPECT_ALLOC_COUNT(4);
    EXPECT_ALLOC_ALIVE(3); // a, expected, result
    EXPECT_EQ(result, expected);
}
TEST(RowVector, matrixVectorMultiplyMoveAB) {
    RESET_ALLOC_COUNT();
    RowVector a        = {11, 13, 17};
    Matrix b           = {{11, 21}, {12, 22}, {13, 23}};
    RowVector expected = {498, 908};
    EXPECT_ALLOC_COUNT(3);
    RowVector result = std::move(a) * std::move(b);
    EXPECT_ALLOC_COUNT(4);
    EXPECT_ALLOC_ALIVE(2); // expected, result
}
