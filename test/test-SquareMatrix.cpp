#include <gtest/gtest.h>

#include <linalg/Matrix.hpp>

#include "CountAllocationsTests.hpp"

TEST(SquareMatrix, zeros) {
    SquareMatrix m = SquareMatrix::zeros(3);
    SquareMatrix expected = {
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
    };
    EXPECT_EQ(m, expected);
}

TEST(SquareMatrix, ones) {
    SquareMatrix m = SquareMatrix::ones(3);
    SquareMatrix expected = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1},
    };
    EXPECT_EQ(m, expected);
}

TEST(SquareMatrix, constant) {
    SquareMatrix m = SquareMatrix::constant(3, .21);
    SquareMatrix expected = {
        {.21, .21, .21},
        {.21, .21, .21},
        {.21, .21, .21},
    };
    EXPECT_EQ(m, expected);
}

TEST(SquareMatrix, identity) {
    SquareMatrix m = SquareMatrix::identity(3);
    SquareMatrix expected = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
    };
    EXPECT_EQ(m, expected);
}

TEST(SquareMatrix, random) {
    SquareMatrix m = SquareMatrix::random(128, -10, +100);
    EXPECT_EQ(m.rows(), 128);
    EXPECT_EQ(m.cols(), 128);
    EXPECT_TRUE(std::all_of(m.begin(), m.end(),
                            [](double d) { return d >= -10 && d <= 100; }));
}

TEST(SquareMatrix, normFro) {
    SquareMatrix m = {
        {11, 12, 13},
        {21, 22, 23},
        {31, 32, 33},
    };
    double result1 = m.normFro();
    double result2 = std::move(m).normFro();
    double expected =
        std::sqrt(11. * 11 + 12 * 12 + 13 * 13 + 21 * 21 + 22 * 22 + 23 * 23 +
                  31 * 31 + 32 * 32 + 33 * 33);
    EXPECT_FLOAT_EQ(result1, expected);
    EXPECT_FLOAT_EQ(result2, expected);
}

TEST(SquareMatrix, matrixCast) {
    Matrix m = {
        {11, 12, 13},
        {21, 22, 23},
        {31, 32, 33},
    };
    SquareMatrix s = SquareMatrix(m);
    SquareMatrix expected = {
        {11, 12, 13},
        {21, 22, 23},
        {31, 32, 33},
    };
    EXPECT_EQ(s, expected);
}