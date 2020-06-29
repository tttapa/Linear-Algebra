#include <gtest/gtest.h>

#include <linalg/Matrix.hpp>

#include "CountAllocationsTests.hpp"

TEST(Matrix, reshape) {
    Matrix m = {{1, 2, 3}};
    m.reshape(3, 1);
    Matrix expected = Vector({1, 2, 3});
    EXPECT_EQ(m, expected);
}

TEST(Matrix, reshaped) {
    Matrix m = {{1, 2, 3}};
    Matrix result = m.reshaped(3, 1);
    Matrix expected = Vector({1, 2, 3});
    EXPECT_EQ(result, expected);
}

TEST(Matrix, zeros) {
    Matrix m = Matrix::zeros(3, 2);
    Matrix expected = {
        {0, 0},
        {0, 0},
        {0, 0},
    };
    EXPECT_EQ(m, expected);
}

TEST(Matrix, ones) {
    Matrix m = Matrix::ones(3, 2);
    Matrix expected = {
        {1, 1},
        {1, 1},
        {1, 1},
    };
    EXPECT_EQ(m, expected);
}

TEST(Matrix, constant) {
    Matrix m = Matrix::constant(3, 2, .21);
    Matrix expected = {
        {.21, .21},
        {.21, .21},
        {.21, .21},
    };
    EXPECT_EQ(m, expected);
}

TEST(Matrix, identity) {
    Matrix m = Matrix::identity(3, 2);
    Matrix expected = {
        {1, 0},
        {0, 1},
        {0, 0},
    };
    EXPECT_EQ(m, expected);
}

TEST(Matrix, identitySquare) {
    Matrix m = Matrix::identity(3, 3);
    Matrix expected = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
    };
    EXPECT_EQ(m, expected);
}

TEST(Matrix, random) {
    Matrix m = Matrix::random(128, 64, -10, +100);
    EXPECT_EQ(m.rows(), 128);
    EXPECT_EQ(m.cols(), 64);
    EXPECT_TRUE(std::all_of(m.begin(), m.end(),
                            [](double d) { return d >= -10 && d <= 100; }));
}

TEST(Matrix, swaprows) {
    Matrix m = {
        {11, 12, 13},
        {21, 22, 23},
        {31, 32, 33},
    };
    m.swap_rows(0, 2);
    Matrix expected = {
        {31, 32, 33},
        {21, 22, 23},
        {11, 12, 13},
    };
    EXPECT_EQ(m, expected);
}

TEST(Matrix, swapcolumns) {
    Matrix m = {
        {11, 12, 13},
        {21, 22, 23},
        {31, 32, 33},
    };
    m.swap_columns(0, 2);
    Matrix expected = {
        {13, 12, 11},
        {23, 22, 21},
        {33, 32, 31},
    };
    EXPECT_EQ(m, expected);
}

TEST(Matrix, normFro) {
    Matrix m = {
        {11, 12, 13},
        {21, 22, 23},
    };
    double result1 = m.normFro();
    double result2 = std::move(m).normFro();
    double expected =
        std::sqrt(11. * 11 + 12 * 12 + 13 * 13 + 21 * 21 + 22 * 22 + 23 * 23);
    EXPECT_FLOAT_EQ(result1, expected);
    EXPECT_FLOAT_EQ(result2, expected);
}
