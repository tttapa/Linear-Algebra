#include <gtest/gtest.h>

#include <linalg/Matrix.hpp>

#include "CountAllocationsTests.hpp"

TEST(RowVector, zeros) {
    RowVector m = RowVector::zeros(3);
    RowVector expected = {0, 0, 0};
    EXPECT_EQ(m, expected);
}

TEST(RowVector, ones) {
    RowVector m = RowVector::ones(3);
    RowVector expected = {1, 1, 1};
    EXPECT_EQ(m, expected);
}

TEST(RowVector, constant) {
    RowVector m = RowVector::constant(3, .21);
    RowVector expected = {.21, .21, .21};
    EXPECT_EQ(m, expected);
}

TEST(RowVector, random) {
    RowVector m = RowVector::random(128, -10, +100);
    EXPECT_EQ(m.rows(), 1);
    EXPECT_EQ(m.cols(), 128);
    EXPECT_TRUE(std::all_of(m.begin(), m.end(),
                            [](double d) { return d >= -10 && d <= 100; }));
}

TEST(RowVector, norm2) {
    RowVector m = {11, 12, 13};
    double result1 = m.norm2();
    double result2 = std::move(m).norm2();
    double expected = std::sqrt(11. * 11 + 12 * 12 + 13 * 13);
    EXPECT_FLOAT_EQ(result1, expected);
    EXPECT_FLOAT_EQ(result2, expected);
}