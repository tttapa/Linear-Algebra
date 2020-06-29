#include <gtest/gtest.h>

#include <linalg/Matrix.hpp>

#include "CountAllocationsTests.hpp"

TEST(Vector, zeros) {
    Vector m = Vector::zeros(3);
    Vector expected = {0, 0, 0};
    EXPECT_EQ(m, expected);
}

TEST(Vector, ones) {
    Vector m = Vector::ones(3);
    Vector expected = {1, 1, 1};
    EXPECT_EQ(m, expected);
}

TEST(Vector, constant) {
    Vector m = Vector::constant(3, .21);
    Vector expected = {.21, .21, .21};
    EXPECT_EQ(m, expected);
}

TEST(Vector, random) {
    Vector m = Vector::random(128, -10, +100);
    EXPECT_EQ(m.rows(), 128);
    EXPECT_EQ(m.cols(), 1);
    EXPECT_TRUE(std::all_of(m.begin(), m.end(),
                            [](double d) { return d >= -10 && d <= 100; }));
}

TEST(Vector, norm2) {
    Vector m = {11, 12, 13};
    double result1 = m.norm2();
    double result2 = std::move(m).norm2();
    double expected = std::sqrt(11. * 11 + 12 * 12 + 13 * 13);
    EXPECT_FLOAT_EQ(result1, expected);
    EXPECT_FLOAT_EQ(result2, expected);
}