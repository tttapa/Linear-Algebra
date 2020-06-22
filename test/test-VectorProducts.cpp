#include <gtest/gtest.h>

#include <HouseholderQR.hpp>
#include <Matrix.hpp>

#include "CountAllocationsTests.hpp"

// Vector
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(Vector, dotProduct) {
    Vector a        = {3, 5, 7};
    Vector b        = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    double result   = a.dot(b);
    EXPECT_EQ(result, expected);
}
TEST(Vector, dotProductMoveA) {
    RESET_ALLOC_COUNT();
    Vector a        = {3, 5, 7};
    Vector b        = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    EXPECT_ALLOC_COUNT(2);
    double result = std::move(a).dot(b);
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(1); // b
    EXPECT_EQ(result, expected);
}
TEST(Vector, dotProductMoveB) {
    RESET_ALLOC_COUNT();
    Vector a        = {3, 5, 7};
    Vector b        = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    EXPECT_ALLOC_COUNT(2);
    double result = a.dot(std::move(b));
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(1); // a
    EXPECT_EQ(result, expected);
}
TEST(Vector, dotProductMoveAB) {
    RESET_ALLOC_COUNT();
    Vector a        = {3, 5, 7};
    Vector b        = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    EXPECT_ALLOC_COUNT(2);
    double result = std::move(a).dot(std::move(b));
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(0);
    EXPECT_EQ(result, expected);
}

TEST(Vector, dotProductStatic) {
    Vector a        = {3, 5, 7};
    Vector b        = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    double result   = Vector::dot(a, b);
    EXPECT_EQ(result, expected);
}
TEST(Vector, dotProductStaticMoveA) {
    RESET_ALLOC_COUNT();
    Vector a        = {3, 5, 7};
    Vector b        = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    EXPECT_ALLOC_COUNT(2);
    double result = Vector::dot(std::move(a), b);
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(1); // b
    EXPECT_EQ(result, expected);
}
TEST(Vector, dotProductStaticMoveB) {
    RESET_ALLOC_COUNT();
    Vector a        = {3, 5, 7};
    Vector b        = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    EXPECT_ALLOC_COUNT(2);
    double result = Vector::dot(a, std::move(b));
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(1); // a
    EXPECT_EQ(result, expected);
}
TEST(Vector, dotProductStaticMoveAB) {
    RESET_ALLOC_COUNT();
    Vector a        = {3, 5, 7};
    Vector b        = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    EXPECT_ALLOC_COUNT(2);
    double result = Vector::dot(std::move(a), std::move(b));
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(0);
    EXPECT_EQ(result, expected);
}

TEST(Vector, crossProduct) {
    Vector a        = {2, 3, 7};
    Vector b        = {11, 13, 17};
    Vector expected = {-40, 43, -7};
    Vector result   = a.cross(b);
    EXPECT_EQ(result, expected);
}
TEST(Vector, crossProductMoveA) {
    RESET_ALLOC_COUNT();
    Vector a        = {2, 3, 7};
    Vector b        = {11, 13, 17};
    Vector expected = {-40, 43, -7};
    EXPECT_ALLOC_COUNT(3);
    Vector result = std::move(a).cross(b);
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // b, expected, result
    EXPECT_EQ(result, expected);
}
TEST(Vector, crossProductMoveB) {
    RESET_ALLOC_COUNT();
    Vector a        = {2, 3, 7};
    Vector b        = {11, 13, 17};
    Vector expected = {-40, 43, -7};
    EXPECT_ALLOC_COUNT(3);
    Vector result = a.cross(std::move(b));
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // a, expected, result
    EXPECT_EQ(result, expected);
}
TEST(Vector, crossProductMoveAB) {
    RESET_ALLOC_COUNT();
    Vector a        = {2, 3, 7};
    Vector b        = {11, 13, 17};
    Vector expected = {-40, 43, -7};
    EXPECT_ALLOC_COUNT(3);
    Vector result = std::move(a).cross(std::move(b));
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

TEST(Vector, crossProductStatic) {
    Vector a        = {2, 3, 7};
    Vector b        = {11, 13, 17};
    Vector expected = {-40, 43, -7};
    Vector result   = Vector::cross(a, b);
    EXPECT_EQ(result, expected);
}
TEST(Vector, crossProductStaticMoveA) {
    RESET_ALLOC_COUNT();
    Vector a        = {2, 3, 7};
    Vector b        = {11, 13, 17};
    Vector expected = {-40, 43, -7};
    EXPECT_ALLOC_COUNT(3);
    Vector result = Vector::cross(std::move(a), b);
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // b, expected, result
    EXPECT_EQ(result, expected);
}
TEST(Vector, crossProductStaticMoveB) {
    RESET_ALLOC_COUNT();
    Vector a        = {2, 3, 7};
    Vector b        = {11, 13, 17};
    Vector expected = {-40, 43, -7};
    EXPECT_ALLOC_COUNT(3);
    Vector result = Vector::cross(a, std::move(b));
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // a, expected, result
    EXPECT_EQ(result, expected);
}
TEST(Vector, crossProductStaticMoveAB) {
    RESET_ALLOC_COUNT();
    Vector a        = {2, 3, 7};
    Vector b        = {11, 13, 17};
    Vector expected = {-40, 43, -7};
    EXPECT_ALLOC_COUNT(3);
    Vector result = Vector::cross(std::move(a), std::move(b));
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

// RowVector
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(RowVector, dotProduct) {
    RowVector a     = {3, 5, 7};
    RowVector b     = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    double result   = a.dot(b);
    EXPECT_EQ(result, expected);
}
TEST(RowVector, dotProductMoveA) {
    RESET_ALLOC_COUNT();
    RowVector a     = {3, 5, 7};
    RowVector b     = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    EXPECT_ALLOC_COUNT(2);
    double result = std::move(a).dot(b);
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(1); // b
    EXPECT_EQ(result, expected);
}
TEST(RowVector, dotProductMoveB) {
    RESET_ALLOC_COUNT();
    RowVector a     = {3, 5, 7};
    RowVector b     = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    EXPECT_ALLOC_COUNT(2);
    double result = a.dot(std::move(b));
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(1); // a
    EXPECT_EQ(result, expected);
}
TEST(RowVector, dotProductMoveAB) {
    RESET_ALLOC_COUNT();
    RowVector a     = {3, 5, 7};
    RowVector b     = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    EXPECT_ALLOC_COUNT(2);
    double result = std::move(a).dot(std::move(b));
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(0);
    EXPECT_EQ(result, expected);
}

TEST(RowVector, dotProductStatic) {
    RowVector a     = {3, 5, 7};
    RowVector b     = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    double result   = RowVector::dot(a, b);
    EXPECT_EQ(result, expected);
}
TEST(RowVector, dotProductStaticMoveA) {
    RESET_ALLOC_COUNT();
    RowVector a     = {3, 5, 7};
    RowVector b     = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    EXPECT_ALLOC_COUNT(2);
    double result = RowVector::dot(std::move(a), b);
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(1); // b
    EXPECT_EQ(result, expected);
}
TEST(RowVector, dotProductStaticMoveB) {
    RESET_ALLOC_COUNT();
    RowVector a     = {3, 5, 7};
    RowVector b     = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    EXPECT_ALLOC_COUNT(2);
    double result = RowVector::dot(a, std::move(b));
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(1); // a
    EXPECT_EQ(result, expected);
}
TEST(RowVector, dotProductStaticMoveAB) {
    RESET_ALLOC_COUNT();
    RowVector a     = {3, 5, 7};
    RowVector b     = {11, 13, 17};
    double expected = 3 * 11 + 5 * 13 + 7 * 17;
    EXPECT_ALLOC_COUNT(2);
    double result = RowVector::dot(std::move(a), std::move(b));
    EXPECT_ALLOC_COUNT(2);
    EXPECT_ALLOC_ALIVE(0);
    EXPECT_EQ(result, expected);
}

TEST(RowVector, crossProduct) {
    RowVector a        = {2, 3, 7};
    RowVector b        = {11, 13, 17};
    RowVector expected = {-40, 43, -7};
    RowVector result   = a.cross(b);
    EXPECT_EQ(result, expected);
}
TEST(RowVector, crossProductMoveA) {
    RESET_ALLOC_COUNT();
    RowVector a        = {2, 3, 7};
    RowVector b        = {11, 13, 17};
    RowVector expected = {-40, 43, -7};
    EXPECT_ALLOC_COUNT(3);
    RowVector result = std::move(a).cross(b);
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // b, expected, result
    EXPECT_EQ(result, expected);
}
TEST(RowVector, crossProductMoveB) {
    RESET_ALLOC_COUNT();
    RowVector a        = {2, 3, 7};
    RowVector b        = {11, 13, 17};
    RowVector expected = {-40, 43, -7};
    EXPECT_ALLOC_COUNT(3);
    RowVector result = a.cross(std::move(b));
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // a, expected, result
    EXPECT_EQ(result, expected);
}
TEST(RowVector, crossProductMoveAB) {
    RESET_ALLOC_COUNT();
    RowVector a        = {2, 3, 7};
    RowVector b        = {11, 13, 17};
    RowVector expected = {-40, 43, -7};
    EXPECT_ALLOC_COUNT(3);
    RowVector result = std::move(a).cross(std::move(b));
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}

TEST(RowVector, crossProductStatic) {
    RowVector a        = {2, 3, 7};
    RowVector b        = {11, 13, 17};
    RowVector expected = {-40, 43, -7};
    RowVector result   = RowVector::cross(a, b);
    EXPECT_EQ(result, expected);
}
TEST(RowVector, crossProductStaticMoveA) {
    RESET_ALLOC_COUNT();
    RowVector a        = {2, 3, 7};
    RowVector b        = {11, 13, 17};
    RowVector expected = {-40, 43, -7};
    EXPECT_ALLOC_COUNT(3);
    RowVector result = RowVector::cross(std::move(a), b);
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // b, expected, result
    EXPECT_EQ(result, expected);
}
TEST(RowVector, crossProductStaticMoveB) {
    RESET_ALLOC_COUNT();
    RowVector a        = {2, 3, 7};
    RowVector b        = {11, 13, 17};
    RowVector expected = {-40, 43, -7};
    EXPECT_ALLOC_COUNT(3);
    RowVector result = RowVector::cross(a, std::move(b));
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(3); // a, expected, result
    EXPECT_EQ(result, expected);
}
TEST(RowVector, crossProductStaticMoveAB) {
    RESET_ALLOC_COUNT();
    RowVector a        = {2, 3, 7};
    RowVector b        = {11, 13, 17};
    RowVector expected = {-40, 43, -7};
    EXPECT_ALLOC_COUNT(3);
    RowVector result = RowVector::cross(std::move(a), std::move(b));
    EXPECT_ALLOC_COUNT(3);
    EXPECT_ALLOC_ALIVE(2); // expected, result
    EXPECT_EQ(result, expected);
}
