#include <gtest/gtest.h>

#include <HouseholderQR.hpp>
#include <Matrix.hpp>

#ifdef MATRIX_COUNT_ALLOCATIONS
#define RESET_ALLOC_COUNT() CountingAllocator<double>::total = 0
#define EXPECT_ALLOC_COUNT(n) EXPECT_EQ(CountingAllocator<double>::total, (n))
#else
#define RESET_ALLOC_COUNT()
#define EXPECT_ALLOC_COUNT(n)
#endif

TEST(HouseholderQR, QR) {
    Matrix A = {
        {1, 2, 1},
        {3, 4, 3},
        {1, 2, 3},
        {6, 5, 4},
    };
    HouseholderQR qr(A);

    // std::cout << std::scientific;
    std::cout << std::setprecision(8);
    std::cout << qr << std::endl;

    Matrix QR = qr.get_R();
    qr.apply_Q_inplace(QR);

    std::cout << "Q×R = " << std::endl << QR;

    for (size_t r = 0; r < A.rows(); ++r)
        for (size_t c = 0; c < A.cols(); ++c)
            EXPECT_FLOAT_EQ(A(r, c), QR(r, c)) << "(" << r << ", " << c << ")";
}

TEST(Matrix, addition) {
    Matrix a = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {3, 2, 1},
    };
    Matrix b = {
        {10, 11, 12},
        {13, 14, 15},
        {16, 17, 18},
        {23, 20, 19},
    };
    Matrix expected = {
        {11, 13, 15},
        {17, 19, 21},
        {23, 25, 27},
        {26, 22, 20},
    };
    Matrix sum = a + b;
    EXPECT_EQ(sum, expected);
}

TEST(Matrix, additionAllocationCount) {
    RESET_ALLOC_COUNT();
    Matrix a = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {3, 2, 1},
    };
    EXPECT_ALLOC_COUNT(1);
    Matrix b = {
        {10, 11, 12},
        {13, 14, 15},
        {16, 17, 18},
        {23, 20, 19},
    };
    EXPECT_ALLOC_COUNT(2);
    Matrix expected = {
        {11, 13, 15},
        {17, 19, 21},
        {23, 25, 27},
        {26, 22, 20},
    };
    EXPECT_ALLOC_COUNT(3);
    Matrix sum = std::move(a) + b;
    EXPECT_ALLOC_COUNT(3);
    EXPECT_EQ(sum, expected);
}

TEST(SquareMatrix, addition) {
    SquareMatrix a = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
    };
    SquareMatrix b = {
        {10, 11, 12},
        {13, 14, 15},
        {16, 17, 18},
    };
    SquareMatrix expected = {
        {11, 13, 15},
        {17, 19, 21},
        {23, 25, 27},
    };
    SquareMatrix sum = a + b;
    EXPECT_EQ(sum, expected);
}

TEST(Vector, addition) {
    Vector a        = {1, 2, 3};
    Vector b        = {10, 11, 12};
    Vector expected = {11, 13, 15};
    Vector sum      = a + b;
    EXPECT_EQ(sum, expected);
}

TEST(RowVector, addition) {
    RowVector a        = {1, 2, 3};
    RowVector b        = {10, 11, 12};
    RowVector expected = {11, 13, 15};
    RowVector sum      = a + b;
    EXPECT_EQ(sum, expected);
}

TEST(Matrix, negation) {
    Matrix a = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {3, 2, 1},
    };
    Matrix expected = {
        {-1, -2, -3},
        {-4, -5, -6},
        {-7, -8, -9},
        {-3, -2, -1},
    };
    Matrix negated = -a;
    EXPECT_EQ(negated, expected);
}

TEST(Matrix, negationAllocationCount) {
    RESET_ALLOC_COUNT();
    Matrix a = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {3, 2, 1},
    };
    EXPECT_ALLOC_COUNT(1);
    Matrix expected = {
        {-1, -2, -3},
        {-4, -5, -6},
        {-7, -8, -9},
        {-3, -2, -1},
    };
    EXPECT_ALLOC_COUNT(2);
    Matrix negated = -std::move(a);
    EXPECT_ALLOC_COUNT(2);
    EXPECT_EQ(negated, expected);
}