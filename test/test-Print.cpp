#include <gtest/gtest.h>

#include <linalg/Matrix.hpp>

#include "CountAllocationsTests.hpp"

#include <sstream>

// Matrix
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

TEST(Matrix, print) {
    Matrix a = {{1, 2}, {3, 4}, {5, 6}};
    std::stringstream ss;
    ss.precision(2);
    ss << a;
    std::string expected = //
        "          1          2\n"
        "          3          4\n"
        "          5          6\n";
    EXPECT_EQ(ss.str(), expected);
}
