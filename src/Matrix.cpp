#include "Matrix.hpp"

#include <iomanip>
#include <iostream>

void Matrix::print(std::ostream &os, int w) const {
    w = w ? w : os.precision() + 9;
    for (size_t r = 0; r < rows(); ++r) {
        for (size_t c = 0; c < cols(); ++c)
            os << std::setw(w) << (*this)(r, c);
        os << std::endl;
    }
}

std::ostream &operator<<(std::ostream &os, const Matrix &M) {
    M.print(os);
    return os;
}