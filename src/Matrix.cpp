#include "Matrix.hpp"

#ifndef NO_IOSTREAM_SUPPORT

#include <iomanip>
#include <iostream>

void Matrix::print(std::ostream &os, uint8_t precision, uint8_t width) const {
    int backup_precision = os.precision();
    precision = precision > 0 ? precision : backup_precision;
    width = width > 0 ? width : precision + 9;
    os.precision(precision);
    for (size_t r = 0; r < rows(); ++r) {
        for (size_t c = 0; c < cols(); ++c)
            os << std::setw(width) << (*this)(r, c);
        os << std::endl;
    }
    os.precision(backup_precision);
}


std::ostream &operator<<(std::ostream &os, const Matrix &M) {
    M.print(os);
    return os;
}

#endif
