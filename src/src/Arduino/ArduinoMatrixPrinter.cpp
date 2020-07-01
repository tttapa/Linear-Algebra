/**
 * @file
 * @brief   Fallback functions for printing matrices without relying on 
 *          std::ostream or printf.
 */

#include <include/linalg/Arduino/ArduinoConfig.hpp>

#ifdef AVR // AVR has no printf float support, so use dtostre.

#include <stdlib.h>

namespace detail {
char *pad_spaces(char *buf, uint8_t count) {
    while (count-- > 0)
        *buf++ = ' ';
    return buf;
}
void format_double(double d, char *buffer, uint8_t width, uint8_t precision) {
    int pad = width - 6 - precision;
    pad = pad > 0 ? pad : 0;
    buffer = pad_spaces(buffer, pad);
    dtostre(d, buffer, precision - 1, DTOSTR_ALWAYS_SIGN);
}
} // namespace detail

#else // If printf float support available, use it.
#include <stdio.h>
#include <stdint.h>

namespace detail {
void format_double(double d, char *buf, uint8_t width, uint8_t precision) {
    int len = snprintf(buf, width + 1, "%*.*e", (int)width, (int)precision, d);
    // if the text didn't fit the field, fill buffer with *, Fortran style:
    if (len > width)
        while (width-- > 0)
            *buf++ = '*';
}
} // namespace detail

#endif

#ifndef NO_ARDUINO_PRINT_SUPPORT

#include <include/linalg/Matrix.hpp>
#include <new>

void Matrix::print(Print &print, uint8_t precision, uint8_t width) const {
    precision = precision > 0 ? precision : 6;
    width = width > 0 ? width : precision + 9;
    char *buffer = new char[width + 1];
    for (size_t r = 0; r < rows(); ++r) {
        for (size_t c = 0; c < cols(); ++c) {
            detail::format_double((*this)(r, c), buffer, width, precision);
            print.print(buffer);
        }
        print.println();
    }
    delete[] buffer;
}

Print &operator<<(Print &print, const Matrix &M) {
    M.print(print);
    return print;
}

#endif // NO_ARDUINO_PRINT_SUPPORT