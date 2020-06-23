[![Build Status](https://github.com/tttapa/Linear-Algebra/workflows/CI%20Tests/badge.svg)](https://github.com/tttapa/Linear-Algebra/actions)
[![Test Coverage](https://img.shields.io/endpoint?url=https://tttapa.github.io/Linear-Algebra/Coverage/shield.io.coverage.json)](https://tttapa.github.io/Linear-Algebra/Coverage/index.html)
[![GitHub](https://img.shields.io/github/stars/tttapa/Linear-Algebra?label=GitHub&logo=github)](https://github.com/tttapa/Linear-Algebra)


# Linear Algebra

This repo aims to implement some well-known linear algebra algorithms in a 
readable and easy to understand way.  
Established libraries such as BLAS, LAPACK, Eigen etc. offer great performance
and precision, but the code is often hard to read.

## Documentation

[**Documentation**](https://tttapa.github.io/Linear-Algebra/Doxygen/index.html)

The [**modules**](https://tttapa.github.io/Linear-Algebra/Doxygen/modules.html)
page is the best place to start.

## Disclaimer

While the code is definitely useful as a linear algebra library, don't use it
in critical applications. Use Eigen instead, it'll be faster, numerically more
accurate, and more reliable.

## Arduino

This library requires the C++ Standard Library to work correctly. It should work
out of the box on more powerful boards (ARM and Espressif) that ship with the
STL. On platforms that don't support the STL out of the box, like AVR, you can
install the STL through a third-party library.

The main platform used for testing is the ESP32 (currently version 1.0.4 of the
Arduino ESP32 Core).
