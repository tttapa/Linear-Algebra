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

Direct links to the algorithms:
 - [LU factorization without pivoting](https://tttapa.github.io/Linear-Algebra/Doxygen/d2/d7b/classNoPivotLU.html#a756ec297e96edf953ce640744892a0cc)
 - [LU factorization with row pivoting](https://tttapa.github.io/Linear-Algebra/Doxygen/d6/d1a/classRowPivotLU.html#a756ec297e96edf953ce640744892a0cc)
 - [QR factorization using Householder reflectors](https://tttapa.github.io/Linear-Algebra/Doxygen/d1/dac/classHouseholderQR.html#a756ec297e96edf953ce640744892a0cc)

## Arduino

This library has an Arduino version as well. It's available on the 
[`arduino`](https://github.com/tttapa/Linear-Algebra/tree/arduino) branch.

## Disclaimer

While the code is definitely useful as a linear algebra library, don't use it
in critical applications. Use Eigen instead, it'll be faster, numerically more
accurate, and more reliable.