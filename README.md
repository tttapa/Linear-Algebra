[![Build Status](https://github.com/tttapa/Linear-Algebra/workflows/CI%20Tests/badge.svg)](https://github.com/tttapa/Linear-Algebra/actions)
[![Test Coverage](https://img.shields.io/endpoint?url=https://tttapa.github.io/Linear-Algebra/Coverage/shield.io.coverage.json)](https://tttapa.github.io/Linear-Algebra/Coverage/index.html)
[![GitHub](https://img.shields.io/github/stars/tttapa/Linear-Algebra?label=GitHub&logo=github)](https://github.com/tttapa/Linear-Algebra/tree/arduino)


# Linear Algebra

This repo aims to implement some well-known linear algebra algorithms in a 
readable and easy to understand way.  
Established libraries such as BLAS, LAPACK, Eigen etc. offer great performance
and precision, but the code is often hard to read.

## Documentation

[**Documentation**](https://tttapa.github.io/Linear-Algebra/arduino/Doxygen/index.html)

The [**modules**](https://tttapa.github.io/Linear-Algebra/arduino/Doxygen/modules.html)
page is the best place to start browsing the documentation.

Also see the [**installation instructions**](https://tttapa.github.io/Linear-Algebra/arduino/Doxygen/d8/da8/md_pages_Installation.html)
and the [**examples**](https://tttapa.github.io/Linear-Algebra/arduino/Doxygen/examples.html).

## Arduino

This library requires the C++ Standard Library to work correctly. It should work
out of the box on more powerful boards (ARM and Espressif) that ship with the
STL. On platforms that don't support the STL out of the box, like AVR, you 
might be able to install the STL through a third-party library, but I haven't 
tried this.

The main platforms used for testing are ESP32 (currently version 1.0.4 of the
Arduino ESP32 Core) and Teensy 4.0.

|    Board            | Supported | Comments                                             |
|:--------------------|:---------:|:-----------------------------------------------------|
| ESP32               |     ✔     | Full support.                                        |
| ESP8266             |     ✔     | Full support.                                        |
| Teensy 3.x          |     ✔     | No `std::iostream` support                           |
| Teensy 4.x          |     ✔     | No `std::cout` support, use `arduino::cout` instead. |
| Arduino Zero        |     ✔*    | `std::cout` support unknown.                         |
| Arduino Nano 33 BLE |     ✔*    | `std::cout` support unknown.                         |
| AVR (Uno, Mega ...) |    ❌     | No STL support.                                      |
| Arduino Due         |    ❌     | Toolchain's STL is configured incorrectly.           |

(*) Compiled but untested. If you own such a board, feel free to open an issue
to let me know if it works or not!

For the STL to work correctly on Teensy 4.x, you need a Teensyduino version that
includes [this patch](https://github.com/PaulStoffregen/cores/commit/2f8568659cb7553ca12e5ca2d0358df9d30427a6) 
to the linker scripts.

## Arduino

This library has an Arduino version as well. It's available on the 
[`arduino`](https://github.com/tttapa/Linear-Algebra/tree/arduino) branch.

## Disclaimer

While the code is definitely useful as a linear algebra library, don't use it
in critical applications. If you can, use Eigen instead, it'll be faster, 
numerically more accurate, and more reliable.