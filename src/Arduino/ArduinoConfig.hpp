#pragma once

/**
 * @file
 * @brief   Preprocessor logic for configuring the library to make it compatible
 *          with the Arduino environment.
 */

#ifdef ARDUINO

#include <Arduino/ArduinoMacroFix.hpp>

#ifdef AVR

#define NO_IOSTREAM_SUPPORT

#elif defined(TEENSYDUINO)

#if defined(ARDUINO_TEENSY31) || defined(ARDUINO_TEENSY32) ||                  \
    defined(ARDUINO_TEENSY35) || defined(ARDUINO_TEENSY36)
// Teensy 3.x doesn't support iostream
#define NO_IOSTREAM_SUPPORT
// The STL is partly broken because of the -fsingle-precision-constant flag
#define NO_RANDOM_SUPPORT
#else
// Teensy 4.x supports iostream and random
#endif

#else
// Assume both Arduino Print and std::ostream support, but
// keep in mind that std::cout doesn't work on most boards.
#endif

#ifdef ESP32
#define ARDUINO_HAS_WORKING_COUT
#else
// std::cout doesn't work (Teensy 4.x, for example)
#endif

#else // ARDUINO
#define NO_ARDUINO_PRINT_SUPPORT
#endif