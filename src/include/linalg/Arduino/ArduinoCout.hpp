#pragma once

/**
 * @file 
 * @brief   Arduino std::ostream Serial wrapper.
 * 
 * The usual "desktop" C/C++ constructs like printf and std::cout don't work on
 * most Arduinos.
 * This class defines an alternative to std::cout: arduino::cout. It behaves 
 * much in the same way as the former, but it writes everything to the Serial
 * output of the Arduino using Serial.write.
 */

#include "ArduinoConfig.hpp"

#ifndef NO_IOSTREAM_SUPPORT

#ifdef ARDUINO_HAS_WORKING_COUT

#include <iostream> // std::cout

namespace arduino {
std::ostream &cout = std::cout;
} // namespace arduino

#else

#include <ostream>   // std::ostream
#include <streambuf> // std::streambuf

namespace arduino {
/// "Buffer" custom std::ostream that wraps the Serial output of the Arduino.
struct SerialBuf : std::streambuf {
    std::streamsize xsputn(const std::streambuf::char_type *s,
                           std::streamsize n) override {
        return Serial.write(s, n);
    }
    std::streambuf::int_type overflow(std::streambuf::int_type c) override {
        Serial.write(char(c));
        return 0; // TODO
    }
} sbuf;
/// Output stream that prints to the Serial monitor.
std::ostream cout(&sbuf);
} // namespace arduino

#endif // ARDUINO_HAS_WORKING_COUT

#endif // NO_IOSTREAM_SUPPORT
