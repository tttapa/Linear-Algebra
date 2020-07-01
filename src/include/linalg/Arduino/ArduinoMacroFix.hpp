#pragma once

/**
 * @file
 * @brief   Fixes for problematic macros defined by the Arduino Core.
 * 
 * The Arduino Core defines some macros for abs, min, max, etc. that cause
 * compilation problems when including the C++ standard library headers that
 * define these names as functions (as they should be defined).
 */

#ifdef ARDUINO
#include <Arduino.h>

#ifdef abs
#undef abs
#endif
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#ifdef round
#undef round
#endif

#endif