#pragma once

#include <vector>

#ifdef MATRIX_COUNT_ALLOCATIONS

#include "CountingAllocator.hpp"

namespace util {
template <class T>
using storage_t = std::vector<T, CountingAllocator<T>>;
} // namespace util

#else

namespace util {
/// Container to store the elements of a matrix internally.
template <class T>
using storage_t = std::vector<T>;
} // namespace util

#endif
