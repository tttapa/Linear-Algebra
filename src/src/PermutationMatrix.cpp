#ifndef ARDUINO
#include <linalg/PermutationMatrix.hpp>
#else
#include <include/linalg/PermutationMatrix.hpp>
#endif

#ifndef NO_RANDOM_SUPPORT
PermutationMatrix::Permutation PermutationMatrix::random_permutation(
    size_t length, std::default_random_engine::result_type seed) {
    // Create a random engine for shuffling
    std::default_random_engine gen(seed);
    // Start with the identity permutation (0, 1, 2, 3, ..., length-1)
    Permutation permutation = identity_permutation(length);
    // Then shuffle it randomly
    std::random_shuffle(
        permutation.begin(), permutation.end(), [&gen](size_t n) {
            std::uniform_int_distribution<size_t> dist(0, n ? n - 1 : 0);
            return dist(gen);
        });
    return permutation;
}
#endif

PermutationMatrix::Permutation
PermutationMatrix::identity_permutation(size_t length) {
    Permutation permutation(length);
    std::iota(permutation.begin(), permutation.end(), size_t(0));
    return permutation;
}

void PermutationMatrix::fill_from_permutation(Permutation permutation) {
    resize(permutation.size());
    // Convert the permutation to a sequence of swaps.
    //
    // Sort the permuted sequence using selection sort, starting from the
    // right, and record all swaps necessary to sort. This sequence of swaps
    // will be used as the internal representation of the permutation
    // matrix.
    //
    // TODO: Selection sort is O(n²), is there a better way?
    for (size_t i = size(); i-- > 0;) {
        // Boundaries of the sorted and unsorted sublists:
        // | unsorted | sorted |
        auto unsorted_begin = permutation.begin();
        auto unsorted_end = permutation.begin() + i + 1; // one-past-end
        // Find the element that comes at the i-th place in the sorted list:
        auto next_element = std::find(unsorted_begin, unsorted_end, i);
        // Ensure the number exists in the list. If it doesn't, the list
        // didn't contain a valid permutation in the first place.
        assert(next_element != unsorted_end && "Invalid permutation");
        // Find the index of that element:
        auto swap_idx = next_element - permutation.begin();
        // Swap it with the current element, so the *next_element is in the
        // correct place for a sorted list:
        std::swap(permutation[i], *next_element);
        // Record the swap:
        (*this)(i) = swap_idx;
    }
}

#ifndef NO_IOSTREAM_SUPPORT

#include <iomanip>
#include <iostream>

// LCOV_EXCL_START

void PermutationMatrix::print(std::ostream &os, uint8_t precision,
                              uint8_t width) const {
    int backup_precision = os.precision();
    precision = precision > 0 ? precision : backup_precision;
    width = width > 0 ? width : precision + 9;
    os.precision(precision);
    Permutation permutation = to_permutation();
    for (size_t i = 0; i < size(); ++i) {
        os << std::setw(width) << permutation[i];
    }
    os.precision(backup_precision);
}

std::ostream &operator<<(std::ostream &os, const PermutationMatrix &P) {
    P.print(os);
    return os;
}

// LCOV_EXCL_STOP

#endif
