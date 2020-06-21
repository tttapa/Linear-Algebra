#pragma once

#include <memory>

template <class T>
class CountingAllocator {
  public:
    using value_type      = typename std::allocator<T>::value_type;
    using size_type       = typename std::allocator<T>::size_type;
    using difference_type = typename std::allocator<T>::difference_type;
    using propagate_on_container_move_assignment =
        typename std::allocator<T>::propagate_on_container_move_assignment;
    using is_always_equal = typename std::allocator<T>::is_always_equal;

    CountingAllocator() = default;
    CountingAllocator(const CountingAllocator &) = default;
    CountingAllocator(CountingAllocator &&) = default;
    CountingAllocator &operator=(const CountingAllocator &) = default;
    CountingAllocator &operator=(CountingAllocator &&) = default;

    typename std::allocator<T>::pointer allocate(std::size_t n) {
        ++total;
        ++alive;
        return a.allocate(n);
    }
    void deallocate(typename std::allocator<T>::pointer p, std::size_t n) {
        --alive;
        a.deallocate(p, n);
    }

    bool operator==(const CountingAllocator &other) const {
        return this->a == other.a;
    }
    bool operator!=(const CountingAllocator &other) const {
        return this->a != other.a;
    }

    static std::size_t total;
    static std::size_t alive;

  private:
    std::allocator<T> a;
};

template <class T>
std::size_t CountingAllocator<T>::total = 0;
template <class T>
std::size_t CountingAllocator<T>::alive = 0;