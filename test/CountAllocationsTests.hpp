#pragma once

#ifdef MATRIX_COUNT_ALLOCATIONS
#define RESET_ALLOC_COUNT() CountingAllocator<double>::total = 0;
#define EXPECT_ALLOC_COUNT(n) EXPECT_EQ(CountingAllocator<double>::total, (n))
#define EXPECT_ALLOC_ALIVE(n) EXPECT_EQ(CountingAllocator<double>::alive, (n))
#else
#define RESET_ALLOC_COUNT()
#define EXPECT_ALLOC_COUNT(n)
#define EXPECT_ALLOC_ALIVE(n)
#endif
