cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 11)
project(matrix)

# find_package(GTest REQUIRED PATHS "$ENV{HOME}/.local/lib/cmake" NO_DEFAULT_PATH)
find_package(GTest REQUIRED NO_CMAKE_SYSTEM_PATH)
include(GoogleTest)
enable_testing()

# Set the compiler warnings
set(CMAKE_CPP_FLAGS "${CMAKE_CPP_FLAGS} \
    -Wmissing-include-dirs")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
    -Wall -Wextra -Werror -pedantic -pedantic-errors -fdiagnostics-show-option")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} \
    -Wall -Wextra -Werror -pedantic -pedantic-errors -fdiagnostics-show-option")

# Build-type specific flags
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb3 -O0 -D_GLIBCXX_DEBUG")
set(CMAKE_C_FLAGS_DEBUG   "-ggdb3 -O0 -D_GLIBCXX_DEBUG")

set(CMAKE_CXX_FLAGS_ASAN "-g3 -O0 -D_GLIBCXX_DEBUG \
    -fsanitize=address,leak,undefined,pointer-compare,pointer-subtract")
set(CMAKE_C_FLAGS_ASAN   "-g3 -O0 -D_GLIBCXX_DEBUG \
    -fsanitize=address,leak,undefined,pointer-compare,pointer-subtract")

set(CMAKE_CXX_FLAGS_TSAN "-g3 -O0 -D_GLIBCXX_DEBUG \
    -fsanitize=thread,undefined")
set(CMAKE_C_FLAGS_TSAN   "-g3 -O0 -D_GLIBCXX_DEBUG \
    -fsanitize=thread,undefined")

add_compile_definitions(MATRIX_COUNT_ALLOCATIONS)

add_subdirectory(src)
add_subdirectory(test)

# ---- 

add_executable(qr-perftest qr-perftest.cpp)
target_link_libraries(qr-perftest PRIVATE matrix)