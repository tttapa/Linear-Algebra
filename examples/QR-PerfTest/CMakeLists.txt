find_package(Eigen3)

if (Eigen3_FOUND)
add_executable(QR-PerfTest QR-PerfTest.cpp)
target_link_libraries(QR-PerfTest PRIVATE LinearAlgebra::linalg Eigen3::Eigen)
endif()