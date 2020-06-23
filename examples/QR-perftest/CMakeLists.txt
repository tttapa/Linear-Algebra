find_package(Eigen3)

if (Eigen3_FOUND)
add_executable(qr-perftest qr-perftest.cpp)
target_link_libraries(qr-perftest PRIVATE matrix Eigen3::Eigen)
endif()