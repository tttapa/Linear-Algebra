# Test executable compilation and linking
add_executable(tests test.cpp)
target_include_directories(tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tests PRIVATE GTest::gtest_main GTest::gmock matrix)

# Add tests
gtest_add_tests(TARGET tests)