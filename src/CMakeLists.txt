add_library(matrix 
    Matrix.cpp
    PermutationMatrix.cpp
    HouseholderQR.cpp
    LU.cpp
    RowPivotLU.cpp
)
target_include_directories(matrix PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_precompile_headers(matrix
  PUBLIC
    Matrix.hpp
)