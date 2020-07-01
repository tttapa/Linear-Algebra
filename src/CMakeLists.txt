add_library(linalg SHARED
    "src/Matrix.cpp"
    "src/PermutationMatrix.cpp"
    "src/HouseholderQR.cpp"
    "src/NoPivotLU.cpp"
    "src/RowPivotLU.cpp"
)
add_library(LinearAlgebra::linalg ALIAS linalg)
set_target_properties(linalg PROPERTIES EXPORT_NAME LinearAlgebra::linalg)
target_include_directories(linalg
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

include(GNUInstallDirs)

set(INSTALL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/LinearAlgebra")

# Add the linalg library to the "export-set", install the library files
install(TARGETS linalg
    EXPORT LinearAlgebraTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        COMPONENT shlib
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" 
        COMPONENT lib
)

# Install the header files
INSTALL (DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        COMPONENT dev
    FILES_MATCHING PATTERN "*.h*"
)

# Install the export set for use with the install-tree
install(EXPORT LinearAlgebraTargets 
    FILE LinearAlgebraTargets.cmake
    DESTINATION "${INSTALL_CMAKE_DIR}" 
        COMPONENT dev
)

# Generate the config file that includes the exports
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in"
    "${PROJECT_BINARY_DIR}/LinearAlgebraConfig.cmake"
    INSTALL_DESTINATION "${INSTALL_CMAKE_DIR}"
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/LinearAlgebraConfigVersion.cmake"
    VERSION "${LinearAlgebra_VERSION}"
    COMPATIBILITY SameMajorVersion
)

# Install the LinearAlgebraConfig.cmake and LinearAlgebraConfigVersion.cmake
install(FILES
    "${PROJECT_BINARY_DIR}/LinearAlgebraConfig.cmake"
    "${PROJECT_BINARY_DIR}/LinearAlgebraConfigVersion.cmake"
    DESTINATION "${INSTALL_CMAKE_DIR}" 
        COMPONENT dev)

# Add all targets to the build tree export set
export(EXPORT LinearAlgebraTargets
    FILE "${PROJECT_BINARY_DIR}/LinearAlgebraTargets.cmake")

# Export the package for use from the build tree
# (this registers the build tree with a global CMake-registry, so you can use
# find_package(LinearAlgebra) to find the package in the build tree
export(PACKAGE LinearAlgebra)
