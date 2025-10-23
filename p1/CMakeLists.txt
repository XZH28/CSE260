cmake_minimum_required(VERSION 3.10)
project(BlockedGEMM CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable optimization by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add common optimization flags
if(MSVC)
    add_compile_options(/O2)
else()
    add_compile_options(-O3 -march=native -funroll-loops)
endif()

# Add source files
set(SOURCES
    test.cpp
    bgemm_naive.cpp
)

# Add the executable
add_executable(BlockedGEMM ${SOURCES})

# Include directory for header
target_include_directories(BlockedGEMM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
