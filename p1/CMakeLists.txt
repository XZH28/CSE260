cmake_minimum_required(VERSION 3.10)
project(BlockedGEMM CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile_commands.json (handy for tooling)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# If not specified, default to Debug (better for development)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Option to enable sanitizers (only meaningful for Debug/RelWithDebInfo)
option(ENABLE_ASAN "Enable AddressSanitizer" ON)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" ON)

# Sources
set(SOURCES
  main.cpp
)

add_executable(BlockedGEMM ${SOURCES})
target_include_directories(BlockedGEMM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Compiler / linker flags per compiler and configuration
if(MSVC)
  # Debug: symbols, no optimization, runtime checks
  target_compile_options(BlockedGEMM PRIVATE
    $<$<CONFIG:Debug>:/Zi /Od /RTC1>
    $<$<CONFIG:RelWithDebInfo>:/Zi /O2>
    $<$<CONFIG:Release>:/O2>
  )
  target_link_options(BlockedGEMM PRIVATE
    $<$<CONFIG:Debug>:/INCREMENTAL>
    $<$<CONFIG:RelWithDebInfo>:/INCREMENTAL>
  )
else()
  # GCC/Clang
  target_compile_options(BlockedGEMM PRIVATE
    # Debug: full symbols, no opts, keep frame pointers for better stacks
    $<$<CONFIG:Debug>:-g3 -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls>
    # RelWithDebInfo: optimized but with symbols
    $<$<CONFIG:RelWithDebInfo>:-g3 -O2 -fno-omit-frame-pointer>
    # Release: your original tuning
    $<$<CONFIG:Release>:-O3 -march=native -funroll-loops>
  )
  # Linker opts (mostly relevant for sanitizers)
  # (empty by default; populated below when sanitizers are enabled)
endif()

# Sanitizers (only non-MSVC)
if(NOT MSVC)
  set(_SAN_FLAGS "")
  if(ENABLE_ASAN)
    list(APPEND _SAN_FLAGS "-fsanitize=address")
  endif()
  if(ENABLE_UBSAN)
    list(APPEND _SAN_FLAGS "-fsanitize=undefined")
  endif()

  if(_SAN_FLAGS)
    # Only apply sanitizers to Debug / RelWithDebInfo
    target_compile_options(BlockedGEMM PRIVATE
      $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:${_SAN_FLAGS}>
    )
    target_link_options(BlockedGEMM PRIVATE
      $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:${_SAN_FLAGS}>
    )
  endif()
endif()
