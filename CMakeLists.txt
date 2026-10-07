cmake_minimum_required(VERSION 3.16)
project(chess_engine C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Default target: AVX2 + BMI2 (laptop and Kaggle). Override with -DENGINE_ARCH=...
set(ENGINE_ARCH "x86-64-v3" CACHE STRING "-march value")
option(USE_PEXT "Use PEXT sliding attacks (slow on Zen 2; off by default)" OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -march=${ENGINE_ARCH}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=${ENGINE_ARCH}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer")

add_library(core STATIC
  src/bitboard.cpp
  src/position.cpp
  src/movegen.cpp
  src/perft.cpp
  src/tt.cpp
  src/eval.cpp
  src/search.cpp
  src/bench.cpp
  src/tablebase.cpp
  third_party/pyrrhic/tbprobe.c)
find_package(Threads REQUIRED)
target_link_libraries(core PUBLIC Threads::Threads)
target_include_directories(core PUBLIC src third_party/pyrrhic)
set_source_files_properties(third_party/pyrrhic/tbprobe.c PROPERTIES COMPILE_FLAGS "-O2 -w")
if(USE_PEXT)
  target_compile_definitions(core PUBLIC USE_PEXT)
endif()

add_executable(bulwark src/main.cpp)
target_link_libraries(bulwark core)

add_executable(perft tools/perft_main.cpp)
target_link_libraries(perft core)

add_executable(run_tests tests/test_main.cpp)
target_link_libraries(run_tests core)

enable_testing()
add_test(NAME unit COMMAND run_tests ${CMAKE_SOURCE_DIR}/tests/data)
