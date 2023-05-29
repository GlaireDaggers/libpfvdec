cmake_minimum_required(VERSION 3.20)
project(libpfvdec)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sdl2)

add_subdirectory(lib)

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

# TEST EXECUTABLE

add_executable(pfvdec_test main.c)
set_property(TARGET pfvdec_test PROPERTY C_STANDARD 99)

target_include_directories(pfvdec_test PRIVATE lib)
target_link_libraries(pfvdec_test
  libpfvdec
  ${SDL2_LIBRARIES})