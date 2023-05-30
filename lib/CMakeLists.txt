cmake_minimum_required(VERSION 3.20)
project(libpfvdec)

set(USE_OPENCL OFF CACHE BOOL "Enable GPU decoding via OpenCL")

find_package(OpenMP)

if(OPENMP_FOUND)
    message("Compiling with OpenMP support")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(USE_OPENCL)
    message("Compiling with OpenCL support")
    find_package(OpenCL REQUIRED)
endif()

# TEST EXECUTABLE

add_library(libpfvdec src/bitstream.c src/dec.c src/huffman.c src/dct.c)
set_property(TARGET libpfvdec PROPERTY C_STANDARD 99)

target_include_directories(libpfvdec
    PRIVATE
    inc
)

target_include_directories(libpfvdec
    PUBLIC
    .
)

if(OPENMP_FOUND)
    target_link_libraries(libpfvdec PRIVATE OpenMP::OpenMP_C)
endif()

if(USE_OPENCL)
    target_link_libraries(libpfvdec PRIVATE OpenCL::OpenCL)
    add_definitions(-DUSE_OPENCL)
endif()