cmake_minimum_required(VERSION 3.3)
project(opencl_example)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.c)
add_executable(md5-bruteforce ${SOURCE_FILES})
target_link_libraries(md5-bruteforce OpenCL m)