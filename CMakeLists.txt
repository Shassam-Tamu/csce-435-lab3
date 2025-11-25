cmake_minimum_required(VERSION 3.12)

set(CMAKE_CUDA_ARCHITECTURES 80)
project(filter LANGUAGES CUDA CXX)

find_package(caliper REQUIRED)
find_package(adiak REQUIRED)

add_executable(filter filter.cu)

# Enable extended lambda support for CUDA
set_target_properties(filter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(filter PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

message(STATUS "CUDA includes : ${CUDA_INCLUDE_DIRS}")
message(STATUS "Caliper includes : ${caliper_INCLUDE_DIR}")
message(STATUS "Adiak includes : ${adiak_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${caliper_INCLUDE_DIR})
include_directories(${adiak_INCLUDE_DIRS})

target_link_libraries(filter PRIVATE caliper)
