cmake_minimum_required(VERSION 3.20)

project(cpp_ml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cpp_ml
    core/Tensor.cpp
)

target_include_directories(cpp_ml
    PUBLIC
        ${PROJECT_SOURCE_DIR}/core
        ${PROJECT_SOURCE_DIR}/include
)

add_executable(cpp_ml_example
    nn/main.cpp
)

target_link_libraries(cpp_ml_example
    PRIVATE cpp_ml
)

