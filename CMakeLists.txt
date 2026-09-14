cmake_minimum_required(VERSION 3.20)
project(cpp_ml LANGUAGES CXX)

add_library(cpp_ml
    core/Tensor.cpp
    nn/Linear.cpp
    nn/ReLU.cpp
    nn/NeuralNetwork.cpp
    losses/MSELoss.cpp
)
target_compile_features(cpp_ml PUBLIC cxx_std_23)
set_target_properties(cpp_ml PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(cpp_ml PUBLIC
    ${PROJECT_SOURCE_DIR}/core
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(cpp_ml_example examples/xor.cpp)
target_link_libraries(cpp_ml_example PRIVATE cpp_ml)

include(CTest)
if(BUILD_TESTING)
    add_executable(cpp_ml_tests tests/TensorTests.cpp)
    target_link_libraries(cpp_ml_tests PRIVATE cpp_ml)
    add_test(NAME components COMMAND cpp_ml_tests)
    add_test(NAME xor COMMAND cpp_ml_example)
endif()
