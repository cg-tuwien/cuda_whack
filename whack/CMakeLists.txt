cmake_minimum_required(VERSION 3.18)
project(cuda_whack_lib LANGUAGES CUDA CXX)

########################################## projects ##############################################
add_library(whack
    kernel.h kernel.cpp
    indexing.h
    array.h
    macros.h
    TensorView.h
    RandomNumberGenerator.h
    Tensor.h
    enums.h
)
target_link_libraries(whack PUBLIC OpenMP::OpenMP_CXX glm::glm)
target_compile_definitions(whack PUBLIC GLM_FORCE_XYZW_ONLY GLM_FORCE_INLINE)
target_include_directories(whack PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
