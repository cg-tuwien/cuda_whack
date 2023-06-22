cmake_minimum_required(VERSION 3.18)
project(cuda_whack_lib LANGUAGES CUDA CXX)

########################################## projects ##############################################
add_library(whack
    whack/kernel.h whack/kernel.cpp
    whack/indexing.h
    whack/array.h
    whack/macros.h
    whack/TensorView.h
    whack/RandomNumberGenerator.h
    whack/Tensor.h
    whack/enums.h
)
target_link_libraries(whack PUBLIC OpenMP::OpenMP_CXX glm::glm)
target_compile_definitions(whack PUBLIC GLM_FORCE_XYZW_ONLY GLM_FORCE_INLINE)
target_include_directories(whack PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (WHACK_ENABLE_ASSERTS)
    if (UNIX)
        target_compile_options(whack PUBLIC -U NDEBUG)
    else()
        target_compile_options(whack PUBLIC /U NDEBUG)
    endif()
endif()
