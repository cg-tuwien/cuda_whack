cmake_minimum_required(VERSION 3.19)
project(cuda_whack_lib LANGUAGES CUDA CXX)

########################################## projects ##############################################
add_library(whack INTERFACE
    whack/kernel.h
    whack/indexing.h
    whack/array.h
    whack/macros.h
    whack/TensorView.h
    whack/Tensor.h
    whack/enums.h
    whack/random/generators.h
    whack/random/state.h
)
target_link_libraries(whack INTERFACE OpenMP::OpenMP_CXX glm::glm)
target_compile_definitions(whack INTERFACE GLM_FORCE_XYZW_ONLY GLM_FORCE_INLINE)
target_include_directories(whack INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if (WHACK_ENABLE_ASSERTS)
    if (UNIX)
        target_compile_options(whack INTERFACE -U NDEBUG)
    else()
        target_compile_options(whack INTERFACE /U NDEBUG)
    endif()
endif()
