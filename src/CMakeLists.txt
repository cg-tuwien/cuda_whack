cmake_minimum_required(VERSION 3.19)
project(cuda_whack_lib LANGUAGES CUDA CXX)

option(BUILD_WITH_ENABLED_ASSERTS "enable asserts (do not define NDEBUG)" OFF)

if (UNIX)
    option(BUILD_USING_LLVM_LINKER "use lld (llvm) for linking. it's parallel and much faster, but not installed by default.
        if it's not installed, you'll get errors, that openmp or other stuff is not installed (hard to track down)" OFF)
endif()

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
target_compile_options(whack INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)

if(MSVC)
    target_compile_options(whack INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/MP>)
    target_compile_options(whack INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /MP>)
endif()

if (BUILD_USING_LLVM_LINKER)
    target_link_options(whack INTERFACE -fuse-ld=lld)
endif()

if (BUILD_WITH_ENABLED_ASSERTS)
    if (UNIX)
        target_compile_options(whack INTERFACE -U NDEBUG)
    else()
        target_compile_options(whack INTERFACE /U NDEBUG)
    endif()
endif()
