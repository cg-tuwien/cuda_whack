cmake_minimum_required(VERSION 3.18)
project(cuda_whack_unittests LANGUAGES CUDA CXX)

include(FetchContent)
FetchContent_Declare(catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.0
    SOURCE_DIR 		${CMAKE_SOURCE_DIR}/3rdparty/Catch2
)
FetchContent_MakeAvailable(catch2)
find_package(CUDAToolkit)

add_executable(whack_unittests
    main.cpp
    cpu_only_kernel_starter.cu
    indexing.cu
    kernel_starter.cu
    tensor_view.cu
    random_number_generator.cu
    rng_state.cu
    Tensor.cu
)
        
target_link_libraries(whack_unittests PUBLIC Catch2::Catch2WithMain CUDA::nvToolsExt whack)

