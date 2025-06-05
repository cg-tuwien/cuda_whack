cmake_minimum_required(VERSION 3.18)
project(cuda_whack_unittests LANGUAGES CUDA CXX)

alp_add_git_repository(catch2 URL https://github.com/catchorg/Catch2.git COMMITISH v3.6.0)
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

