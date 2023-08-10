cmake_minimum_required(VERSION 3.18)
project(cuda_whack_unittests LANGUAGES CUDA CXX)

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

