cmake_minimum_required(VERSION 3.18)
project(cuda_whack_unittests LANGUAGES CUDA CXX)

add_executable(unittests 
    main.cpp
    cpu_only_kernel_starter.cpp
    indexing.cpp
    tensor_view.cpp
    )

if (CMAKE_CUDA_COMPILER AND WHACK_BUILD_UNITTESTS_FOR_CUDA)
    target_sources(unittests
        PRIVATE
            kernel_starter.cu
            tensor_view.cu
            random_number_generator.cu
            random_number_generator_benchmark.cu
            rng_state.cu
            Tensor.cu
    )
    target_link_libraries(unittests PUBLIC CUDA::nvToolsExt)
endif()

target_link_libraries(unittests PUBLIC Catch2::Catch2WithMain whack)
