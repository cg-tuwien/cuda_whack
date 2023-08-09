cmake_minimum_required(VERSION 3.18)
project(cuda_whack LANGUAGES CUDA CXX)

option(BUILD_WHACK_UNITTESTS "include unit test targets for whack in the buildsystem" OFF)


include(FetchContent)
if(NOT TARGET glm::glm)
    FetchContent_Declare(glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        SOURCE_DIR 		${CMAKE_SOURCE_DIR}/3rdparty/glm
    )
    FetchContent_MakeAvailable(glm)
endif()
if(BUILD_WHACK_UNITTESTS OR BUILD_TESTING)
    FetchContent_Declare(catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.3.2
        SOURCE_DIR 		${CMAKE_SOURCE_DIR}/3rdparty/Catch2
    )
    FetchContent_MakeAvailable(catch2)
endif()

find_package(CUDAToolkit)
find_package(OpenMP REQUIRED)

add_subdirectory(src)
if (BUILD_WHACK_UNITTESTS OR BUILD_TESTING)
    add_subdirectory(unittests)
endif()
