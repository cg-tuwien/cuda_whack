cmake_minimum_required(VERSION 3.18)
project(cuda_whack LANGUAGES CUDA CXX)

option(BUILD_WHACK_UNITTESTS "include unit test targets for whack in the buildsystem" OFF)

add_subdirectory(src)
if (BUILD_WHACK_UNITTESTS OR BUILD_TESTING)
    add_subdirectory(unittests)
endif()
