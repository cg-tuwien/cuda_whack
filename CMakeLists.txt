cmake_minimum_required(VERSION 3.25)
project(cuda_whack LANGUAGES CUDA CXX)

option(A_BUILD_WHACK_UNITTESTS "include unit test targets for whack in the buildsystem" OFF)

include(cmake/AddRepo.cmake)

add_subdirectory(src)
if (A_BUILD_WHACK_UNITTESTS)
    add_subdirectory(unittests)
endif()
