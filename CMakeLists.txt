cmake_minimum_required(VERSION 3.18)
project(cuda_whack LANGUAGES CUDA CXX)

option(WHACK_BUILD_UNITTESTS "include unit test targets in the buildsystem" OFF)
option(WHACK_BUILD_UNITTESTS_FOR_CUDA "include unit tests for the cuda side" OFF)
option(WHACK_USE_LLVM_LINKER "use lld (llvm) for linking. it's parallel and much faster, but not installed by default. if it's not installed, you'll get errors, that openmp or other stuff is not installed (hard to track down)" OFF)

######################################## dependencies ###########################################
include(FetchContent)
if(NOT TARGET glm::glm)
    FetchContent_Declare(glm GIT_REPOSITORY https://github.com/g-truc/glm.git)
    FetchContent_MakeAvailable(glm)
endif()
if(WHACK_BUILD_UNITTESTS OR BUILD_TESTING)
    FetchContent_Declare(catch2 GIT_REPOSITORY https://github.com/catchorg/Catch2.git GIT_TAG v3.3.2)
    FetchContent_MakeAvailable(catch2)
endif()

find_package(CUDAToolkit)

############################################ setup ##############################################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

string(APPEND CMAKE_CUDA_FLAGS " --expt-extended-lambda --expt-relaxed-constexpr")
string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -g -G")

if(NOT MSVC)
    #string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler '-fPIC,-fopenmp,-ffast-math,-fno-finite-math-only,-std=c++17'  --use_fast_math")
    string(APPEND CMAKE_CUDA_FLAGS_RELEASE " --generate-line-info -Xcompiler '-O4,-march=native' -O3")

    # gcc 10 is buggy (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=100102)
    #string(APPEND CMAKE_CUDA_FLAGS " -ccbin /usr/bin/g++-9")

    string(APPEND CMAKE_CXX_FLAGS " -ffast-math -fno-finite-math-only")
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O4 -march=native")
endif()

if (WHACK_USE_LLVM_LINKER)
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -fuse-ld=lld")
endif()

find_package(OpenMP REQUIRED)
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

#set(MY_INCLUDE_PATHS
    #${MY_ANACONDA_PATH}/include/python3.9/
    #../glm
    #../yamc/include
    #../gcem/include
    #../autodiff
#)
#include_directories(SYSTEM ${MY_INCLUDE_PATHS} SYSTEM ${TORCH_INCLUDE_DIRS})

########################################## projects ##############################################

add_subdirectory(whack)
if (WHACK_BUILD_UNITTESTS)
    add_subdirectory(unittests)
endif()
