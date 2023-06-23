/*
 * Whack
 * Copyright (C) 2023 Adam Celarek
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#include <catch2/catch_test_macros.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "whack/kernel.h"

#define WHACK_UNUSED_THREAD_INDICES WHACK_UNUSED(whack_gridDim) WHACK_UNUSED(whack_blockDim) WHACK_UNUSED(whack_blockIdx) WHACK_UNUSED(whack_threadIdx)

// windows is only happy, if the enclosing function of a host device lambda has external linkage

void kernel_starter_interface() {
    dim3 dimBlock = dim3(1, 1, 1);
    dim3 dimGrid = dim3(1, 1, 1);
    whack::start_parallel(whack::ComputeDevice::CUDA, dimGrid, dimBlock, WHACK_KERNEL() { WHACK_UNUSED_THREAD_INDICES });
}

void kernel_starter_start_on_cuda() {
    thrust::device_vector<int> v(16);
    int* v_ptr = thrust::raw_pointer_cast(v.data());
    dim3 dimBlock = dim3(32, 1, 1);
    dim3 dimGrid = dim3(1, 1, 1);
    whack::start_parallel(
        whack::ComputeDevice::CUDA, dimGrid, dimBlock, WHACK_KERNEL(v_ptr) {
            WHACK_UNUSED_THREAD_INDICES
            if (whack_threadIdx.x >= 16)
                return;
            v_ptr[whack_threadIdx.x] = whack_threadIdx.x;
        });

    thrust::host_vector<int> host_v(v);
    REQUIRE(host_v.size() == 16);
    for (int i = 0; i < 16; ++i) {
        CHECK(host_v[i] == i);
    }
}

void kernel_starter_start_on_cpu() {
    thrust::host_vector<int> v(16);
    int* v_ptr = thrust::raw_pointer_cast(v.data());
    dim3 dimBlock = dim3(32, 1, 1);
    dim3 dimGrid = dim3(1, 1, 1);
    whack::start_parallel(
        whack::ComputeDevice::CPU, dimGrid, dimBlock, WHACK_KERNEL(v_ptr) {
            WHACK_UNUSED_THREAD_INDICES
            if (whack_threadIdx.x >= 16)
                return;
            v_ptr[whack_threadIdx.x] = whack_threadIdx.x;
        });

    REQUIRE(v.size() == 16);
    for (int i = 0; i < 16; ++i) {
        CHECK(v[i] == i);
    }
}

TEST_CASE("kernel_starter.cu")
{

    SECTION("interface")
    {
        kernel_starter_interface();
    }

    SECTION("start on cuda")
    {
        kernel_starter_start_on_cuda();
    }

    SECTION("start on cpu")
    {
        kernel_starter_start_on_cpu();
    }
}
