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

// windows is only happy, if the enclosing function of a host device lambda has external linkage

void kernel_starter_interface() {
    dim3 dimBlock = dim3(1, 1, 1);
    dim3 dimGrid = dim3(1, 1, 1);
    whack::start_parallel(whack::ComputeDevice::CUDA, dimGrid, dimBlock, [] __host__ __device__(const dim3&, const dim3&, const dim3&, const dim3&) {});
}

void kernel_starter_start_on_cuda() {
    thrust::device_vector<int> v(16);
    int* v_ptr = thrust::raw_pointer_cast(v.data());
    dim3 dimBlock = dim3(32, 1, 1);
    dim3 dimGrid = dim3(1, 1, 1);
    whack::start_parallel(whack::ComputeDevice::CUDA, dimGrid, dimBlock, [v_ptr] __host__ __device__(const dim3 & gpe_gridDim, const dim3 & gpe_blockDim, const dim3 & gpe_blockIdx, const dim3 & gpe_threadIdx) {
        if (gpe_threadIdx.x >= 16)
            return;
        v_ptr[gpe_threadIdx.x] = gpe_threadIdx.x;
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
    whack::start_parallel(whack::ComputeDevice::CPU, dimGrid, dimBlock, [v_ptr] __host__ __device__(const dim3 & gpe_gridDim, const dim3 & gpe_blockDim, const dim3 & gpe_blockIdx, const dim3 & gpe_threadIdx) {
        if (gpe_threadIdx.x >= 16)
            return;
        v_ptr[gpe_threadIdx.x] = gpe_threadIdx.x;
    });

    REQUIRE(v.size() == 16);
    for (int i = 0; i < 16; ++i) {
        CHECK(v[i] == i);
    }
}

TEST_CASE("kernel_starter: start cuda kernel")
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
