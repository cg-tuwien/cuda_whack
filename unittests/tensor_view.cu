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

#include <cstdint>

#include <catch2/catch_test_macros.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "whack/kernel.h"
#include "whack/tensor_view.h"

void tensor_view_cuda_read_write_multi_dim_cuda()
{
    const thrust::device_vector<int> tensor_1 = std::vector { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    thrust::device_vector<int> tensor_2 = std::vector { 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11 };

    const whack::Array<uint32_t, 4> dimensions = { 1u, 2u, 3u, 2u };
    const auto tensor_1_view = whack::make_tensor_view(tensor_1, dimensions);
    auto tensor_2_view = whack::make_tensor_view(tensor_2, dimensions);

    dim3 dimBlock = dim3(2, 3, 2);
    dim3 dimGrid = dim3(1, 1, 1);
    whack::start_parallel(whack::ComputeDevice::CUDA, dimGrid, dimBlock, [tensor_1_view, tensor_2_view] __host__ __device__(const dim3&, const dim3&, const dim3&, const dim3& gpe_threadIdx) mutable {
        tensor_2_view(0u, gpe_threadIdx.x, gpe_threadIdx.y, gpe_threadIdx.z) = tensor_1_view(0u, gpe_threadIdx.x, gpe_threadIdx.y, gpe_threadIdx.z) * 2;
    });

    thrust::host_vector<int> host_v(tensor_2);
    REQUIRE(host_v.size() == 12);
    for (int i = 0; i < 12; ++i) {
        CHECK(host_v[i] == i * 2);
    }
}

void tensor_view_cuda_read_write_multi_dim_cpu()
{
    const thrust::host_vector<int> tensor_1 = std::vector{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    thrust::host_vector<int> tensor_2 = std::vector{ 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11 };

    const whack::Array<uint32_t, 4> dimensions = { 1u, 2u, 3u, 2u };
    const auto tensor_1_view = whack::make_tensor_view(tensor_1, dimensions);
    auto tensor_2_view = whack::make_tensor_view(tensor_2, dimensions);

    dim3 dimBlock = dim3(2, 3, 2);
    dim3 dimGrid = dim3(1, 1, 1);
    whack::start_parallel(whack::ComputeDevice::CPU, dimGrid, dimBlock, [tensor_1_view, tensor_2_view] __host__ __device__(const dim3&, const dim3&, const dim3&, const dim3 & gpe_threadIdx) mutable {
        tensor_2_view(0u, gpe_threadIdx.x, gpe_threadIdx.y, gpe_threadIdx.z) = tensor_1_view(0u, gpe_threadIdx.x, gpe_threadIdx.y, gpe_threadIdx.z) * 2;
    });

    thrust::host_vector<int> host_v(tensor_2);
    REQUIRE(host_v.size() == 12);
    for (int i = 0; i < 12; ++i) {
        CHECK(host_v[i] == i * 2);
    }
}

TEST_CASE("tensor view (cuda)")
{
    SECTION("read/write multi dim cuda")
    {
        tensor_view_cuda_read_write_multi_dim_cuda();
    }
    SECTION("read/write multi dim cpu")
    {
        tensor_view_cuda_read_write_multi_dim_cpu();
    }
}
