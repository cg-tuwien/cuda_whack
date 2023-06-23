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

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nvtx3/nvToolsExt.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "whack/TensorView.h"
#include "whack/kernel.h"

#define WHACK_UNUSED_THREAD_INDICES WHACK_UNUSED(whack_gridDim) WHACK_UNUSED(whack_blockDim) WHACK_UNUSED(whack_blockIdx) WHACK_UNUSED(whack_threadIdx)

void tensor_view_cuda_read_write_multi_dim_cuda()
{
    const thrust::device_vector<int> tensor_1 = std::vector { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    thrust::device_vector<int> tensor_2 = std::vector { 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11 };

    const whack::Array<uint32_t, 4> dimensions = { 1u, 2u, 3u, 2u };
    const auto tensor_1_view = whack::make_tensor_view(tensor_1, dimensions);
    auto tensor_2_view = whack::make_tensor_view(tensor_2, dimensions);

    dim3 dimBlock = dim3(2, 3, 2);
    dim3 dimGrid = dim3(1, 1, 1);
    whack::start_parallel(
        whack::ComputeDevice::CUDA, dimGrid, dimBlock, WHACK_KERNEL(tensor_1_view, tensor_2_view) {
            WHACK_UNUSED_THREAD_INDICES
            tensor_2_view(0u, whack_threadIdx.x, whack_threadIdx.y, whack_threadIdx.z) = tensor_1_view(0u, whack_threadIdx.x, whack_threadIdx.y, whack_threadIdx.z) * 2;
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
    whack::start_parallel(
        whack::ComputeDevice::CPU, dimGrid, dimBlock, WHACK_KERNEL(tensor_1_view, tensor_2_view) {
            WHACK_UNUSED_THREAD_INDICES
            tensor_2_view(0u, whack_threadIdx.x, whack_threadIdx.y, whack_threadIdx.z) = tensor_1_view(0u, whack_threadIdx.x, whack_threadIdx.y, whack_threadIdx.z) * 2;
        });

    thrust::host_vector<int> host_v(tensor_2);
    REQUIRE(host_v.size() == 12);
    for (int i = 0; i < 12; ++i) {
        CHECK(host_v[i] == i * 2);
    }
}

template <typename IndexCalculationType>
void tensor_view_cuda_benchmark_read_write_multi_dim_cuda()
{
    auto batch_dim = 5000u;
    auto thread_dim = 512u;
    auto vector_dim = 256u;
    thrust::device_vector<int> tensor_1(batch_dim * thread_dim * vector_dim);
    thrust::device_vector<float> tensor_2(batch_dim * thread_dim);
    thrust::sequence(tensor_1.begin(), tensor_1.end());
    thrust::fill(tensor_2.begin(), tensor_2.end(), 0);

    const auto tensor_1_view = whack::make_tensor_view<uint32_t, IndexCalculationType>(tensor_1, batch_dim, vector_dim, thread_dim);
    auto tensor_2_view = whack::make_tensor_view<uint32_t, IndexCalculationType>(tensor_2, batch_dim, thread_dim);

    dim3 dimBlock = dim3(thread_dim, 1, 1);
    dim3 dimGrid = dim3(1, batch_dim, 1);

    cudaDeviceSynchronize();
    BENCHMARK("tensor api")
    {
        const auto nvtx_range = nvtxRangeStart("tensor_api");
        whack::start_parallel(
            whack::ComputeDevice::CUDA, dimGrid, dimBlock, WHACK_KERNEL(tensor_1_view, tensor_2_view, batch_dim, thread_dim, vector_dim) {
                WHACK_UNUSED_THREAD_INDICES
                const auto batch_id = whack_blockIdx.y;
                const auto thread_id = whack_threadIdx.x;
                float tmp = 0;
                for (unsigned i = 0; i < vector_dim; ++i) {
                    tmp += tensor_1_view(batch_id, i, thread_id) / float(batch_dim * thread_dim * vector_dim);
                    const auto i_l = i - unsigned(i > 0);
                    tmp -= tensor_1_view(batch_id, i_l, thread_id);
                    const auto i_h = i + unsigned(i < (vector_dim - 1));
                    tmp += tensor_1_view(batch_id, i_h, thread_id);
                }
                tensor_2_view(batch_id, thread_id) = tmp;
            });
        nvtxRangeEnd(nvtx_range);
        cudaDeviceSynchronize(); // sync not needed, because the kernels depend on each other.
    };

    cudaDeviceSynchronize();
    BENCHMARK("manual api")
    {
        const auto nvtx_range = nvtxRangeStart("manual_api");
        int* tensor_1_ptr = thrust::raw_pointer_cast(tensor_1.data());
        float* tensor_2_ptr = thrust::raw_pointer_cast(tensor_2.data());
        whack::start_parallel(
            whack::ComputeDevice::CUDA, dimGrid, dimBlock, WHACK_KERNEL(tensor_1_ptr, tensor_2_ptr, batch_dim, thread_dim, vector_dim) {
                WHACK_UNUSED_THREAD_INDICES
                const auto batch_id = whack_blockIdx.y;
                const auto thread_id = whack_threadIdx.x;
                float tmp = 0;
                for (unsigned i = 0; i < vector_dim; ++i) {
                    tmp += *(tensor_1_ptr + (IndexCalculationType(batch_id) * thread_dim * vector_dim + i * thread_dim + thread_id)) / float(batch_dim * thread_dim * vector_dim);
                    const auto i_l = i - unsigned(i > 0);
                    tmp -= *(tensor_1_ptr + (IndexCalculationType(batch_id) * thread_dim * vector_dim + i_l * thread_dim + thread_id));
                    const auto i_h = i + unsigned(i < (vector_dim - 1));
                    tmp += *(tensor_1_ptr + (IndexCalculationType(batch_id) * thread_dim * vector_dim + i_h * thread_dim + thread_id));
                }
                *(tensor_2_ptr + (IndexCalculationType(batch_id) * thread_dim + thread_id)) = tmp;
            });
        nvtxRangeEnd(nvtx_range);
        cudaDeviceSynchronize(); // sync not needed, because the kernels depend on each other.
    };
}

TEST_CASE("tensor_view.cu")
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

TEMPLATE_TEST_CASE("tensor_view.cu/benchmark", "", uint32_t, uint64_t)
{
    //SECTION("benchmark read/write multi dim cuda")
    //{
        tensor_view_cuda_benchmark_read_write_multi_dim_cuda<TestType>();
    //}
}
