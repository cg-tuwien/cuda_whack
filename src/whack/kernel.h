/*****************************************************************************
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

#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

#include "enums.h"
#include "indexing.h"

#define WHACK_KERNEL(...) [__VA_ARGS__] __host__ __device__(const dim3& whack_gridDim, const dim3& whack_blockDim, const dim3& whack_blockIdx, const dim3& whack_threadIdx) mutable
#define WHACK_DEVICE_KERNEL(...) [__VA_ARGS__] __device__(const dim3& whack_gridDim, const dim3& whack_blockDim, const dim3& whack_blockIdx, const dim3& whack_threadIdx) mutable
#define WHACK_UNUSED(x) (void)(x);

namespace whack {
namespace detail {

#ifdef __CUDACC__
    template <typename Fun>
    __global__ void lambda_caller_kernel(Fun function)
    {
        function(gridDim, blockDim, blockIdx, threadIdx);
    }

    inline void gpu_assert(cudaError_t code)
    {
        if (code != cudaSuccess) {
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(code)));
        }
    }

    template <typename Fun>
    void run_cuda_kernel(const dim3& grid_dim, const dim3& block_dim, const Fun& function)
    {
        lambda_caller_kernel<<<grid_dim, block_dim>>>(function);
        gpu_assert(cudaPeekAtLastError());
        gpu_assert(cudaDeviceSynchronize());
    }
#endif

    template <typename Fun>
    void run_cpu_kernel(const dim3& grid_dim, const dim3& block_dim, Fun function)
    {
        const auto n = block_dim.x * block_dim.y * block_dim.z;
        const auto thread_count = n; // std::min(n, 64u);
        (void)thread_count;

        //        gpe::detail::CpuSynchronisationPoint::setThreadCount(thread_count);

        for (unsigned blockIdxZ = 0; blockIdxZ < grid_dim.z; ++blockIdxZ) {
            for (unsigned blockIdxY = 0; blockIdxY < grid_dim.y; ++blockIdxY) {
                for (unsigned blockIdxX = 0; blockIdxX < grid_dim.x; ++blockIdxX) {
                    const auto blockIdx = dim3 { blockIdxX, blockIdxY, blockIdxZ };
#pragma omp parallel for num_threads(thread_count)
                    for (int i = 0; i < int(n); ++i) { // msvc only supports int index variables in OpenMP
                        const auto threadIdx_arr = split_n_dim_index<unsigned, 3>({ block_dim.x, block_dim.y, block_dim.z }, i);
                        const auto threadIdx = dim3 { threadIdx_arr[0], threadIdx_arr[1], threadIdx_arr[2] };
                        function(grid_dim, block_dim, blockIdx, threadIdx);
                    }
                }
            }
        }
    }
} // namespace detail

inline dim3 grid_dim_from_total_size(const dim3& total_size, const dim3& block_dim)
{
    return {
        (total_size.x + block_dim.x - 1) / block_dim.x,
        (total_size.y + block_dim.y - 1) / block_dim.y,
        (total_size.z + block_dim.z - 1) / block_dim.z
    };
}

template <typename Fun>
void start_parallel(Location device, const dim3& grid_dim, const dim3& block_dim, const Fun& function)
{
    switch (device) {
#ifdef __CUDACC__
    case Location::Device:
        static_assert(__nv_is_extended_host_device_lambda_closure_type(Fun), "function must be annotated with __host__ __device__");
        detail::run_cuda_kernel(grid_dim, block_dim, function);
        break;
#endif
    case Location::Host:
        detail::run_cpu_kernel(grid_dim, block_dim, function);
        break;
    default:
        throw std::runtime_error("start_parallel: unsupported device");
        break;
    }
}
} // namespace whack
