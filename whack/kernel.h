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
 */

#include <cuda_runtime.h>
#include <stdexcept>

#include "enums.h"
#include "indexing.h"

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
    void run_cuda_kernel(const dim3& gridDim, const dim3& blockDim, const Fun& function)
    {
        lambda_caller_kernel<<<gridDim, blockDim>>>(function);
        gpu_assert(cudaPeekAtLastError());
        gpu_assert(cudaDeviceSynchronize());
    }
#endif

    template <typename Fun>
    void run_cpu_kernel(const dim3& gridDim, const dim3& blockDim, Fun function)
    {
        const auto n = blockDim.x * blockDim.y * blockDim.z;
        const auto thread_count = n; // std::min(n, 64u);
        (void)thread_count;

        //        gpe::detail::CpuSynchronisationPoint::setThreadCount(thread_count);

        for (unsigned blockIdxZ = 0; blockIdxZ < gridDim.z; ++blockIdxZ) {
            for (unsigned blockIdxY = 0; blockIdxY < gridDim.y; ++blockIdxY) {
                for (unsigned blockIdxX = 0; blockIdxX < gridDim.x; ++blockIdxX) {
                    const auto blockIdx = dim3 { blockIdxX, blockIdxY, blockIdxZ };
#pragma omp parallel for num_threads(thread_count)
                    for (unsigned i = 0; i < n; ++i) {
                        const auto threadIdx_arr = split_n_dim_index<unsigned, 3>({ blockDim.x, blockDim.y, blockDim.z }, i);
                        const auto threadIdx = dim3 { threadIdx_arr[0], threadIdx_arr[1], threadIdx_arr[2] };
                        function(gridDim, blockDim, blockIdx, threadIdx);
                    }
                }
            }
        }
    }
}

template <typename Fun>
void start_parallel(ComputeDevice device, const dim3& gridDim, const dim3& blockDim, const Fun& function)
{
    switch (device) {
#ifdef __CUDACC__
    case ComputeDevice::CUDA:
        detail::run_cuda_kernel(gridDim, blockDim, function);
        break;
#endif
    case ComputeDevice::CPU:
        detail::run_cpu_kernel(gridDim, blockDim, function);
        break;
    default:
        throw std::runtime_error("start_parallel: unsupported device");
        break;
    }
}
}
