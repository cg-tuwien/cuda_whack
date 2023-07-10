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

#include <thrust/host_vector.h>

#include "whack/kernel.h"

TEST_CASE("cpu_only_kernel_starter.cpp")
{
    thrust::host_vector<int> v(16);
    int* v_ptr = thrust::raw_pointer_cast(v.data());
    dim3 dimBlock = dim3(32, 1, 1);
    dim3 dimGrid = dim3(1, 1, 1);
    whack::start_parallel(whack::Location::Host, dimGrid, dimBlock, [v_ptr] __host__ __device__(const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
        if (gpe_threadIdx.x >= 16)
            return;
        v_ptr[gpe_threadIdx.x] = gpe_threadIdx.x;
    });

    REQUIRE(v.size() == 16);
    for (int i = 0; i < 16; ++i) {
        CHECK(v[i] == i);
    }
}
