/****************************************************************************
 *  Copyright (C) 2023 Adam Celarek (github.com/adam-ce, github.com/cg-tuwien)
 * 
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights to
 *  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 *  of the Software, and to permit persons to whom the Software is furnished to do so,
 *  subject to the following conditions:
 * 
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 * 
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 ****************************************************************************/

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
