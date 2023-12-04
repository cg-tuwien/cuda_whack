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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "whack/kernel.h"

#define WHACK_UNUSED_THREAD_INDICES WHACK_UNUSED(whack_gridDim) WHACK_UNUSED(whack_blockDim) WHACK_UNUSED(whack_blockIdx) WHACK_UNUSED(whack_threadIdx)

inline bool operator==(const dim3& a, const dim3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

// windows is only happy, if the enclosing function of a host device lambda has external linkage
namespace {
void kernel_starter_interface()
{
    dim3 dimBlock = dim3(1, 1, 1);
    dim3 dimGrid = dim3(1, 1, 1);
    whack::start_parallel(whack::Location::Device, dimGrid, dimBlock, WHACK_KERNEL() { WHACK_UNUSED_THREAD_INDICES });
}

void kernel_starter_start_on_cuda()
{
    thrust::device_vector<int> v(16);
    int* v_ptr = thrust::raw_pointer_cast(v.data());
    dim3 dimBlock = dim3(32, 1, 1);
    dim3 dimGrid = dim3(1, 1, 1);
    whack::start_parallel(
        whack::Location::Device, dimGrid, dimBlock, WHACK_DEVICE_KERNEL(v_ptr) {
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

void kernel_starter_start_on_cpu()
{
    thrust::host_vector<int> v(16);
    int* v_ptr = thrust::raw_pointer_cast(v.data());
    dim3 dimBlock = dim3(32, 1, 1);
    dim3 dimGrid = dim3(1, 1, 1);
    whack::start_parallel(
        whack::Location::Host, dimGrid, dimBlock, WHACK_KERNEL(v_ptr) {
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
} // namespace

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

    SECTION("compute grid dim")
    {
        CHECK(whack::grid_dim_from_total_size({ 1, 1, 1 }, { 1, 1, 1 }) == dim3(1, 1, 1));
        CHECK(whack::grid_dim_from_total_size({ 1, 1, 1 }, { 16, 8, 4 }) == dim3(1, 1, 1));
        CHECK(whack::grid_dim_from_total_size({ 16, 16, 16 }, { 16, 8, 4 }) == dim3(1, 2, 4));
        CHECK(whack::grid_dim_from_total_size({ 17, 17, 17 }, { 16, 8, 4 }) == dim3(2, 3, 5));
    }
}
