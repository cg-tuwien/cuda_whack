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

#include <type_traits>

#include <catch2/catch_template_test_macros.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "whack/RandomNumberGenerator.h"
#include "whack/Tensor.h"
#include "whack/kernel.h"


constexpr auto n_batches = 8;

namespace {
template <typename RNG>
whack::Tensor<glm::vec2, 3> compute_random_numbers_with_fixed_seed2()
{
    whack::Tensor<glm::vec2, 3> retval = whack::make_device_tensor<glm::vec2>(n_batches, 16, 1024);

    dim3 dimBlock = dim3(32, 4, 1);
    dim3 dimGrid = dim3(1, 4, n_batches);
    auto view = retval.view();

    whack::start_parallel(retval.device(), dimGrid, dimBlock, [=] __host__ __device__(const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
        const auto sequence_nr = whack::join_n_dim_index<uint64_t, 6, unsigned>(
            { gpe_blockDim.x, gpe_blockDim.y, gpe_blockDim.z, gpe_gridDim.x, gpe_gridDim.y, gpe_gridDim.z },
            { gpe_threadIdx.x, gpe_threadIdx.y, gpe_threadIdx.z, gpe_blockIdx.x, gpe_blockIdx.y, gpe_blockIdx.z });
        auto rng = RNG(55, sequence_nr);
        const unsigned idX = (gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x) * 32;
        const unsigned idY = gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y;
        const unsigned idZ = gpe_blockIdx.z * gpe_blockDim.z + gpe_threadIdx.z;

        for (auto i = 0u; i < 32; ++i)
            view(idZ, idY, idX + i) = rng.normal2();
    });
    return retval;
}

template <typename RNG>
void run_random_number_generator_2d()
{
    compute_random_numbers_with_fixed_seed2<RNG>();
}
}

TEMPLATE_TEST_CASE("random_number_generator benchmark 2d", "", whack::GpuRNGFastGeneration, whack::GpuRNGFastInit)
{
    run_random_number_generator_2d<TestType>();
}
