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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "whack/RandomNumberGenerator.h"
#include "whack/kernel.h"
#include "whack/rng/state.h"

namespace {

void run_rng_state_tensor_test()
{
    whack::Tensor<whack::CpuRNG, 1> random_number_state = whack::rng::make_host_state(1);
    whack::TensorView<whack::CpuRNG, 1> random_number_state_view = random_number_state.view();
    auto result = whack::make_host_tensor<float>(1000);
    auto result_view = result.view();

    whack::start_parallel(
        random_number_state.device(), 1, 1, WHACK_KERNEL(random_number_state_view, result_view) {
            WHACK_UNUSED(whack_gridDim);
            WHACK_UNUSED(whack_blockDim);
            WHACK_UNUSED(whack_threadIdx);
            WHACK_UNUSED(whack_blockIdx);

            for (auto i = 0u; i < 1000; ++i)
                result_view(i) = random_number_state_view(0).normal();
        });

    const auto result_vector = result.host_vector();
    const auto mean = thrust::reduce(result_vector.begin(), result_vector.end(), 0.0f, thrust::plus<float>()) / float(result_vector.size());
    const auto two_standard_deviations = 2.f / std::sqrt(float(result_vector.size()));
    CHECK(std::abs(mean) < two_standard_deviations);
}
}

TEST_CASE("rng_state: api", "")
{
    run_rng_state_tensor_test();
}
