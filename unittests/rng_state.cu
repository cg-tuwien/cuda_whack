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

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "whack/RandomNumberGenerator.h"
#include "whack/kernel.h"
#include "whack/rng/state.h"

namespace {
void run_rng_state_tensor_test(whack::Location location)
{
    auto s1 = whack::rng::make_state(location, 1, 1);
    auto s1_view = s1.view();

    whack::start_parallel(
        location, 1, 1, WHACK_KERNEL(=) {
            WHACK_UNUSED(whack_gridDim);
            WHACK_UNUSED(whack_blockDim);
            WHACK_UNUSED(whack_threadIdx);
            WHACK_UNUSED(whack_blockIdx);
            s1_view(0, 0) = whack::KernelRNG(0, 0);
        });

    auto s2 = whack::rng::make_state(location, 1);
    auto s2_view = s2.view();

    whack::start_parallel(
        location, 1, 1, WHACK_KERNEL(=) {
            WHACK_UNUSED(whack_gridDim);
            WHACK_UNUSED(whack_blockDim);
            WHACK_UNUSED(whack_threadIdx);
            WHACK_UNUSED(whack_blockIdx);
            s2_view(0) = whack::KernelRNG(10, 0);
        });

    auto s3 = whack::rng::make_state<whack::rng::FastInitType>(location, 1);
    auto s3_view = s3.view();

    whack::start_parallel(
        location, 1, 1, WHACK_KERNEL(=) {
            WHACK_UNUSED(whack_gridDim);
            WHACK_UNUSED(whack_blockDim);
            WHACK_UNUSED(whack_threadIdx);
            WHACK_UNUSED(whack_blockIdx);
            s3_view(0) = whack::KernelRNGFastInit(20, 0);
        });

    auto result = whack::make_tensor<float>(location, 1000);
    auto result_view = result.view();

    whack::start_parallel(
        location, 1, 1, WHACK_KERNEL(=) {
            WHACK_UNUSED(whack_gridDim);
            WHACK_UNUSED(whack_blockDim);
            WHACK_UNUSED(whack_threadIdx);
            WHACK_UNUSED(whack_blockIdx);

            assert(s1_view(0).normal() != s2_view(0).normal());

            for (auto i = 0u; i < 1000; ++i)
                result_view(i) = s1_view(0).normal();
        });

    const auto host_copy = result.host_copy();
    const auto result_vector = host_copy.host_vector();
    const auto mean = thrust::reduce(result_vector.begin(), result_vector.end(), 0.0f, thrust::plus<float>()) / float(result_vector.size());
    const auto two_standard_deviations = 2.f / std::sqrt(float(result_vector.size()));
    CHECK(std::abs(mean) < two_standard_deviations);
}

template <typename RngType>
void run_rng_state_tensor_benchmark(const std::string& rng_name)
{
    //    constexpr auto n_batch = 10;
    //    constexpr auto n_blocks = 256;
    //    constexpr auto n_threads = 1024;
    //    constexpr auto n_random_numbers = 1024;
    constexpr auto location = whack::Location::Device;
    constexpr auto n_batch = 10;
    constexpr auto n_blocks = 256;
    constexpr auto n_threads = 1024;
    constexpr auto n_random_numbers = 128;

    auto s1 = whack::rng::make_state<RngType>(location, n_batch, n_blocks, n_threads);
    auto s1_view = s1.view();

    BENCHMARK(std::to_string(n_batch * n_blocks * n_threads) + " calls to whack::" + rng_name + "()")
    {
        whack::start_parallel(
            location, { n_batch, n_blocks, 1 }, { n_threads, 1, 1 }, WHACK_KERNEL(=) {
                WHACK_UNUSED(whack_gridDim);
                WHACK_UNUSED(whack_blockDim);
                // note: using large sequence numbers is very expensive with the fast generation type
                s1_view(whack_blockIdx.x, whack_blockIdx.y, whack_threadIdx.x) = { whack_blockIdx.x * n_threads * n_blocks + whack_blockIdx.y * n_threads + whack_threadIdx.x, 0 };
            });
    };

    auto rnd_sum = whack::make_tensor<float>(location, n_batch, n_blocks, n_threads);
    auto rnd_sum_view = rnd_sum.view();

    BENCHMARK(std::to_string(n_batch * n_blocks * n_threads) + " x " + std::to_string(n_random_numbers) + " calls to whack::" + rng_name + "::normal()")
    {
        whack::start_parallel(
            location, { n_batch, n_blocks, 1 }, { n_threads, 1, 1 }, WHACK_KERNEL(=) {
                WHACK_UNUSED(whack_gridDim);
                WHACK_UNUSED(whack_blockDim);
                WHACK_UNUSED(whack_threadIdx);
                WHACK_UNUSED(whack_blockIdx);

                auto& rng = s1_view(whack_blockIdx.x, whack_blockIdx.y, whack_threadIdx.x);
                float result = 0;
                for (auto i = 0u; i < n_random_numbers; ++i)
                    result += rng.normal();
                rnd_sum_view(whack_blockIdx.x, whack_blockIdx.y, whack_threadIdx.x) = result / n_random_numbers;
            });
    };

    auto rnd_sum_sum = whack::make_tensor<float>(location, n_batch, n_blocks);
    auto rnd_sum_sum_view = rnd_sum_sum.view();

    whack::start_parallel(
        location, { n_batch, 1, 1 }, { n_blocks, 1, 1 }, WHACK_KERNEL(=) {
            WHACK_UNUSED(whack_gridDim);
            WHACK_UNUSED(whack_blockDim);

            float result = 0;
            for (auto i = 0u; i < n_threads; ++i)
                result += rnd_sum_view(whack_blockIdx.x, whack_threadIdx.x, i);
            rnd_sum_sum_view(whack_blockIdx.x, whack_threadIdx.x) = result / n_threads;
        });

    const auto host_copy = rnd_sum_sum.host_copy();
    const auto result_vector = host_copy.host_vector();
    const auto mean = thrust::reduce(result_vector.begin(), result_vector.end(), 0.0f, thrust::plus<float>()) / float(result_vector.size());
    const auto four_standard_deviations = 4.f / std::sqrt(float(n_batch * n_blocks * n_threads * unsigned(n_random_numbers)));
    CHECK(std::abs(mean) < four_standard_deviations);
}
}

TEST_CASE("rng_state: api")
{
    CHECK(whack::rng::StateTensor<whack::rng::FastGenerationType, 1>().location() == whack::Location::Invalid);
    CHECK(whack::rng::StateTensor<whack::rng::FastGenerationType, 1>().raw_pointer() == nullptr);
    CHECK(whack::rng::StateTensor<whack::rng::FastGenerationType, 1>().dimensions()[0] == 0);
    run_rng_state_tensor_test(whack::Location::Host);
    run_rng_state_tensor_test(whack::Location::Device);
}

TEST_CASE("rng_state: benchmark")
{
    run_rng_state_tensor_benchmark<whack::rng::FastGenerationType>("GpuRNGFastGeneration");
    run_rng_state_tensor_benchmark<whack::rng::FastInitType>("GpuRNGFastInit");
}
