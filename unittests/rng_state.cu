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

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "whack/kernel.h"
#include "whack/random/generators.h"
#include "whack/random/state.h"

struct BenchmarkResults {
    float mean;
    float four_std_dev;
};

namespace {
void run_rng_state_tensor_test(whack::Location location)
{
    auto s1 = whack::random::make_state(location, 1, 1);
    auto s1_view = s1.view();

    whack::start_parallel(
        location, 1, 1, WHACK_KERNEL(=) {
            WHACK_UNUSED(whack_gridDim);
            WHACK_UNUSED(whack_blockDim);
            WHACK_UNUSED(whack_threadIdx);
            WHACK_UNUSED(whack_blockIdx);
            s1_view(0, 0) = whack::random::KernelGenerator(0, 0);
        });

    auto s2 = whack::random::make_state(location, 1);
    auto s2_view = s2.view();

    whack::start_parallel(
        location, 1, 1, WHACK_KERNEL(=) {
            WHACK_UNUSED(whack_gridDim);
            WHACK_UNUSED(whack_blockDim);
            WHACK_UNUSED(whack_threadIdx);
            WHACK_UNUSED(whack_blockIdx);
            s2_view(0) = whack::random::KernelGenerator(10, 0);
        });

    auto s3 = whack::random::make_state<whack::random::FastInitType>(location, 1);
    auto s3_view = s3.view();

    whack::start_parallel(
        location, 1, 1, WHACK_KERNEL(=) {
            WHACK_UNUSED(whack_gridDim);
            WHACK_UNUSED(whack_blockDim);
            WHACK_UNUSED(whack_threadIdx);
            WHACK_UNUSED(whack_blockIdx);
            s3_view(0) = whack::random::KernelGeneratorWithFastInit(20, 0);
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

    auto result2 = whack::make_tensor<float>(location, 1000);
    auto result2_view = result2.view();

    whack::start_parallel(
        location, 1, 1, WHACK_KERNEL(=) {
            WHACK_UNUSED(whack_gridDim);
            WHACK_UNUSED(whack_blockDim);
            WHACK_UNUSED(whack_threadIdx);
            WHACK_UNUSED(whack_blockIdx);

            for (auto i = 0u; i < 1000; ++i)
                result2_view(i) = s1_view(0).uniform();
        });

    const auto host2_copy = result2.host_copy();
    const auto result2_vector = host2_copy.host_vector();
    for (auto i = 0u; i < 1000; ++i) {
        CHECK(result2_vector[i] > 0);
        CHECK(result2_vector[i] < 1);
    }
}

template <typename RngType>
BenchmarkResults run_rng_state_tensor_benchmark(const std::string& rng_name)
{
    //    constexpr auto n_batch = 10;
    //    constexpr auto n_blocks = 256;
    //    constexpr auto n_threads = 1024;
    //    constexpr auto n_random_numbers = 1024;
    constexpr auto location = whack::Location::Device;
    constexpr unsigned int n_batch = 10;
    constexpr unsigned int n_blocks = 64;
    constexpr unsigned int n_threads = 1024;
    constexpr unsigned int n_random_numbers = 64;

    auto s1 = whack::random::make_state<RngType>(location, n_batch, n_blocks, n_threads);
    auto s1_view = s1.view();

    BENCHMARK(std::to_string(n_batch * n_blocks * n_threads) + " calls to " + rng_name + " constructor")
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

    BENCHMARK(std::to_string(n_batch * n_blocks * n_threads) + " x " + std::to_string(n_random_numbers) + " calls to " + rng_name + "::normal()")
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
    return { mean, four_standard_deviations };
}
} // namespace

TEST_CASE("rng_state: api")
{
    CHECK(whack::random::StateTensor<whack::random::FastGenerationType, 1>().location() == whack::Location::Invalid);
    CHECK(whack::random::StateTensor<whack::random::FastGenerationType, 1>().raw_pointer() == nullptr);
    CHECK(whack::random::StateTensor<whack::random::FastGenerationType, 1>().dimensions()[0] == 0);
    run_rng_state_tensor_test(whack::Location::Host);
    run_rng_state_tensor_test(whack::Location::Device);
}

TEST_CASE("rng_state: benchmark")
{
    BenchmarkResults results = run_rng_state_tensor_benchmark<whack::random::FastGenerationType>("FastGenerationDeviceGenerator");
    CHECK(std::abs(results.mean) < results.four_std_dev);
    results = run_rng_state_tensor_benchmark<whack::random::FastInitType>("FastInitDeviceGenerator");
    CHECK(std::abs(results.mean) < results.four_std_dev);
}
