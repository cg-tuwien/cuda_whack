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

#include <type_traits>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "whack/Tensor.h"
#include "whack/kernel.h"
#include "whack/random/generators.h"

constexpr auto n_batches = 8;

TEST_CASE("random_number_generator.cu: (single threaded)")
{
    auto rng = whack::random::HostGenerator<float>(55, 0);
    const auto rnd1 = rng.normal();
    rng = whack::random::HostGenerator<float>(55, 0);
    const auto rnd2 = rng.normal();
    CHECK(rnd1 == Catch::Approx(rnd2));
}

namespace {
struct ConfigCudaFastGen {
    using enable_cuda = std::true_type;
    using RNG = whack::random::KernelGeneratorWithFastGeneration;
};
struct ConfigCudaFastOffset {
    using enable_cuda = std::true_type;
    using RNG = whack::random::KernelGeneratorWithFastInit;
};
struct ConfigCpu {
    using enable_cuda = std::false_type;
    using RNG = whack::random::KernelGeneratorWithFastInit;
};

template <typename Config>
whack::Tensor<float, 3> compute_random_numbers_with_fixed_seed()
{
    using RNG = typename Config::RNG;

    auto retval = whack::make_host_tensor<float>(n_batches, 16, 1024);
    if (Config::enable_cuda::value) {
        retval = retval.device_copy();
    }

    dim3 dimBlock = dim3(32, 4, 1);
    dim3 dimGrid = dim3(1, 4, n_batches);
    auto view = retval.view();

    whack::start_parallel(
        retval.location(), dimGrid, dimBlock, WHACK_KERNEL(=) {
            const auto sequence_nr = whack::join_n_dim_index<uint64_t, 6, unsigned>(
                { whack_blockDim.x, whack_blockDim.y, whack_blockDim.z, whack_gridDim.x, whack_gridDim.y, whack_gridDim.z },
                { whack_threadIdx.x, whack_threadIdx.y, whack_threadIdx.z, whack_blockIdx.x, whack_blockIdx.y, whack_blockIdx.z });
            auto rng = RNG(55, sequence_nr);
            const unsigned idX = (whack_blockIdx.x * whack_blockDim.x + whack_threadIdx.x) * 32;
            const unsigned idY = whack_blockIdx.y * whack_blockDim.y + whack_threadIdx.y;
            const unsigned idZ = whack_blockIdx.z * whack_blockDim.z + whack_threadIdx.z;

            for (auto i = 0u; i < 32; ++i)
                view(idZ, idY, idX + i) = rng.normal();
        });
    return retval;
}
} // namespace

struct Results1D {
    float mean;
    float var;
    float two_std_dev;
};

template <typename Config>
Results1D run_random_number_generator_1d()
{
    auto rnd = compute_random_numbers_with_fixed_seed<Config>();
    static_assert(std::is_constructible_v<thrust::host_vector<float>, thrust::host_vector<float>>);
    static_assert(std::is_constructible_v<thrust::host_vector<float>, thrust::device_vector<float>>);
    //    std::cout << rnd << std::endl;

    const auto rnd_host = rnd.host_copy();
    thrust::host_vector<float> host_vector = rnd_host.host_vector();
#ifndef _MSC_VER
    REQUIRE(host_vector.size() == n_batches * 16 * 1024);
#endif

    // mean
    const auto mean = thrust::reduce(host_vector.begin(), host_vector.end(), 0.0f, thrust::plus<float>()) / float(host_vector.size());
    const auto two_standard_deviations = 2.f / std::sqrt(float(host_vector.size()));

    // var
    const auto sqr = [](float v) { return v * v; };
    const float variance = thrust::reduce(
                               thrust::make_transform_iterator(host_vector.begin(), sqr),
                               thrust::make_transform_iterator(host_vector.end(), sqr),
                               0.f, thrust::plus<float>())
        / float(host_vector.size());

    auto rnd2 = compute_random_numbers_with_fixed_seed<Config>();
    const auto rnd_v = rnd.view();
    const auto rnd2_v = rnd2.view();
    whack::start_parallel(
        rnd2.location(), dim3(32, 16, n_batches), dim3(32, 1, 1), WHACK_KERNEL(=) {
            const unsigned idX = whack_blockIdx.x * whack_blockDim.x + whack_threadIdx.x;
            const unsigned idY = whack_blockIdx.y * whack_blockDim.y + whack_threadIdx.y;
            const unsigned idZ = whack_blockIdx.z * whack_blockDim.z + whack_threadIdx.z;
            (void)idX;
            (void)idY;
            (void)idZ;
            assert(rnd_v(idZ, idY, idX) == rnd2_v(idZ, idY, idX));
        });
    return { mean, variance, two_standard_deviations };
}

template <typename Config>
whack::Tensor<glm::vec2, 3> compute_random_numbers_with_fixed_seed2()
{
    using RNG = typename Config::RNG;
    const whack::Location location = Config::enable_cuda::value ? whack::Location::Device : whack::Location::Host;

    auto retval = whack::make_tensor<glm::vec2>(location, n_batches, 16, 1024);

    dim3 dimBlock = dim3(32, 4, 1);
    dim3 dimGrid = dim3(1, 4, n_batches);
    auto view = retval.view();

    whack::start_parallel(
        retval.location(), dimGrid, dimBlock, WHACK_KERNEL(=) {
            const auto sequence_nr = whack::join_n_dim_index<uint64_t, 6, unsigned>(
                { whack_blockDim.x, whack_blockDim.y, whack_blockDim.z, whack_gridDim.x, whack_gridDim.y, whack_gridDim.z },
                { whack_threadIdx.x, whack_threadIdx.y, whack_threadIdx.z, whack_blockIdx.x, whack_blockIdx.y, whack_blockIdx.z });
            auto rng = RNG(55, sequence_nr);
            const unsigned idX = (whack_blockIdx.x * whack_blockDim.x + whack_threadIdx.x) * 32;
            const unsigned idY = whack_blockIdx.y * whack_blockDim.y + whack_threadIdx.y;
            const unsigned idZ = whack_blockIdx.z * whack_blockDim.z + whack_threadIdx.z;

            for (auto i = 0u; i < 32; ++i)
                view(idZ, idY, idX + i) = rng.normal2();
        });
    return retval;
}

struct Results2D {
    glm::vec2 means;
    glm::vec2 vars;
    float cov;
    float two_std_dev;
};

template <typename Config>
Results2D run_random_number_generator_2d()
{
    whack::Tensor<glm::vec2, 3> rnd = compute_random_numbers_with_fixed_seed2<Config>();

    static_assert(std::is_constructible_v<thrust::host_vector<float>, thrust::host_vector<float>>);
    static_assert(std::is_constructible_v<thrust::host_vector<float>, thrust::device_vector<float>>);
    //    std::cout << rnd << std::endl;

    const auto rnd_host = rnd.host_copy();
    thrust::host_vector<glm::vec2> host_vector = rnd_host.host_vector();
    std::vector<glm::vec2> std_vec;
    std_vec.resize(host_vector.size());
    thrust::copy(host_vector.begin(), host_vector.end(), std_vec.begin());

#ifndef _MSC_VER
    REQUIRE(host_vector.size() == n_batches * 16 * 1024);
#endif

    // mean
    const auto means = thrust::reduce(host_vector.begin(), host_vector.end(), glm::vec2(0.0f), thrust::plus<glm::vec2>()) / glm::vec2(host_vector.size(), host_vector.size());
    const auto two_standard_deviations = 2.f / std::sqrt(float(host_vector.size()));

    // var
    const auto sqr = [](glm::vec2 v) { return v * v; };
    const auto variances = thrust::reduce(
                               thrust::make_transform_iterator(host_vector.begin(), sqr),
                               thrust::make_transform_iterator(host_vector.end(), sqr),
                               glm::vec2(0, 0), thrust::plus<glm::vec2>())
        / glm::vec2(host_vector.size(), host_vector.size());

    // cov
    const auto cov_comp = [](glm::vec2 v) { return v.x * v.y; };
    auto cov = thrust::reduce(
                   thrust::make_transform_iterator(host_vector.begin(), cov_comp),
                   thrust::make_transform_iterator(host_vector.end(), cov_comp),
                   0.f, thrust::plus<float>())
        / float(host_vector.size());

    // equality when sampling a second time
    whack::Tensor<glm::vec2, 3> rnd2 = compute_random_numbers_with_fixed_seed2<Config>();

    const auto rnd_v = rnd.view();
    const auto rnd2_v = rnd2.view();
    whack::start_parallel(
        rnd2.location(), dim3(32, 16, n_batches), dim3(32, 1, 1), WHACK_KERNEL(=) {
            const unsigned idX = whack_blockIdx.x * whack_blockDim.x + whack_threadIdx.x;
            const unsigned idY = whack_blockIdx.y * whack_blockDim.y + whack_threadIdx.y;
            const unsigned idZ = whack_blockIdx.z * whack_blockDim.z + whack_threadIdx.z;
            (void)idX;
            (void)idY;
            (void)idZ;
            assert(rnd_v(idZ, idY, idX) == rnd2_v(idZ, idY, idX));
        });
    return { means, variances, cov, two_standard_deviations };
}

TEST_CASE("random_number_generator 1d", "")
{
    // msvc + cuda can't run cuda kernels inside TEST_CASES
    Results1D results = run_random_number_generator_1d<ConfigCudaFastGen>();

    // msvc/nvcc also can't run CHECK/REQUIRE inside tempalted function
    CHECK(std::abs(results.mean) < results.two_std_dev);
    CHECK(results.var == Catch::Approx(1.0).scale(1).epsilon(0.01));

    results = run_random_number_generator_1d<ConfigCudaFastOffset>(); // msvc + cuda can't run cuda kernels inside TEST_CASES

    // msvc/nvcc also can't run CHECK/REQUIRE inside tempalted function
    CHECK(std::abs(results.mean) < results.two_std_dev);
    CHECK(results.var == Catch::Approx(1.0).scale(1).epsilon(0.01));

    results = run_random_number_generator_1d<ConfigCpu>(); // msvc + cuda can't run cuda kernels inside TEST_CASES

    // msvc/nvcc also can't run CHECK/REQUIRE inside tempalted function
    CHECK(std::abs(results.mean) < results.two_std_dev);
    CHECK(results.var == Catch::Approx(1.0).scale(1).epsilon(0.01));
}

// TEMPLATE_TEST_CASE("random_number_generator 2d", "", ConfigCudaFastGen, ConfigCudaFastOffset, ConfigCpu)
TEST_CASE("random_number_generator 2d", "")
{
    Results2D results = run_random_number_generator_2d<ConfigCudaFastGen>();

    CHECK(std::abs(results.means.x) < results.two_std_dev);
    CHECK(std::abs(results.means.y) < results.two_std_dev);
    CHECK(results.vars.x == Catch::Approx(1.0).scale(2).epsilon(0.01));
    CHECK(results.vars.y == Catch::Approx(1.0).scale(2).epsilon(0.01));
    CHECK(results.cov == Catch::Approx(0.0).scale(2).epsilon(0.01));

    results = run_random_number_generator_2d<ConfigCudaFastOffset>();

    CHECK(std::abs(results.means.x) < results.two_std_dev);
    CHECK(std::abs(results.means.y) < results.two_std_dev);
    CHECK(results.vars.x == Catch::Approx(1.0).scale(2).epsilon(0.01));
    CHECK(results.vars.y == Catch::Approx(1.0).scale(2).epsilon(0.01));
    CHECK(results.cov == Catch::Approx(0.0).scale(2).epsilon(0.01));

    results = run_random_number_generator_2d<ConfigCpu>();

    CHECK(std::abs(results.means.x) < results.two_std_dev);
    CHECK(std::abs(results.means.y) < results.two_std_dev);
    CHECK(results.vars.x == Catch::Approx(1.0).scale(2).epsilon(0.01));
    CHECK(results.vars.y == Catch::Approx(1.0).scale(2).epsilon(0.01));
    CHECK(results.cov == Catch::Approx(0.0).scale(2).epsilon(0.01));
}
