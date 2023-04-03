#include <type_traits>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "whack/RandomNumberGenerator.h"
#include "whack/Tensor.h"
#include "whack/kernel.h"

constexpr auto n_batches = 8;

TEST_CASE("random_number_generator.cu: (single threaded)")

{
    {
        auto rng = whack::RandomNumberGenerator<float>(55, { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 });
        const auto rnd1 = rng.normal();
        rng = whack::RandomNumberGenerator<float>(55, { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 });
        const auto rnd2 = rng.normal();
        CHECK(rnd1 == Catch::Approx(rnd2));
    }
}

namespace {
whack::Tensor<float, 3> compute_random_numbers_with_fixed_seed(bool cuda)
{
    auto retval = whack::make_host_tensor<float>(n_batches, 16, 1024);
    if (cuda) {
        retval = retval.device_copy();
    }

    dim3 dimBlock = dim3(32, 4, 1);
    dim3 dimGrid = dim3(1, 4, n_batches);
    auto view = retval.view();

    whack::start_parallel(retval.device(), dimGrid, dimBlock, [=] __host__ __device__(const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
        auto rng = whack::RandomNumberGenerator<float>(55, gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx);
        const unsigned idX = (gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x) * 32;
        const unsigned idY = gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y;
        const unsigned idZ = gpe_blockIdx.z * gpe_blockDim.z + gpe_threadIdx.z;

        for (auto i = 0; i < 32; ++i)
            view(idZ, idY, idX + i) = rng.normal();
    });
    return retval;
}

void run_random_number_generator_1d(bool use_cuda)
{
    auto rnd = compute_random_numbers_with_fixed_seed(use_cuda);
    static_assert(std::is_constructible_v<thrust::host_vector<float>, thrust::host_vector<float>>);
    static_assert(std::is_constructible_v<thrust::host_vector<float>, thrust::device_vector<float>>);
    //    std::cout << rnd << std::endl;

    thrust::host_vector<float> host_vector;
    if (rnd.device() == whack::ComputeDevice::CPU)
        host_vector = std::any_cast<thrust::host_vector<float>>(rnd.memory());
    else
        host_vector = std::any_cast<thrust::device_vector<float>>(rnd.memory());

    REQUIRE(host_vector.size() == n_batches * 16 * 1024);

    // mean
    const auto mean = thrust::reduce(host_vector.begin(), host_vector.end(), 0.0f, thrust::plus<float>()) / host_vector.size();
    const auto two_standard_deviations = 2.f / std::sqrt(float(host_vector.size()));
    CHECK(std::abs(mean) < two_standard_deviations);

    // var
    const auto sqr = [](float v) { return v * v; };
    const float variance = thrust::reduce(
                               thrust::make_transform_iterator(host_vector.begin(), sqr),
                               thrust::make_transform_iterator(host_vector.end(), sqr),
                               0.f, thrust::plus<float>())
        / host_vector.size();
    CHECK(variance == Catch::Approx(1.0).scale(1).epsilon(0.01));

    auto rnd2 = compute_random_numbers_with_fixed_seed(use_cuda);
    const auto rnd_v = rnd.view();
    const auto rnd2_v = rnd2.view();
    whack::start_parallel(rnd2.device(), dim3(32, 16, n_batches), dim3(32, 1, 1), [=] __host__ __device__(const dim3&, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
        const unsigned idX = gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x;
        const unsigned idY = gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y;
        const unsigned idZ = gpe_blockIdx.z * gpe_blockDim.z + gpe_threadIdx.z;
        assert(rnd_v(idZ, idY, idX) == rnd2_v(idZ, idY, idX));
    });
}
}

TEMPLATE_TEST_CASE("random_number_generator 1d", "", std::true_type, std::false_type)
{
    constexpr bool use_cuda = TestType::value;
    run_random_number_generator_1d(use_cuda); // msvc + cuda can't run cuda kernels inside TEST_CASES
}

namespace {
whack::Tensor<glm::vec2, 3> compute_random_numbers_with_fixed_seed2(bool cuda)
{
    whack::Tensor<glm::vec2, 3> retval;
    if (cuda) {
        retval = whack::make_device_tensor<glm::vec2>(n_batches, 16, 1024);
    } else {
        retval = whack::make_host_tensor<glm::vec2>(n_batches, 16, 1024);
    }

    dim3 dimBlock = dim3(32, 4, 1);
    dim3 dimGrid = dim3(1, 4, n_batches);
    auto view = retval.view();

    whack::start_parallel(retval.device(), dimGrid, dimBlock, [=] __host__ __device__(const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
        auto rng = whack::RandomNumberGenerator<float>(55, gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx);
        const unsigned idX = (gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x) * 32;
        const unsigned idY = gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y;
        const unsigned idZ = gpe_blockIdx.z * gpe_blockDim.z + gpe_threadIdx.z;

        for (auto i = 0; i < 32; ++i)
            view(idZ, idY, idX + i) = rng.normal2();
    });
    return retval;
}

void run_random_number_generator_2d(bool use_cuda)
{
    auto rnd = compute_random_numbers_with_fixed_seed2(use_cuda);
    static_assert(std::is_constructible_v<thrust::host_vector<float>, thrust::host_vector<float>>);
    static_assert(std::is_constructible_v<thrust::host_vector<float>, thrust::device_vector<float>>);
    //    std::cout << rnd << std::endl;

    thrust::host_vector<glm::vec2> host_vector = rnd.host_copy().host_vector();
    std::vector<glm::vec2> std_vec;
    std_vec.resize(host_vector.size());
    thrust::copy(host_vector.begin(), host_vector.end(), std_vec.begin());

    REQUIRE(host_vector.size() == n_batches * 16 * 1024);

    // mean
    const auto means = thrust::reduce(host_vector.begin(), host_vector.end(), glm::vec2(0.0f), thrust::plus<glm::vec2>()) / glm::vec2(host_vector.size(), host_vector.size());
    const auto two_standard_deviations = 2.f / std::sqrt(float(host_vector.size()));
    CHECK(std::abs(means.x) < two_standard_deviations);
    CHECK(std::abs(means.y) < two_standard_deviations);

    // var
    const auto sqr = [](glm::vec2 v) { return v * v; };
    const auto variances = thrust::reduce(
                               thrust::make_transform_iterator(host_vector.begin(), sqr),
                               thrust::make_transform_iterator(host_vector.end(), sqr),
                               glm::vec2(0, 0), thrust::plus<glm::vec2>())
        / glm::vec2(host_vector.size(), host_vector.size());
    CHECK(variances.x == Catch::Approx(1.0).scale(2).epsilon(0.01));
    CHECK(variances.y == Catch::Approx(1.0).scale(2).epsilon(0.01));

    // cov
    const auto cov_comp = [](glm::vec2 v) { return v.x * v.y; };
    auto cov = thrust::reduce(
                   thrust::make_transform_iterator(host_vector.begin(), cov_comp),
                   thrust::make_transform_iterator(host_vector.end(), cov_comp),
                   0.f, thrust::plus<float>())
        / host_vector.size();
    CHECK(cov == Catch::Approx(0.0).scale(2).epsilon(0.001));

    // equality when sampling a second time
    auto rnd2 = compute_random_numbers_with_fixed_seed2(use_cuda);
    const auto rnd_v = rnd.view();
    const auto rnd2_v = rnd2.view();
    whack::start_parallel(rnd2.device(), dim3(32, 16, n_batches), dim3(32, 1, 1), [=] __host__ __device__(const dim3&, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
        const unsigned idX = gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x;
        const unsigned idY = gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y;
        const unsigned idZ = gpe_blockIdx.z * gpe_blockDim.z + gpe_threadIdx.z;
        assert(rnd_v(idZ, idY, idX) == rnd2_v(idZ, idY, idX));
    });
}
}

TEMPLATE_TEST_CASE("random_number_generator 2d", "", std::true_type, std::false_type)
{
    constexpr bool use_cuda = TestType::value;
    run_random_number_generator_2d(use_cuda);
}
