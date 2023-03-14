#include <type_traits>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "whack/RandomNumberGenerator.h"
#include "whack/Tensor.h"
#include "whack/kernel.h"

TEST_CASE("random_number_generator single threaded")
{
    {
        auto rng = whack::RandomNumberGenerator<float>(55, { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 });
        const auto rnd1 = rng.normal();
        rng = whack::RandomNumberGenerator<float>(55, { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 });
        const auto rnd2 = rng.normal();
        CHECK(rnd1 == Catch::Approx(rnd2));
    }
}

whack::Tensor<float, 3> compute_random_numbers_with_fixed_seed(bool cuda)
{
    whack::Tensor<float, 3> retval;
    if (cuda) {
        thrust::device_vector<float> memory(16 * 16 * 1024);
        static_assert(std::is_move_assignable_v<thrust::device_vector<float>>, "thrust::host_vector<float> must be movable");
        retval.view = whack::make_tensor_view(memory, 16u, 16u, 1024u);
        retval.memory = std::move(memory);
        retval.device = whack::ComputeDevice::CUDA;
    } else {
        thrust::host_vector<float> memory(16 * 16 * 1024);
        static_assert(std::is_move_assignable_v<thrust::host_vector<float>>, "thrust::host_vector<float> must be movable");
        retval.view = whack::make_tensor_view(memory, 16u, 16u, 1024u);
        retval.memory = std::move(memory);
        retval.device = whack::ComputeDevice::CPU;
    }

    dim3 dimBlock = dim3(32, 4, 1);
    dim3 dimGrid = dim3(1, 4, 16);
    auto view = retval.view;
    //        gpe::start_serial<gpe::ComputeDevice::CPU>(gpe::device(out_mixture), dimGrid, dimBlock, [=]
    whack::start_parallel(retval.device, dimGrid, dimBlock, [=] __host__ __device__(const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
        auto rng = whack::RandomNumberGenerator<float>(55, gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx);
        const unsigned idX = (gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x) * 32;
        const unsigned idY = gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y;
        const unsigned idZ = gpe_blockIdx.z * gpe_blockDim.z + gpe_threadIdx.z;

        for (auto i = 0; i < 32; ++i)
            view(idZ, idY, idX + i) = rng.normal();
    });
    return retval;
}

TEMPLATE_TEST_CASE("random_number_generator", "", std::true_type, std::false_type)
{
    constexpr bool use_cuda = TestType::value;
    auto rnd = compute_random_numbers_with_fixed_seed(use_cuda);
    static_assert(std::is_constructible_v<thrust::host_vector<float>, thrust::host_vector<float>>);
    static_assert(std::is_constructible_v<thrust::host_vector<float>, thrust::device_vector<float>>);
    //    std::cout << rnd << std::endl;

    thrust::host_vector<float> host_vector;
    if (rnd.device == whack::ComputeDevice::CPU)
        host_vector = std::any_cast<thrust::host_vector<float>>(rnd.memory);
    else
        host_vector = std::any_cast<thrust::device_vector<float>>(rnd.memory);

    REQUIRE(host_vector.size() == 16 * 16 * 1024);

    // mean
    CHECK(thrust::reduce(host_vector.begin(), host_vector.end(), 0.0f, thrust::plus<float>()) / host_vector.size() == Catch::Approx(0.0).scale(1).epsilon(0.001));

    // var
    const auto sqr = [](float v) { return v * v; };
    CHECK(thrust::reduce(
              thrust::make_transform_iterator(host_vector.begin(), sqr),
              thrust::make_transform_iterator(host_vector.end(), sqr),
              0.0f, thrust::plus<float>())
            / host_vector.size()
        == Catch::Approx(1.0).scale(1).epsilon(0.01));

    //    CHECK((rnd == torch::zeros_like(rnd)).sum().cpu().item<int64_t>() < rnd.numel() * 0.001);
    //    CHECK(rnd.view({-1, 2}).cov().cpu().item<float>() == Approx(1.0).scale(1).epsilon(0.001));
    auto rnd2 = compute_random_numbers_with_fixed_seed(use_cuda);
    const auto rnd_v = rnd.view;
    const auto rnd2_v = rnd2.view;
    whack::start_parallel(rnd2.device, dim3(32, 16, 16), dim3(32, 1, 1), [=] __host__ __device__(const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
        const unsigned idX = gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x;
        const unsigned idY = gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y;
        const unsigned idZ = gpe_blockIdx.z * gpe_blockDim.z + gpe_threadIdx.z;
        assert(rnd_v(idZ, idY, idX) == rnd2_v(idZ, idY, idX));
    });
}

// template <int n_dims>
// torch::Tensor compute_Nd_normal(bool cuda)
//{
//     const auto n_batch = 1;
//     const auto n_samples = 1000000;
//     const auto sample_batch_size = 128;
//     const auto n_sample_batches = (n_samples + sample_batch_size - 1) / sample_batch_size;
//     auto rnd = torch::zeros({ n_batch, n_samples, n_dims }, torch::TensorOptions(torch::ScalarType::Float)); //.cuda();
//     if (cuda)
//         rnd = rnd.cuda();
//     auto rnd_a = gpe::struct_accessor<glm::vec<n_dims, float>, 2>(rnd);

//    dim3 dimBlock = dim3(32, 4, 1);
//    dim3 dimGrid = dim3((unsigned(n_sample_batches) + dimBlock.x - 1) / dimBlock.x,
//        (unsigned(n_batch) + dimBlock.y - 1) / dimBlock.y,
//        1);
//    //        gpe::start_serial<gpe::ComputeDevice::CPU>(gpe::device(out_mixture), dimGrid, dimBlock, [=]
//    gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(rnd), dimGrid, dimBlock, [=] __host__ __device__(const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
//        auto rng = gpe::RandomNumberGenerator<float>(0, gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx);
//        const unsigned idX = gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x;
//        const unsigned idY = gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y;
//        //        const unsigned idZ = gpe_blockIdx.z * gpe_blockDim.z + gpe_threadIdx.z;
//        if (idX >= n_sample_batches || idY >= n_batch)
//            return;

//        const auto start_sample_id = idX * sample_batch_size;
//        for (auto i = 0; i < sample_batch_size; ++i) {
//            const auto sample_id = start_sample_id + i;
//            if (sample_id >= n_samples)
//                break;

//            rnd_a[idY][sample_id] = gpe::random_normal_vec<float, n_dims>(&rng);
//        }
//    });
//    return rnd;
//}

// template torch::Tensor compute_Nd_normal<2>(bool cuda);

// template torch::Tensor compute_Nd_normal<3>(bool cuda);
