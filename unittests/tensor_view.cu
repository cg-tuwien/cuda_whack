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

#include <cstdint>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <nvtx3/nvToolsExt.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "whack/TensorView.h"
#include "whack/kernel.h"

#define WHACK_UNUSED_THREAD_INDICES WHACK_UNUSED(whack_gridDim) WHACK_UNUSED(whack_blockDim) WHACK_UNUSED(whack_blockIdx) WHACK_UNUSED(whack_threadIdx)

void tensor_view_read_write_multi_dim_cuda()
{
    const thrust::device_vector<int> tensor_1 = std::vector { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    thrust::device_vector<int> tensor_2 = std::vector { 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11 };

    const whack::Array<uint32_t, 4> dimensions = { 1u, 2u, 3u, 2u };
    const auto tensor_1_view = whack::make_tensor_view(tensor_1, dimensions);
    auto tensor_2_view = whack::make_tensor_view(tensor_2, dimensions);

    dim3 dimBlock = dim3(2, 3, 2);
    dim3 dimGrid = dim3(1, 1, 1);
    whack::start_parallel(
        whack::Location::Device, dimGrid, dimBlock, WHACK_KERNEL(tensor_1_view, tensor_2_view) {
            WHACK_UNUSED_THREAD_INDICES
            tensor_2_view(0u, whack_threadIdx.x, whack_threadIdx.y, whack_threadIdx.z) = tensor_1_view(0u, whack_threadIdx.x, whack_threadIdx.y, whack_threadIdx.z) * 2;
        });

    thrust::host_vector<int> host_v(tensor_2);
    REQUIRE(host_v.size() == 12);
    for (int i = 0; i < 12; ++i) {
        CHECK(host_v[i] == i * 2);
    }
}

void tensor_view_read_write_multi_dim_cpu()
{
    const thrust::host_vector<int> tensor_1 = std::vector { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    thrust::host_vector<int> tensor_2 = std::vector { 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11 };

    const whack::Array<uint32_t, 4> dimensions = { 1u, 2u, 3u, 2u };
    const auto tensor_1_view = whack::make_tensor_view(tensor_1, dimensions);
    auto tensor_2_view = whack::make_tensor_view(tensor_2, dimensions);

    dim3 dimBlock = dim3(2, 3, 2);
    dim3 dimGrid = dim3(1, 1, 1);
    whack::start_parallel(
        whack::Location::Host, dimGrid, dimBlock, WHACK_KERNEL(tensor_1_view, tensor_2_view) {
            WHACK_UNUSED_THREAD_INDICES
            tensor_2_view(0u, whack_threadIdx.x, whack_threadIdx.y, whack_threadIdx.z) = tensor_1_view(0u, whack_threadIdx.x, whack_threadIdx.y, whack_threadIdx.z) * 2;
        });

    thrust::host_vector<int> host_v(tensor_2);
    REQUIRE(host_v.size() == 12);
    for (int i = 0; i < 12; ++i) {
        CHECK(host_v[i] == i * 2);
    }
}

void tensor_view_pointer_api()
{
    const thrust::device_vector<int> tensor_1 = std::vector { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    thrust::device_vector<int> tensor_2 = std::vector { 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11 };

    const whack::Array<uint32_t, 3> dimensions = { 1u, 2u, 3u };
    const auto tensor_1_view = whack::make_tensor_view<glm::ivec2>(thrust::raw_pointer_cast(tensor_1.data()), whack::Location::Device, dimensions);
    auto tensor_2_view = whack::make_tensor_view<glm::ivec2>(thrust::raw_pointer_cast(tensor_2.data()), whack::Location::Device, 1, 2, 3);

    dim3 dimBlock = dim3(2, 3, 1);
    dim3 dimGrid = dim3(1, 1, 1);
    whack::start_parallel(
        whack::Location::Device, dimGrid, dimBlock, WHACK_KERNEL(tensor_1_view, tensor_2_view) {
            WHACK_UNUSED_THREAD_INDICES
            const glm::ivec2& v = tensor_1_view(0u, whack_threadIdx.x, whack_threadIdx.y);
            tensor_2_view(0u, whack_threadIdx.x, whack_threadIdx.y) = glm::ivec2(v.x + v.y, 2);
        });

    thrust::host_vector<int> host_v(tensor_2);
    REQUIRE(host_v.size() == 12);
    for (int i = 0; i < 6; ++i) {
        CHECK(host_v[i * 2] == i * 4 + 1);
        CHECK(host_v[i * 2 + 1] == 2);
    }
}

void tensor_view_shape_and_size_on_device()
{
    const thrust::device_vector<int> tensor = std::vector { 0 };
    const auto* ptr = thrust::raw_pointer_cast(tensor.data());
    const auto v1 = whack::make_tensor_view<glm::ivec2>(ptr, whack::Location::Device, 42);
    auto v2 = whack::make_tensor_view<glm::ivec2>(ptr, whack::Location::Device, 12, 34, 56);

    dim3 dimBlock = dim3(1, 1, 1);
    dim3 dimGrid = dim3(1, 1, 1);
    whack::start_parallel(
        whack::Location::Device, dimGrid, dimBlock, WHACK_KERNEL(v1, v2) {
            WHACK_UNUSED_THREAD_INDICES
            const auto s1 = v1.shape();
            const auto s2 = v2.shape();
            assert(s1.size() == 1);
            assert(s1[0] == 42);
            assert(v1.size<0>() == 42);
            assert(v1.size(0) == 42);

            assert(s2.size() == 3);
            assert(s2[0] == 12);
            assert(s2[1] == 34);
            assert(s2[2] == 56);
            assert(v2.size<0>() == 12);
            assert(v2.size(0) == 12);
            assert(v2.size<1>() == 34);
            assert(v2.size(1) == 34);
            assert(v2.size<2>() == 56);
            assert(v2.size(2) == 56);
        });
}

template <typename IndexCalculationType>
void tensor_view_cuda_benchmark_read_write_multi_dim_cuda()
{
    auto batch_dim = 5000u;
    auto thread_dim = 512u;
    auto vector_dim = 256u;
    thrust::device_vector<int> tensor_1(batch_dim * thread_dim * vector_dim);
    thrust::device_vector<float> tensor_2(batch_dim * thread_dim);
    thrust::sequence(tensor_1.begin(), tensor_1.end());
    thrust::fill(tensor_2.begin(), tensor_2.end(), 0);

    const auto tensor_1_view = whack::make_tensor_view<uint32_t, IndexCalculationType>(tensor_1, batch_dim, vector_dim, thread_dim);
    auto tensor_2_view = whack::make_tensor_view<uint32_t, IndexCalculationType>(tensor_2, batch_dim, thread_dim);

    dim3 dimBlock = dim3(thread_dim, 1, 1);
    dim3 dimGrid = dim3(1, batch_dim, 1);

    cudaDeviceSynchronize();
    BENCHMARK("tensor api")
    {
        const auto nvtx_range = nvtxRangeStart("tensor_api");
        whack::start_parallel(
            whack::Location::Device, dimGrid, dimBlock, WHACK_KERNEL(tensor_1_view, tensor_2_view, batch_dim, thread_dim, vector_dim) {
                WHACK_UNUSED_THREAD_INDICES
                const auto batch_id = whack_blockIdx.y;
                const auto thread_id = whack_threadIdx.x;
                float tmp = 0;
                for (unsigned i = 0; i < vector_dim; ++i) {
                    tmp += tensor_1_view(batch_id, i, thread_id) / float(batch_dim * thread_dim * vector_dim);
                    const auto i_l = i - unsigned(i > 0);
                    tmp -= tensor_1_view(batch_id, i_l, thread_id);
                    const auto i_h = i + unsigned(i < (vector_dim - 1));
                    tmp += tensor_1_view(batch_id, i_h, thread_id);
                }
                tensor_2_view(batch_id, thread_id) = tmp;
            });
        nvtxRangeEnd(nvtx_range);
        cudaDeviceSynchronize(); // sync not needed, because the kernels depend on each other.
    };

    cudaDeviceSynchronize();
    BENCHMARK("manual api")
    {
        const auto nvtx_range = nvtxRangeStart("manual_api");
        int* tensor_1_ptr = thrust::raw_pointer_cast(tensor_1.data());
        float* tensor_2_ptr = thrust::raw_pointer_cast(tensor_2.data());
        whack::start_parallel(
            whack::Location::Device, dimGrid, dimBlock, WHACK_KERNEL(tensor_1_ptr, tensor_2_ptr, batch_dim, thread_dim, vector_dim) {
                WHACK_UNUSED_THREAD_INDICES
                const auto batch_id = whack_blockIdx.y;
                const auto thread_id = whack_threadIdx.x;
                float tmp = 0;
                for (unsigned i = 0; i < vector_dim; ++i) {
                    tmp += *(tensor_1_ptr + (IndexCalculationType(batch_id) * thread_dim * vector_dim + i * thread_dim + thread_id)) / float(batch_dim * thread_dim * vector_dim);
                    const auto i_l = i - unsigned(i > 0);
                    tmp -= *(tensor_1_ptr + (IndexCalculationType(batch_id) * thread_dim * vector_dim + i_l * thread_dim + thread_id));
                    const auto i_h = i + unsigned(i < (vector_dim - 1));
                    tmp += *(tensor_1_ptr + (IndexCalculationType(batch_id) * thread_dim * vector_dim + i_h * thread_dim + thread_id));
                }
                *(tensor_2_ptr + (IndexCalculationType(batch_id) * thread_dim + thread_id)) = tmp;
            });
        nvtxRangeEnd(nvtx_range);
        cudaDeviceSynchronize(); // sync not needed, because the kernels depend on each other.
    };
}

TEST_CASE("tensor_view (cuda)")
{
    SECTION("read/write multi dim cuda")
    {
        tensor_view_read_write_multi_dim_cuda();
    }
    SECTION("read/write multi dim cpu")
    {
        tensor_view_read_write_multi_dim_cpu();
    }
    SECTION("pointer api")
    {
        tensor_view_pointer_api();
    }
}

TEMPLATE_TEST_CASE("tensor_view cuda benchmark", "", uint32_t, uint64_t)
{
    // SECTION("benchmark read/write multi dim cuda")
    //{
    tensor_view_cuda_benchmark_read_write_multi_dim_cuda<TestType>();
    //}
}

TEST_CASE("tensor_view (cpp)")
{
    SECTION("read single dim")
    {
        const thrust::host_vector<int> tensor_data = std::vector { 42, 43 };
        REQUIRE(tensor_data.size() == 2);
        REQUIRE(tensor_data[0] == 42);
        REQUIRE(tensor_data[1] == 43);
        const auto dimensions = whack::Array<uint32_t, 1> { 2u };
        const auto tensor_view = whack::make_tensor_view(tensor_data, dimensions);

        static_assert(std::is_const_v<std::remove_reference_t<decltype(tensor_view({ 0 }))>>);
        CHECK(tensor_view({ 0 }) == 42);
        CHECK(tensor_view({ 1 }) == 43);
    }
    SECTION("read multi dim")
    {
        const thrust::host_vector<int> tensor_data = std::vector { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        REQUIRE(tensor_data.size() == 12);
        const auto dimensions = whack::Array<uint32_t, 4> { 1, 2, 3, 2 };
        const auto tensor_view = whack::make_tensor_view(tensor_data, dimensions);

        CHECK(tensor_view({ 0, 0, 0, 0 }) == 0);
        CHECK(tensor_view({ 0, 0, 0, 1 }) == 1);
        CHECK(tensor_view({ 0, 0, 1, 0 }) == 2);
        CHECK(tensor_view({ 0, 0, 1, 1 }) == 3);
        CHECK(tensor_view({ 0, 0, 2, 0 }) == 4);
        CHECK(tensor_view({ 0, 1, 0, 0 }) == 6);
        CHECK(tensor_view({ 0, 1, 2, 1 }) == 11);
    }

    SECTION("write single dim")
    {
        thrust::host_vector<int> tensor_data = std::vector { 42, 43 };
        REQUIRE(tensor_data.size() == 2);
        REQUIRE(tensor_data[0] == 42);
        REQUIRE(tensor_data[1] == 43);
        const auto dimensions = whack::Array<uint32_t, 1> { 2u };
        auto tensor_view = whack::make_tensor_view(tensor_data, dimensions);

        CHECK(tensor_view({ 0 }) == 42);
        CHECK(tensor_view({ 1 }) == 43);

        tensor_view({ 0 }) = 2;
        tensor_view({ 1 }) = 3;

        CHECK(tensor_view({ 0 }) == 2);
        CHECK(tensor_view({ 1 }) == 3);
    }

    SECTION("const tensor views to a writable vectors are actually const")
    {
        thrust::host_vector<int> tensor_data = std::vector { 42, 43 };
        REQUIRE(tensor_data.size() == 2);
        REQUIRE(tensor_data[0] == 42);
        REQUIRE(tensor_data[1] == 43);
        const auto dimensions = whack::Array<uint32_t, 1> { 2u };
        const auto tensor_view = whack::make_tensor_view(tensor_data, dimensions);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(tensor_view({ 0 }))>>);
    }

    SECTION("read struct")
    {
        struct Foo {
            int a = 0;
            int b = 0;
        };
        const thrust::host_vector<Foo> tensor_data = std::vector { Foo { 1, 2 }, Foo { 3, 4 } };
        REQUIRE(tensor_data.size() == 2);
        REQUIRE(tensor_data[0].a == 1);
        REQUIRE(tensor_data[0].b == 2);
        REQUIRE(tensor_data[1].a == 3);
        REQUIRE(tensor_data[1].b == 4);
        const auto dimensions = whack::Array<uint32_t, 1> { 2u };
        const auto tensor_view = whack::make_tensor_view(tensor_data, dimensions);

        static_assert(std::is_const_v<std::remove_reference_t<decltype(tensor_view({ 0 }))>>);
        CHECK(tensor_view({ 0 }).a == 1);
        CHECK(tensor_view({ 0 }).b == 2);
        CHECK(tensor_view({ 1 }).a == 3);
        CHECK(tensor_view({ 1 }).b == 4);
    }

    SECTION("parametre pack api for dimensions (single dim read)")
    {
        const thrust::host_vector<int> tensor_data = std::vector { 42, 43 };
        REQUIRE(tensor_data.size() == 2);
        REQUIRE(tensor_data[0] == 42);
        REQUIRE(tensor_data[1] == 43);
        const auto tensor_view = whack::make_tensor_view(tensor_data, 2u);

        static_assert(std::is_const_v<std::remove_reference_t<decltype(tensor_view({ 0 }))>>);
        CHECK(tensor_view({ 0 }) == 42);
        CHECK(tensor_view({ 1 }) == 43);
    }
    SECTION("parametre pack api for dimensions (multiple dims read / write)")
    {
        thrust::host_vector<int> tensor_data = std::vector { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        REQUIRE(tensor_data.size() == 12);
        const auto tensor_view = whack::make_tensor_view(tensor_data, 1u, 2u, 3u, 2u);
        auto tensor_writer_view = whack::make_tensor_view(tensor_data, 1u, 2u, 3u, 2u);

        CHECK(tensor_view({ 0, 0, 0, 0 }) == 0);
        CHECK(tensor_view({ 0, 0, 0, 1 }) == 1);
        CHECK(tensor_view({ 0, 0, 1, 0 }) == 2);
        CHECK(tensor_view({ 0, 0, 1, 1 }) == 3);
        CHECK(tensor_view({ 0, 0, 2, 0 }) == 4);
        CHECK(tensor_view({ 0, 1, 0, 0 }) == 6);
        CHECK(tensor_view({ 0, 1, 2, 1 }) == 11);

        tensor_writer_view({ 0, 0, 0, 0 }) = 10;
        tensor_writer_view({ 0, 0, 0, 1 }) = 11;
        tensor_writer_view({ 0, 0, 1, 0 }) = 12;
        tensor_writer_view({ 0, 0, 1, 1 }) = 13;
        tensor_writer_view({ 0, 0, 2, 0 }) = 14;
        tensor_writer_view({ 0, 1, 0, 0 }) = 16;
        tensor_writer_view({ 0, 1, 2, 1 }) = 111;

        CHECK(tensor_writer_view({ 0, 0, 0, 0 }) == 10);
        CHECK(tensor_writer_view({ 0, 0, 0, 1 }) == 11);
        CHECK(tensor_writer_view({ 0, 0, 1, 0 }) == 12);
        CHECK(tensor_writer_view({ 0, 0, 1, 1 }) == 13);
        CHECK(tensor_writer_view({ 0, 0, 2, 0 }) == 14);
        CHECK(tensor_writer_view({ 0, 1, 0, 0 }) == 16);
        CHECK(tensor_writer_view({ 0, 1, 2, 1 }) == 111);
    }

    SECTION("parametre pack api for dimensions (single dim write)")
    {
        thrust::host_vector<int> tensor_data = std::vector { 42, 43 };
        REQUIRE(tensor_data.size() == 2);
        REQUIRE(tensor_data[0] == 42);
        REQUIRE(tensor_data[1] == 43);
        auto tensor_view = whack::make_tensor_view(tensor_data, 2u);

        CHECK(tensor_view({ 0 }) == 42);
        CHECK(tensor_view({ 1 }) == 43);

        tensor_view({ 0 }) = 2;
        tensor_view({ 1 }) = 3;

        CHECK(tensor_view({ 0 }) == 2);
        CHECK(tensor_view({ 1 }) == 3);
    }

    SECTION("parametre pack api for indices (single dim read)")
    {
        const thrust::host_vector<int> tensor_data = std::vector { 42, 43 };
        const auto tensor_view = whack::make_tensor_view(tensor_data, 2u);

        static_assert(std::is_const_v<std::remove_reference_t<decltype(tensor_view(0))>>);
        CHECK(tensor_view(0) == 42);
        CHECK(tensor_view(1) == 43);
    }

    SECTION("parametre pack api for indices (single dim write)")
    {
        thrust::host_vector<int> tensor_data = std::vector { 42, 43 };
        auto tensor_view = whack::make_tensor_view(tensor_data, 2u);
        CHECK(tensor_view(0) == 42);
        CHECK(tensor_view(1) == 43);

        tensor_view(0) = 2;
        tensor_view(1) = 3;

        CHECK(tensor_view(0) == 2);
        CHECK(tensor_view(1) == 3);
    }

    SECTION("parametre pack api for indices (multiple dims read / write)")
    {
        thrust::host_vector<int> tensor_data = std::vector { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        REQUIRE(tensor_data.size() == 12);
        const auto tensor_view = whack::make_tensor_view(tensor_data, 1, 2, 3, 2);
        auto tensor_writer_view = whack::make_tensor_view(tensor_data, 1, 2, 3, 2);

        CHECK(tensor_view(0, 0, 0, 0) == 0);
        CHECK(tensor_view(0, 0, 0, 1) == 1);
        CHECK(tensor_view(0, 0, 1, 0) == 2);
        CHECK(tensor_view(0, 0, 1, 1) == 3);
        CHECK(tensor_view(0, 0, 2, 0) == 4);
        CHECK(tensor_view(0, 1, 0, 0) == 6);
        CHECK(tensor_view(0, 1, 2, 1) == 11);

        tensor_writer_view(0, 0, 0, 0) = 10;
        tensor_writer_view(0, 0, 0, 1) = 11;
        tensor_writer_view(0, 0, 1, 0) = 12;
        tensor_writer_view(0, 0, 1, 1) = 13;
        tensor_writer_view(0, 0, 2, 0) = 14;
        tensor_writer_view(0, 1, 0, 0) = 16;
        tensor_writer_view(0, 1, 2, 1) = 111;

        CHECK(tensor_writer_view(0, 0, 0, 0) == 10);
        CHECK(tensor_writer_view(0, 0, 0, 1) == 11);
        CHECK(tensor_writer_view(0, 0, 1, 0) == 12);
        CHECK(tensor_writer_view(0, 0, 1, 1) == 13);
        CHECK(tensor_writer_view(0, 0, 2, 0) == 14);
        CHECK(tensor_writer_view(0, 1, 0, 0) == 16);
        CHECK(tensor_writer_view(0, 1, 2, 1) == 111);
    }

    SECTION("tensor view shape")
    {
        {
            const auto view = whack::make_tensor_view<glm::ivec2>((int*)(nullptr), whack::Location::Host, 1);
            CHECK(view.size(0) == 1);
            CHECK(view.size<0>() == 1);
            CHECK(view.shape() == decltype(view)::Shape { 1 });
        }
        {
            const auto view = whack::make_tensor_view<glm::ivec2>((int*)(nullptr), whack::Location::Host, 44);
            CHECK(view.size(0) == 44);
            CHECK(view.size<0>() == 44);
            CHECK(view.shape() == decltype(view)::Shape { 44 });
        }
        {
            const auto view = whack::make_tensor_view<glm::ivec2>((int*)(nullptr), whack::Location::Host, 1, 4, 8);
            CHECK(view.size(0) == 1);
            CHECK(view.size(1) == 4);
            CHECK(view.size(2) == 8);
            CHECK(view.size<0>() == 1);
            CHECK(view.size<1>() == 4);
            CHECK(view.size<2>() == 8);
            CHECK(view.shape() == decltype(view)::Shape { 1, 4, 8 });
        }
        {
            const auto view = whack::make_tensor_view<glm::ivec2>((int*)(nullptr), whack::Location::Host, 45, 321, 99);
            CHECK(view.size(0) == 45);
            CHECK(view.size(1) == 321);
            CHECK(view.size(2) == 99);
            CHECK(view.size<0>() == 45);
            CHECK(view.size<1>() == 321);
            CHECK(view.size<2>() == 99);
            CHECK(view.shape() == decltype(view)::Shape { 45, 321, 99 });
        }
        tensor_view_shape_and_size_on_device();
    }
}
