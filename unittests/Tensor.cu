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

#include <catch2/catch_test_macros.hpp>
#include <thrust/device_vector.h>

#include "whack/Tensor.h"
#include "whack/kernel.h"

// windows is only happy, if the enclosing function of a host device lambda has external linkage

void Tensor_interface()
{
    {
        auto tensor = whack::make_tensor<int>(whack::ComputeDevice::CPU, 16);
        CHECK(tensor.device() == whack::ComputeDevice::CPU);
        CHECK(tensor.host_vector().size() == 16);
        CHECK_THROWS(tensor.device_vector());
        tensor.view()(0) = 42;
        CHECK(tensor.view()(0) == 42);
    }
    {
        auto tensor = whack::make_tensor<int>(whack::ComputeDevice::CPU, 2, 3, 4);
        CHECK(tensor.device() == whack::ComputeDevice::CPU);
        CHECK(tensor.host_vector().size() == 2 * 3 * 4);
        CHECK_THROWS(tensor.device_vector());
        tensor.view()(0, 0, 0) = 42;
        CHECK(tensor.view()(0, 0, 0) == 42);
    }
    {
        const whack::Array<uint32_t, 1> dimensions = { 16 };
        auto tensor = whack::make_tensor<int>(whack::ComputeDevice::CPU, dimensions);
        CHECK(tensor.device() == whack::ComputeDevice::CPU);
        CHECK(tensor.host_vector().size() == 16);
        CHECK_THROWS(tensor.device_vector());
        tensor.view()(0) = 42;
        CHECK(tensor.view()(0) == 42);
    }
    {
        const whack::Array<uint32_t, 3> dimensions = { 2, 3, 4 };
        auto tensor = whack::make_tensor<int>(whack::ComputeDevice::CPU, dimensions);
        CHECK(tensor.device() == whack::ComputeDevice::CPU);
        CHECK(tensor.host_vector().size() == 2 * 3 * 4);
        CHECK_THROWS(tensor.device_vector());
        tensor.view()(0, 0, 0) = 42;
        CHECK(tensor.view()(0, 0, 0) == 42);
    }

    {
        auto tensor = whack::make_tensor<float>(whack::ComputeDevice::CUDA, 16);
        CHECK(tensor.device() == whack::ComputeDevice::CUDA);
        CHECK(tensor.device_vector().size() == 16);
        CHECK_THROWS(tensor.host_vector());

        auto view = tensor.view();

        dim3 dimBlock = dim3(1, 1, 1);
        dim3 dimGrid = dim3(1, 1, 1);
        whack::start_parallel(whack::ComputeDevice::CUDA, dimGrid, dimBlock, [=] __host__ __device__(const dim3&, const dim3&, const dim3&, const dim3&) mutable {
            view(0) = 42;
        });
        CHECK(tensor.device_vector()[0] == 42);
    }

    {
        const whack::Array<uint32_t, 3> dimensions = { 2, 3, 4 };
        auto tensor = whack::make_tensor<float>(whack::ComputeDevice::CUDA, dimensions);
        CHECK(tensor.device() == whack::ComputeDevice::CUDA);
        CHECK(tensor.device_vector().size() == 2 * 3 * 4);
        CHECK_THROWS(tensor.host_vector());

        auto view = tensor.view();

        dim3 dimBlock = dim3(1, 1, 1);
        dim3 dimGrid = dim3(1, 1, 1);
        whack::start_parallel(tensor.device(), dimGrid, dimBlock, [=] __host__ __device__(const dim3&, const dim3&, const dim3&, const dim3&) mutable {
            view(1, 2, 3) = 42;
        });
        CHECK(tensor.device_vector()[2 * 3 * 4 - 1] == 42);
    }
}

void Tensor_copy()
{
    {
        auto a = whack::make_tensor<int>(whack::ComputeDevice::CPU, 16);
        auto b = a;
        CHECK(a.host_vector().begin() != b.host_vector().begin());

        CHECK(&a.view()(0) == &a.host_vector().front()); // views must point to something else
        CHECK(&b.view()(0) == &b.host_vector().front()); // views must point to something else
    }
    {
        auto a = whack::make_tensor<int>(whack::ComputeDevice::CUDA, 16);
        auto b = a;
        CHECK(a.device_vector().begin() != b.device_vector().begin());
        CHECK(thrust::raw_pointer_cast(a.device_vector().data()) != thrust::raw_pointer_cast(b.device_vector().data()));

        CHECK(&a.view()(0) == thrust::raw_pointer_cast(a.device_vector().data())); // views must point to something else
        CHECK(&b.view()(0) == thrust::raw_pointer_cast(b.device_vector().data())); // views must point to something else
    }
    {
        auto a = whack::make_tensor<int>(whack::ComputeDevice::CPU, 16);
        auto b = whack::make_tensor<int>(whack::ComputeDevice::CUDA, 16);
        a = b;
        CHECK(a.device() == whack::ComputeDevice::CUDA);

        CHECK(a.device_vector().begin() != b.device_vector().begin());
        CHECK(thrust::raw_pointer_cast(a.device_vector().data()) != thrust::raw_pointer_cast(b.device_vector().data()));

        CHECK(&a.view()(0) == thrust::raw_pointer_cast(a.device_vector().data())); // views must point to something else
        CHECK(&b.view()(0) == thrust::raw_pointer_cast(b.device_vector().data())); // views must point to something else
    }
}

void Tensor_copy_to_device_and_back()
{
    auto h1 = whack::make_tensor<int>(whack::ComputeDevice::CPU, 16);
    h1.host_vector()[0] = 12387;
    auto d1 = h1.device_copy();
    REQUIRE(d1.device() == whack::ComputeDevice::CUDA);
    CHECK(d1.device_vector().size() == 16);
    CHECK(d1.device_vector()[0] == 12387);
    d1.device_vector()[1] = 68543;

    auto d2 = d1.device_copy();
    auto h2 = d2.host_copy();

    CHECK(h2.device() == whack::ComputeDevice::CPU);
    CHECK(h2.host_vector().size() == 16);
    CHECK(h2.host_vector()[0] == 12387);
    CHECK(h2.host_vector()[1] == 68543);

    auto h3 = h2.host_copy();
    CHECK(h3.device() == whack::ComputeDevice::CPU);
    CHECK(h3.host_vector().size() == 16);
    CHECK(h3.host_vector()[0] == 12387);
    CHECK(h3.host_vector()[1] == 68543);
}

namespace {
    
class FailOnCopy {
    int v = 0;

    public:
        FailOnCopy() = default;
        FailOnCopy(const FailOnCopy& other)
            : v(other.v)
        {
            assert(false);
        }
        FailOnCopy(FailOnCopy&& other) noexcept
            : v(other.v)
        {
            assert(true);
        }
        FailOnCopy& operator=(const FailOnCopy& other)
        {
            v = other.v;
            assert(false);
            return *this;
        }
        FailOnCopy& operator=(FailOnCopy&& other) noexcept
        {
            v = other.v;
            assert(true);
            return *this;
        }
        ~FailOnCopy() = default;
};
}

TEST_CASE("Tensor.cu")
{
    SECTION("interface")
    {
        Tensor_interface();
    }

    SECTION("copy")
    {
        Tensor_copy();
    }

    SECTION("copy to device and back")
    {
        Tensor_copy_to_device_and_back();
    }
}
