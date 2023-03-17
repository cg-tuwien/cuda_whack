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
        auto tensor = whack::make_host_tensor<int>(16);
        CHECK(tensor.device() == whack::ComputeDevice::CPU);
        CHECK(std::any_cast<thrust::host_vector<int>>(&tensor.memory()) != nullptr);
        tensor.view()(0) = 42;
        CHECK(tensor.view()(0) == 42);
        CHECK(tensor.host_vector().size() == 16);
    }
    {
        auto tensor = whack::make_host_tensor<int>(2, 3, 4);
        CHECK(tensor.device() == whack::ComputeDevice::CPU);
        CHECK(std::any_cast<thrust::host_vector<int>>(&tensor.memory()) != nullptr);
        tensor.view()(0, 0, 0) = 42;
        CHECK(tensor.view()(0, 0, 0) == 42);
        CHECK(tensor.host_vector().size() == 2 * 3 * 4);
    }

    {
        auto tensor = whack::make_device_tensor<float>(16);
        CHECK(tensor.device() == whack::ComputeDevice::CUDA);
        CHECK(std::any_cast<thrust::device_vector<float>>(&tensor.memory()) != nullptr);

        auto view = tensor.view();

        dim3 dimBlock = dim3(1, 1, 1);
        dim3 dimGrid = dim3(1, 1, 1);
        whack::start_parallel(whack::ComputeDevice::CUDA, dimGrid, dimBlock, [=] __host__ __device__(const dim3&, const dim3&, const dim3&, const dim3&) mutable {
            view(0) = 42;
        });
        CHECK(tensor.device_vector().size() == 16);
        CHECK(tensor.device_vector()[0] == 42);
    }
}

void Tensor_copy_to_device()
{
    //    auto tensor = whack::make_device_tensor<float>(16);
    //    CHECK(tensor.device() == whack::ComputeDevice::CUDA);
    //    CHECK(std::any_cast<thrust::device_vector<float>>(&tensor.memory()) != nullptr);
}

class FailOnCopy {
    int v = 0;

public:
    FailOnCopy() = default;
    FailOnCopy(const FailOnCopy& other)
        : v(other.v)
    {
        CHECK(false);
    }
    FailOnCopy(FailOnCopy&& other) noexcept
        : v(other.v)
    {
        CHECK(true);
    }
    FailOnCopy& operator=(const FailOnCopy& other)
    {
        v = other.v;
        CHECK(false);
        return *this;
    }
    FailOnCopy& operator=(FailOnCopy&& other) noexcept
    {
        v = other.v;
        CHECK(true);
        return *this;
    }
    ~FailOnCopy() = default;
};

TEST_CASE("Tensor")
{
    SECTION("interface")
    {
        Tensor_interface();
    }
    SECTION("copy to device")
    {
        Tensor_copy_to_device();
    }

    SECTION("thrust vector is actually movable into an any")
    {
        thrust::host_vector<FailOnCopy> thrust_vector(16);
        std::any dud = std::move(thrust_vector);
    }
}
