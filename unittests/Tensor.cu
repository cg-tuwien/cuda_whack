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

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <thrust/device_vector.h>

#include "whack/Tensor.h"
#include "whack/kernel.h"
#include "whack/tensor_operations.h"

// windows is only happy, if the enclosing function of a host device lambda has external linkage

void Tensor_interface()
{
    {
        auto tensor = whack::make_tensor<int>(whack::Location::Host, 16);
        CHECK(tensor.location() == whack::Location::Host);
        CHECK(tensor.host_vector().size() == 16);
        CHECK_THROWS(tensor.device_vector());
        tensor.view()(0) = 42;
        CHECK(tensor.view()(0) == 42);
    }
    {
        auto tensor = whack::make_tensor<int>(whack::Location::Host, 2, 3, 4);
        CHECK(tensor.location() == whack::Location::Host);
        CHECK(tensor.host_vector().size() == 2 * 3 * 4);
        CHECK_THROWS(tensor.device_vector());
        tensor.view()(0, 0, 0) = 42;
        CHECK(tensor.view()(0, 0, 0) == 42);
    }
    {
        const whack::Array<uint32_t, 1> dimensions = { 16 };
        auto tensor = whack::make_tensor<int>(whack::Location::Host, dimensions);
        CHECK(tensor.location() == whack::Location::Host);
        CHECK(tensor.host_vector().size() == 16);
        CHECK_THROWS(tensor.device_vector());
        tensor.view()(0) = 42;
        CHECK(tensor.view()(0) == 42);
    }
    {
        const whack::Array<uint32_t, 3> dimensions = { 2, 3, 4 };
        auto tensor = whack::make_tensor<int>(whack::Location::Host, dimensions);
        CHECK(tensor.location() == whack::Location::Host);
        CHECK(tensor.host_vector().size() == 2 * 3 * 4);
        CHECK_THROWS(tensor.device_vector());
        tensor.view()(0, 0, 0) = 42;
        CHECK(tensor.view()(0, 0, 0) == 42);
    }

    {
        auto tensor = whack::make_tensor<float>(whack::Location::Device, 16);
        CHECK(tensor.location() == whack::Location::Device);
        CHECK(tensor.device_vector().size() == 16);
        CHECK_THROWS(tensor.host_vector());

        auto view = tensor.view();

        dim3 dimBlock = dim3(1, 1, 1);
        dim3 dimGrid = dim3(1, 1, 1);
        whack::start_parallel(whack::Location::Device, dimGrid, dimBlock, [=] __host__ __device__(const dim3&, const dim3&, const dim3&, const dim3&) mutable {
            view(0) = 42;
        });
        CHECK(tensor.device_vector()[0] == 42);
    }

    {
        const whack::Array<uint32_t, 3> dimensions = { 2, 3, 4 };
        auto tensor = whack::make_tensor<float>(whack::Location::Device, dimensions);
        CHECK(tensor.location() == whack::Location::Device);
        CHECK(tensor.device_vector().size() == 2 * 3 * 4);
        CHECK_THROWS(tensor.host_vector());

        auto view = tensor.view();

        dim3 dimBlock = dim3(1, 1, 1);
        dim3 dimGrid = dim3(1, 1, 1);
        whack::start_parallel(tensor.location(), dimGrid, dimBlock, [=] __host__ __device__(const dim3&, const dim3&, const dim3&, const dim3&) mutable {
            view(1, 2, 3) = 42;
        });
        CHECK(tensor.device_vector()[2 * 3 * 4 - 1] == 42);
    }

    {
        const whack::Array<uint32_t, 2> dimensions = { 2, 3 };

        auto tensor = whack::make_tensor<float>(whack::Location::Device, { 1, 2, 3, 4, 5, 6 }, dimensions);
        CHECK(tensor.location() == whack::Location::Device);
        CHECK(tensor.device_vector().size() == 2 * 3);
        CHECK_THROWS(tensor(0, 0));
    }
    {
        auto tensor = whack::make_tensor<float>(whack::Location::Host, { 1, 2, 3, 4, 5, 6 }, 3, 2);
        CHECK(tensor.location() == whack::Location::Host);
        CHECK(tensor.host_vector().size() == 3 * 2);
        CHECK(tensor(0, 0) == 1);
        CHECK(tensor(0, 1) == 2);
        CHECK(tensor(1, 0) == 3);
        CHECK(tensor(1, 1) == 4);
        CHECK(tensor(2, 0) == 5);
        CHECK(tensor(2, 1) == 6);
    }
    {
        const std::vector<float> data = { 1, 2, 3, 4, 5, 6 };
        const whack::Array<uint32_t, 2> dimensions = { 2, 3 };
        auto tensor = whack::make_tensor<float>(whack::Location::Host, data.begin(), data.end(), dimensions);
        CHECK(tensor.location() == whack::Location::Host);
        CHECK(tensor.host_vector().size() == 3 * 2);
        CHECK(tensor(0, 0) == 1);
        CHECK(tensor(0, 1) == 2);
        CHECK(tensor(0, 2) == 3);
        CHECK(tensor(1, 0) == 4);
        CHECK(tensor(1, 1) == 5);
        CHECK(tensor(1, 2) == 6);
    }
    {
        auto tensor = whack::make_tensor<float>(whack::Location::Host, { 1, 2, 3, 4, 5, 6 }, 3, 2);
        auto vec_view = tensor.view<glm::vec3>(1, 2);
        CHECK(vec_view.size<0>() == 1);
        CHECK(vec_view.size<1>() == 2);
        CHECK(vec_view.shape().size() == 2);
        CHECK(vec_view(0, 0) == glm::vec3(1, 2, 3));
        CHECK(vec_view(0, 1) == glm::vec3(4, 5, 6));
        vec_view(0, 1) = glm::vec3(3, 2, 11);
        CHECK(vec_view(0, 1) == glm::vec3(3, 2, 11));
    }
    {
        const auto tensor = whack::make_tensor<float>(whack::Location::Host, { 1, 2, 3, 4, 5, 6 }, 3, 2);
        auto vec_view = tensor.view<glm::vec3>(1, 2);
        CHECK(vec_view.size<0>() == 1);
        CHECK(vec_view.size<1>() == 2);
        CHECK(vec_view.shape().size() == 2);
        CHECK(vec_view(0, 0) == glm::vec3(1, 2, 3));
        CHECK(vec_view(0, 1) == glm::vec3(4, 5, 6));
    }
    {
        auto tensor = whack::make_tensor<float>(whack::Location::Host, { 1, 2, 3, 4, 5, 6 }, 3, 2);
        CHECK_THROWS(const_cast<const decltype(tensor)*>(&tensor)->view<glm::vec3>(3, 2));
        CHECK_THROWS(tensor.view<glm::vec3>(3, 2));
    }
}

void Tensor_copy()
{
    {
        auto a = whack::make_tensor<int>(whack::Location::Host, 16);
        auto b = a;
        CHECK(b.location() == whack::Location::Host);
        CHECK(a.raw_pointer() != b.raw_pointer());
    }
    {
        auto a = whack::make_tensor<int>(whack::Location::Device, 16);
        auto b = a;
        CHECK(b.location() == whack::Location::Device);
        CHECK(a.raw_pointer() != b.raw_pointer());
    }
    {
        auto a = whack::make_tensor<int>(whack::Location::Host, 16);
        auto b = whack::make_tensor<int>(whack::Location::Device, 16);
        a = b;
        CHECK(a.location() == whack::Location::Device);
        CHECK(a.raw_pointer() != b.raw_pointer());
    }
}

void Tensor_copy_to_device_and_back()
{
    auto h1 = whack::make_tensor<int>(whack::Location::Host, 16);
    h1.host_vector()[0] = 12387;
    auto d1 = h1.device_copy();
    REQUIRE(d1.location() == whack::Location::Device);
    CHECK(d1.device_vector().size() == 16);
    CHECK(d1.device_vector()[0] == 12387);
    d1.device_vector()[1] = 68543;

    auto d2 = d1.device_copy();
    auto h2 = d2.host_copy();

    CHECK(h2.location() == whack::Location::Host);
    CHECK(h2.host_vector().size() == 16);
    CHECK(h2.host_vector()[0] == 12387);
    CHECK(h2.host_vector()[1] == 68543);

    auto h3 = h2.host_copy();
    CHECK(h3.location() == whack::Location::Host);
    CHECK(h3.host_vector().size() == 16);
    CHECK(h3.host_vector()[0] == 12387);
    CHECK(h3.host_vector()[1] == 68543);
}

void Tensor_concat_and_split(whack::Location location)
{
    {
        auto a = whack::make_tensor<int>(location, { 1, 2, 3, 4, 5, 6, 7, 8 }, 2, 4);
        const auto b = whack::make_tensor<int>(location, { 1, 2, 3 }, 3, 1);
        auto c = whack::make_tensor<int>(location, { 1, 2, 3, 4, 5 }, 5);

        const auto cc = concat(a, b, c);
        static_assert(std::is_same_v<std::remove_cv_t<decltype(cc)::value_type>, int>);
        CHECK(cc.numel() == 2 * 4 + 3 * 1 + 5); // 16
        const auto cc_host = cc.host_copy();
        CHECK(cc_host(0) == 1);
        CHECK(cc_host(1) == 2);
        CHECK(cc_host(2) == 3);
        CHECK(cc_host(3) == 4);
        CHECK(cc_host(4) == 5);
        CHECK(cc_host(5) == 6);
        CHECK(cc_host(6) == 7);
        CHECK(cc_host(7) == 8);

        CHECK(cc_host(8) == 1);
        CHECK(cc_host(9) == 2);
        CHECK(cc_host(10) == 3);

        CHECK(cc_host(11) == 1);
        CHECK(cc_host(12) == 2);
        CHECK(cc_host(13) == 3);
        CHECK(cc_host(14) == 4);
        CHECK(cc_host(15) == 5);
    }

    {
        auto a = whack::make_tensor<int>(location, { 1, 2, 3, 4, 5, 6, 7, 8 }, 8);
        const auto [b, c] = whack::split<int>(a, 3, 5);
        const auto bh = b.host_copy();
        const auto ch = c.host_copy();
        CHECK(bh(0) == 1);
        CHECK(bh(1) == 2);
        CHECK(bh(2) == 3);
        CHECK(ch(0) == 4);
        CHECK(ch(1) == 5);
        CHECK(ch(2) == 6);
        CHECK(ch(3) == 7);
        CHECK(ch(4) == 8);
    }

    {
        auto a = whack::make_tensor<int>(location, { 1, 2, 3, 4, 5, 6, 7, 8 }, 8);
        const auto [b, c, d, e] = whack::split<int>(a, 1, 2, 3, 2);
        const auto bh = b.host_copy();
        const auto ch = c.host_copy();
        const auto dh = d.host_copy();
        const auto eh = e.host_copy();
        CHECK(bh(0) == 1);
        CHECK(ch(0) == 2);
        CHECK(ch(1) == 3);
        CHECK(dh(0) == 4);
        CHECK(dh(1) == 5);
        CHECK(dh(2) == 6);
        CHECK(eh(0) == 7);
        CHECK(eh(1) == 8);
    }
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
} // namespace

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

    SECTION("concat and split")
    {
        Tensor_concat_and_split(whack::Location::Device);
        Tensor_concat_and_split(whack::Location::Host);
    }
}
