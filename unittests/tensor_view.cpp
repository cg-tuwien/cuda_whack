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

#include <thrust/host_vector.h>

#include "whack/TensorView.h"

TEST_CASE("tensor_view.cpp")
{
    SECTION("read single dim")
    {
        const thrust::host_vector<int> tensor_data = std::vector { 42, 43 };
        REQUIRE(tensor_data.size() == 2);
        REQUIRE(tensor_data[0] == 42);
        REQUIRE(tensor_data[1] == 43);
        const auto dimensions = whack::Array<uint32_t, 1>({ 2u });
        const auto tensor_view = whack::make_tensor_view(tensor_data, dimensions);

        static_assert(std::is_const_v<std::remove_reference_t<decltype(tensor_view({ 0 }))>>);
        CHECK(tensor_view({ 0 }) == 42);
        CHECK(tensor_view({ 1 }) == 43);
    }
    SECTION("read multi dim")
    {
        const thrust::host_vector<int> tensor_data = std::vector { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        REQUIRE(tensor_data.size() == 12);
        const auto dimensions = whack::Array<uint32_t, 4>({ 1, 2, 3, 2 });
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
        const auto dimensions = whack::Array<uint32_t, 1>({ 2u });
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
        const auto dimensions = whack::Array<uint32_t, 1>({ 2u });
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
        const auto dimensions = whack::Array<uint32_t, 1>({ 2u });
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
}
