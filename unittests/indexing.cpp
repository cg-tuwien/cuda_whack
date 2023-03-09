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

#include "whack/indexing.h"

#include <cstdint>

#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE("util helpers", "[util]", int, unsigned, int64_t, uint64_t)
{
    SECTION("indexing")
    {
        using whack::join_n_dim_index;
        {
            const auto dims = whack::Array<TestType, 4> { 5, 4, 3, 2 };
            REQUIRE(join_n_dim_index<TestType>(dims, { 0, 0, 0, 0 }) == 0);
            REQUIRE(join_n_dim_index<TestType>(dims, { 0, 0, 0, 1 }) == 1);
            REQUIRE(join_n_dim_index<TestType>(dims, { 0, 0, 1, 0 }) == 2);
            REQUIRE(join_n_dim_index<TestType>(dims, { 0, 0, 1, 1 }) == 3);
            REQUIRE(join_n_dim_index<TestType>(dims, { 0, 1, 0, 0 }) == 6);
            REQUIRE(join_n_dim_index<TestType>(dims, { 0, 1, 1, 1 }) == 9);
            REQUIRE(join_n_dim_index<TestType>(dims, { 1, 0, 0, 0 }) == 24);
            REQUIRE(join_n_dim_index<TestType>(dims, { 1, 1, 1, 1 }) == 33);

            TestType idx = 0;
            for (TestType l = 0; l < 5; ++l) {
                for (TestType k = 0; k < 4; ++k) {
                    for (TestType j = 0; j < 3; ++j) {
                        for (TestType i = 0; i < 2; ++i) {
                            const auto joined = join_n_dim_index<TestType>(dims, { l, k, j, i });
                            CHECK(joined == idx);
                            idx++;
                            const auto split = split_n_dim_index(dims, joined);
                            CHECK(split[0] == l);
                            CHECK(split[1] == k);
                            CHECK(split[2] == j);
                            CHECK(split[3] == i);
                        }
                    }
                }
            }
        }
        {
            const auto dims = whack::Array<TestType, 3> { (1 << 4), (1 << 12), (1 << 17) };
            REQUIRE(join_n_dim_index<uint64_t>(dims, { 0, 0, 0 }) == 0);
            REQUIRE(join_n_dim_index<uint64_t>(dims, { 0, 0, 1 }) == 1);
            REQUIRE(join_n_dim_index<uint64_t>(dims, { 0, 1, 0 }) == (1 << 17));
            REQUIRE(join_n_dim_index<uint64_t>(dims, { 1, 0, 0 }) == (1 << 17) * (1 << 12));
            {
                for (TestType i = 0; i < 2; ++i) {
                    for (TestType j = 0; j < 3; ++j) {
                        for (TestType k = 0; k < 4; ++k) {
                            const auto joined = join_n_dim_index<uint64_t>(dims, { k, j, i });
                            const auto split = split_n_dim_index(dims, joined);
                            REQUIRE(split[0] == k);
                            REQUIRE(split[1] == j);
                            REQUIRE(split[2] == i);
                        }
                    }
                }
            }
        }
    }
}
