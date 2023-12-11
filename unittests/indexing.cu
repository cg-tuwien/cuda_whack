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

#include "whack/indexing.h"

#include <cstdint>

#ifdef _MSC_VER
#undef REQUIRE
#undef CHECK
#include <cassert>
#define REQUIRE assert
#define CHECK assert
#endif // MSVC

namespace {
template<typename TestType>
void test_indexing()
{
    using whack::join_n_dim_index;
    using whack::split_n_dim_index;
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


TEST_CASE("whack test indexing")
{
    //int, unsigned, int64_t, uint64_t
    test_indexing<int32_t>();
    test_indexing<uint32_t>();
    test_indexing<int64_t>();
    test_indexing<uint64_t>();
}
