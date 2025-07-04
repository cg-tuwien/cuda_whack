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

#pragma once

#include "array.h"

namespace whack {

template <typename large_type, whack::size_t n_dims, typename small_type = large_type>
WHACK_DEVICES_INLINE Array<small_type, n_dims> split_n_dim_index(const Array<small_type, n_dims>& dimensions, large_type idx) noexcept
{
    Array<small_type, n_dims> tmp;
    tmp.back() = 1;
    for (unsigned i = n_dims - 2; i < n_dims; --i) {
        tmp[i] = dimensions[i + 1] * tmp[i + 1];
    }

    for (unsigned i = 0; i < n_dims; ++i) {
        const auto tmp_idx = idx / tmp[i];
        idx -= tmp_idx * tmp[i];
        tmp[i] = tmp_idx;
    }
    return tmp;
}

template <typename large_type, whack::size_t n_dims, typename small_type = large_type>
WHACK_DEVICES_INLINE large_type join_n_dim_index(const Array<small_type, n_dims>& dimensions, const Array<small_type, n_dims>& idx) noexcept
{
    large_type joined_idx = 0;
    large_type cum_dims = 1;
    for (unsigned i = n_dims - 1; i < n_dims; --i) {
        joined_idx += idx[i] * cum_dims;
        cum_dims *= dimensions[i];
    }
    return joined_idx;
}
} // namespace whack
