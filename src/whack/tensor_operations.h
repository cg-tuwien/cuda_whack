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

#include "Tensor.h"

namespace whack {

template <typename T, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, whack::size_t n_dims_a, whack::size_t n_dims_b>
Tensor<T, 1, IndexStoreType, IndexCalculateType> concat(const Tensor<T, n_dims_a>& a, const Tensor<T, n_dims_b>& b)
{
    assert(a.location() == b.location());
    Tensor<T, 1> new_tensor = make_tensor<T, IndexStoreType, IndexCalculateType>(a.location(), a.numel() + b.numel());

    switch (new_tensor.location()) {
    case Location::Host: {
        const auto copy_end = thrust::copy(a.host_vector().begin(), a.host_vector().end(), new_tensor.host_vector().begin());
        const auto copy2_end = thrust::copy(b.host_vector().begin(), b.host_vector().end(), copy_end);
        assert(copy2_end == new_tensor.host_vector().end());
        break;
    }
    case Location::Device: {
        const auto copy_end = thrust::copy(a.device_vector().begin(), a.device_vector().end(), new_tensor.device_vector().begin());
        const auto copy2_end = thrust::copy(b.device_vector().begin(), b.device_vector().end(), copy_end);
        assert(copy2_end == new_tensor.device_vector().end());
        break;
    }
    case Location::Invalid:
        return {};
    }
    return new_tensor;
}

/// warning: concatenating more than 2 elements is currently unnecessarily slow.
template <typename T, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, whack::size_t n_dims_a, whack::size_t n_dims_b, typename... TensorTypes>
Tensor<T, 1, IndexStoreType, IndexCalculateType> concat(const Tensor<T, n_dims_a>& a, const Tensor<T, n_dims_b>& b, TensorTypes... params)
{
    return concat(concat(a, b), params...);
}

template <typename T, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, whack::size_t n_dims_a, unsigned n_splits>
std::array<Tensor<T, 1, IndexStoreType, IndexCalculateType>, n_splits> split(const Tensor<T, n_dims_a, IndexStoreType, IndexCalculateType>& a, const whack::Array<IndexStoreType, n_splits>& splits)
{
    assert(a.numel() == std::accumulate(splits.begin(), splits.end(), 0));
    std::array<Tensor<T, 1, IndexStoreType, IndexCalculateType>, n_splits> splitted;
    for (int i = 0; i < splits.size(); ++i) {
        splitted[i] = make_tensor<T, IndexCalculateType>(a.location(), splits[i]);
    }

    switch (a.location()) {
    case Location::Host: {
        IndexCalculateType pos = 0;
        for (int i = 0; i < splits.size(); ++i) {
            thrust::copy_n(a.host_vector().begin() + pos, splits[i], splitted[i].host_vector().begin());
            pos += splits[i];
        }
        break;
    }
    case Location::Device: {
        IndexCalculateType pos = 0;
        for (int i = 0; i < splits.size(); ++i) {
            thrust::copy_n(a.device_vector().begin() + pos, splits[i], splitted[i].device_vector().begin());
            pos += splits[i];
        }
        break;
    }
    case Location::Invalid:
        return {};
    }
    return std::move(splitted);
}

template <typename T, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, whack::size_t n_dims_a, typename... SizeTypes>
std::array<Tensor<T, 1, IndexStoreType, IndexCalculateType>, sizeof...(SizeTypes)> split(const Tensor<T, n_dims_a, IndexStoreType, IndexCalculateType>& a, SizeTypes... dim)
{
    using Sizes = whack::Array<IndexStoreType, sizeof...(SizeTypes)>;

    const auto sizes = Sizes { IndexStoreType(dim)... };
    return split(a, sizes);
}

} // namespace whack
