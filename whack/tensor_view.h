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

#pragma once

#include <cinttypes>

#include <thrust/detail/raw_pointer_cast.h>

#include "indexing.h"

namespace whack {

template <typename T, uint32_t n_dims, typename IndexType = uint32_t, typename DimensionType = IndexType>
class TensorView {
    using Index = whack::Array<DimensionType, n_dims>;
    T* data = nullptr;
    Index dimensions = {};

public:
    TensorView() = default;
    TensorView(T* data, const Index& dimensions)
        : data(data)
        , dimensions(dimensions)
    {
    }
    //    const typename std::remove_const_t<T>& operator()(const Index& index) const
    template <typename U = T>
    typename std::enable_if<(n_dims > 1), const U&>::type operator()(const Index& index) const
    {
        return *(data + whack::join_n_dim_index<IndexType, n_dims, DimensionType>(dimensions, index));
    }

    template <typename U = T>
    typename std::enable_if<(n_dims > 1), U&>::type operator()(const Index& index)
    {
        return *(data + whack::join_n_dim_index<IndexType, n_dims, DimensionType>(dimensions, index));
    }

    const T& operator()(const IndexType& index) const
    {
        return *(data + index);
    }

    T& operator()(const IndexType& index)
    {
        return *(data + index);
    }

    template <typename... IndexTypes>
    const T& operator()(const IndexType& index0, const IndexTypes&... other_indices) const
    {
        return operator()({ index0, other_indices... });
    }

    template <typename... IndexTypes>
    T& operator()(const IndexType& index0, const IndexTypes&... other_indices)
    {
        return operator()({ index0, other_indices... });
    }
};

// whack::Array api
template <typename ThrustVector, uint32_t n_dims, typename IndexType = uint32_t, typename DimensionType = IndexType>
TensorView<typename std::remove_pointer_t<decltype(thrust::raw_pointer_cast(ThrustVector().data()))>, n_dims, IndexType, DimensionType>
make_tensor_view(ThrustVector& data, const whack::Array<DimensionType, n_dims>& dimensions)
{
    return { thrust::raw_pointer_cast(data.data()), dimensions };
}

// parameter pack api
template <typename ThrustVector, typename IndexType = uint32_t, typename DimensionType = IndexType, typename... DimensionTypes>
TensorView<typename std::remove_pointer_t<decltype(thrust::raw_pointer_cast(ThrustVector().data()))>, sizeof...(DimensionTypes), IndexType, DimensionType> make_tensor_view(ThrustVector& data, DimensionTypes... dimensions)
{
    return { thrust::raw_pointer_cast(data.data()), { dimensions... } };
}

}
