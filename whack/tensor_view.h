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
    const T& operator()(const Index& index) const
    {
        return *(data + whack::join_n_dim_index<IndexType, n_dims, DimensionType>(dimensions, index));
    }
    T& operator()(const Index& index)
    {
        return *(data + whack::join_n_dim_index<IndexType, n_dims, DimensionType>(dimensions, index));
    }
};

// whack::Array api
template <typename ThrustVector, uint32_t n_dims, typename IndexType = uint32_t, typename DimensionType = IndexType>
TensorView<const typename ThrustVector::value_type, n_dims, IndexType, DimensionType> make_tensor_view(const ThrustVector& data, const whack::Array<DimensionType, n_dims>& dimensions)
{
    return { thrust::raw_pointer_cast(data.data()), dimensions };
}

template <typename ThrustVector, uint32_t n_dims, typename IndexType = uint32_t, typename DimensionType = IndexType>
TensorView<typename ThrustVector::value_type, n_dims, IndexType, DimensionType> make_tensor_view(ThrustVector& data, const whack::Array<DimensionType, n_dims>& dimensions)
{
    return { thrust::raw_pointer_cast(data.data()), dimensions };
}

// parameter pack api
template <typename ThrustVector, typename IndexType = uint32_t, typename DimensionType = IndexType, typename... DimensionTypes>
TensorView<const typename ThrustVector::value_type, sizeof...(DimensionTypes), IndexType, DimensionType> make_tensor_view(const ThrustVector& data, DimensionTypes... dimensions)
{
    return { thrust::raw_pointer_cast(data.data()), { dimensions... } };
}

template <typename ThrustVector, typename IndexType = uint32_t, typename DimensionType = IndexType, typename... DimensionTypes>
TensorView<typename ThrustVector::value_type, sizeof...(DimensionTypes), IndexType, DimensionType> make_tensor_view(ThrustVector& data, DimensionTypes... dimensions)
{
    return { thrust::raw_pointer_cast(data.data()), { dimensions... } };
}

}
