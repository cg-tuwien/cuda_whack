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

template <typename T, uint32_t n_dims, typename DimensionType = uint32_t, typename IndexType = DimensionType>
class TensorView {
    using Index = whack::Array<DimensionType, n_dims>;
    const T* data = nullptr;
    Index dimensions = {};

public:
    TensorView() = default;
    TensorView(const T* data, const Index& dimensions)
        : data(data)
        , dimensions(dimensions)
    {
    }
    T operator()(const Index& index) const
    {
        return *(data + whack::join_n_dim_index<IndexType, n_dims, DimensionType>(dimensions, index));
    }
};

template <typename ThrustVector, uint32_t n_dims, typename DimensionType = uint32_t, typename IndexType = DimensionType>
TensorView<typename ThrustVector::value_type, n_dims, DimensionType, IndexType> make_tensor_view(const ThrustVector& data, const whack::Array<DimensionType, n_dims>& dimensions)
{
    return { thrust::raw_pointer_cast(data.data()), dimensions };
}

}
