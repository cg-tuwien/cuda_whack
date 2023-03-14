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
#include <any>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "TensorView.h"
#include "enums.h"

namespace whack {

template <typename T, uint32_t n_dims, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType>
struct Tensor {
    std::any memory;
    TensorView<T, n_dims, IndexStoreType, IndexCalculateType> view;
    ComputeDevice m_device = ComputeDevice::Invalid;

    ComputeDevice device() const { return m_device; }
};

template <typename T, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename... DimensionTypes>
Tensor<T, sizeof...(DimensionTypes)> make_host_tensor(DimensionTypes... dim)
{
    static_assert(std::is_integral_v<IndexStoreType>);
    static_assert(std::is_integral_v<IndexCalculateType>);
    static_assert(std::is_unsigned_v<IndexStoreType>);
    static_assert(std::is_unsigned_v<IndexCalculateType>);
    static_assert(std::is_move_assignable_v<thrust::host_vector<T>>, "thrust::host_vector<T> must be movable");

    constexpr auto n_dims = sizeof...(DimensionTypes);
    const IndexCalculateType size = (std::make_unsigned_t<IndexCalculateType>(dim) * ...);
    thrust::host_vector<T> memory(size);
    Tensor<T, n_dims> retval;
    retval.view = whack::make_tensor_view(memory, dim...);
    retval.memory = std::move(memory);
    retval.m_device = whack::ComputeDevice::CPU;
    return retval;
}

template <typename T, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename... DimensionTypes>
Tensor<T, sizeof...(DimensionTypes)> make_device_tensor(DimensionTypes... dim)
{
    static_assert(std::is_integral_v<IndexStoreType>);
    static_assert(std::is_integral_v<IndexCalculateType>);
    static_assert(std::is_unsigned_v<IndexStoreType>);
    static_assert(std::is_unsigned_v<IndexCalculateType>);
    static_assert(std::is_move_assignable_v<thrust::host_vector<T>>, "thrust::host_vector<T> must be movable");

    constexpr auto n_dims = sizeof...(DimensionTypes);
    const IndexCalculateType size = (std::make_unsigned_t<IndexCalculateType>(dim) * ...);
    thrust::device_vector<T> memory(size);
    Tensor<T, n_dims> retval;
    retval.view = whack::make_tensor_view(memory, dim...);
    retval.memory = std::move(memory);
    retval.m_device = whack::ComputeDevice::CUDA;
    return retval;
}

}
