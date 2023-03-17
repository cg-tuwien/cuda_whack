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
class Tensor {
    std::any m_memory;
    ComputeDevice m_device = ComputeDevice::Invalid;
    TensorView<T, n_dims, IndexStoreType, IndexCalculateType> m_view;

public:
    Tensor() = default;
    __host__ Tensor(thrust::host_vector<T>&& memory, TensorView<T, n_dims, IndexStoreType, IndexCalculateType> view)
        : m_memory(std::move(memory))
        , m_view(view)
        , m_device(ComputeDevice::CPU)
    {
    }
    __host__ Tensor(thrust::device_vector<T>&& memory, TensorView<T, n_dims, IndexStoreType, IndexCalculateType> view)
        : m_memory(std::move(memory))
        , m_view(view)
        , m_device(ComputeDevice::CUDA)
    {
    }
    __host__ [[nodiscard]] TensorView<T, n_dims, IndexStoreType, IndexCalculateType> view() const { return m_view; }
    __host__ [[nodiscard]] ComputeDevice device() const { return m_device; }
    __host__ [[nodiscard]] const std::any& memory() const { return m_memory; }
};

template <typename T, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename... DimensionTypes>
Tensor<T, sizeof...(DimensionTypes)> make_host_tensor(DimensionTypes... dim)
{
    static_assert(std::is_integral_v<IndexStoreType>);
    static_assert(std::is_integral_v<IndexCalculateType>);
    static_assert(std::is_unsigned_v<IndexStoreType>);
    static_assert(std::is_unsigned_v<IndexCalculateType>);
    static_assert(std::is_move_assignable_v<thrust::host_vector<T>>, "thrust::host_vector<T> must be movable");

    const IndexCalculateType size = (std::make_unsigned_t<IndexCalculateType>(dim) * ...);
    thrust::host_vector<T> memory(size);
    const auto view = whack::make_tensor_view(memory, dim...);
    return { std::move(memory), view };
}

template <typename T, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename... DimensionTypes>
Tensor<T, sizeof...(DimensionTypes)> make_device_tensor(DimensionTypes... dim)
{
    static_assert(std::is_integral_v<IndexStoreType>);
    static_assert(std::is_integral_v<IndexCalculateType>);
    static_assert(std::is_unsigned_v<IndexStoreType>);
    static_assert(std::is_unsigned_v<IndexCalculateType>);
    static_assert(std::is_move_assignable_v<thrust::host_vector<T>>, "thrust::host_vector<T> must be movable");

    const IndexCalculateType size = (std::make_unsigned_t<IndexCalculateType>(dim) * ...);
    thrust::device_vector<T> memory(size);
    const auto view = whack::make_tensor_view(memory, dim...);
    return { std::move(memory), view }; // something not working. overthink design. this way we loose access to thrust vectors (which we probably don't want).
}

}
