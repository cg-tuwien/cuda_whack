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

    using Dimensions = whack::Array<IndexStoreType, n_dims>;
    Dimensions m_dimensions = {};

public:
    Tensor() = default;
    Tensor(thrust::host_vector<T>&& memory, const Dimensions& dimensions)
        : m_memory(std::move(memory))
        , m_dimensions(dimensions)
        , m_device(ComputeDevice::CPU)
    {
    }
    Tensor(thrust::device_vector<T>&& memory, const Dimensions& dimensions)
        : m_memory(std::move(memory))
        , m_dimensions(dimensions)
        , m_device(ComputeDevice::CUDA)
    {
    }

    [[nodiscard]] thrust::host_vector<T>& host_vector()
    {
        auto* memory_vector = std::any_cast<thrust::host_vector<T>>(&m_memory);
        assert(memory_vector != nullptr);
        return *memory_vector;
    }

    [[nodiscard]] thrust::device_vector<T>& device_vector()
    {
        auto* memory_vector = std::any_cast<thrust::device_vector<T>>(&m_memory);
        assert(memory_vector != nullptr);
        return *memory_vector;
    }

    [[nodiscard]] const thrust::host_vector<T>& host_vector() const
    {
        auto* memory_vector = std::any_cast<thrust::host_vector<T>>(&m_memory);
        assert(memory_vector != nullptr);
        return *memory_vector;
    }

    [[nodiscard]] const thrust::device_vector<T>& device_vector() const
    {
        auto* memory_vector = std::any_cast<thrust::device_vector<T>>(&m_memory);
        assert(memory_vector != nullptr);
        return *memory_vector;
    }

    [[nodiscard]] TensorView<const T, n_dims, IndexStoreType, IndexCalculateType> view() const
    {
        switch (m_device) {
        case ComputeDevice::CPU:
            return make_tensor_view(host_vector(), m_dimensions);
        case ComputeDevice::CUDA:
            return make_tensor_view(device_vector(), m_dimensions);
        }
        assert(false);
        return {};
    }

    [[nodiscard]] TensorView<T, n_dims, IndexStoreType, IndexCalculateType> view()
    {
        switch (m_device) {
        case ComputeDevice::CPU:
            return make_tensor_view(host_vector(), m_dimensions);
        case ComputeDevice::CUDA:
            return make_tensor_view(device_vector(), m_dimensions);
        }
        assert(false);
        return {};
    }

    [[nodiscard]] Tensor device_copy() const;
    [[nodiscard]] Tensor host_copy() const;

    [[nodiscard]] ComputeDevice device() const { return m_device; }
    [[nodiscard]] Dimensions dimensions() const { return m_dimensions; }
    [[nodiscard]] const std::any& memory() const { return m_memory; }
};

// whack::Array api
template <typename T, uint32_t n_dims, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType>
Tensor<T, n_dims> make_host_tensor(const whack::Array<IndexStoreType, n_dims>& dimensions)
{
    static_assert(std::is_integral_v<IndexStoreType>);
    static_assert(std::is_integral_v<IndexCalculateType>);
    static_assert(std::is_unsigned_v<IndexStoreType>);
    static_assert(std::is_unsigned_v<IndexCalculateType>);
    static_assert(std::is_move_assignable_v<thrust::host_vector<T>>, "thrust::host_vector<T> must be movable");

    IndexCalculateType size = 1;
    for (unsigned i = 0; i < n_dims; ++i)
        size *= dimensions[i];
    thrust::host_vector<T> memory(size);

    return { std::move(memory), dimensions };
}

template <typename T, uint32_t n_dims, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType>
Tensor<T, n_dims> make_device_tensor(const whack::Array<IndexStoreType, n_dims>& dimensions)
{
    static_assert(std::is_integral_v<IndexStoreType>);
    static_assert(std::is_integral_v<IndexCalculateType>);
    static_assert(std::is_unsigned_v<IndexStoreType>);
    static_assert(std::is_unsigned_v<IndexCalculateType>);
    static_assert(std::is_move_assignable_v<thrust::device_vector<T>>, "thrust::device_vector<T> must be movable");

    IndexCalculateType size = 1;
    for (unsigned i = 0; i < n_dims; ++i)
        size *= dimensions[i];
    thrust::device_vector<T> memory(size);

    return { std::move(memory), dimensions };
}

// parameter pack api
template <typename T, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename... DimensionTypes>
Tensor<T, sizeof...(DimensionTypes)> make_host_tensor(DimensionTypes... dim)
{
    static_assert(std::is_integral_v<IndexStoreType>);
    static_assert(std::is_integral_v<IndexCalculateType>);
    static_assert(std::is_unsigned_v<IndexStoreType>);
    static_assert(std::is_unsigned_v<IndexCalculateType>);
    static_assert(std::is_move_assignable_v<thrust::host_vector<T>>, "thrust::host_vector<T> must be movable");

    using Dimensions = whack::Array<IndexStoreType, sizeof...(DimensionTypes)>;

    const IndexCalculateType size = (std::make_unsigned_t<IndexCalculateType>(dim) * ...);
    thrust::host_vector<T> memory(size);

    return { std::move(memory), Dimensions { IndexStoreType(dim)... } };
}

template <typename T, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename... DimensionTypes>
Tensor<T, sizeof...(DimensionTypes)> make_device_tensor(DimensionTypes... dim)
{
    static_assert(std::is_integral_v<IndexStoreType>);
    static_assert(std::is_integral_v<IndexCalculateType>);
    static_assert(std::is_unsigned_v<IndexStoreType>);
    static_assert(std::is_unsigned_v<IndexCalculateType>);
    static_assert(std::is_move_assignable_v<thrust::device_vector<T>>, "thrust::device_vector<T> must be movable");

    using Dimensions = whack::Array<IndexStoreType, sizeof...(DimensionTypes)>;

    const IndexCalculateType size = (std::make_unsigned_t<IndexCalculateType>(dim) * ...);
    thrust::device_vector<T> memory(size);

    return { std::move(memory), Dimensions { IndexStoreType(dim)... } };
}

template <typename T, uint32_t n_dims, typename IndexStoreType, typename IndexCalculateType>
Tensor<T, n_dims, IndexStoreType, IndexCalculateType> Tensor<T, n_dims, IndexStoreType, IndexCalculateType>::device_copy() const
{
    Tensor t = make_device_tensor<T>(m_dimensions);
    switch (m_device) {
    case ComputeDevice::CPU:
        t.device_vector() = host_vector();
        break;
    case ComputeDevice::CUDA:
        t.device_vector() = device_vector();
        break;
    }
    return t;
}

template <typename T, uint32_t n_dims, typename IndexStoreType, typename IndexCalculateType>
Tensor<T, n_dims, IndexStoreType, IndexCalculateType> Tensor<T, n_dims, IndexStoreType, IndexCalculateType>::host_copy() const
{
    Tensor t = make_host_tensor<T>(m_dimensions);
    switch (m_device) {
    case ComputeDevice::CPU:
        t.host_vector() = host_vector();
        break;
    case ComputeDevice::CUDA:
        t.host_vector() = device_vector();
        break;
    }
    return t;
}

}
