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
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <variant>

#include "TensorView.h"
#include "enums.h"

namespace whack {

/**
 * @brief The Tensor class is a device type erased holder of memory. Use the view() member function to get an accessor object.
 */
template <typename T, uint32_t n_dims, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType>
class Tensor {
public:
    static constexpr uint32_t n_dims_value = n_dims;
    using index_store_type = IndexStoreType;
    using index_calculate_type = IndexCalculateType;

private:
    std::variant<thrust::host_vector<T>, thrust::device_vector<T>> m_memory;
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
        assert(memory.size() == std::reduce(dimensions.begin(), dimensions.end(), IndexCalculateType(0), std::multiplies<IndexCalculateType>()));
    }
    Tensor(thrust::device_vector<T>&& memory, const Dimensions& dimensions)
        : m_memory(std::move(memory))
        , m_dimensions(dimensions)
        , m_device(ComputeDevice::CUDA)
    {
        assert(memory.size() == std::reduce(dimensions.begin(), dimensions.end(), IndexCalculateType(0), std::multiplies<IndexCalculateType>()));
    }

    [[nodiscard]] thrust::host_vector<T>& host_vector() &
    {
        return std::get<thrust::host_vector<T>>(m_memory);
    }

    [[nodiscard]] const thrust::host_vector<T>& host_vector() const&
    {
        return std::get<thrust::host_vector<T>>(m_memory);
    }

    /// disalow calling on a temporary
    [[nodiscard]] thrust::host_vector<T>& host_vector() && = delete;

    [[nodiscard]] thrust::device_vector<T>& device_vector() &
    {
        return std::get<thrust::device_vector<T>>(m_memory);
    }

    [[nodiscard]] const thrust::device_vector<T>& device_vector() const &
    {
        return std::get<thrust::device_vector<T>>(m_memory);
    }

    /// disalow calling on a temporary
    [[nodiscard]] thrust::host_vector<T>& device_vector() && = delete;

    [[nodiscard]] const T* raw_pointer() const
    {
        switch (m_device) {
        case ComputeDevice::CPU:
            return thrust::raw_pointer_cast(host_vector().data());
        case ComputeDevice::CUDA:
            return thrust::raw_pointer_cast(device_vector().data());
        }
        assert(false);
        return {};
    }

    [[nodiscard]] T* raw_pointer()
    {
        switch (m_device) {
        case ComputeDevice::CPU:
            return thrust::raw_pointer_cast(host_vector().data());
        case ComputeDevice::CUDA:
            return thrust::raw_pointer_cast(device_vector().data());
        }
        assert(false);
        return {};
    }

    [[nodiscard]] TensorView<const T, n_dims, IndexStoreType, IndexCalculateType> view() const
    {
        return { raw_pointer(), m_dimensions };
    }

    [[nodiscard]] TensorView<T, n_dims, IndexStoreType, IndexCalculateType> view()
    {
        return { raw_pointer(), m_dimensions };
    }

    [[nodiscard]] Tensor device_copy() const;
    [[nodiscard]] Tensor host_copy() const;

    [[nodiscard]] ComputeDevice device() const { return m_device; }
    [[nodiscard]] Dimensions dimensions() const { return m_dimensions; }
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
