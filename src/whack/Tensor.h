/*****************************************************************************
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

#ifndef __CUDACC__
static_assert(false, "whack::Tensor is only supported in cuda files!");
// because at the time of writing, thrust did not support even including its headers in cpp mode.
#endif

namespace whack {

/**
 * @brief The Tensor class is a device type erased holder of memory. Use the view() member function to get an accessor object.
 */
template <typename T, whack::size_t n_dims, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType>
class Tensor {
public:
    static constexpr whack::size_t n_dims_value = n_dims;
    using index_store_type = IndexStoreType;
    using index_calculate_type = IndexCalculateType;

private:
    std::variant<thrust::host_vector<T>, thrust::device_vector<T>> m_memory;
    Location m_device = Location::Invalid;

    using Dimensions = whack::Array<IndexStoreType, n_dims>;
    Dimensions m_dimensions = {};

public:
    Tensor() = default;
    Tensor(thrust::host_vector<T>&& memory, const Dimensions& dimensions)
        : m_memory(std::move(memory))
        , m_dimensions(dimensions)
        , m_device(Location::Host)
    {
        assert(host_vector().size() == std::reduce(dimensions.begin(), dimensions.end(), IndexCalculateType(1), std::multiplies<IndexCalculateType>()));
    }
    Tensor(thrust::device_vector<T>&& memory, const Dimensions& dimensions)
        : m_memory(std::move(memory))
        , m_dimensions(dimensions)
        , m_device(Location::Device)
    {
        assert(device_vector().size() == std::reduce(dimensions.begin(), dimensions.end(), IndexCalculateType(1), std::multiplies<IndexCalculateType>()));
    }

    [[nodiscard]] thrust::host_vector<T>& host_vector() &
    {
        try {
            return std::get<thrust::host_vector<T>>(m_memory);
        } catch (...) {
            throw std::logic_error("whack::Tensor::host_vector() called on a tensor that is not on the host!");
        }
    }

    [[nodiscard]] const thrust::host_vector<T>& host_vector() const&
    {
        try {
            return std::get<thrust::host_vector<T>>(m_memory);
        } catch (...) {
            throw std::logic_error("whack::Tensor::host_vector() called on a tensor that is not on the host!");
        }
    }

    /// disalow calling on a temporary
    [[nodiscard]] thrust::host_vector<T>& host_vector() && = delete;

    [[nodiscard]] thrust::device_vector<T>& device_vector() &
    {
        try {
            return std::get<thrust::device_vector<T>>(m_memory);
        } catch (...) {
            throw std::logic_error("whack::Tensor::device_vector() called on a tensor that is not on the device!");
        }
    }

    [[nodiscard]] const thrust::device_vector<T>& device_vector() const&
    {
        try {
            return std::get<thrust::device_vector<T>>(m_memory);
        } catch (...) {
            throw std::logic_error("whack::Tensor::device_vector() called on a tensor that is not on the device!");
        }
    }

    /// disalow calling on a temporary
    [[nodiscard]] thrust::host_vector<T>& device_vector() && = delete;

    [[nodiscard]] const T* raw_pointer() const
    {
        switch (m_device) {
        case Location::Host:
            return thrust::raw_pointer_cast(host_vector().data());
        case Location::Device:
            return thrust::raw_pointer_cast(device_vector().data());
        case Location::Invalid:
            return nullptr;
        }
        assert(false);
        return nullptr;
    }

    [[nodiscard]] T* raw_pointer()
    {
        switch (m_device) {
        case Location::Host:
            return thrust::raw_pointer_cast(host_vector().data());
        case Location::Device:
            return thrust::raw_pointer_cast(device_vector().data());
        case Location::Invalid:
            return nullptr;
        }
        assert(false);
        return nullptr;
    }

    [[nodiscard]] TensorView<const T, n_dims, IndexStoreType, IndexCalculateType> view() const
    {
        return { raw_pointer(), location(), m_dimensions };
    }

    [[nodiscard]] TensorView<T, n_dims, IndexStoreType, IndexCalculateType> view()
    {
        return { raw_pointer(), location(), m_dimensions };
    }

    [[nodiscard]] Tensor device_copy() const;
    [[nodiscard]] Tensor host_copy() const;

    [[nodiscard]] Location location() const { return m_device; }
    [[nodiscard]] Dimensions dimensions() const { return m_dimensions; }
};

template <typename T, whack::size_t n_dims, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType>
Tensor<T, n_dims> make_tensor(Location device, const whack::Array<IndexStoreType, n_dims>& dimensions)
{
    static_assert(std::is_integral_v<IndexStoreType>);
    static_assert(std::is_integral_v<IndexCalculateType>);
    static_assert(std::is_unsigned_v<IndexStoreType>);
    static_assert(std::is_unsigned_v<IndexCalculateType>);
    static_assert(std::is_move_assignable_v<thrust::host_vector<T>>, "thrust::host_vector<T> must be movable");

    IndexCalculateType size = 1;
    for (unsigned i = 0; i < n_dims; ++i)
        size *= dimensions[i];

    switch (device) {
    case Location::Host:
        return { thrust::host_vector<T>(size), dimensions };
    case Location::Device:
        return { thrust::device_vector<T>(size), dimensions };
    case Location::Invalid:
        return {};
    }
    return {};
}

template <typename T, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename... DimensionTypes>
Tensor<T, sizeof...(DimensionTypes)> make_tensor(Location device, DimensionTypes... dim)
{
    static_assert(std::is_integral_v<IndexStoreType>);
    static_assert(std::is_integral_v<IndexCalculateType>);
    static_assert(std::is_unsigned_v<IndexStoreType>);
    static_assert(std::is_unsigned_v<IndexCalculateType>);
    static_assert(std::is_move_assignable_v<thrust::host_vector<T>>, "thrust::host_vector<T> must be movable");

    using Dimensions = whack::Array<IndexStoreType, sizeof...(DimensionTypes)>;

    const IndexCalculateType size = (std::make_unsigned_t<IndexCalculateType>(dim) * ...);
    const auto dimensions = Dimensions { IndexStoreType(dim)... };

    switch (device) {
    case Location::Host:
        return { thrust::host_vector<T>(size), dimensions };
    case Location::Device:
        return { thrust::device_vector<T>(size), dimensions };
    case Location::Invalid:
        return {};
    }
    return {};
}

template <typename T, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename... DimensionTypes>
Tensor<T, sizeof...(DimensionTypes)> make_host_tensor(DimensionTypes... dim)
{
    return make_tensor<T, IndexStoreType, IndexCalculateType>(Location::Host, dim...);
}

template <typename T, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename... DimensionTypes>
Tensor<T, sizeof...(DimensionTypes)> make_device_tensor(DimensionTypes... dim)
{
    return make_tensor<T, IndexStoreType, IndexCalculateType>(Location::Device, dim...);
}

template <typename T, whack::size_t n_dims, typename IndexStoreType, typename IndexCalculateType>
Tensor<T, n_dims, IndexStoreType, IndexCalculateType> Tensor<T, n_dims, IndexStoreType, IndexCalculateType>::device_copy() const
{
    Tensor t = make_tensor<T>(whack::Location::Device, m_dimensions);
    switch (m_device) {
    case Location::Host:
        t.device_vector() = host_vector();
        break;
    case Location::Device:
        t.device_vector() = device_vector();
        break;
    }
    return t;
}

template <typename T, whack::size_t n_dims, typename IndexStoreType, typename IndexCalculateType>
Tensor<T, n_dims, IndexStoreType, IndexCalculateType> Tensor<T, n_dims, IndexStoreType, IndexCalculateType>::host_copy() const
{
    Tensor t = make_tensor<T>(whack::Location::Host, m_dimensions);
    switch (m_device) {
    case Location::Host:
        t.host_vector() = host_vector();
        break;
    case Location::Device:
        t.host_vector() = device_vector();
        break;
    }
    return t;
}

} // namespace whack
