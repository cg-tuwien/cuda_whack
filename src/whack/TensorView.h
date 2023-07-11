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
#define WHACK_STRIDE_BASED_CALCULATION

#include <cinttypes>
#include <type_traits>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "enums.h"
#include "indexing.h"
#include "macros.h"

namespace whack {

/**
 * @brief Use TensorView for passing a tensor to kernels. Unlike a Tensor, it can be copied to the gpu using lambda capture.
 */
template <typename T, uint32_t n_dims, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType>
class TensorView {
    static_assert(std::is_integral_v<IndexStoreType>);
    static_assert(std::is_integral_v<IndexCalculateType>);
    static_assert(std::is_unsigned_v<IndexStoreType>);
    static_assert(std::is_unsigned_v<IndexCalculateType>);

    using Index = whack::Array<IndexStoreType, n_dims>;
    T* m_data = nullptr;
#ifdef WHACK_STRIDE_BASED_CALCULATION
    whack::Array<IndexCalculateType, n_dims> m_strides = {};
#endif
#if !defined(NDEBUG) || !defined(WHACK_STRIDE_BASED_CALCULATION)
    Index m_dimensions = {};
#endif

#if !defined(NDEBUG)
    Location m_location = Location::Invalid;
#endif

    WHACK_DEVICES_INLINE void assert_access_location() const
    {
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
        // device code trajectory
        assert(m_location == Location::Device);
#else
        // nvcc host code trajectory
        // or non-nvcc code trajectory
        assert(m_location == Location::Host);
#endif
    }

    WHACK_DEVICES_INLINE IndexCalculateType offset(const Index& index) const
    {
#ifdef WHACK_STRIDE_BASED_CALCULATION
        IndexCalculateType offset = index[n_dims - 1];
        for (unsigned i = 0; i < n_dims - 1; ++i) {
            offset += index[i] * m_strides[i];
        }
        return offset;
#else
        return whack::join_n_dim_index<IndexCalculateType, n_dims, IndexStoreType>(m_dimensions, index);
#endif
    }

public:
    TensorView() = default;

    TensorView(T* data, Location location, const Index& dimensions)
        : m_data(data)
#if !defined(NDEBUG) || !defined(WHACK_STRIDE_BASED_CALCULATION)
        , m_dimensions(dimensions)
#endif
#if !defined(NDEBUG)
        , m_location(location)
#endif
    {
#ifdef WHACK_STRIDE_BASED_CALCULATION
        IndexCalculateType cum_dims = 1;
        for (unsigned i = n_dims - 1; i < n_dims; --i) {
            m_strides[i] = cum_dims;
            cum_dims *= dimensions[i];
        }
#endif

        cudaPointerAttributes attr;
        cudaPointerGetAttributes(&attr, (void*)data);
        if (location == Location::Device) {
            if (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged)
                throw std::logic_error("TensorView created with Location::Device, but data points to something on the host!");
        } else if (location == Location::Host) {
            if (attr.type != cudaMemoryTypeHost && attr.type != cudaMemoryTypeManaged && attr.type != cudaMemoryTypeUnregistered)
                throw std::logic_error("TensorView created with Location::Host, but data points to something on the device!");
        } else {
            throw std::logic_error("TensorView created with invalid location!");
        }
    }

    template <typename U = T>
    WHACK_DEVICES_INLINE typename std::enable_if<(n_dims > 1), const U&>::type operator()(const Index& index) const
    {
        assert_access_location();
        for (unsigned i = 0; i < n_dims; ++i)
            assert(index[i] < m_dimensions[i]);

        return *(m_data + offset(index));
    }

    template <typename U = T>
    WHACK_DEVICES_INLINE typename std::enable_if<(n_dims > 1), U&>::type operator()(const Index& index)
    {
        assert_access_location();
        for (unsigned i = 0; i < n_dims; ++i)
            assert(index[i] < m_dimensions[i]);
        return *(m_data + offset(index));
    }

    WHACK_DEVICES_INLINE
    const T& operator()(const IndexStoreType& index) const
    {
        assert_access_location();
        assert(index < m_dimensions[0]);
        return *(m_data + index);
    }

    WHACK_DEVICES_INLINE
    T& operator()(const IndexStoreType& index)
    {
        assert_access_location();
        assert(index < m_dimensions[0]);
        return *(m_data + index);
    }

    template <typename... IndexTypes>
    WHACK_DEVICES_INLINE const T& operator()(const IndexStoreType& index0, const IndexTypes&... other_indices) const
    {
        return operator()(Index { index0, IndexStoreType(other_indices)... });
    }

    template <typename... IndexTypes>
    WHACK_DEVICES_INLINE T& operator()(const IndexStoreType& index0, const IndexTypes&... other_indices)
    {
        return operator()(Index { index0, IndexStoreType(other_indices)... });
    }
};

// whack::Array api
namespace detail {
    template <template <typename...> class, template <typename...> class>
    struct is_same_template : std::false_type { };

    template <template <typename...> class T>
    struct is_same_template<T, T> : std::true_type { };

    template <template <typename...> class T, template <typename...> class U>
    inline constexpr bool is_same_template_v = is_same_template<T, U>::value;

    template <template <typename...> class Vector, typename T>
    std::enable_if_t<detail::is_same_template_v<Vector, thrust::host_vector>, Location>
    location_of(const Vector<T>&)
    {
        return Location::Host;
    }

    template <template <typename...> class Vector, typename T>
    std::enable_if_t<detail::is_same_template_v<Vector, thrust::device_vector>, Location>
    location_of(const Vector<T>&)
    {
        return Location::Device;
    }
}

template <typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, template <typename...> class Vector, typename T, uint32_t n_dims>
std::enable_if_t<detail::is_same_template_v<thrust::host_vector, Vector> || detail::is_same_template_v<Vector, thrust::device_vector>, TensorView<T, n_dims, IndexStoreType, IndexCalculateType>>
make_tensor_view(Vector<T>& data, const whack::Array<IndexStoreType, n_dims>& dimensions)
{
#ifndef NDEBUG
    IndexCalculateType dimension_size = 1;
    for (unsigned i = 0; i < n_dims; ++i)
        dimension_size *= dimensions[i];
    assert(dimension_size == data.size());
#endif
    return { thrust::raw_pointer_cast(data.data()), detail::location_of(data), dimensions };
}

template <typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, template <typename...> class Vector, typename T, uint32_t n_dims>
std::enable_if_t<detail::is_same_template_v<thrust::host_vector, Vector> || detail::is_same_template_v<Vector, thrust::device_vector>, TensorView<const T, n_dims, IndexStoreType, IndexCalculateType>>
make_tensor_view(const Vector<T>& data, const whack::Array<IndexStoreType, n_dims>& dimensions)
{
#ifndef NDEBUG
    IndexCalculateType dimension_size = 1;
    for (unsigned i = 0; i < n_dims; ++i)
        dimension_size *= dimensions[i];
    assert(dimension_size == data.size());
#endif
    return { thrust::raw_pointer_cast(data.data()), detail::location_of(data), dimensions };
}

// parameter pack api
template <typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, template <typename...> class Vector, typename T, typename... DimensionTypes>
std::enable_if_t<detail::is_same_template_v<thrust::host_vector, Vector> || detail::is_same_template_v<Vector, thrust::device_vector>, TensorView<T, sizeof...(DimensionTypes), IndexStoreType, IndexCalculateType>>
make_tensor_view(Vector<T>& data, DimensionTypes... dimensions)
{
    assert((std::make_unsigned_t<IndexCalculateType>(dimensions) * ...) == data.size());
    return { thrust::raw_pointer_cast(data.data()), detail::location_of(data), { IndexStoreType(dimensions)... } };
}

template <typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, template <typename...> class Vector, typename T, typename... DimensionTypes>
std::enable_if_t<detail::is_same_template_v<thrust::host_vector, Vector> || detail::is_same_template_v<Vector, thrust::device_vector>, TensorView<const T, sizeof...(DimensionTypes), IndexStoreType, IndexCalculateType>>
make_tensor_view(const Vector<T>& data, DimensionTypes... dimensions)
{
    assert((IndexCalculateType(dimensions) * ...) == data.size());
    return { thrust::raw_pointer_cast(data.data()), detail::location_of(data), { IndexStoreType(dimensions)... } };
}

}
