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

#include <thrust/detail/raw_pointer_cast.h>

#include "indexing.h"
#include "macros.h"

namespace whack {

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

    WHACK_DEVICES_INLINE
    TensorView(T* data, const Index& dimensions)
        : m_data(data)
#if !defined(NDEBUG) || !defined(WHACK_STRIDE_BASED_CALCULATION)
        , m_dimensions(dimensions)
#endif
    {
#ifdef WHACK_STRIDE_BASED_CALCULATION
        IndexCalculateType cum_dims = 1;
        for (unsigned i = n_dims - 1; i < n_dims; --i) {
            m_strides[i] = cum_dims;
            cum_dims *= dimensions[i];
        }
#endif
    }

    template <typename U = T>
    WHACK_DEVICES_INLINE typename std::enable_if<(n_dims > 1), const U&>::type operator()(const Index& index) const
    {
        for (unsigned i = 0; i < n_dims; ++i)
            assert(index[i] < m_dimensions[i]);

        return *(m_data + offset(index));
    }

    template <typename U = T>
    WHACK_DEVICES_INLINE typename std::enable_if<(n_dims > 1), U&>::type operator()(const Index& index)
    {
        for (unsigned i = 0; i < n_dims; ++i)
            assert(index[i] < m_dimensions[i]);
        return *(m_data + offset(index));
    }

    WHACK_DEVICES_INLINE
    const T& operator()(const IndexStoreType& index) const
    {
        assert(index < m_dimensions[0]);
        return *(m_data + index);
    }

    WHACK_DEVICES_INLINE
    T& operator()(const IndexStoreType& index)
    {
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
template <typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename ThrustVector, uint32_t n_dims>
TensorView<typename std::remove_pointer_t<decltype(thrust::raw_pointer_cast(ThrustVector().data()))>, n_dims, IndexStoreType, IndexCalculateType>
make_tensor_view(ThrustVector& data, const whack::Array<IndexStoreType, n_dims>& dimensions)
{
    static_assert(std::is_integral_v<IndexStoreType>);
    static_assert(std::is_integral_v<IndexCalculateType>);
    static_assert(std::is_unsigned_v<IndexStoreType>);
    static_assert(std::is_unsigned_v<IndexCalculateType>);

#ifndef NDEBUG
    IndexCalculateType dimension_size = 1;
    for (unsigned i = 0; i < n_dims; ++i)
        dimension_size *= dimensions[i];
    assert(dimension_size == data.size());
#endif
    return { thrust::raw_pointer_cast(data.data()), dimensions };
}

// parameter pack api
template <typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename ThrustVector, typename... DimensionTypes>
TensorView<
    typename std::remove_pointer_t<decltype(thrust::raw_pointer_cast(ThrustVector().data()))>,
    sizeof...(DimensionTypes),
    std::make_unsigned_t<IndexStoreType>,
    std::make_unsigned_t<IndexCalculateType>>
make_tensor_view(ThrustVector& data, DimensionTypes... dimensions)
{
    assert((std::make_unsigned_t<IndexCalculateType>(dimensions) * ...) == data.size());
    return { thrust::raw_pointer_cast(data.data()), { std::make_unsigned_t<IndexStoreType>(dimensions)... } };
}

}
