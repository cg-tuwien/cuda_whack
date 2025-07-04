/****************************************************************************
 *  Copyright (C) 2023 Adam Celarek (github.com/adam-ce, github.com/cg-tuwien)
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights to
 *  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 *  of the Software, and to permit persons to whom the Software is furnished to do so,
 *  subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 ****************************************************************************/

#pragma once
#define WHACK_STRIDE_BASED_CALCULATION

#include "enums.h"
#include "indexing.h"
#include "macros.h"
#include <cinttypes>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace whack {

/**
 * @brief Use TensorView for passing a tensor to kernels. Unlike a Tensor, it can be copied to the gpu using lambda capture.
 */
template <typename T, whack::size_t n_dims, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType>
class TensorView {
    static_assert(std::is_integral_v<IndexStoreType>);
    static_assert(std::is_integral_v<IndexCalculateType>);
    static_assert(std::is_unsigned_v<IndexStoreType>);
    static_assert(std::is_unsigned_v<IndexCalculateType>);

    using Index = whack::Array<IndexStoreType, n_dims>;
    T* m_data = nullptr;
#ifdef WHACK_STRIDE_BASED_CALCULATION
    // strides shifted by one dim, so that we can fit the total size in. see constructor and offset() for details
    whack::Array<IndexCalculateType, n_dims> m_cum_dims = {};
#endif
#if !defined(NDEBUG) || !defined(WHACK_STRIDE_BASED_CALCULATION)
    Index m_dimensions = {};
#endif

#if !defined(NDEBUG)
    Location m_location = Location::Invalid;
#endif

    WHACK_DEVICES_INLINE void assert_access_location() const noexcept
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

    WHACK_DEVICES_INLINE IndexCalculateType offset(const Index& index) const noexcept
    {
#ifdef WHACK_STRIDE_BASED_CALCULATION
        IndexCalculateType offset = index[n_dims - 1];
        for (unsigned i = 0; i < n_dims - 1; ++i) {
            offset += index[i] * m_cum_dims[i + 1];
        }
        return offset;
#else
        return whack::join_n_dim_index<IndexCalculateType, n_dims, IndexStoreType>(m_dimensions, index);
#endif
    }

    // compile time for from https://stackoverflow.com/a/47563100
    template <std::size_t N>
    struct num {
        static const constexpr auto value = N;
    };

    template <class F, std::size_t... Is>
    WHACK_DEVICES_INLINE static void for_(F func, std::index_sequence<Is...>) noexcept
    {
        (func(num<Is> {}), ...);
    }

    template <std::size_t N, typename F>
    WHACK_DEVICES_INLINE static void for_(F func) noexcept
    {
        for_(func, std::make_index_sequence<N>());
    }

public:
    using Shape = Index;
    using value_type = T;

    TensorView() = default;

    WHACK_DEVICES_INLINE TensorView(const TensorView<std::remove_const_t<T>, n_dims, IndexStoreType, IndexCalculateType>& other) noexcept
    {
        // not using copy and swap idiom since we would have to implement a swap function using WHACK_DEVICES_INLINE
        *this = other;
    };

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
            cum_dims *= dimensions[i];
            m_cum_dims[i] = cum_dims;
        }
#endif

        for (const auto s : shape()) {
            if (s == 0)
                return; // torch might not allocate the tensor on the gpu if the size is 0 (empty tensor). don't test the pointer in that case.
        }
        cudaPointerAttributes attr;
        const auto result = cudaPointerGetAttributes(&attr, (void*)data);
        if (location == Location::Device) {
            if (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged)
                throw std::logic_error("TensorView created with Location::Device, but data points to something on the host!");
        } else if (location == Location::Host) {
            if (result == cudaErrorInsufficientDriver)
                return; // happens when running without cuda capable gpu (with cpu only code).
            if (attr.type != cudaMemoryTypeHost && attr.type != cudaMemoryTypeManaged && attr.type != cudaMemoryTypeUnregistered)
                throw std::logic_error("TensorView created with Location::Host, but data points to something on the device!");
        } else {
            throw std::logic_error("TensorView created with invalid location!");
        }
    }

    template <whack::size_t dimension>
    WHACK_DEVICES_INLINE IndexStoreType size() const noexcept
    {
        static_assert(dimension < n_dims);
#if defined(WHACK_STRIDE_BASED_CALCULATION)
        if (dimension == n_dims - 1)
            return m_cum_dims[n_dims - 1];
        return m_cum_dims[dimension] / m_cum_dims[dimension + 1];
#else
        return m_dimensions[dimension];
#endif
    }

    WHACK_DEVICES_INLINE IndexStoreType size(unsigned dimension) const noexcept
    {
        assert(dimension < n_dims);
#if defined(WHACK_STRIDE_BASED_CALCULATION)
        if (dimension == n_dims - 1)
            return m_cum_dims[n_dims - 1];
        return m_cum_dims[dimension] / m_cum_dims[dimension + 1];
#else
        return m_dimensions[dimension];
#endif
    }

    WHACK_DEVICES_INLINE Shape shape() const noexcept
    {
        Shape s;
        for_<n_dims>([&s, this](auto i) {
            IndexStoreType size = this->size<whack::size_t(i.value)>();
            s[i.value] = size;
        });
        return s;
    }

    template <typename U = T>
    WHACK_DEVICES_INLINE typename std::enable_if<(n_dims > 1), const U&>::type operator()(const Index& index) const noexcept
    {
        assert_access_location();
        for (unsigned i = 0; i < n_dims; ++i)
            assert(index[i] < m_dimensions[i]);

        return *(m_data + offset(index));
    }

    template <typename U = T>
    WHACK_DEVICES_INLINE typename std::enable_if<(n_dims > 1), U&>::type operator()(const Index& index) noexcept
    {
        assert_access_location();
        for (unsigned i = 0; i < n_dims; ++i)
            assert(index[i] < m_dimensions[i]);
        return *(m_data + offset(index));
    }

    WHACK_DEVICES_INLINE
    const T& operator()(const IndexStoreType& index) const noexcept
    {
        assert_access_location();
        assert(index < m_dimensions[0]);
        return *(m_data + index);
    }

    WHACK_DEVICES_INLINE
    T& operator()(const IndexStoreType& index) noexcept
    {
        assert_access_location();
        assert(index < m_dimensions[0]);
        return *(m_data + index);
    }

    template <typename... IndexTypes>
    WHACK_DEVICES_INLINE const T& operator()(const IndexStoreType& index0, const IndexTypes&... other_indices) const noexcept
    {
        static_assert(sizeof...(other_indices) + 1 == n_dims);
        return operator()(Index { index0, IndexStoreType(other_indices)... });
    }

    template <typename... IndexTypes>
    WHACK_DEVICES_INLINE T& operator()(const IndexStoreType& index0, const IndexTypes&... other_indices) noexcept
    {
        static_assert(sizeof...(other_indices) + 1 == n_dims);
        return operator()(Index { index0, IndexStoreType(other_indices)... });
    }

    WHACK_DEVICES_INLINE T* data() noexcept
    {
        return m_data;
    }

    WHACK_DEVICES_INLINE const T* data() const noexcept
    {
        return m_data;
    }

    WHACK_DEVICES_INLINE TensorView& operator=(const TensorView<std::remove_const_t<T>, n_dims, IndexStoreType, IndexCalculateType>& other) noexcept
    {
        this->m_data = other.m_data;
#ifdef WHACK_STRIDE_BASED_CALCULATION
        this->m_cum_dims = other.m_cum_dims;
#endif

#if !defined(NDEBUG) || !defined(WHACK_STRIDE_BASED_CALCULATION)
        this->m_dimensions = other.m_dimensions;
#endif

#if !defined(NDEBUG)
        this->m_location = other.m_location;
#endif
        return *this;
    };
    friend class TensorView<const T, n_dims, IndexStoreType, IndexCalculateType>;
};

// whack::Array api
namespace detail {
    template <template <typename...> class, template <typename...> class>
    struct is_same_template : std::false_type {
    };

    template <template <typename...> class T>
    struct is_same_template<T, T> : std::true_type {
    };

    template <template <typename...> class T, template <typename...> class U>
    inline constexpr bool is_same_template_v = is_same_template<T, U>::value;
} // namespace detail

// // api for raw pointers
// dimensions array api
template <typename InterpretedType, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename T, whack::size_t n_dims>
TensorView<InterpretedType, n_dims, IndexStoreType, IndexCalculateType>
make_tensor_view(T* pointer, Location location, const whack::Array<IndexStoreType, n_dims>& dimensions)
{
    return { reinterpret_cast<InterpretedType*>(pointer), location, dimensions };
}

template <typename InterpretedType, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename T, whack::size_t n_dims>
TensorView<const InterpretedType, n_dims, IndexStoreType, IndexCalculateType>
make_tensor_view(const T* pointer, Location location, const whack::Array<IndexStoreType, n_dims>& dimensions)
{
    return { reinterpret_cast<const InterpretedType*>(pointer), location, dimensions };
}

// parameter pack api
template <typename InterpretedType, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename T, typename... DimensionTypes>
TensorView<InterpretedType, sizeof...(DimensionTypes), IndexStoreType, IndexCalculateType>
make_tensor_view(T* pointer, Location location, DimensionTypes... dimensions)
{
    return { reinterpret_cast<InterpretedType*>(pointer), location, { IndexStoreType(dimensions)... } };
}

template <typename InterpretedType, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename T, typename... DimensionTypes>
TensorView<const InterpretedType, sizeof...(DimensionTypes), IndexStoreType, IndexCalculateType>
make_tensor_view(const T* pointer, Location location, DimensionTypes... dimensions)
{
    return { reinterpret_cast<const InterpretedType*>(pointer), location, { IndexStoreType(dimensions)... } };
}

} // namespace whack
