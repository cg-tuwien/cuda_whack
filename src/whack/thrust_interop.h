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

#include "TensorView.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <type_traits>

namespace whack {

// whack::Array api
namespace detail {
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
} // namespace detail

// // api for thrust arrays
// dimensions array api
template <typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, template <typename...> class Vector, typename T, whack::size_t n_dims>
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

template <typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, template <typename...> class Vector, typename T, whack::size_t n_dims>
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

} // namespace whack
