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

#include <torch/types.h>

#include "TensorView.h"
#include "enums.h"

namespace whack {
namespace detail {
    inline whack::Location location_of(const torch::Tensor& t)
    {
        if (t.device().is_cuda() && t.device().index() == 0)
            return whack::Location::Device;
        if (t.is_cpu())
            return whack::Location::Host;
        std::cerr << "Unknown device " << t.device() << std::endl;
        return whack::Location::Invalid;
    }

} // namespace detail

// dimensions array api
template <typename T, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, whack::size_t n_dims>
TensorView<T, n_dims, IndexStoreType, IndexCalculateType>
make_tensor_view(torch::Tensor& data, const whack::Array<IndexStoreType, n_dims>& dimensions)
{
#ifndef NDEBUG
    IndexCalculateType dimension_size = 1;
    for (unsigned i = 0; i < n_dims; ++i)
        dimension_size *= dimensions[i];
    assert(dimension_size * sizeof(T) == torch::numel(data) * data.element_size());
#endif
    return { reinterpret_cast<T*>(data.data_ptr()), detail::location_of(data), dimensions };
}

template <typename T, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, whack::size_t n_dims>
TensorView<T, n_dims, IndexStoreType, IndexCalculateType>
make_tensor_view(const torch::Tensor& data, const whack::Array<IndexStoreType, n_dims>& dimensions)
{
#ifndef NDEBUG
    IndexCalculateType dimension_size = 1;
    for (unsigned i = 0; i < n_dims; ++i)
        dimension_size *= dimensions[i];
    if (dimension_size * sizeof(T) != torch::numel(data) * data.element_size()) {
        for (unsigned i = 0; i < n_dims; ++i) {
            std::cout << "dimensions[" << i << "] = " << dimensions[i] << std::endl;
        }
        std::cout << "dimension_size = " << dimension_size << ", sizeof(T)=" << sizeof(T)
                  << ", torch::numel(data)=" << torch::numel(data) << ", data.element_size()=" << data.element_size() << std::endl;
    }
    assert(dimension_size * sizeof(T) == torch::numel(data) * data.element_size());
#endif
    return { reinterpret_cast<const T*>(data.data_ptr()), detail::location_of(data), dimensions };
}

// parameter pack api
template <typename T, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename... DimensionTypes>
TensorView<T, sizeof...(DimensionTypes), IndexStoreType, IndexCalculateType>
make_tensor_view(torch::Tensor& data, DimensionTypes... dimensions)
{
    return make_tensor_view<T>(data, whack::Array<IndexStoreType, sizeof...(DimensionTypes)> { IndexStoreType(dimensions)... });
}

template <typename T, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename... DimensionTypes>
TensorView<const T, sizeof...(DimensionTypes), IndexStoreType, IndexCalculateType>
make_tensor_view(const torch::Tensor& data, DimensionTypes... dimensions)
{
    return make_tensor_view<T>(data, whack::Array<IndexStoreType, sizeof...(DimensionTypes)> { IndexStoreType(dimensions)... });
}

template <typename T>
T* raw_pointer(const torch::Tensor& tensor)
{
    return reinterpret_cast<T*>(tensor.data_ptr());
}

} // namespace whack
