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

#include <variant>

#include "whack/RandomNumberGenerator.h"
#include "whack/Tensor.h"
#include "whack/kernel.h"

namespace whack::rng {

struct FastGenerationType;
struct FastInitType;
using DefaultType = FastGenerationType;

namespace {
    struct UnderlyingRngStateDummy;
}

template <typename RngType, uint32_t n_dims, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType>
class StateTensorView : TensorView<UnderlyingRngStateDummy, n_dims, IndexStoreType, IndexCalculateType> {

    using Index = whack::Array<IndexStoreType, n_dims>;

public:
    StateTensorView() = default;

    StateTensorView(UnderlyingRngStateDummy* data, Location location, const Index& dimensions)
        : TensorView<UnderlyingRngStateDummy, n_dims, IndexStoreType, IndexCalculateType>(data, location, dimensions)
    {
    }
    template <typename U = RngType, typename... IndexTypes>
    WHACK_DEVICES_INLINE typename std::enable_if_t<std::is_same_v<U, FastInitType>, KernelRNGFastInit&> operator()(const IndexTypes&... indices)
    {
        return reinterpret_cast<TensorView<KernelRNGFastInit, n_dims, IndexStoreType, IndexCalculateType>*>(this)->operator()(indices...);
    }

    template <typename U = RngType, typename... IndexTypes>
    WHACK_DEVICES_INLINE typename std::enable_if_t<std::is_same_v<U, FastGenerationType>, KernelRNGFastGeneration&> operator()(const IndexTypes&... indices)
    {
        return reinterpret_cast<TensorView<KernelRNGFastGeneration, n_dims, IndexStoreType, IndexCalculateType>*>(this)->operator()(indices...);
    }
};

template <typename RngType, uint32_t n_dims, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType>
class StateTensor {
    template <typename RngState>
    using UnderlyingTensor = Tensor<RngState, n_dims, IndexStoreType, IndexCalculateType>;
    using Dimensions = whack::Array<IndexStoreType, n_dims>;

    std::variant<UnderlyingTensor<CpuRNG>, UnderlyingTensor<GpuRNGFastGeneration>, UnderlyingTensor<GpuRNGFastInit>> m_tensor;

public:
    StateTensor() = default;

    template <typename T>
    StateTensor(UnderlyingTensor<T> t)
        : m_tensor(std::move(t))
    {
    }

    Location location() const
    {
        return std::visit([](const auto& tensor) { return tensor.location(); }, m_tensor);
    }

    UnderlyingRngStateDummy* raw_pointer()
    {
        return std::visit([](auto& tensor) -> UnderlyingRngStateDummy* { return reinterpret_cast<UnderlyingRngStateDummy*>(tensor.raw_pointer()); }, m_tensor);
    }

    Dimensions dimensions() const
    {
        return std::visit([](const auto& tensor) { return tensor.dimensions(); }, m_tensor);
    }

    StateTensorView<RngType, n_dims, IndexStoreType, IndexCalculateType> view()
    {
        return StateTensorView<RngType, n_dims, IndexStoreType, IndexCalculateType>(raw_pointer(), location(), dimensions());
    }
};

template <typename GeneratorType = DefaultType, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename... DimensionTypes>
inline StateTensor<GeneratorType, sizeof...(DimensionTypes)> make_host_state(DimensionTypes... dims)
{
    static_assert(std::is_same_v<GeneratorType, FastGenerationType> || std::is_same_v<GeneratorType, FastInitType>);
    auto t = make_tensor<CpuRNG, IndexStoreType, IndexCalculateType>(whack::Location::Host, dims...);
    using TensorType = decltype(t);
    auto st = StateTensor<GeneratorType, TensorType::n_dims_value, typename TensorType::index_store_type, typename TensorType::index_calculate_type>(t);
    return st;
}

template <typename GeneratorType = DefaultType, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename... DimensionTypes>
inline StateTensor<GeneratorType, sizeof...(DimensionTypes)> make_device_state(DimensionTypes... dims)
{
    static_assert(std::is_same_v<GeneratorType, FastGenerationType> || std::is_same_v<GeneratorType, FastInitType>);
    if constexpr (std::is_same_v<GeneratorType, FastGenerationType>) {
        auto t = make_tensor<GpuRNGFastGeneration, IndexStoreType, IndexCalculateType>(whack::Location::Device, dims...);
        using TensorType = decltype(t);
        return StateTensor<GeneratorType, TensorType::n_dims_value, IndexStoreType, IndexCalculateType>(t);
    }
    if constexpr (std::is_same_v<GeneratorType, FastInitType>) {
        auto t = make_tensor<GpuRNGFastInit, IndexStoreType, IndexCalculateType>(whack::Location::Device, dims...);
        using TensorType = decltype(t);
        return StateTensor<GeneratorType, TensorType::n_dims_value, IndexStoreType, IndexCalculateType>(t);
    }
    assert(false);
    return {};
}

template <typename GeneratorType = DefaultType, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename... DimensionTypes>
inline StateTensor<GeneratorType, sizeof...(DimensionTypes)> make_state(whack::Location location, DimensionTypes... dims)
{
    static_assert(std::is_same_v<GeneratorType, FastGenerationType> || std::is_same_v<GeneratorType, FastInitType>);
    switch (location) {
    case whack::Location::Host:
        return make_host_state<GeneratorType, IndexStoreType, IndexCalculateType, DimensionTypes...>(dims...);
    case whack::Location::Device:
        return make_device_state<GeneratorType, IndexStoreType, IndexCalculateType, DimensionTypes...>(dims...);
    case whack::Location::Invalid:
        break;
    }
    assert(false);
    return {};
}
}
