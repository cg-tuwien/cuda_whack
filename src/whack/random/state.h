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

#include <variant>

#include "whack/Tensor.h"
#include "whack/kernel.h"
#include "whack/random/generators.h"

namespace whack::random {

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
    WHACK_DEVICES_INLINE typename std::enable_if_t<std::is_same_v<U, FastInitType>, KernelGeneratorWithFastInit&> operator()(const IndexTypes&... indices)
    {
        return reinterpret_cast<TensorView<KernelGeneratorWithFastInit, n_dims, IndexStoreType, IndexCalculateType>*>(this)->operator()(indices...);
    }

    template <typename U = RngType, typename... IndexTypes>
    WHACK_DEVICES_INLINE typename std::enable_if_t<std::is_same_v<U, FastGenerationType>, KernelGeneratorWithFastGeneration&> operator()(const IndexTypes&... indices)
    {
        return reinterpret_cast<TensorView<KernelGeneratorWithFastGeneration, n_dims, IndexStoreType, IndexCalculateType>*>(this)->operator()(indices...);
    }
};

template <typename RngType, uint32_t n_dims, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType>
class StateTensor {
    template <typename RngState>
    using UnderlyingTensor = Tensor<RngState, n_dims, IndexStoreType, IndexCalculateType>;
    using Dimensions = whack::Array<IndexStoreType, n_dims>;

    std::variant<UnderlyingTensor<HostGenerator<float>>, UnderlyingTensor<FastGenerationDeviceGenerator>, UnderlyingTensor<FastInitDeviceGenerator>> m_tensor;

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
    auto t = make_tensor<HostGenerator<float>, IndexStoreType, IndexCalculateType>(whack::Location::Host, dims...);
    using TensorType = decltype(t);
    auto st = StateTensor<GeneratorType, TensorType::n_dims_value, typename TensorType::index_store_type, typename TensorType::index_calculate_type>(t);
    return st;
}

template <typename GeneratorType = DefaultType, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename... DimensionTypes>
inline StateTensor<GeneratorType, sizeof...(DimensionTypes)> make_device_state(DimensionTypes... dims)
{
    static_assert(std::is_same_v<GeneratorType, FastGenerationType> || std::is_same_v<GeneratorType, FastInitType>);
    if constexpr (std::is_same_v<GeneratorType, FastGenerationType>) {
        auto t = make_tensor<FastGenerationDeviceGenerator, IndexStoreType, IndexCalculateType>(whack::Location::Device, dims...);
        using TensorType = decltype(t);
        return StateTensor<GeneratorType, TensorType::n_dims_value, IndexStoreType, IndexCalculateType>(t);
    }
    if constexpr (std::is_same_v<GeneratorType, FastInitType>) {
        auto t = make_tensor<FastInitDeviceGenerator, IndexStoreType, IndexCalculateType>(whack::Location::Device, dims...);
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
} // namespace whack::random
