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

namespace {
    struct UnderlyingRngStateDummy;
}

template <typename RngType, uint32_t n_dims, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType>
class StateTensorView : TensorView<UnderlyingRngStateDummy, n_dims, IndexStoreType, IndexCalculateType> {

    using Index = whack::Array<IndexStoreType, n_dims>;

public:
    WHACK_DEVICES_INLINE
    StateTensorView(UnderlyingRngStateDummy* data, const Index& dimensions)
        : TensorView<UnderlyingRngStateDummy, n_dims, IndexStoreType, IndexCalculateType>(data, dimensions)
    {
    }

    template <typename U = RngType, typename... IndexTypes>
    WHACK_DEVICES_INLINE typename std::enable_if<std::is_same<U, FastInitType>::value, const KernelRNGFastInit&>::type operator()(const IndexTypes&... indices) const
    {
        return reinterpret_cast<const TensorView<KernelRNGFastInit, n_dims, IndexStoreType, IndexCalculateType>*>(this)->operator()(indices...);
    }

    template <typename U = RngType, typename... IndexTypes>
    WHACK_DEVICES_INLINE typename std::enable_if<std::is_same<U, FastInitType>::value, KernelRNGFastInit&>::type operator()(const IndexTypes&... indices)
    {
        return reinterpret_cast<TensorView<KernelRNGFastInit, n_dims, IndexStoreType, IndexCalculateType>*>(this)->operator()(indices...);
    }

    template <typename U = RngType, typename... IndexTypes>
    WHACK_DEVICES_INLINE typename std::enable_if<std::is_same<U, FastGenerationType>::value, const KernelRNGFastInit&>::type operator()(const IndexTypes&... indices) const
    {
        return reinterpret_cast<const TensorView<KernelRNGFastGeneration, n_dims, IndexStoreType, IndexCalculateType>*>(this)->operator()(indices...);
    }

    template <typename U = RngType, typename... IndexTypes>
    WHACK_DEVICES_INLINE typename std::enable_if<std::is_same<U, FastGenerationType>::value, KernelRNGFastGeneration&>::type operator()(const IndexTypes&... indices)
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
    template <typename T>
    StateTensor(UnderlyingTensor<T> t)
        : m_tensor(std::move(t))
    {
    }

    ComputeDevice device() const
    {
        return std::visit([](const auto& tensor) { return tensor.device(); }, m_tensor);
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
        const auto ptr = raw_pointer();
        const auto dims = dimensions();
        return StateTensorView<RngType, n_dims, IndexStoreType, IndexCalculateType>(ptr, dims);
    }
};

// template <typename Functor>
inline StateTensor<FastGenerationType, 1> make_host_state(/*Functor seed_and_sequence, */ int)
{
    auto t = make_tensor<CpuRNG>(whack::ComputeDevice::CPU, 1);
    using TensorType = decltype(t);
    auto st = StateTensor<FastGenerationType, TensorType::n_dims_value, TensorType::index_store_type, TensorType::index_calculate_type>(t);
    //    auto v = t.view();
    //    whack::start_parallel(
    //        t.device(), 1, 1, WHACK_KERNEL(=) {
    //            unsigned index = whack_threadIdx.x;
    //            uint64_t seed;
    //            uint64_t sequence_nr;
    //            thrust::tie(seed, sequence_nr) = seed_and_sequence(index);
    //            v(index) = CpuRNG(seed, sequence_nr);
    //        });
    return st;
}

}
