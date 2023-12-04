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
    using value_type = T;

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

    template <typename InterpretedType = T, typename... DimensionTypes>
    [[nodiscard]] TensorView<const InterpretedType, sizeof...(DimensionTypes), IndexStoreType, IndexCalculateType> view(DimensionTypes... dimensions) const
    {
        const auto v = make_tensor_view<const InterpretedType>(raw_pointer(), location(), dimensions...);
        check_sizes(v);
        return v;
    }

    template <typename InterpretedType = T, typename... DimensionTypes>
    [[nodiscard]] TensorView<InterpretedType, sizeof...(DimensionTypes), IndexStoreType, IndexCalculateType> view(DimensionTypes... dimensions)
    {
        const auto v = make_tensor_view<InterpretedType>(raw_pointer(), location(), dimensions...);
        check_sizes(v);
        return v;
    }

    template <typename... ViewIndex>
    [[nodiscard]] const T& operator()(const ViewIndex&... params) const
    {
        if (m_device != Location::Host)
            throw std::logic_error("whack::Tensor::operator(): element access attempted on a tensor that is not on the host! ");
        return view()(params...);
    }

    template <typename... ViewIndex>
    [[nodiscard]] T& operator()(const ViewIndex&... params)
    {
        if (m_device != Location::Host)
            throw std::logic_error("whack::Tensor::operator(): element access attempted on a tensor that is not on the host! ");
        return view()(params...);
    }

    [[nodiscard]] Tensor device_copy() const;
    [[nodiscard]] Tensor host_copy() const;

    [[nodiscard]] Location location() const { return m_device; }
    [[nodiscard]] Dimensions dimensions() const { return m_dimensions; }

    [[nodiscard]] IndexCalculateType numel() const
    {
        IndexCalculateType s = 1;
        for (unsigned i = 0; i < n_dims; ++i)
            s *= m_dimensions[i];
        return s;
    }

    [[nodiscard]] constexpr IndexCalculateType n_dimensions() const
    {
        return n_dims;
    }

private:
    template <typename View>
    void check_sizes(const View& view) const
    {
        IndexCalculateType orig_size = sizeof(T) * numel();

        const auto v_shape = view.shape();
        IndexCalculateType v_size = sizeof(typename View::value_type);
        for (unsigned i = 0; i < v_shape.size(); ++i)
            v_size *= v_shape[i];

        if (orig_size != v_size)
            throw std::logic_error("whack::Tensor::view<..>(...): Tried to create a view with incompatible dimensions! Original size is "
                + std::to_string(orig_size) + " bytes, while the view would have " + std::to_string(v_size) + " bytes.");
    }
};

template <typename T, whack::size_t n_dims, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename BeginIterator, typename EndIterator>
Tensor<T, n_dims> make_tensor(Location device, BeginIterator begin, EndIterator end, const whack::Array<IndexStoreType, n_dims>& dimensions)
{
    static_assert(std::is_integral_v<IndexStoreType>);
    static_assert(std::is_integral_v<IndexCalculateType>);
    static_assert(std::is_unsigned_v<IndexStoreType>);
    static_assert(std::is_unsigned_v<IndexCalculateType>);
    static_assert(std::is_move_assignable_v<thrust::host_vector<T>>, "thrust::host_vector<T> must be movable");

    IndexCalculateType size = 1;
    for (unsigned i = 0; i < n_dims; ++i)
        size *= dimensions[i];
    assert(end - begin == size);
    if (end - begin != size)
        return {};

    switch (device) {
    case Location::Host:
        return { thrust::host_vector<T>(begin, end), dimensions };
    case Location::Device:
        return { thrust::device_vector<T>(begin, end), dimensions };
    case Location::Invalid:
        return {};
    }
    return {};
}

template <typename T, whack::size_t n_dims, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType>
Tensor<T, n_dims> make_tensor(Location device, std::initializer_list<T> data, const whack::Array<IndexStoreType, n_dims>& dimensions)
{
    return make_tensor<T, n_dims, IndexStoreType, IndexCalculateType>(device, data.begin(), data.end(), dimensions);
}

template <typename T, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename... DimensionTypes>
Tensor<T, sizeof...(DimensionTypes)> make_tensor(Location device, std::initializer_list<T> data, DimensionTypes... dim)
{
    using Dimensions = whack::Array<IndexStoreType, sizeof...(DimensionTypes)>;

    const auto dimensions = Dimensions { IndexStoreType(dim)... };
    return make_tensor<T, dimensions.size(), IndexStoreType, IndexCalculateType>(device, data.begin(), data.end(), dimensions);
}

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
    using Dimensions = whack::Array<IndexStoreType, sizeof...(DimensionTypes)>;

    const auto dimensions = Dimensions { IndexStoreType(dim)... };
    return make_tensor<T, dimensions.size(), IndexStoreType, IndexCalculateType>(device, dimensions);
}

template <typename T, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename... ParameterTypes>
Tensor<T, sizeof...(ParameterTypes)> make_host_tensor(ParameterTypes... params)
{
    return make_tensor<T, IndexStoreType, IndexCalculateType>(Location::Host, params...);
}

template <typename T, typename IndexStoreType = uint32_t, typename IndexCalculateType = IndexStoreType, typename... ParameterTypes>
Tensor<T, sizeof...(ParameterTypes)> make_device_tensor(ParameterTypes... params)
{
    return make_tensor<T, IndexStoreType, IndexCalculateType>(Location::Device, params...);
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
