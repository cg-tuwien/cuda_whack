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

#include <cassert>
#include <cinttypes>
#include <cuda/std/utility>

#define WHACK_USE_OWN_ARRAY

#ifndef WHACK_USE_OWN_ARRAY
#include <cuda/std/array>
#endif

#include "macros.h"

namespace whack {

#ifndef WHACK_USE_OWN_ARRAY

using size_t = cuda::std::size_t;

// template <typename T, size_t N>
// using Array = cuda::std::array<T, N>;

template <typename T, size_t N>
struct Array : cuda::std::array<T, N> { };

#else
// array is now available in the cuda stl. and it's possible to replace it by undefining WHACK_USE_OWN_ARRAY

// unfortunately, at the time of writing, it doesn't compile yet in c++ files (https://github.com/NVIDIA/libcudacxx/issues/354)
// also, cudas version doesn't assert in operator[] (only in at() i think). I consider the bounds checks handy for debugging.

// you might want to benchmark your code with the cuda version, in my brief tests the performance was the same.

using size_t = uint32_t;

template <typename T, uint32_t N>
struct Array {
    T data[N];
    static_assert(N > 0, "an array of size 0 doesn't appear usefull and would break front and back functions.");
    static_assert(std::is_nothrow_default_constructible_v<T>);
    static_assert(std::is_nothrow_copy_constructible_v<T>);

    constexpr Array() noexcept
    {
        for (int i = 0; i < N; ++i) {
            data[i] = {};
        }
    }

    template <typename... Args,
        typename = std::enable_if_t<sizeof...(Args) == N && (std::is_nothrow_constructible_v<T, Args> && ...)>>
    constexpr Array(Args&&... args) noexcept
        : data { cuda::std::forward<Args>(args)... }
    {
    }

    explicit constexpr Array(const T& fill_value) noexcept
    {
        for (int i = 0; i < N; ++i) {
            data[i] = fill_value;
        }
    }

    WHACK_DEVICES_INLINE T& operator[](uint32_t i) noexcept
    {
        assert(i < N);
        return data[i];
    }
    WHACK_DEVICES_INLINE
    constexpr const T& operator[](uint32_t i) const noexcept
    {
        assert(i < N);
        return data[i];
    }
    WHACK_DEVICES_INLINE
    static constexpr uint32_t size() noexcept
    {
        return N;
    }

    WHACK_DEVICES_INLINE
    T& front() noexcept
    {
        return data[0];
    }
    WHACK_DEVICES_INLINE
    constexpr const T& front() const noexcept
    {
        return data[0];
    }

    WHACK_DEVICES_INLINE
    T& back() noexcept
    {
        return data[N - 1];
    }
    WHACK_DEVICES_INLINE
    constexpr const T& back() const noexcept
    {
        return data[N - 1];
    }

    WHACK_DEVICES_INLINE
    T* begin() noexcept
    {
        return data;
    }
    WHACK_DEVICES_INLINE
    constexpr const T* begin() const noexcept
    {
        return data;
    }
    WHACK_DEVICES_INLINE
    T* end() noexcept
    {
        return data + N;
    }
    WHACK_DEVICES_INLINE
    constexpr const T* end() const noexcept
    {
        return data + N;
    }
};

template <typename T, uint32_t N>
constexpr bool operator==(const Array<T, N>& a, const Array<T, N>& b) noexcept
{
    for (uint32_t i = 0; i < N; ++i) {
        if (a[i] != b[i])
            return false;
    }
    return true;
}
#endif

} // namespace whack
