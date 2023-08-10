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

#include <cassert>
#include <cinttypes>

#include "macros.h"

namespace whack {

// array is now available in the cuda stl. so in theory we should be able to replace it here.
// but unfortunately, at the time of writing, it doesn't compile yet in c++ files (https://github.com/NVIDIA/libcudacxx/issues/354)

// once the bugfix is widely available, we should test for performance, as the cuda stl version uses size_t for the size, and it might be 64bits.
// in general this shouldn't be a problem as we should be using it only in constexpr context, but better check, maybe auto takes a 64bit size somewher.

// also, cudas version doesn't assert in operator[] (only in at() i think). I consider the bounds checks handy for debugging. so we might wrap cudas version
// and only use a typedef in release mode.

template <typename T, uint32_t N>
struct Array {
    T data[N];
    static_assert(N > 0, "an array of size 0 doesn't appear usefull and would break front and back functions.");

    WHACK_DEVICES_INLINE T& operator[](uint32_t i)
    {
        assert(i < N);
        return data[i];
    }
    WHACK_DEVICES_INLINE
    const T& operator[](uint32_t i) const
    {
        assert(i < N);
        return data[i];
    }
    WHACK_DEVICES_INLINE
    constexpr uint32_t size() const
    {
        return N;
    }

    WHACK_DEVICES_INLINE
    T& front()
    {
        return data[0];
    }
    WHACK_DEVICES_INLINE
    const T& front() const
    {
        return data[0];
    }

    WHACK_DEVICES_INLINE
    T& back()
    {
        return data[N - 1];
    }
    WHACK_DEVICES_INLINE
    const T& back() const
    {
        return data[N - 1];
    }

    WHACK_DEVICES_INLINE
    T* begin()
    {
        return data;
    }
    WHACK_DEVICES_INLINE
    const T* begin() const
    {
        return data;
    }
    WHACK_DEVICES_INLINE
    T* end()
    {
        return data + N;
    }
    WHACK_DEVICES_INLINE
    const T* end() const
    {
        return data + N;
    }
};

template <typename T, uint32_t N>
bool operator==(const Array<T, N>& a, const Array<T, N>& b)
{
    for (uint32_t i = 0; i < N; ++i) {
        if (a[i] != b[i])
            return false;
    }
    return true;
}

} // namespace whack
