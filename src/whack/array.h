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
template <typename T, uint32_t N>
struct Array {
    T data[N];
    static_assert(N > 0, "an array of size 0 doesn't appear usefull and would break front and back functions.");

    WHACK_DEVICES_INLINE
    T& operator[](uint32_t i)
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
} // namespace whack
