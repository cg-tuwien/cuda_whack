/*
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

#include "array.h"

namespace whack {

template <typename large_type, unsigned n_dims, typename small_type = large_type>
WHACK_DEVICES_INLINE
    Array<small_type, n_dims>
    split_n_dim_index(const Array<small_type, n_dims>& dimensions, large_type idx)
{
    Array<small_type, n_dims> tmp;
    tmp.front() = 1;
    for (unsigned i = 1; i < n_dims; ++i) {
        tmp[i] = dimensions[i - 1] * tmp[i - 1];
    }

    for (unsigned i = n_dims - 1; i < n_dims; --i) {
        const auto tmp_idx = idx / tmp[i];
        idx -= tmp_idx * tmp[i];
        tmp[i] = tmp_idx;
    }
    return tmp;
}

template <typename large_type, unsigned n_dims, typename small_type = large_type>
WHACK_DEVICES_INLINE
    large_type
    join_n_dim_index(const Array<small_type, n_dims>& dimensions, const Array<small_type, n_dims>& idx)
{
    large_type joined_idx = 0;
    large_type cum_dims = 1;
    for (unsigned i = 0; i < n_dims; ++i) {
        joined_idx += idx[i] * cum_dims;
        cum_dims *= dimensions[i];
    }
    return joined_idx;
}
}
