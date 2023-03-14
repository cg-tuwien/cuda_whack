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
 */

#pragma once
#include <any>

#include "tensor_view.h"

namespace whack {
enum class ComputeDevice {
    CPU,
    CUDA,
    Invalid
};

template <typename T, uint32_t n_dims, typename IndexType = uint32_t, typename DimensionType = IndexType>
struct Tensor {
    std::any memory;
    TensorView<T, n_dims, IndexType, DimensionType> view;
    ComputeDevice device = ComputeDevice::Invalid;
};
}
