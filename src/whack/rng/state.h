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

#include "whack/RandomNumberGenerator.h"
#include "whack/Tensor.h"
#include "whack/kernel.h"

namespace whack::rng {

// template <typename Functor>
inline Tensor<CpuRNG, 1> make_host_state(/*Functor seed_and_sequence, */ int)
{
    auto t = make_host_tensor<CpuRNG>(1);
    //    auto v = t.view();
    //    whack::start_parallel(
    //        t.device(), 1, 1, WHACK_KERNEL(=) {
    //            unsigned index = whack_threadIdx.x;
    //            uint64_t seed;
    //            uint64_t sequence_nr;
    //            thrust::tie(seed, sequence_nr) = seed_and_sequence(index);
    //            v(index) = CpuRNG(seed, sequence_nr);
    //        });
    return t;
}

}
