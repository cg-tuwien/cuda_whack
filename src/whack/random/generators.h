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
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <glm/glm.hpp>
#include <random>

#include "whack/macros.h"

namespace whack::random {

template <typename Scalar, typename engine>
class DeviceGenerator {
    engine m_state;

public:
    __device__
    DeviceGenerator() noexcept
        = default;

    __device__
    DeviceGenerator(uint64_t seed, uint64_t sequence_nr) noexcept
    {
        curand_init(seed, sequence_nr, 0, &m_state);
    }

    __device__ Scalar normal() noexcept
    {
        return curand_normal(&m_state);
    }

    __device__ glm::vec<2, Scalar> normal2() noexcept
    {
        const auto r = curand_normal2(&m_state); // no specialisation for double, but not necessary atm.
        return glm::vec<2, Scalar>(r.x, r.y);
    }

    __device__ glm::vec<3, Scalar> normal3() noexcept
    {
        const auto r = curand_normal2(&m_state); // no specialisation for double, but not necessary atm.
        const auto rz = curand_normal(&m_state);
        return glm::vec<3, Scalar>(r.x, r.y, rz);
    }

    __device__ Scalar uniform() noexcept
    {
        return curand_uniform(&m_state);
    }
};

template <typename Scalar, typename Unused = void>
class HostGenerator {
    std::mt19937_64 m_engine;

public:
    HostGenerator(uint64_t seed = 0, uint64_t sequence_nr = 0)
        : m_engine((seed + 47616198) * 5687969629871 + sequence_nr)
    {
        // std::default_random_engine seeded with 0 or one results in the same random sequence
        // should be enough to just add one, but let's just shuffle a bit more for good measure.
    }

    Scalar normal()
    {
        std::normal_distribution<Scalar> normal_distribution;
        return normal_distribution(m_engine);
    }

    glm::vec<2, Scalar> normal2()
    {
        std::normal_distribution<Scalar> normal_distribution;
        return glm::vec<2, Scalar>(normal_distribution(m_engine), normal_distribution(m_engine));
    }

    glm::vec<3, Scalar> normal3()
    {
        std::normal_distribution<Scalar> normal_distribution;
        return glm::vec<3, Scalar>(normal_distribution(m_engine), normal_distribution(m_engine), normal_distribution(m_engine));
    }

    Scalar uniform()
    {
        std::uniform_real_distribution<Scalar> uniform_distribution(0.0, 1.0);
        return uniform_distribution(m_engine);
    }

    glm::vec<2, Scalar> uniform2()
    {
        std::uniform_real_distribution<Scalar> uniform_distribution(0.0, 1.0);
        return { uniform_distribution(m_engine), uniform_distribution(m_engine) };
    }

    glm::vec<3, Scalar> uniform3()
    {
        std::uniform_real_distribution<Scalar> uniform_distribution(0.0, 1.0);
        return {uniform_distribution(m_engine), uniform_distribution(m_engine), uniform_distribution(m_engine)};
    }
};

// note: using large sequence numbers is very expensive with the fast generation type
using FastGenerationDeviceGenerator = DeviceGenerator<float, curandStateXORWOW_t>;
using FastInitDeviceGenerator = DeviceGenerator<float, curandStatePhilox4_32_10_t>;

// KernelGenerator* must be used only from within the kernel (otherwise the cpu type will be used, that is, don't forward it as a template parameter or similar to the kernel)!
#ifdef __CUDACC__
// warning using nvcc
#ifdef __CUDA_ARCH__
// device code trajectory
#define DEPENDENT_WHACK_NOEXCEPT noexcept
using KernelGenerator = FastGenerationDeviceGenerator;
// note: using large sequence numbers is very expensive with the fast generation type
using KernelGeneratorWithFastGeneration = FastGenerationDeviceGenerator;
using KernelGeneratorWithFastInit = FastInitDeviceGenerator;
#else
#define DEPENDENT_WHACK_NOEXCEPT
// nvcc host code trajectory
using KernelGenerator = HostGenerator<float>;
// note: using large sequence numbers is very expensive with the fast generation type
using KernelGeneratorWithFastGeneration = HostGenerator<float>;
using KernelGeneratorWithFastInit = HostGenerator<float>;
#endif
#else
#define DEPENDENT_WHACK_NOEXCEPT
// non-nvcc code trajectory
using KernelGenerator = HostGenerator<float>;
// note: using large sequence numbers is very expensive with the fast generation type
using KernelGeneratorWithFastGeneration = HostGenerator<float>;
using KernelGeneratorWithFastInit = HostGenerator<float>;
#endif

// compiler errors if moved into the RandomNumberGenerator class
template <typename Scalar, int n_dims, typename Rng>
WHACK_DEVICES_INLINE glm::vec<n_dims, Scalar> random_normal_vec(Rng* rng) DEPENDENT_WHACK_NOEXCEPT
{
    if constexpr (n_dims == 2)
        return rng->normal2();
    else
        return rng->normal3();
}
} // namespace whack::random

#undef DEPENDENT_WHACK_NOEXCEPT
