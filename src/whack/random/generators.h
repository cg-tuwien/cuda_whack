#pragma once
#include <random>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <glm/glm.hpp>

#include "whack/macros.h"

namespace whack::random {

template <typename scalar_t, typename engine>
class DeviceGenerator {
    engine m_state;

public:
    __device__
    DeviceGenerator()
        = default;

    __device__
    DeviceGenerator(uint64_t seed, uint64_t sequence_nr)
    {
        curand_init(seed, sequence_nr, 0, &m_state);
    }

    __device__
        scalar_t
        normal()
    {
        return curand_normal(&m_state);
    }

    __device__
        glm::vec<2, scalar_t>
        normal2()
    {
        const auto r = curand_normal2(&m_state); // no specialisation for double, but not necessary atm.
        return glm::vec<2, scalar_t>(r.x, r.y);
    }

    __device__
        glm::vec<3, scalar_t>
        normal3()
    {
        const auto r = curand_normal2(&m_state); // no specialisation for double, but not necessary atm.
        const auto rz = curand_normal(&m_state);
        return glm::vec<3, scalar_t>(r.x, r.y, rz);
    }
};

template <typename scalar_t, typename Unused = void>
class HostGenerator {
    std::mt19937_64 m_engine;

public:
    HostGenerator(uint64_t seed = 0, uint64_t sequence_nr = 0)
        : m_engine((seed + 47616198) * 5687969629871 + sequence_nr)
    {
        // std::default_random_engine seeded with 0 or one results in the same random sequence
        // should be enough to just add one, but let's just shuffle a bit more for good measure.
    }
    scalar_t normal()
    {
        std::normal_distribution<scalar_t> normal_distribution;
        return normal_distribution(m_engine);
    }

    glm::vec<2, scalar_t> normal2()
    {
        std::normal_distribution<scalar_t> normal_distribution;
        return glm::vec<2, scalar_t>(normal_distribution(m_engine), normal_distribution(m_engine));
    }

    glm::vec<3, scalar_t> normal3()
    {
        std::normal_distribution<scalar_t> normal_distribution;
        return glm::vec<3, scalar_t>(normal_distribution(m_engine), normal_distribution(m_engine), normal_distribution(m_engine));
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
using KernelGenerator = FastGenerationDeviceGenerator;
// note: using large sequence numbers is very expensive with the fast generation type
using KernelGeneratorWithFastGeneration = FastGenerationDeviceGenerator;
using KernelGeneratorWithFastInit = FastInitDeviceGenerator;
#else
// nvcc host code trajectory
using KernelGenerator = HostGenerator<float>;
// note: using large sequence numbers is very expensive with the fast generation type
using KernelGeneratorWithFastGeneration = HostGenerator<float>;
using KernelGeneratorWithFastInit = HostGenerator<float>;
#endif
#else
// non-nvcc code trajectory
using KernelGenerator = HostGenerator<float>;
// note: using large sequence numbers is very expensive with the fast generation type
using KernelGeneratorWithFastGeneration = HostGenerator<float>;
using KernelGeneratorWithFastInit = HostGenerator<float>;
#endif

// compiler errors if moved into the RandomNumberGenerator class
template <typename scalar_t, int n_dims, typename Rng>
WHACK_DEVICES_INLINE glm::vec<n_dims, scalar_t> random_normal_vec(Rng* rng)
{
    if constexpr (n_dims == 2)
        return rng->normal2();
    else
        return rng->normal3();
}
}
