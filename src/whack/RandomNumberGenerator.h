#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "indexing.h"
#include "macros.h"

#ifdef __CUDACC__
#include <curand_kernel.h>
#endif

#if defined(__CUDA_ARCH__)

namespace whack {

template <typename scalar_t, typename engine>
class RandomNumberGenerator {
    engine m_state;

public:
    __device__
    RandomNumberGenerator(uint64_t seed, uint64_t sequence_nr)
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

typedef RandomNumberGenerator<float, curandStateXORWOW_t> RNGFastGeneration;
typedef RandomNumberGenerator<float, curandStatePhilox4_32_10_t> RNGFastOffset;

}
#else
#include <random>

namespace whack {

template <typename scalar_t, typename engine>
class RandomNumberGenerator {
    engine m_engine;

public:
    RandomNumberGenerator(uint64_t seed, uint64_t sequence_nr)
        : m_engine(seed + sequence_nr)
    {
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

typedef RandomNumberGenerator<float, std::default_random_engine> RNGFastGeneration;
typedef RandomNumberGenerator<float, std::default_random_engine> RNGFastOffset;

}

#endif

// namespace whack {
//// compiler errors if moved into the RandomNumberGenerator class
// template <typename scalar_t, int n_dims>
// WHACK_DEVICES_INLINE glm::vec<n_dims, scalar_t> random_normal_vec(RandomNumberGenerator<scalar_t>* rng)
//{
//     if constexpr (n_dims == 2)
//         return rng->normal2();
//     else
//         return rng->normal3();
// }
// }
