/// \file RNG.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_RNG_H_
#define VECGEOM_BASE_RNG_H_

#include "VecGeom/base/Global.h"

#ifdef VECCORE_CUDA
#include <cuda.h>
#include <curand_kernel.h>
#else
#include <random>
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifdef VECCORE_CUDA
class RNG;

// Emulating static class member ..
namespace RNGvar {
extern VECCORE_ATT_DEVICE unsigned long gMaxInstance;
extern VECCORE_ATT_DEVICE RNG **gInstances;
}
#endif

/**
 * @brief Singleton random number generator.
 */
class RNG {

private:
#ifdef VECCORE_CUDA

#ifdef __CUDA_ARCH__
  curandState fState;
#else
// Using rand in C++03
#endif

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetUniform()
  {
#ifdef __CUDA_ARCH__
    return curand_uniform(&fState);
#else
    return (Precision)rand() / RAND_MAX;
#endif
  }

#else

  std::mt19937 rng;
  std::uniform_real_distribution<> uniform_dist;

  VECGEOM_FORCE_INLINE
  Precision GetUniform() { return uniform_dist(rng); }

#endif

public:
#ifdef VECCORE_CUDA
  // The state should really be 'thread' specific
  VECCORE_ATT_HOST_DEVICE
  RNG()
  {
#ifdef __CUDA_ARCH__
    curand_init(0 /*seed */, 0 /* subsequence */, 0 /* offset */, &fState);
#else
// using rand in C++03
#endif
  }
#else
  RNG() : rng(0), uniform_dist(0, 1) {}
#endif

public:
/**
 * Init thread specific singleton instance.
 */
#ifdef __CUDA_ARCH__
  VECCORE_ATT_DEVICE
  static void InitInstances(unsigned long nthreads)
  {
    unsigned int tid = (threadIdx.x + blockIdx.x * blockDim.x);

    if (tid == 0) {
      RNGvar::gMaxInstance = nthreads;
      RNGvar::gInstances   = new RNG *[nthreads];
    }
    __syncthreads();

    for (int i = tid; i < nthreads; i += blockDim.x * gridDim.x) {
      RNGvar::gInstances[i] = new RNG;
    }
  }
#endif

#ifndef VECCORE_CUDA
  void seed(unsigned long seed_val) { rng.seed(seed_val); }
#endif
  /**
   * Access singleton instance.
   */
  VECCORE_ATT_HOST_DEVICE
  static RNG &Instance()
  {
#ifdef __CUDA_ARCH__
    unsigned int tid = (threadIdx.x + blockIdx.x * blockDim.x);
    if (tid < RNGvar::gMaxInstance)
      return *(RNGvar::gInstances[tid]);
    else
      return *(new RNG);
#else
    static RNG instance;
    return instance;
#endif
  }

  /**
   * @return Uniformly distributed floating point number between 0 and 1 unless
   *         range arguments are passed.
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision uniform(const Precision min = 0., const Precision max = 1.) { return min + (max - min) * GetUniform(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int Poisson(const Precision lambda)
  {
    int k                  = 0;
    const Precision target = exp(-lambda);
    Precision p            = GetUniform();
    while (p < target) {
      p *= GetUniform();
      ++k;
    }
    return k;
  }

  // interface for ROOT compatibility
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision Gaus(Precision ave = 0.0, Precision sig = 1.0) { return Gauss(ave, sig); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision Gauss(Precision ave = 0.0, Precision sig = 1.0)
  {
    Precision x1, x2, w;

    do {
      x1 = 2.0 * GetUniform() - 1.0;
      x2 = 2.0 * GetUniform() - 1.0;
      w  = x1 * x1 + x2 * x2;
    } while (w >= 1.0);

    w = std::sqrt((-2.0 * std::log(w)) / w);
    return ave + (x1 * w * sig);
  }

  /**
   * Uniformly distributed array of floating point number between 0 and 1 unless
   *         range arguments are passed.
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void uniform_array(size_t n, Precision *array, const Precision min = 0., const Precision max = 1.)
  {
    for (size_t i = 0; i < n; ++i) {
      array[i] = min + (max - min) * GetUniform();
    }
  }

private:
  RNG(RNG const &);
  RNG &operator=(RNG const &);
};
}
} // End global namespace

#endif // VECGEOM_BASE_RNG_H_
