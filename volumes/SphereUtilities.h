#ifndef VECGEOM_VOLUMES_SPHEREUTILITIES_H_
#define VECGEOM_VOLUMES_SPHEREUTILITIES_H_

#include "base/Global.h"

#ifndef VECCORE_CUDA
#include "base/RNG.h"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename T>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
T sqr(T x)
{
  return x * x;
}

#ifndef VECCORE_CUDA
// Generate radius in annular ring according to uniform area
template <typename T>
VECGEOM_FORCE_INLINE
T GetRadiusInRing(T rmin, T rmax)
{
  if (rmin == rmax) return rmin;

  T rng(RNG::Instance().uniform(0.0, 1.0));

  if (rmin <= T(0.0)) return rmax * Sqrt(rng);

  T rmin2 = rmin * rmin;
  T rmax2 = rmax * rmax;

  return Sqrt(rng * (rmax2 - rmin2) + rmin2);
}
#endif
}
} // End global namespace

#endif // VECGEOM_VOLUMES_SPHEREUTILITIES_H_
