#ifndef VECGEOM_MATH_H
#define VECGEOM_MATH_H

#include <cmath>
#include <limits>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifdef VECGEOM_FLOAT_PRECISION
using Precision = float;
#else
using Precision = double;
#endif

using namespace vecCore::math;

#ifdef __CUDA_ARCH__
#define VECGEOM_CONST static __constant__ const
#else
#define VECGEOM_CONST static constexpr
#endif

VECGEOM_CONST Precision kAngTolerance = 1e-9;
VECGEOM_CONST Precision kAvogadro     = 6.02214085774e23;
VECGEOM_CONST Precision kEpsilon      = std::numeric_limits<Precision>::epsilon();
// VECGEOM_CONST Precision kInfinity         = std::numeric_limits<Precision>::infinity();
// a special constant to indicate a "miss" length
VECGEOM_CONST Precision kInfLength        = std::numeric_limits<Precision>::max();
VECGEOM_CONST Precision kMaximum          = std::numeric_limits<Precision>::max();
VECGEOM_CONST Precision kMinimum          = std::numeric_limits<Precision>::min();
VECGEOM_CONST Precision kPi               = 3.14159265358979323846;
VECGEOM_CONST Precision kHalfPi           = 0.5 * kPi;
VECGEOM_CONST Precision kTwoPi            = 2. * kPi;
VECGEOM_CONST Precision kTwoPiInv         = 1. / kTwoPi;
VECGEOM_CONST Precision kDegToRad         = kPi / 180.;
VECGEOM_CONST Precision kRadToDeg         = 180. / kPi;
VECGEOM_CONST Precision kRadTolerance     = 1e-9;
VECGEOM_CONST Precision kTiny             = 1e-30;
VECGEOM_CONST Precision kTolerance        = 1e-9;
VECGEOM_CONST Precision kHalfTolerance    = 0.5 * kTolerance;
VECGEOM_CONST Precision kToleranceSquared = kTolerance * kTolerance;

template <typename T>
struct Tiny {
  static constexpr T kValue = 1.e-30;
};

template <template <typename, typename> class ImplementationType, typename T, typename Q>
struct Tiny<ImplementationType<T, Q>> {
  static constexpr typename ImplementationType<T, Q>::value_type kValue = 1.e-30;
};
#undef VECGEOM_CONST
}
}

#endif
