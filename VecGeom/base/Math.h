#ifndef VECGEOM_MATH_H
#define VECGEOM_MATH_H

#include <cmath>
#include <limits>
#include "VecCore/Limits.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifdef __CUDA_ARCH__
#define VECGEOM_CONST static __constant__ const
#else
#define VECGEOM_CONST static constexpr
#endif

#ifdef VECGEOM_FLOAT_PRECISION
using Precision                        = float;
VECGEOM_CONST Precision kTolerance     = 1e-3;
VECGEOM_CONST Precision kPushTolerance = 1e-3;
VECGEOM_CONST Precision kSqrtTolerance = 3.1622777e-2;
VECGEOM_CONST Precision kAngTolerance  = 1e-2;
VECGEOM_CONST Precision kConeTolerance = 1e-3;
VECGEOM_CONST Precision kFarAway       = 1e5;
#else
using Precision                        = double;
VECGEOM_CONST Precision kTolerance     = 1e-9;
VECGEOM_CONST Precision kPushTolerance = 1e-6;
VECGEOM_CONST Precision kSqrtTolerance = 3.1622777e-5;
VECGEOM_CONST Precision kAngTolerance  = 1e-9;
VECGEOM_CONST Precision kConeTolerance = 1e-7;
VECGEOM_CONST Precision kFarAway       = 1e10;
#endif

using namespace vecCore::math;

VECGEOM_CONST Precision kAvogadro = 6.02214085774e23;
VECGEOM_CONST Precision kEpsilon  = std::numeric_limits<Precision>::epsilon();
// VECGEOM_CONST Precision kInfinity         = std::numeric_limits<Precision>::infinity();
// a special constant to indicate a "miss" length
VECGEOM_CONST Precision kInfLength         = vecCore::NumericLimits<Precision>::Max();
VECGEOM_CONST Precision kMaximum           = vecCore::NumericLimits<Precision>::Max();
VECGEOM_CONST Precision kMinimum           = vecCore::NumericLimits<Precision>::Min();
VECGEOM_CONST Precision kPi                = 3.14159265358979323846;
VECGEOM_CONST Precision kHalfPi            = 0.5 * kPi;
VECGEOM_CONST Precision kTwoPi             = 2. * kPi;
VECGEOM_CONST Precision kTwoPiInv          = 1. / kTwoPi;
VECGEOM_CONST Precision kDegToRad          = kPi / 180.;
VECGEOM_CONST Precision kRadToDeg          = 180. / kPi;
VECGEOM_CONST Precision kRadTolerance      = 1e-9;
VECGEOM_CONST Precision kTiny              = 1e-30;
VECGEOM_CONST Precision kHalfTolerance     = 0.5 * kTolerance;
VECGEOM_CONST Precision kHalfConeTolerance = 0.5 * kConeTolerance;
VECGEOM_CONST Precision kToleranceSquared  = kTolerance * kTolerance;

template <typename T>
struct Tiny {
  static constexpr T kValue = 1.e-30;
};

template <template <typename, typename> class ImplementationType, typename T, typename Q>
struct Tiny<ImplementationType<T, Q>> {
  static constexpr typename ImplementationType<T, Q>::value_type kValue = 1.e-30;
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
