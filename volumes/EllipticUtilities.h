/// @file EllipticUtilities.h
/// @author Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

#ifndef VECGEOM_ELLIPTICUTILITIES_H_
#define VECGEOM_ELLIPTICUTILITIES_H_

#include "base/Global.h"
#include "base/RNG.h"
#include "base/Vector2D.h"
#include "base/Vector3D.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

namespace EllipticUtilities {

// Compute Complete Elliptical Integral of the Second Kind
//
// The algorithm is based upon Carlson B.C., "Computation of real
// or complex elliptic integrals", Numerical Algorithms,
// Volume 10, Issue 1, 1995 (see equations 2.36 - 2.39)
//
VECCORE_ATT_HOST_DEVICE
VECGEOM_FORCE_INLINE
Precision comp_ellint_2(Precision e)
{
  const Precision eps = 1. / 134217728.; // 1 / 2^27

  Precision a = 1.;
  Precision b = vecCore::math::Sqrt((1. - e) * (1. + e));
  if (b == 1.) return kHalfPi;
  if (b == 0.) return 1.;

  Precision x = 1.;
  Precision y = b;
  Precision S = 0.;
  Precision M = 1.;
  while (x - y > eps * y) {
    Precision tmp = (x + y) * 0.5;

    y = vecCore::math::Sqrt(x * y);
    x = tmp;
    M += M;
    S += M * (x - y) * (x - y);
  }
  return 0.5 * kHalfPi * ((a + b) * (a + b) - S) / (x + y);
}

// Compute the perimeter of an ellipse (X/A)^2 + (Y/B)^2 = 1
//
VECCORE_ATT_HOST_DEVICE
VECGEOM_FORCE_INLINE
Precision EllipsePerimeter(Precision A, Precision B)
{
  Precision dx  = vecCore::math::Abs(A);
  Precision dy  = vecCore::math::Abs(B);
  Precision a   = vecCore::math::Max(dx, dy);
  Precision b   = vecCore::math::Min(dx, dy);
  Precision b_a = b / a;
  Precision e   = vecCore::math::Sqrt((1. - b_a) * (1. + b_a));
  return 4. * a * comp_ellint_2(e);
}

// Pick a random point in ellipse (x/a)^2 + (y/b)^2 = 1
// (rejection sampling)
//
VECCORE_ATT_HOST_DEVICE
VECGEOM_FORCE_INLINE
Vector2D<Precision> RandomPointInEllipse(Precision a, Precision b)
{
  Precision aa = (a * a == 0.) ? 0. : 1. / (a * a);
  Precision bb = (b * b == 0.) ? 0. : 1. / (b * b);
  for (int i = 0; i < 1000; ++i) {
    Precision x = a * (2. * RNG::Instance().uniform() - 1.);
    Precision y = b * (2. * RNG::Instance().uniform() - 1.);
    if (x * x * aa + y * y * bb <= 1.) return Vector2D<Precision>(x, y);
  }
  return Vector2D<Precision>(0., 0.);
}

// Pick a random point on ellipse (x/a)^2 + (y/b)^2 = 1
// (rejection sampling)
//
VECCORE_ATT_HOST_DEVICE
VECGEOM_FORCE_INLINE
Vector2D<Precision> RandomPointOnEllipse(Precision a, Precision b)
{
  Precision A      = vecCore::math::Abs(a);
  Precision B      = vecCore::math::Abs(b);
  Precision mu_max = vecCore::math::Max(A, B);

  Precision x, y;
  for (int i = 0; i < 1000; ++i) {
    Precision phi = kTwoPi * RNG::Instance().uniform();
    x             = vecCore::math::Cos(phi);
    y             = vecCore::math::Sin(phi);
    Precision Bx  = B * x;
    Precision Ay  = A * y;
    Precision mu  = vecCore::math::Sqrt(Bx * Bx + Ay * Ay);
    if (mu_max * RNG::Instance().uniform() <= mu) break;
  }
  return Vector2D<Precision>(A * x, B * y);
}

} /* namespace EllipticUtilities */
} /* namespace VECGEOM_IMPL_NAMESPACE */
} /* namespace vecgeom */

#endif /* VECGEOM_ELLIPTICUTILITIES_H_ */
