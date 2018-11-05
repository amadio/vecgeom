///
/// \file Utils3D.cpp
/// \author Andrei Gheata (andrei.gheata@cern.ch)

#include "base/Utils3D.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

namespace Utils3D {

using vecCore::math::Abs;
using vecCore::math::Max;

#ifndef VECCORE_CUDA
std::ostream &operator<<(std::ostream &os, HPlane const &hpl)
{
  os << "   plane normal: " << hpl.fNorm << "  distance = " << hpl.fDist << std::endl;
  return os;
}
std::ostream &operator<<(std::ostream &os, HRectangle const &hrect)
{
  os << "   rectangle (" << hrect.fDx << ", " << hrect.fDy << ") center: " << hrect.fCenter
     << " normal: " << hrect.fNorm << " up vector: " << hrect.fUpVect << " distance = " << hrect.fDist << std::endl;
  return os;
}
#endif

void TransformPlane(Transformation3D const &tr, HPlane const &localp, HPlane &masterp)
{
  // Transform niormal vector
  tr.InverseTransformDirection(localp.fNorm, masterp.fNorm);
  masterp.fDist = localp.fDist - masterp.fNorm.Dot(tr.Translation());
}

void TransformRectangle(Transformation3D const &tr, HRectangle const &local, HRectangle &master)
{
  // Transform niormal vector
  master.fDx = local.fDx;
  master.fDy = local.fDy;
  tr.InverseTransformDirection(local.fNorm, master.fNorm);
  tr.InverseTransformDirection(local.fUpVect, master.fUpVect);
  tr.InverseTransform(local.fCenter, master.fCenter);
  master.fDist = local.fDist - master.fNorm.Dot(tr.Translation());
}

EPlaneXing_t PlaneXing(Vector3D<double> const &n1, double p1, Vector3D<double> const &n2, double p2,
                       Vector3D<double> &point, Vector3D<double> &direction)
{
  direction        = n1.Cross(n2);
  const double det = direction.Mag2();
  if (Abs(det) < kTolerance) {
    // The 2 planes are parallel, let's find the distance between them
    const double d12 = Abs(Abs(p1) - Abs(p2));
    if (d12 < kTolerance) {
      if (p1 * p2 * n1.Dot(n2) > 0.) return kIdentical;
    }
    return kParallel;
  }
  // The planes do intersect
  point = ((p1 * direction.Cross(n2) - p2 * direction.Cross(n1))) / det;
  direction.Normalize();
  return kIntersecting;
}

EBodyXing_t RectangleXing(HRectangle const &rect1, HRectangle const &rect2)
{
  using Vec_t = Vector3D<double>;
  Vec_t point, direction;
  EPlaneXing_t crossing = PlaneXing(rect1.fNorm, rect1.fDist, rect2.fNorm, rect2.fDist, point, direction);
  if (crossing == kParallel) return kDisjoint;
  if (crossing == kIdentical) {
    // We use the separate axis theorem
    Vec_t side1 = rect1.fNorm.Cross(rect1.fUpVect);
    Vec_t side2 = rect2.fNorm.Cross(rect2.fUpVect);
    Vec_t vertices1[4], vertices2[4], sides1[4], sides2[4];

    sides1[0] = 2. * rect1.fDx * side1;
    sides1[1] = - 2. * rect1.fDy * rect1.fUpVect;
    sides1[2] = - 2. * rect1.fDx * side1;
    sides1[3] = 2. * rect1.fDy * rect1.fUpVect;

    vertices1[0] = rect1.fCenter + rect1.fDy * rect1.fUpVect - rect1.fDx * side1;
    vertices1[1] = rect1.fCenter + rect1.fDy * rect1.fUpVect + rect1.fDx * side1;
    vertices1[2] = rect1.fCenter - rect1.fDy * rect1.fUpVect + rect1.fDx * side1;
    vertices1[3] = rect1.fCenter - rect1.fDy * rect1.fUpVect - rect1.fDx * side1;

    sides2[0] = 2. * rect2.fDx * side2;
    sides2[1] = - 2. * rect2.fDy * rect2.fUpVect;
    sides2[2] = - 2. * rect2.fDx * side2;
    sides2[3] = 2. * rect2.fDy * rect2.fUpVect;

    vertices2[0] = rect2.fCenter + rect2.fDy * rect2.fUpVect - rect2.fDx * side2;
    vertices2[1] = rect2.fCenter + rect2.fDy * rect2.fUpVect + rect2.fDx * side2;
    vertices2[2] = rect2.fCenter - rect2.fDy * rect2.fUpVect + rect2.fDx * side2;
    vertices2[3] = rect2.fCenter - rect2.fDy * rect2.fUpVect - rect2.fDx * side2;

    // loop segments of 1
    for (int i = 0; i < 4; ++i) {
      // loop vertices of 2
      bool outside = false;
      for (int j = 0; j < 4; ++j) {
        outside = rect1.fNorm.Dot((vertices2[j] - vertices1[i]).Cross(sides1[i])) > 0;
        if (!outside) break;
      }
      if (outside) return kDisjoint;
      // loop vertices of 1
      for (int j = 0; j < 4; ++j) {
        outside = rect2.fNorm.Dot((vertices1[j] - vertices2[i]).Cross(sides2[i])) > 0;
        if (!outside) break;
      }
      if (outside) return kDisjoint;
    }
    return kTouching;
  }
  // The rectangles do cross each other along a line

  return kDisjoint;
}

EBodyXing_t BoxXing(Vector3D<double> const &box1, Transformation3D const &tr1, Vector3D<double> const &box2,
                    Transformation3D const &tr2)
{
  // A fast check if the bounding spheres touch
  Vector3D<double> orig1 = tr1.Translation();
  Vector3D<double> orig2 = tr2.Translation();
  double r1sq            = box1.Mag2();
  double r2sq            = box2.Mag2();
  double dsq             = (orig2 - orig1).Mag2();
  if (dsq > r1sq + r2sq + 2. * Sqrt(r1sq * r2sq)) return kDisjoint;

  if (!tr1.HasRotation() && !tr2.HasRotation()) {
    // Aligned boxes case
    Vector3D<double> eps1 = (orig2 - box2) - (orig1 + box1);
    Vector3D<double> eps2 = (orig1 - box1) - (orig2 + box2);
    double deps           = Max(eps1.Max(), eps2.Max());
    if (deps > kTolerance)
      return kDisjoint;
    else if (deps > -kTolerance)
      return kTouching;
    return kOverlapping;
  }
  // General case
  // compute matrix to go from 2 to 1
  Transformation3D tr12;
  tr1.Inverse(tr12);
  tr12.MultiplyFromRight(tr2);

  return kDisjoint;
}

} // namespace Utils3D
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
