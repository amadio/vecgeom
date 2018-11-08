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
  os << "   plane normal: " << hpl.fNorm << "  distance = " << hpl.fDist;
  return os;
}
std::ostream &operator<<(std::ostream &os, Polygon const &poly)
{
  os << "   rectangle (";
  for (size_t i = 0; i < poly.fN; ++i)
    os << i << ":" << poly.fVert[i] << "  ";
  os << "  normal: " << poly.fNorm << "  distance = " << poly.fDist;
  return os;
}
#endif

void TransformPlane(Transformation3D const &tr, HPlane const &localp, HPlane &masterp)
{
  // Transform niormal vector
  tr.InverseTransformDirection(localp.fNorm, masterp.fNorm);
  masterp.fDist = localp.fDist - masterp.fNorm.Dot(tr.Translation());
}

Polygon::Polygon(size_t n, bool convex) : fN(n), fConvex(convex)
{
  assert(fN > 2);
  fVert.reserve(fN);
  fSides.reserve(fN);
  for (size_t i = 0; i < fN; ++i) {
    fVert.push_back(Vec_t());
    fSides.push_back(Vec_t());
  }
}

Polygon::Polygon(size_t n, double normx, double normy, double normz, bool is_normalized, bool is_convex)
    : fN(n), fConvex(is_convex), fHasNorm(true)
{
  assert(fN > 2);
  fVert.reserve(fN);
  fSides.reserve(fN);
  for (size_t i = 0; i < fN; ++i) {
    fVert.push_back(Vec_t());
    fSides.push_back(Vec_t());
  }
  fNorm.Set(normx, normy, normz);
  if (!is_normalized) fNorm.Normalize();
}

void Polygon::Init()
{
  // Compute sides
  for (size_t i = 0; i < fN - 1; ++i) {
    fSides[i] = fVert[i + 1] - fVert[i];
    assert(fSides[i].Mag2() > kTolerance);
  }
  fSides[fN - 1] = fVert[0] - fVert[fN - 2];
  assert(fSides[fN - 1].Mag2() > kTolerance);
  // Compute normal if not already set
  if (!fHasNorm) {
    fNorm = fSides[0].Cross(fSides[1]);
    fNorm.Normalize();
  } else {
    if ((fSides[0].Cross(fSides[1])).Dot(fNorm) < 0) {
      // We need to invert the vertices and sides
      size_t i = 1;
      while (i < fN - i) {
        Vector3D<double> temp = fVert[i];
        fVert[i]              = fVert[fN - i];
        fVert[fN - i]         = temp;
      }
      for (i = 0; i < fN - 1; ++i)
        fSides[i] = fVert[i + 1] - fVert[i];
      fSides[fN - 1] = fVert[0] - fVert[fN - 2];
    }
  }
  // Compute convexity if not supplied
  if (!fConvex) {
    fConvex = true;
    for (size_t i = 0; i < fN; ++i) {
      for (size_t k = 0; k < fN - 2; ++k) {
        size_t j = (i + k + 2) % fN; // remaining vertices
        if ((fSides[i].Cross(fVert[j] - fVert[i])).Dot(fNorm) < 0) {
          fConvex = false;
          break;
        }
      }
    }
  }

  // Compute distance to origin
  fDist = -fNorm.Dot(fVert[0]);
}

void Polygon::Transform(Transformation3D const &tr)
{
  for (size_t i = 0; i < fN; ++i) {
    Vector3D<double> temp;
    tr.InverseTransform(fVert[i], temp);
    fVert[i] = temp;
  }
  if (fHasNorm) {
    Vector3D<double> temp;
    tr.InverseTransformDirection(fNorm, temp);
    fNorm = temp;
  }
  Init();
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

EBodyXing_t PolygonXing(Polygon const &poly1, Polygon const &poly2, Line *line)
{
  using Vec_t = Vector3D<double>;
  using vecCore::math::CopySign;
  using vecCore::math::Max;
  using vecCore::math::Min;
  using vecCore::math::Sqrt;

  Vec_t point, direction;
  EPlaneXing_t crossing = PlaneXing(poly1.fNorm, poly1.fDist, poly2.fNorm, poly2.fDist, point, direction);
  if (crossing == kParallel) return kDisjoint;

  if (crossing == kIdentical) {
    // We use the separate axis theorem
    // loop segments of 1
    for (size_t i = 0; i < poly1.fN; ++i) {
      // loop vertices of 2
      bool outside = false;
      for (size_t j = 0; j < poly2.fN; ++j) {
        outside = poly1.fNorm.Dot((poly2.fVert[j] - poly1.fVert[i]).Cross(poly1.fSides[i])) > 0;
        if (!outside) break;
      }
      if (outside) return kDisjoint;
    }
    // loop segments of 2
    for (size_t i = 0; i < poly2.fN; ++i) {
      // loop vertices of 1
      bool outside = false;
      for (size_t j = 0; j < poly1.fN; ++j) {
        outside = poly2.fNorm.Dot((poly1.fVert[j] - poly2.fVert[i]).Cross(poly2.fSides[i])) > 0;
        if (!outside) break;
      }
      if (outside) return kDisjoint;
    }
    return kTouching;
  }

  // The polygons do cross each other along a line
  if (!(poly1.fConvex | poly2.fConvex)) return kDisjoint; // cannot solve yet non-convex case
  double smin1 = InfinityLength<double>();
  double smax1 = -InfinityLength<double>();
  std::cout << "IP: " << point << "  direction: " << direction << std::endl;
  std::cout << "Checking poly1\n";
  for (size_t i = 0; i < poly1.fN; ++i) {
    std::cout << "segment " << i << ": (" << poly1.fVert[i] << ", " << poly1.fVert[(i + 1) % poly1.fN] << ")\n";
    Vec_t crossdirs      = poly1.fSides[i].Cross(direction);
    double mag2crossdirs = crossdirs.Mag2();
    if (mag2crossdirs < kTolerance) continue; // Crossing line parallel to edge
    Vec_t crosspts = (point - poly1.fVert[i]).Cross(direction);
    if (crossdirs.Dot(crosspts) < 0) {
      std::cout << " miss (before)\n";
      continue;
    } // crossing line hits edge on lower prolongation
    if (crosspts.Mag2() < mag2crossdirs + kTolerance) {
      // Crossing line hits the current edge
      crosspts      = (point - poly1.fVert[i]).Cross(poly1.fSides[i]);
      double distsq = CopySign(crosspts.Mag2() / mag2crossdirs, crosspts.Dot(crossdirs));
      std::cout << "  dist = " << CopySign(Sqrt(crosspts.Mag2() / mag2crossdirs), distsq) << std::endl;
      smin1 = Min(smin1, distsq);
      smax1 = Max(smax1, distsq);
    } else {
      std::cout << " miss (after)\n";
    }
  }
  std::cout << "smin1 = " << smin1 << "  smax1 = " << smax1 << std::endl;
  if (smax1 <= smin1) return kDisjoint;

  double smin2 = InfinityLength<double>();
  double smax2 = -InfinityLength<double>();
  std::cout << "Checking poly2\n";
  for (size_t i = 0; i < poly2.fN; ++i) {
    std::cout << "segment " << i << ": (" << poly2.fVert[i] << ", " << poly2.fVert[(i + 1) % poly2.fN] << ")\n";
    Vec_t crossdirs      = poly2.fSides[i].Cross(direction);
    double mag2crossdirs = crossdirs.Mag2();
    if (mag2crossdirs < kTolerance) continue; // Crossing line parallel to edge
    Vec_t crosspts = (point - poly2.fVert[i]).Cross(direction);
    if (crossdirs.Dot(crosspts) < 0) {
      std::cout << " miss (before)\n";
      continue;
    } // crossing line hits edge on lower prolongation
    if (crosspts.Mag2() < mag2crossdirs + kTolerance) {
      // Crossing line hits the current edge
      crosspts      = (point - poly2.fVert[i]).Cross(poly2.fSides[i]);
      double distsq = CopySign(crosspts.Mag2() / mag2crossdirs, crosspts.Dot(crossdirs));
      std::cout << "  dist = " << CopySign(Sqrt(crosspts.Mag2() / mag2crossdirs), distsq) << std::endl;
      smin2 = Min(smin2, distsq);
      smax2 = Max(smax2, distsq);
    } else {
      std::cout << " miss (after)\n";
    }
  }
  std::cout << "smin2 = " << smin2 << "  smax2 = " << smax2 << std::endl;
  if (smax2 <= smin2) return kDisjoint;
  if (smin2 - smax1 > -kTolerance || smin1 - smax2 > -kTolerance) return kDisjoint;
  if (line != nullptr) {
    std::cout << "point: " << point << "  direction: " << direction << std::endl;
    std::cout << " smin1 = " << smin1 << "  smax1 = " << smax1 << " smin2 = " << smin2 << "  smax2 = " << smax2
              << std::endl;
    double dmin = Max(smin1, smin2);
    double dmax = Min(smax1, smax2);
    std::cout << "dmin = " << dmin << "  dmax = " << dmax << std::endl;
    assert(dmax - dmin > -kTolerance);
    line->fPts[0] = point + direction * CopySign(Sqrt(Abs(dmin)), dmin);
    line->fPts[1] = point + direction * CopySign(Sqrt(Abs(dmax)), dmax);
  }
  return kOverlapping;
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
