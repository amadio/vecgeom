/// \file Rectangles.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_RECTANGLES_H_
#define VECGEOM_VOLUMES_RECTANGLES_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/Planes.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class Rectangles;);
VECGEOM_DEVICE_DECLARE_CONV(class, Rectangles);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// \class Rectangles
///
/// \brief Stores a number of rectangles in SOA form to allow vectorized
///        operations.
///
/// To allow efficient computation, two corners, one normalized side and
/// the plane equation in which the rectangle lies is stored.
/// If the set of rectangles are assumed to be convex, the convex methods
/// can be called for faster computation, falling back on the implementations
/// for planes.
class Rectangles : public AlignedBase {

  //   fCorners[0]
  //             p0----------
  //       |      |         |
  // (Normalized) |         |
  //    fSides    |   -o-   |
  //       |      |         |
  //       v      |         |
  //             p1---------p2
  //                          fCorners[1]

private:
  Planes fPlanes;
  SOA3D<Precision> fSides;
  SOA3D<Precision> fCorners[2];

public:
  typedef SOA3D<Precision> Corners_t[2];

  VECCORE_ATT_HOST_DEVICE
  Rectangles(int size);

  VECCORE_ATT_HOST_DEVICE
  ~Rectangles();

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int size() const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<Precision> GetNormal(int i) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  SOA3D<Precision> const &GetNormals() const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDistance(int i) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Array<Precision> const &GetDistances() const;

  VECCORE_ATT_HOST_DEVICE
  inline Vector3D<Precision> GetCenter(int i) const;

  VECCORE_ATT_HOST_DEVICE
  inline Vector3D<Precision> GetCorner(int i, int j) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Corners_t const &GetCorners() const;

  VECCORE_ATT_HOST_DEVICE
  inline Vector3D<Precision> GetSide(int i) const;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  SOA3D<Precision> const &GetSides() const;

  VECCORE_ATT_HOST_DEVICE
  void Set(int index, Vector3D<Precision> const &p0, Vector3D<Precision> const &p1, Vector3D<Precision> const &p2);

  VECCORE_ATT_HOST_DEVICE
  inline Precision Distance(Vector3D<Precision> const &point, Vector3D<Precision> const &direction) const;
};

VECCORE_ATT_HOST_DEVICE
int Rectangles::size() const
{
  return fPlanes.size();
}

VECCORE_ATT_HOST_DEVICE
Vector3D<Precision> Rectangles::GetNormal(int i) const
{
  return fPlanes.GetNormal(i);
}

VECCORE_ATT_HOST_DEVICE
SOA3D<Precision> const &Rectangles::GetNormals() const
{
  return fPlanes.GetNormals();
}

VECCORE_ATT_HOST_DEVICE
Precision Rectangles::GetDistance(int i) const
{
  return fPlanes.GetDistance(i);
}

VECCORE_ATT_HOST_DEVICE
Array<Precision> const &Rectangles::GetDistances() const
{
  return fPlanes.GetDistances();
}

VECCORE_ATT_HOST_DEVICE
Vector3D<Precision> Rectangles::GetCenter(int i) const
{
  return -GetDistance(i) * GetNormal(i);
}

VECCORE_ATT_HOST_DEVICE
Vector3D<Precision> Rectangles::GetCorner(int i, int j) const
{
  return Vector3D<Precision>(fCorners[i][0][j], fCorners[i][1][j], fCorners[i][2][j]);
}

VECCORE_ATT_HOST_DEVICE
Rectangles::Corners_t const &Rectangles::GetCorners() const
{
  return fCorners;
}

VECCORE_ATT_HOST_DEVICE
Vector3D<Precision> Rectangles::GetSide(int i) const
{
  return Vector3D<Precision>(fSides[0][i], fSides[1][i], fSides[2][i]);
}

VECCORE_ATT_HOST_DEVICE
SOA3D<Precision> const &Rectangles::GetSides() const
{
  return fSides;
}

VECCORE_ATT_HOST_DEVICE
void Rectangles::Set(int index, Vector3D<Precision> const &p0, Vector3D<Precision> const &p1,
                     Vector3D<Precision> const &p2)
{

  // Store corners and sides
  fCorners[0].set(index, p0);
  fCorners[1].set(index, p2);
  Vector3D<Precision> side = p1 - p0;
  side.Normalize();
  fSides.set(index, side);

  // Compute plane equation to retrieve normal and distance to origin
  // ax + by + cz + d = 0
  Precision a, b, c, d;
  a = p0[1] * (p1[2] - p2[2]) + p1[1] * (p2[2] - p0[2]) + p2[1] * (p0[2] - p1[2]);
  b = p0[2] * (p1[0] - p2[0]) + p1[2] * (p2[0] - p0[0]) + p2[2] * (p0[0] - p1[0]);
  c = p0[0] * (p1[1] - p2[1]) + p1[0] * (p2[1] - p0[1]) + p2[0] * (p0[1] - p1[1]);
  d = -p0[0] * (p1[1] * p2[2] - p2[1] * p1[2]) - p1[0] * (p2[1] * p0[2] - p0[1] * p2[2]) -
      p2[0] * (p0[1] * p1[2] - p1[1] * p0[2]);
  Vector3D<Precision> normal(a, b, c);
  // Normalize the plane equation
  // (ax + by + cz + d) / sqrt(a^2 + b^2 + c^2) = 0 =>
  // n0*x + n1*x + n2*x + p = 0
  Precision inverseLength = 1. / normal.Length();
  normal *= inverseLength;
  d *= inverseLength;
  if (d >= 0) {
    // Ensure normal is pointing away from origin
    normal = -normal;
    d      = -d;
  }

  fPlanes.Set(index, normal, d);
}

VECCORE_ATT_HOST_DEVICE
Precision Rectangles::Distance(Vector3D<Precision> const &point, Vector3D<Precision> const &direction) const
{
  Precision bestDistance = kInfLength;
  for (int i = 0, iMax = size(); i < iMax; ++i) {
    Vector3D<Precision> normal       = GetNormal(i);
    Vector3D<Precision> side         = GetSide(i);
    Precision t                      = -(normal.Dot(point) + GetDistance(i)) / normal.Dot(direction);
    Vector3D<Precision> intersection = point + t * direction;
    Vector3D<Precision> fromP0       = intersection - fCorners[0][i];
    Vector3D<Precision> fromP2       = intersection - fCorners[1][i];
    if (t >= 0 && t < bestDistance && side.Dot(fromP0) >= 0 && (-side).Dot(fromP2) >= 0) {
      bestDistance = t;
    }
  }
  return bestDistance;
}

std::ostream &operator<<(std::ostream &os, Rectangles const &rhs);

} // End inline impl namespace

} // End global namespace

#endif // VECGEOM_VOLUMES_RECTANGLES_H_
