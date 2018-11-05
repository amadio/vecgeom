///
/// \file Utils3D.h
/// \author Andrei Gheata (andrei.gheata@cern.ch)

#ifndef VECGEOM_BASE_UTILS3D_H_
#define VECGEOM_BASE_UTILS3D_H_

#include "base/Vector3D.h"
#include "base/Transformation3D.h"

/// A set of 3D geometry utilities
namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

namespace Utils3D {

enum EPlaneXing_t { kParallel = 0, kIdentical, kIntersecting };

enum EBodyXing_t { kDisjoint = 0, kTouching, kOverlapping };

/// @brief A basic plane in Hessian normal form
struct HPlane {
  Vector3D<double> fNorm; ///< Unit normal vector to plane
  double fDist = 0.;      ///< Distance to plane (positive if origin on same side as normal)

  HPlane() : fNorm() {}
  HPlane(Vector3D<double> const &norm, double dist)
  {
    fNorm = norm;
    fDist = dist;
  }
};

struct Line {
  Vector3D<double> fPts[2];
};

/// @brief A rectangle defined by half widths, normal and distance to origin, and an up vector
struct HRectangle {
  double fDx   = 0.;        ///< half dimension in X
  double fDy   = 0.;        ///< half dimension in Y
  double fDist = 0.;        ///< Distance to plane (positive if origin on same side as normal)
  Vector3D<double> fNorm;   ///< Unit normal vector to plane
  Vector3D<double> fCenter; ///< Center of the rectangle
  Vector3D<double> fUpVect; ///< up vector along local Y

  HRectangle() : fNorm(), fUpVect() {}
  HRectangle(double dx, double dy, double dist, Vector3D<double> const &norm, Vector3D<double> const &center,
             Vector3D<double> const &up)
  {
    fDx     = dx;
    fDy     = dy;
    fDist   = dist;
    fNorm   = norm;
    fCenter = center;
    fUpVect = up;
    assert(vecCore::math::Abs(fNorm.Dot(fUpVect)) < kTolerance);
  }
  void Transform(Transformation3D const &tr);
};

#ifndef VECCORE_CUDA
std::ostream &operator<<(std::ostream &os, HPlane const &hpl);
std::ostream &operator<<(std::ostream &os, HRectangle const &hrect);
#endif

/// @brief Transform a plane from master frame to local frame given by a transformation
void TransformPlane(Transformation3D const &tr, HPlane const &localp, HPlane &masterp);

/// @brief Transform a plane from master frame to local frame given by a transformation
void TransformRectangle(Transformation3D const &tr, HRectangle const &local, HRectangle &master);

/// @brief Function to find the crossing line between two planes.
/** The function takes as input the 2 plane definition in Hessian form (n, p), where:
    n = normal to the plane
    p = distance from origin to plane (for p > 0 the origin is on the side of the normal)
    The return value is a point on the crossing line (if any) and a normalized direction vector.
    The function returns a status flag of type EPlaneXing_t */
EPlaneXing_t PlaneXing(Vector3D<double> const &n1, double p1, Vector3D<double> const &n2, double p2,
                       Vector3D<double> &point, Vector3D<double> &direction);

// @brief Function to find if 2 3D rectangles cross each other.
EBodyXing_t RectangleXing(HRectangle const &rect1, HRectangle const &rect2, Line *line = nullptr);

/// @brief Function to determine crossing of two arbitrary placed boxes
/** The function takes the box parameters and their transformations in a common frame.
    A fast check is performed if both transformations are identity. */
EBodyXing_t BoxXing(Vector3D<double> const &box1, Transformation3D const &tr1, Vector3D<double> const &box2,
                    Transformation3D const &tr2);

} // namespace Utils3D
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
#endif
