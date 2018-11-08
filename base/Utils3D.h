///
/// \file Utils3D.h
/// \author Andrei Gheata (andrei.gheata@cern.ch)

#ifndef VECGEOM_BASE_UTILS3D_H_
#define VECGEOM_BASE_UTILS3D_H_

#include "base/Vector3D.h"
#include "base/Transformation3D.h"
#include "base/Vector.h"

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

struct Polygon {
  using Vec_t = Vector3D<double>;
  template <typename T>
  using Vector_t = Vector<T>;

  size_t fN     = 0;      ///< Number of vertices
  bool fConvex  = false;  ///< Convexity
  bool fHasNorm = false;  ///< Normal is already supplied
  double fDist  = 0.;     ///< Distance to plane in the Hessian form
  Vec_t fNorm;            ///< Unit normal vector to plane
  Vector_t<Vec_t> fVert;  ///< Vertices
  Vector_t<Vec_t> fSides; ///< Side vectors

  Polygon(){};
  Polygon(size_t n, bool convex = false);
  Polygon(size_t n, double normx, double normy, double normz, bool is_normalized = true, bool is_convex = false);

  VECGEOM_FORCE_INLINE
  void AddVertex(int i, double vx, double vy, double vz) { fVert[i].Set(vx, vy, vz); }
  void Init();
  void Set(size_t n, double normx, double normy, double normz, bool is_normalized = true, bool is_convex = false);
  void Transform(Transformation3D const &tr);
};

#ifndef VECCORE_CUDA
std::ostream &operator<<(std::ostream &os, HPlane const &hpl);
std::ostream &operator<<(std::ostream &os, Polygon const &poly);
#endif

/// @brief Transform a plane from master frame to local frame given by a transformation
void TransformPlane(Transformation3D const &tr, HPlane const &localp, HPlane &masterp);

/// @brief Function to find the crossing line between two planes.
/** The function takes as input the 2 plane definition in Hessian form (n, p), where:
    n = normal to the plane
    p = distance from origin to plane (for p > 0 the origin is on the side of the normal)
    The return value is a point on the crossing line (if any) and a normalized direction vector.
    The function returns a status flag of type EPlaneXing_t */
EPlaneXing_t PlaneXing(Vector3D<double> const &n1, double p1, Vector3D<double> const &n2, double p2,
                       Vector3D<double> &point, Vector3D<double> &direction);

// @brief Function to find if 2 arbitrary polygons cross each other.
EBodyXing_t PolygonXing(Polygon const &poly1, Polygon const &poly2, Line *line = nullptr);

/// @brief Function to determine crossing of two arbitrary placed boxes
/** The function takes the box parameters and their transformations in a common frame.
    A fast check is performed if both transformations are identity. */
EBodyXing_t BoxCollision(Vector3D<double> const &box1, Transformation3D const &tr1, Vector3D<double> const &box2,
                         Transformation3D const &tr2);

} // namespace Utils3D
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
#endif
