///
/// \file Utils3D.h
/// \author Andrei Gheata (andrei.gheata@cern.ch)

#ifndef VECGEOM_BASE_UTILS3D_H_
#define VECGEOM_BASE_UTILS3D_H_

#include "base/Vector3D.h"
#include "base/Transformation3D.h"

#ifndef VECCORE_CUDA
#include <vector>
#else
#include "base/Vector.h"
#endif

/// A set of 3D geometry utilities
namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

namespace Utils3D {

using Vec_t = Vector3D<double>;

template <typename T>
#ifndef VECCORE_CUDA
using vector_t = std::vector<T>;
#else
using vector_t = vecgeom::Vector<T>; // this has problems, like in the initializers Vector v = {...};
#endif

enum EPlaneXing_t { kParallel = 0, kIdentical, kIntersecting };

enum EBodyXing_t { kDisjoint = 0, kTouching, kOverlapping }; // do not change order

/// @brief A basic plane in Hessian normal form
struct Plane {
  Vector3D<double> fNorm; ///< Unit normal vector to plane
  double fDist = 0.;      ///< Distance to plane (positive if origin on same side as normal)

  Plane() : fNorm() {}

  Plane(Vector3D<double> const &norm, double dist)
  {
    fNorm = norm;
    fDist = dist;
  }

  /// @brief Transform the plane by a general transformation
  void Transform(Transformation3D const &tr);
};

/// @brief A line segment
struct Line {
  Vector3D<double> fPts[2]; ///< Points defining the line
};

/// @brief A polygon defined by vertices and normal
/* The list of vertices is a reference to an external array. The used vertex indices have to be defined such
   that consecutive segments cross product is on the same side as the normal. */
struct Polygon {
  size_t fN     = 0;      ///< Number of vertices
  bool fConvex  = false;  ///< Convexity
  bool fHasNorm = false;  ///< Normal is already supplied
  bool fValid = false;    ///< Polygon is not degenerate
  double fDist  = 0.;     ///< Distance to plane in the Hessian form
  Vec_t fNorm;            ///< Unit normal vector to plane
  vector_t<Vec_t> &fVert; ///< Global list of vertices shared with other polygons
  vector_t<size_t> fInd;  ///< [fN] Indices of vertices
  vector_t<Vec_t> fSides; ///< [fN] Side vectors

  /// @brief Constructor taking the number of vertices, a reference to a vector of vertices and the convexity
  VECCORE_ATT_HOST_DEVICE
  Polygon(size_t n, vector_t<Vec_t> &vertices, bool convex = false);

  // @brief Fast constructor with no checks in case of convex polygons, providing the normal vector (normalized)
  VECCORE_ATT_HOST_DEVICE
  Polygon(size_t n, vector_t<Vec_t> &vertices, Vec_t const &norm);

  Polygon(size_t n, vector_t<Vec_t>& vertices, vector_t<size_t> const& indices, bool convex);

  /// @brief Copy constructor
  VECCORE_ATT_HOST_DEVICE
  Polygon(const Polygon &other)
      : fN(other.fN), fConvex(other.fConvex), fHasNorm(other.fHasNorm), fValid(other.fValid), fDist(other.fDist), fNorm(other.fNorm),
        fVert(other.fVert), fInd(other.fInd), fSides(other.fSides)
  {

  }

  /// @brief Assignment operator
  VECGEOM_FORCE_INLINE
  Polygon &operator=(const Polygon &other)
  {
    if (&other == this) return *this;
    fN       = other.fN;
    fConvex  = other.fConvex;
    fHasNorm = other.fHasNorm;
    fValid   = other.fValid;
    fDist    = other.fDist;
    fNorm    = other.fNorm;
    fVert    = other.fVert;
    fInd     = other.fInd;
    fSides   = other.fSides;
    return *this;
  }

  /// @brief Setter for a vertex index
  VECGEOM_FORCE_INLINE
  void SetVertex(size_t ind, size_t ivert) { fInd[ind] = ivert; }

  /// @brief Getter for a vertex
  VECGEOM_FORCE_INLINE
  Vec_t const &GetVertex(size_t i) const { return fVert[fInd[i]]; }

  /// @brief Setter from an array of vertex indices
  template <typename T>
  void SetVertices(vector_t<T> indices)
  {
    for (size_t i = 0; i < fN; ++i)
      fInd[i] = size_t(indices[i]);
  }

  /// @brief Initialization is mandatory before first use
  void Init();

  /// @brief Transform the polygon by a general transformation
  void Transform(Transformation3D const &tr);

  void CheckAndFixDegenerate();
};

/// @brief A simple polyhedron defined by vertices and polygons
struct Polyhedron {
  vector_t<Vec_t> fVert;    ///< Vector of vertices
  vector_t<Polygon> fPolys; ///< Vector of polygons

  /// @brief Constructors
  Polyhedron(){};
  Polyhedron(size_t nvert, size_t npolys)
  {
    fVert.reserve(nvert);
    fPolys.reserve(npolys);
  }

  /// @brief Polyhedrons can be re-used
  VECGEOM_FORCE_INLINE
  void Reset(size_t nvert, size_t npolys)
  {
    fVert.reserve(nvert);
    fVert.clear();
    fPolys.reserve(npolys);
    fPolys.clear();
  }

  /// @brief Get number of vertices
  VECGEOM_FORCE_INLINE
  size_t GetNvertices() const { return fVert.size(); }

  /// @brief Get number of polygons
  VECGEOM_FORCE_INLINE
  size_t GetNpolygons() const { return fPolys.size(); }

  /// @brief Get a reference to a specific vertex
  VECGEOM_FORCE_INLINE
  Vec_t const &GetVertex(size_t i) const { return fVert[i]; }

  /// @brief Get a reference to a specific polygon
  VECGEOM_FORCE_INLINE
  Polygon const &GetPolygon(size_t i) const { return fPolys[i]; }

  /// @brief Transform the polygon by a general transformation
  void Transform(Transformation3D const &tr);

  bool AddPolygon(Polygon const& poly);

};

#ifndef VECCORE_CUDA
/// @brief Streamers
std::ostream &operator<<(std::ostream &os, Plane const &hpl);
std::ostream &operator<<(std::ostream &os, Polygon const &poly);
std::ostream &operator<<(std::ostream &os, Polyhedron const &polyh);
#endif

/// @brief Function to find the crossing line between two planes.
EPlaneXing_t PlaneXing(Plane const &pl1, Plane const &pl2, Vector3D<double> &point, Vector3D<double> &direction);

// @brief Function to find if 2 arbitrary polygons cross each other.
EBodyXing_t PolygonXing(Polygon const &poly1, Polygon const &poly2, Line *line = nullptr);

// @brief Function to find if 2 arbitrary polyhedrons cross each other.
EBodyXing_t PolyhedronXing(Polyhedron const &poly1, Polyhedron const &poly2, vector_t<Line> &lines);

/// @brief Function to determine crossing of two arbitrary placed boxes
/** The function takes the box parameters and their transformations in a common frame.
    A fast check is performed if both transformations are identity. */
EBodyXing_t BoxCollision(Vector3D<double> const &box1, Transformation3D const &tr1, Vector3D<double> const &box2,
                         Transformation3D const &tr2);

/// @brief Function filling a polyhedron as a box
void FillBoxPolyhedron(Vec_t const &dimensions, Polyhedron &polyh);

} // namespace Utils3D
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
#endif
