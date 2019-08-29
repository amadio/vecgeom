// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \brief Wrapper class for mesh representation of VecGeom solids.
/// \file volumes/SolidMesh.h
/// \author First version created by Murat Topak (CERN summer student 2019)


#ifndef VECGEOM_VOLUMES_SOLIDMESH_H_
#define VECGEOM_VOLUMES_SOLIDMESH_H_

#include "base/Utils3D.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

class SolidMesh {
private:
  Utils3D::Polyhedron fMesh; ///< Structure storing the mesh data

public:
  /// Gets the mesh object
  Utils3D::Polyhedron const &GetMesh() { return fMesh; }

  /// Gets the vertices
  Utils3D::vector_t<Utils3D::Vec_t> const &GetVertices() { return fMesh.fVert; }

  /// Gets the polygons
  Utils3D::vector_t<Utils3D::Polygon> const &GetPolygons() { return fMesh.fPolys; }

  /// Sets the polygons to the given polygons
  void SetPolygons(const Utils3D::Polygon polys[], size_t n) { fMesh.fPolys.assign(polys, polys + n); }

  /// Sets the vertices to the given vertices
  void SetVertices(const Utils3D::Vec_t vertices[], size_t n) { fMesh.fVert.assign(vertices, vertices + n); }

  /// Clears the mesh and allocates sufficient space to hold given amount of vertices and polygons
  void ResetMesh(size_t nvert, size_t nPoly) { fMesh.Reset(nvert, nPoly); }

  /// Transforms the vertices and polygons
  void ApplyTransformation(const Transformation3D &trans);

  /// Transforms only the vertices
  void TransformVertices(const Transformation3D &trans);

  /// Adds polygon given by its indices only if it is not a line or a point
  bool AddPolygon(size_t n, Utils3D::vector_t<size_t> const &indices, bool convex)
  {
    Utils3D::Polygon poly{n, fMesh.fVert, indices, convex};
    if (!poly.fValid) return false;

    fMesh.AddPolygon(poly, false);
    return true;
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
