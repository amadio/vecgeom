#ifndef VECGEOM_VOLUMES_SOLIDMESH_H_
#define VECGEOM_VOLUMES_SOLIDMESH_H_

#include "base/Utils3D.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

class SolidMesh {
private:
  Utils3D::Polyhedron fMesh;

public:
  Utils3D::Polyhedron const &GetMesh() { return fMesh; }
  Utils3D::vector_t<Utils3D::Vec_t> const &GetVertices() { return fMesh.fVert; }
  Utils3D::vector_t<Utils3D::Polygon> const &GetPolygons() { return fMesh.fPolys; }
  void SetPolygons(const Utils3D::Polygon polys[], size_t count) { fMesh.fPolys.assign(polys, polys + count); }
  void SetVertices(const Utils3D::Vec_t vertices[], size_t count) { fMesh.fVert.assign(vertices, vertices + count); }
  void ResetMesh(size_t nvert, size_t nPoly) { fMesh.Reset(nvert, nPoly); }
  void AddBatchPolygons(size_t n, size_t N, bool convex);
  void AddBatchPolygons(size_t n, size_t N, Utils3D::Vec_t const &normal);
  void SetPolygonIndices(size_t i, const Utils3D::vector_t<size_t> &indices);
  void ApplyTransformation(const Transformation3D &trans);
  void TransformVertices(const Transformation3D &trans);
  void InitConvexHexahedron();
  void InitTetrahedron(Vector3D<Precision> n0, Vector3D<Precision> n1, Vector3D<Precision> n2, Vector3D<Precision> n3);
  void InitSExtruVolume(size_t nMeshVertices, size_t nMeshPolygons, bool convex);
  void InitPolygons();
  bool AddPolygon(size_t n, Utils3D::vector_t<size_t> const &indices, bool convex)
  {
	Utils3D::Polygon poly{n, fMesh.fVert, indices, convex};
	if(!poly.fValid)
		return false;

	poly.Init();
	fMesh.AddPolygon(poly, true);
	return true;

  }

};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
