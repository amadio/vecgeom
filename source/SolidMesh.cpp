#include "volumes/SolidMesh.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {
typedef Utils3D::Polygon Polygon;

void SolidMesh::InitConvexHexahedron()
{
  fMesh.fPolys = {{4, fMesh.fVert, true}, {4, fMesh.fVert, true}, {4, fMesh.fVert, true},
                  {4, fMesh.fVert, true}, {4, fMesh.fVert, true}, {4, fMesh.fVert, true}};

  fMesh.fPolys[0].fInd = {0, 1, 2, 3};
  fMesh.fPolys[1].fInd = {4, 7, 6, 5};
  fMesh.fPolys[2].fInd = {0, 4, 5, 1};
  fMesh.fPolys[3].fInd = {1, 5, 6, 2};
  fMesh.fPolys[4].fInd = {2, 6, 7, 3};
  fMesh.fPolys[5].fInd = {3, 7, 4, 0};

  // compute normals, sides, etc.
  for (size_t i = 0; i < 6; ++i)
    fMesh.fPolys[i].Init();
}

void SolidMesh::InitTetrahedron(Vector3D<Precision> n0, Vector3D<Precision> n1, Vector3D<Precision> n2,
                                Vector3D<Precision> n3)
{
  fMesh.fPolys = {{3, fMesh.fVert, n0}, {3, fMesh.fVert, n1}, {3, fMesh.fVert, n2}, {3, fMesh.fVert, n3}};

  fMesh.fPolys[0].fInd = {0, 1, 2};
  fMesh.fPolys[1].fInd = {2, 1, 3};
  fMesh.fPolys[2].fInd = {3, 2, 0};
  fMesh.fPolys[3].fInd = {3, 0, 1};

  for (size_t i = 0; i < 4; ++i)
    fMesh.fPolys[i].Init();
}

void SolidMesh::InitSExtruVolume(size_t nMeshVertices, size_t nMeshPolygons, bool convex)
{

  size_t nTopVertices = nMeshVertices / 2;
  fMesh.fPolys.push_back({nTopVertices, fMesh.fVert, convex});
  fMesh.fPolys.push_back({nTopVertices, fMesh.fVert, convex});

  // lateral surfaces
  for (size_t i = 2; i < nMeshPolygons; i++) {
    fMesh.fPolys.push_back({4, fMesh.fVert, true});
  }

  // lower surface
  for (size_t i = 0; i < nTopVertices; i++) {
    fMesh.fPolys[0].fInd[i] = (2 * i);
  }

  // upper surface
  for (size_t i = 0; i < nTopVertices; i++) {
    fMesh.fPolys[1].fInd[nTopVertices - i - 1] = (2 * i + 1);
  }

  // lateral surfaces
  for (size_t i = 2; i < nMeshPolygons - 1; i++) {
    fMesh.fPolys[i].fInd = {2 * i - 2, 2 * i - 4, 2 * i - 3, 2 * i - 1};
  }
  fMesh.fPolys[nMeshPolygons - 1].fInd = {0, nMeshVertices - 2, nMeshVertices - 1, 1};

  for (size_t i = 0; i < nMeshPolygons; ++i)
    fMesh.fPolys[i].Init();
}

void SolidMesh::ApplyTransformation(const Transformation3D &trans)
{
  fMesh.Transform(trans);
}

void SolidMesh::TransformVertices(const Transformation3D &trans)
{
  typedef Vector3D<double> Vec_t;
  // Transform vertices
  for (size_t i = 0; i < fMesh.fVert.size(); ++i) {
    Vec_t temp;
    trans.InverseTransform(fMesh.fVert[i], temp);
    fMesh.fVert[i] = temp;
  }
}

void SolidMesh::AddBatchPolygons(size_t n, size_t N, bool convex)
{

  for (size_t i = 0; i < N; i++) {
    fMesh.fPolys.push_back({n, fMesh.fVert, convex});
  }
}

void SolidMesh::AddBatchPolygons(size_t n, size_t N, Utils3D::Vec_t const &normal)
{
  for (size_t i = 0; i < N; i++) {
    fMesh.fPolys.push_back({n, fMesh.fVert, normal});
  }
}

void SolidMesh::SetPolygonIndices(size_t i, const Utils3D::vector_t<size_t> &indices)
{
  fMesh.fPolys[i].fInd = indices;
}

void SolidMesh::InitPolygons()
{
  for (size_t i = 0; i < fMesh.fPolys.size(); ++i)
    fMesh.fPolys[i].Init();
}

} // namespace VECGEOM_IMPL_NAMESPACE

} // namespace vecgeom
