#include "volumes/SolidMesh.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

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

} // namespace VECGEOM_IMPL_NAMESPACE

} // namespace vecgeom
