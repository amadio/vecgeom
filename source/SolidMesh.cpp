// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \brief Wrapper class for mesh representation of VecGeom solids.
/// \file source/SolidMesh.cpp
/// \author First version created by Murat Topak (CERN summer student 2019)

#include "volumes/SolidMesh.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {
typedef Utils3D::Polygon Polygon;

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

} // namespace VECGEOM_IMPL_NAMESPACE

} // namespace vecgeom
