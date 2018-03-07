/// \file TessellatedHelpers.cpp
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)

#include <ostream>
#include "volumes/TessellatedSection.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

std::ostream &operator<<(std::ostream &os, TriangleFacet<double> const &facet)
{
#ifndef VECCORE_ENABLE_UMESIMD
  os << " triangle facet:\n";
  os << "    vertices: {" << facet.fVertices[0] << ", " << facet.fVertices[1] << ", " << facet.fVertices[2] << "}\n";
  os << "    indices:  {" << facet.fIndices << "}\n";
  os << "    normal: {" << facet.fNormal << "}\n";
  os << "    distance: " << facet.fDistance << "}";
#endif
  return os;
}

std::ostream &operator<<(std::ostream &os, QuadrilateralFacet<double> const &facet)
{
#ifndef VECCORE_ENABLE_UMESIMD
  os << " quadrilateral facet:\n";
  os << "    vertices: {" << facet.fVertices[0] << ", " << facet.fVertices[1] << ", " << facet.fVertices[2] << ", "
     << facet.fVertices[3] << "}\n";
  os << "    indices:  {" << facet.fIndices[0] << ", " << facet.fIndices[1] << ", " << facet.fIndices[2] << ", "
     << facet.fIndices[3] << "}\n";
  os << "    normal: {" << facet.fNormal << "}\n";
  os << "    distance: " << facet.fDistance << "}";
#endif
  return os;
}

std::ostream &operator<<(std::ostream &os, TessellatedCluster<3, typename vecgeom::VectorBackend::Real_v> const &tcl)
{
#ifndef VECCORE_ENABLE_UMESIMD
  os << " tessellated cluster of triangles:\n";
  os << " vertices  0: " << tcl.fVertices[0] << " 1: " << tcl.fVertices[1] << " 2: " << tcl.fVertices[2] << std::endl;
  os << " normals: {" << tcl.fNormals << "}\n";
  os << " distances: {" << tcl.fDistances << "}\n";
  os << " side vectors: {" << tcl.fSideVectors[0] << "}\n\t{" << tcl.fSideVectors[1] << "}\n\t{" << tcl.fSideVectors[2]
     << "}";
#endif
  return os;
}

std::ostream &operator<<(std::ostream &os, TessellatedCluster<4, typename vecgeom::VectorBackend::Real_v> const &tcl)
{
#ifndef VECCORE_ENABLE_UMESIMD
  os << " tessellated cluster of quadrilaterals:\n";
  os << " vertices  0: " << tcl.fVertices[0] << " 1: " << tcl.fVertices[1] << " 2: " << tcl.fVertices[2]
     << " 3: " << tcl.fVertices[3] << std::endl;
  os << " normals: {" << tcl.fNormals << "}\n";
  os << " distances: {" << tcl.fDistances << "}\n";
  os << " side vectors: {" << tcl.fSideVectors[0] << "}\n\t{" << tcl.fSideVectors[1] << "}\n\t{" << tcl.fSideVectors[2]
     << "}\n\t{" << tcl.fSideVectors[3] << "}";
#endif
  return os;
}

std::ostream &operator<<(std::ostream &os, vecgeom::TessellatedSection<double> const &ts)
{
#ifndef VECCORE_ENABLE_UMESIMD
  os << " tessellated section with " << ts.GetNclusters() << "clusters:\n";
  for (size_t i = 0; i < ts.GetNclusters(); ++i) {
    os << ts.GetCluster(i) << "\n";
  }
  for (size_t i = 0; i < ts.GetNfacets(); ++i) {
    os << ts.GetFacet(i) << "\n";
  }
#endif
  return os;
}

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
