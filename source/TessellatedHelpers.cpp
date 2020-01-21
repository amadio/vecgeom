/// \file TessellatedHelpers.cpp
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)

#include <ostream>
#include "VecGeom/volumes/TessellatedSection.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA
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
#endif
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
