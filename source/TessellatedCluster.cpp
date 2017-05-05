#include "volumes/TessellatedCluster.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

std::ostream &operator<<(std::ostream &os, TriangleFacet<double> const &facet)
{
  os << "  vertices: {" << facet.fVertices[0] << ", " << facet.fVertices[1] << ", " << facet.fVertices[2] << "}\n";
  os << "    indices:  {" << facet.fIndices << "}\n";
  if (facet.fNeighbors.size()) {
    os << "    neighbors: {";
    for (size_t i = 0; i < facet.fNeighbors.size(); ++i) {
      os << facet.fNeighbors[i] << ", ";
    }
    os << "}\n";
  }
  os << "    normal: {" << facet.fNormal << "}\n";
  os << "    distance: " << facet.fDistance << "}";
  return os;
}

std::ostream &operator<<(std::ostream &os, TessellatedCluster<typename vecgeom::VectorBackend::Real_v> const &tcl)
{
  os << "normals: {" << tcl.fNormals << "}\n";
  os << "distances: {" << tcl.fDistances << "}\n";
  os << "side vectors: {" << tcl.fSideVectors[0] << "}\n\t{" << tcl.fSideVectors[1] << "}\n\t{" << tcl.fSideVectors[2]
     << "}";
  return os;
}

} // End inline implementation namespace

} // End global namespace
