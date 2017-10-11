/// @file PlacedExtruded.cpp
/// @author Mihaela Gheata (mihaela.gheata@cern.ch)

#include "volumes/Extruded.h"
#include "volumes/Tessellated.h"

#ifndef VECCORE_CUDA
#ifdef VECGEOM_USOLIDS
#include "UTessellatedSolid.hh"
#include "UTriangularFacet.hh"
#endif

#ifdef VECGEOM_GEANT4
#include "G4TessellatedSolid.hh"
#include "G4TriangularFacet.hh"
#endif

#endif // VECCORE_CUDA

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedExtruded::ConvertToUnspecialized() const
{
  return new SimpleExtruded(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedExtruded::ConvertToRoot() const
{
  return nullptr; // TO BE IMPLEMENTED
}
#endif

#if defined(VECGEOM_USOLIDS) && !defined(VECGEOM_REPLACE_USOLIDS)
::VUSolid const *PlacedExtruded::ConvertToUSolids() const
{
  UTessellatedSolid *tsl = new UTessellatedSolid("");
  for (size_t ifacet = 0; ifacet < GetUnplacedVolume()->GetNFacets(); ++ifacet) {
    TriangleFacet<double> *facet = GetUnplacedVolume()->GetFacet(ifacet);
    tsl->AddFacet(
        new UTriangularFacet(UVector3(facet->fVertices[0].x(), facet->fVertices[0].y(), facet->fVertices[0].z()),
                             UVector3(facet->fVertices[1].x(), facet->fVertices[1].y(), facet->fVertices[1].z()),
                             UVector3(facet->fVertices[2].x(), facet->fVertices[2].y(), facet->fVertices[2].z()),
                             UFacetVertexType::UABSOLUTE));
  }
  tsl->SetSolidClosed(true);
  return tsl;
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedExtruded::ConvertToGeant4() const
{
  G4TessellatedSolid *tsl = new G4TessellatedSolid("");
  for (size_t ifacet = 0; ifacet < GetUnplacedVolume()->GetNFacets(); ++ifacet) {
    TriangleFacet<double> *facet = GetUnplacedVolume()->GetFacet(ifacet);
    tsl->AddFacet(new G4TriangularFacet(
        G4ThreeVector(facet->fVertices[0].x(), facet->fVertices[0].y(), facet->fVertices[0].z()),
        G4ThreeVector(facet->fVertices[1].x(), facet->fVertices[1].y(), facet->fVertices[1].z()),
        G4ThreeVector(facet->fVertices[2].x(), facet->fVertices[2].y(), facet->fVertices[2].z()), ABSOLUTE));
  }
  tsl->SetSolidClosed(true);
  return tsl;
}
#endif

#endif // VECCORE_CUDA

} // End impl namespace

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedExtruded)

#endif

} // End global namespace
