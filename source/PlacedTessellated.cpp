/// @file PlacedTessellated.cpp
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#include "volumes/Tessellated.h"

#ifndef VECCORE_CUDA
#ifdef VECGEOM_USOLIDS
#include "UTessellatedSolid.hh"
#include "UTriangularFacet.hh"
#endif

#ifdef VECGEOM_GEANT4
#include "G4TessellatedSolid.hh"
#endif

#endif // VECCORE_CUDA

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedTessellated::ConvertToUnspecialized() const
{
  return new SimpleTessellated(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedTessellated::ConvertToRoot() const
{
  return nullptr;
}
#endif

#if defined(VECGEOM_USOLIDS) && !defined(VECGEOM_REPLACE_USOLIDS)
::VUSolid const *PlacedTessellated::ConvertToUSolids() const
{
  std::cout << "Creating tessellated solid...\n";
  UTessellatedSolid *tsl = new UTessellatedSolid("");
  for (int ifacet = 0; ifacet < GetUnplacedVolume()->GetNFacets(); ++ifacet) {
    TriangleFacet<double> *facet = GetUnplacedVolume()->GetFacet(ifacet);
    tsl->AddFacet(
        new UTriangularFacet(UVector3(facet->fVertices[0].x(), facet->fVertices[0].y(), facet->fVertices[0].z()),
                             UVector3(facet->fVertices[1].x(), facet->fVertices[1].y(), facet->fVertices[1].z()),
                             UVector3(facet->fVertices[2].x(), facet->fVertices[2].y(), facet->fVertices[2].z()),
                             UFacetVertexType::UABSOLUTE));
  }
  tsl->SetSolidClosed(true);
  std::cout << "Tessellated solid done\n";
  return tsl;
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedTessellated::ConvertToGeant4() const
{
  return nullptr; // to implement
}
#endif

#endif // VECCORE_CUDA

} // End im%pl namespace

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedTessellated, TessellatedTypes::UniversalTessellated)

#ifndef VECGEOM_NO_SPECIALIZATION
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedTessellated, TessellatedTypes::Tessellated1)
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedTessellated, TessellatedTypes::Tessellated2)
#endif

#endif

} // End global namespace
