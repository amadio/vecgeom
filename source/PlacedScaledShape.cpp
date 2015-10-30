/// \file PlacedScaledShape.cpp
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)

#include "volumes/PlacedScaledShape.h"

#include "base/AOS3D.h"
#include "base/SOA3D.h"
#include "volumes/SpecializedScaledShape.h"

#include <stdio.h>

#ifdef VECGEOM_ROOT
#include "TGeoScaledShape.h"
#include "TGeoMatrix.h"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH
void PlacedScaledShape::PrintType() const {
  printf("PlacedScaledShape");
}

#ifndef VECGEOM_NVCC

VPlacedVolume const* PlacedScaledShape::ConvertToUnspecialized() const {
  return new SimpleScaledShape(GetLabel().c_str(), logical_volume_, GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedScaledShape::ConvertToRoot() const {
  UnplacedScaledShape const *unplaced = const_cast<UnplacedScaledShape*>(GetUnplacedVolume());
  return new TGeoScaledShape(GetLabel().c_str(),
                 (TGeoShape*)unplaced->fPlaced->ConvertToRoot(), 
                 new TGeoScale(unplaced->fScale.Scale()[0], unplaced->fScale.Scale()[1], unplaced->fScale.Scale()[2]));
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedScaledShape::ConvertToUSolids() const {
  // No implementation in USolids
  return 0;
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedScaledShape::ConvertToGeant4() const {
// No implementation in Geant4
  return 0;
}
#endif

#endif // VECGEOM_NVCC

} // End impl namespace

#ifdef VECGEOM_NVCC

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC( SpecializedScaledShape )

#endif // VECGEOM_NVCC

} // End namespace vecgeom
