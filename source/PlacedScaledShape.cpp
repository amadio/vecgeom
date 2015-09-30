/// \file PlacedScaledShape.cpp
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)

#include "volumes/PlacedScaledShape.h"

#include "base/AOS3D.h"
#include "base/SOA3D.h"
#include "volumes/SpecializedScaledShape.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH
void PlacedScaledShape::PrintType() const {
  printf("PlacedScaledShape");
}

#ifndef VECGEOM_NVCC

VPlacedVolume const* PlacedScaledShape::ConvertToUnspecialized() const {
  return 0;
//  return new SimpleAssembly(GetLabel().c_str(), logical_volume_, fTransformation);
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedScaledShape::ConvertToRoot() const {
// To be implemented
//  return new TGeoBBox(GetLabel().c_str(), x(), y(), z());
  return 0;
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedAssembly::ConvertToUSolids() const {
  // No implementation in USolids
  return 0;
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedScaledShape::ConvertToGeant4() const {
// To be implemented
//  return new G4Box("", x(), y(), z());
  return 0;
}
#endif

#endif // VECGEOM_NVCC

} // End impl namespace

#ifdef VECGEOM_NVCC

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC( SpecializedScaledShape )

#endif // VECGEOM_NVCC

} // End namespace vecgeom
