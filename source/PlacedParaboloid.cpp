// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// @file source/PlacedParaboloid.cpp
/// @author Marilena Bandieramonte

#include "volumes/PlacedParaboloid.h"
#include "volumes/SpecializedParaboloid.h"
#ifdef VECGEOM_ROOT
#include "TGeoParaboloid.h"
#endif
#ifdef VECGEOM_GEANT4
#include "G4Paraboloid.hh"
#endif

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedParaboloid::PrintType() const
{
  printf("PlacedParaboloid");
}

void PlacedParaboloid::PrintType(std::ostream &s) const
{
  s << "PlacedParaboloid";
}

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedParaboloid::ConvertToUnspecialized() const
{
  return new SimpleParaboloid(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedParaboloid::ConvertToRoot() const
{
  return new TGeoParaboloid(GetLabel().c_str(), GetRlo(), GetRhi(), GetDz());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedParaboloid::ConvertToGeant4() const
{
  return new G4Paraboloid(GetLabel(), GetDz(), GetRlo(), GetRhi());
}
#endif

#endif // VECCORE_CUDA

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedParaboloid)

#endif // VECCORE_CUDA

} // End namespace vecgeom
