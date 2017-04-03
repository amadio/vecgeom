/// \file PlacedParaboloid.cpp
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)

#include "volumes/PlacedParaboloid.h"
#include "volumes/SpecializedParaboloid.h"
#ifdef VECGEOM_ROOT
#include "TGeoParaboloid.h"
#endif
#if defined(VECGEOM_USOLIDS) && !defined(VECGEOM_REPLACE_USOLIDS)
#include "UBox.hh"
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

#if defined(VECGEOM_USOLIDS) && !defined(VECGEOM_REPLACE_USOLIDS)
::VUSolid const *PlacedParaboloid::ConvertToUSolids() const
{
  std::cerr << "**************************************************************\n";
  std::cerr << "WARNING: Paraboloid unsupported for USolids.; returning NULL\n";
  std::cerr << "**************************************************************\n";
  return nullptr;
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedParaboloid::ConvertToGeant4() const
{
  return new G4Paraboloid(GetLabel(), GetDz(), GetRlo(), GetRhi());
}
#endif

#endif // VECCORE_CUDA

} // End impl namespace

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedParaboloid)

#endif // VECCORE_CUDA

} // End namespace vecgeom
