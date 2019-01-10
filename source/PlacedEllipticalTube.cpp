/// @file PlacedEllipticalTube.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch), Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

#include "volumes/PlacedEllipticalTube.h"
#include "volumes/SpecializedEllipticalTube.h"
#ifdef VECGEOM_ROOT
#include "TGeoEltu.h"
#endif
#ifdef VECGEOM_GEANT4
#include "G4EllipticalTube.h"
#endif

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedEllipticalTube::PrintType() const
{
  printf("PlacedEllipticalTube");
}

void PlacedEllipticalTube::PrintType(std::ostream &s) const
{
  s << "PlacedEllipticalTube";
}

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedEllipticalTube::ConvertToUnspecialized() const
{
  return new SimpleEllipticalTube(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedEllipticalTube::ConvertToRoot() const
{
  return new TGeoEltu(GetLabel().c_str(), GetDx(), GetDy(), GetDz());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedEllipticalTube::ConvertToGeant4() const
{
  return new G4EllipticalTube("", GetDx(), GetDy(), GetDz());
}
#endif

#endif // VECCORE_CUDA

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedEllipticalTube)

#endif // VECCORE_CUDA

} // namespace vecgeom
