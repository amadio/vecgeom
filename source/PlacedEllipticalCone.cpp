/// @file PlacedEllipticalCone.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch), Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

#include "volumes/PlacedEllipticalCone.h"
#include "volumes/SpecializedEllipticalCone.h"
#ifdef VECGEOM_ROOT
// Include header file for ROOT Elliptical Cone as done for Tube below
// #include "TGeoEltu.h"
#endif
#ifdef VECGEOM_GEANT4
#include "G4EllipticalCone.hh"
#endif

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedEllipticalCone::PrintType() const
{
  printf("PlacedEllipticalCone");
}

void PlacedEllipticalCone::PrintType(std::ostream &s) const
{
  s << "PlacedEllipticalCone";
}

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedEllipticalCone::ConvertToUnspecialized() const
{
  return new SimpleEllipticalCone(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedEllipticalCone::ConvertToRoot() const
{
  // Return ROOT Elliptical Cone
  return nullptr; // There is no suitable TGeo shape 2019.02.19
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedEllipticalCone::ConvertToGeant4() const
{
  return new G4EllipticalCone("", GetSemiAxisX(), GetSemiAxisY(), GetZMax(), GetZTopCut());
}
#endif

#endif // VECCORE_CUDA

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedEllipticalCone)

#endif // VECCORE_CUDA

} // namespace vecgeom
