/// @file PlacedParallelepiped.cpp
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/PlacedParallelepiped.h"

#include "volumes/Parallelepiped.h"

#ifdef VECGEOM_ROOT
#include "TGeoPara.h"
#endif
#ifdef VECGEOM_GEANT4
#include "G4Para.hh"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedParallelepiped::ConvertToUnspecialized() const
{
  return new SimpleParallelepiped(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedParallelepiped::ConvertToRoot() const
{
  return new TGeoPara(GetLabel().c_str(), GetX(), GetY(), GetZ(), GetAlpha() * kRadToDeg, GetTheta() * kRadToDeg,
                      GetPhi() * kRadToDeg);
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedParallelepiped::ConvertToGeant4() const
{
  return new G4Para(GetLabel(), GetX(), GetY(), GetZ(), GetAlpha(), GetTheta(), GetPhi());
}
#endif

#endif // VECCORE_CUDA

} // End impl namespace

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedParallelepiped)

#endif // VECCORE_CUDA

} // End global namespace
