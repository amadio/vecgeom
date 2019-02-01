/// @file PlacedEllipticalCone.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch), Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

#include "volumes/PlacedEllipticalCone.h"
#include "volumes/SpecializedEllipticalCone.h"
#ifdef VECGEOM_ROOT
//include header file for ROOT elliptical Cone as done for Tube below
//#include "TGeoEltu.h"
#endif
#ifdef VECGEOM_GEANT4
//include header file for Geant elliptical Cone as done for Tube below
//#include "G4EllipticalCone.h"
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
  //Return the ROOT EllipticCone object like follows
  //return new TGeoEltu(GetLabel().c_str(), GetDx(), GetDy(), GetDz());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedEllipticalCone::ConvertToGeant4() const
{
	//Return the Geant4 EllipticCone object like follows
    //return new G4EllipticalCone("", GetDx(), GetDy(), GetDz());
}
#endif

#endif // VECCORE_CUDA

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedEllipticalCone)

#endif // VECCORE_CUDA

} // namespace vecgeom
