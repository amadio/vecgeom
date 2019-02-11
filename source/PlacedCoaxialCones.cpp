/// @file PlacedCoaxialCones.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch), Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

#include "volumes/PlacedCoaxialCones.h"
#include "volumes/SpecializedCoaxialCones.h"
#ifdef VECGEOM_ROOT
//include header file for ROOT elliptical Cone as done for Tube below
//#include "TGeoEltu.h"
#endif
#ifdef VECGEOM_GEANT4
//include header file for Geant elliptical Cone as done for Tube below
//#include "G4CoaxialCones.h"
#endif

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedCoaxialCones::PrintType() const
{
  printf("PlacedCoaxialCones");
}

void PlacedCoaxialCones::PrintType(std::ostream &s) const
{
  s << "PlacedCoaxialCones";
}

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedCoaxialCones::ConvertToUnspecialized() const
{
  return new SimpleCoaxialCones(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedCoaxialCones::ConvertToRoot() const
{
  //Return the ROOT EllipticCone object like follows
  //return new TGeoEltu(GetLabel().c_str(), GetDx(), GetDy(), GetDz());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedCoaxialCones::ConvertToGeant4() const
{
	//Return the Geant4 EllipticCone object like follows
    //return new G4CoaxialCones("", GetDx(), GetDy(), GetDz());
}
#endif

#endif // VECCORE_CUDA

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedCoaxialCones)

#endif // VECCORE_CUDA

} // namespace vecgeom
