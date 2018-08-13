/// \file PlacedTrapezoid.cpp
/// \author Guilherme Lima (lima at fnal dot gov)

#include "volumes/PlacedTrapezoid.h"
#include "volumes/SpecializedTrapezoid.h"
#ifdef VECGEOM_ROOT
#include "TGeoArb8.h"
#endif
#ifdef VECGEOM_GEANT4
#include "G4Trap.hh"
#endif

#include <cstdio>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedTrapezoid::PrintType() const
{
  printf("PlacedTrapezoid");
}

void PlacedTrapezoid::PrintType(std::ostream &s) const
{
  s << "PlacedTrapezoid";
}

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedTrapezoid::ConvertToUnspecialized() const
{
  return new SimpleTrapezoid(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedTrapezoid::ConvertToRoot() const
{
  return GetUnplacedVolume()->ConvertToRoot(GetName());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedTrapezoid::ConvertToGeant4() const
{
  return GetUnplacedVolume()->ConvertToGeant4(GetName());
}
#endif

#endif // VECCORE_CUDA

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedTrapezoid)
#endif

} // End namespace vecgeom
