/// @file PlacedTrd.cpp
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#include "volumes/Trd.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedTrd::ConvertToUnspecialized() const
{
  return new SimpleTrd(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedTrd::ConvertToRoot() const
{
  return GetUnplacedVolume()->ConvertToRoot(GetName());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedTrd::ConvertToGeant4() const
{
  return GetUnplacedVolume()->ConvertToGeant4(GetName());
}
#endif

#endif // VECCORE_CUDA

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedTrd, TrdTypes::UniversalTrd)

#ifndef VECGEOM_NO_SPECIALIZATION
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedTrd, TrdTypes::Trd1)
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedTrd, TrdTypes::Trd2)
#endif

#endif

} // namespace vecgeom
