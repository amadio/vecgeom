/// @file PlacedOrb.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#include "volumes/PlacedOrb.h"
#include "volumes/SpecializedOrb.h"
#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedOrb::PrintType() const
{
  printf("PlacedOrb");
}

void PlacedOrb::PrintType(std::ostream &s) const
{
  s << "PlacedOrb";
}

#ifndef VECCORE_CUDA
VPlacedVolume const *PlacedOrb::ConvertToUnspecialized() const
{
  return new SimpleOrb(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}
#endif // VECCORE_CUDA

} // End impl namespace

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedOrb)

#endif // VECCORE_CUDA

} // End namespace vecgeom
