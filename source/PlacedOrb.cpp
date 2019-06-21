// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \file source/PlacedOrb.cpp
/// \author Raman Sehgal

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

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedOrb)

#endif // VECCORE_CUDA

} // End namespace vecgeom
