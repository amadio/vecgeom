// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// @file source/PlacedTrd.cpp
/// @author Georgios Bitzes

#include "VecGeom/volumes/Trd.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedTrd::PrintType() const
{
  printf("PlacedTrd");
}

void PlacedTrd::PrintType(std::ostream &s) const
{
  s << "PlacedTrd";
}

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedTrd::ConvertToUnspecialized() const
{
  return new SimpleTrd(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

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
