/// \file PlacedHype.cpp
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)

#include "VecGeom/volumes/PlacedHype.h"
#include "VecGeom/volumes/SpecializedHype.h"
#include "VecGeom/volumes/Hype.h"
#include "VecGeom/base/Global.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedHype::PrintType() const
{
  printf("PlacedHype");
}

void PlacedHype::PrintType(std::ostream &s) const
{
  s << "PlacedHype";
}

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedHype::ConvertToUnspecialized() const
{
  return new SimpleHype(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#endif // VECGEOM_CUDA

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedHype, HypeTypes::UniversalHype)

#ifndef VECGEOM_NO_SPECIALIZATION
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedHype, HypeTypes::NonHollowHype)
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedHype, HypeTypes::HollowHype)
#endif

#endif // VECCORE_CUDA
} // End namespace vecgeom
