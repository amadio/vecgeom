/// \file PlacedTrapezoid.cpp
/// \author Guilherme Lima (lima at fnal dot gov)

#include "VecGeom/volumes/PlacedTrapezoid.h"
#include "VecGeom/volumes/SpecializedTrapezoid.h"

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

#endif // VECCORE_CUDA

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedTrapezoid)
#endif

} // End namespace vecgeom
