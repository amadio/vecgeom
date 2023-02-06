/// @file PlacedExtruded.cpp
/// @author Mihaela Gheata (mihaela.gheata@cern.ch)

#include "VecGeom/volumes/Extruded.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedExtruded::PrintType() const
{
  printf("PlacedExtruded");
}

void PlacedExtruded::PrintType(std::ostream &s) const
{
  s << "PlacedExtruded";
}

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedExtruded::ConvertToUnspecialized() const
{
  return new SimpleExtruded(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#endif // VECCORE_CUDA

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedExtruded)

#endif

} // namespace vecgeom
