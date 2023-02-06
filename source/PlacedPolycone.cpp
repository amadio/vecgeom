/*
 * PlacedPolycone.cpp
 *
 *  Created on: Dec 9, 2014
 *      Author: swenzel
 */

#include "VecGeom/volumes/SpecializedPolycone.h"
#include <iostream>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedPolycone::PrintType() const
{
  printf("PlacedPolycone");
}

void PlacedPolycone::PrintType(std::ostream &s) const
{
  s << "PlacedPolycone";
}

#ifndef VECCORE_CUDA
VPlacedVolume const *PlacedPolycone::ConvertToUnspecialized() const
{
  return new SimplePolycone(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#endif // ! VECCORE_CUDA
} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedPolycone, ConeTypes::UniversalCone)

#ifndef VECGEOM_NO_SPECIALIZATION
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedPolycone, ConeTypes::HollowCone)
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedPolycone, ConeTypes::HollowConeWithSmallerThanPiSector)
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedPolycone, ConeTypes::HollowConeWithBiggerThanPiSector)
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedPolycone, ConeTypes::HollowConeWithPiSector)
#endif

#endif // VECCORE_CUDA

} // namespace vecgeom
