/*
 * PlacedCone.cpp
 *
 *  Created on: Jun 13, 2014
 *      Author: swenzel
 */

#include "VecGeom/volumes/PlacedCone.h"
#include "VecGeom/volumes/Cone.h"
#include "VecGeom/volumes/SpecializedCone.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedCone::PrintType() const
{
  printf("PlacedCone");
}

void PlacedCone::PrintType(std::ostream &s) const
{
  s << "PlacedCone";
}

#ifndef VECCORE_CUDA
VPlacedVolume const *PlacedCone::ConvertToUnspecialized() const
{
  return new SimpleCone(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#endif

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedCone, ConeTypes::UniversalCone)

#endif // VECCORE_CUDA

} // End namespace vecgeom
