// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// @file source/PlacedEllipsoid.cpp
/// @author Evgueni Tcherniaev

#include "VecGeom/volumes/PlacedEllipsoid.h"
#include "VecGeom/volumes/SpecializedEllipsoid.h"

#ifdef VECGEOM_GEANT4
#include "G4Ellipsoid.hh"
#endif

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedEllipsoid::PrintType() const
{
  printf("PlacedEllipsoid");
}

void PlacedEllipsoid::PrintType(std::ostream &s) const
{
  s << "PlacedEllipsoid";
}

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedEllipsoid::ConvertToUnspecialized() const
{
  return new SimpleEllipsoid(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedEllipsoid::ConvertToGeant4() const
{
  return new G4Ellipsoid("", GetDx(), GetDy(), GetDz(), GetZBottomCut(), GetZTopCut());
}
#endif

#endif // VECCORE_CUDA

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedEllipsoid)

#endif // VECCORE_CUDA

} // namespace vecgeom
