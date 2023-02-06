// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// @file source/PlacedTet.cpp
/// @author Raman Sehgal, Evgueni Tcherniaev

#include "VecGeom/volumes/PlacedTet.h"
#include "VecGeom/volumes/SpecializedTet.h"
#ifdef VECGEOM_GEANT4
#include "G4Tet.hh"
#include "G4ThreeVector.hh"
#endif

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedTet::PrintType() const
{
  printf("PlacedTet");
}

void PlacedTet::PrintType(std::ostream &s) const
{
  s << "PlacedTet";
}

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedTet::ConvertToUnspecialized() const
{
  return new SimpleTet(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedTet::ConvertToGeant4() const
{
  Vector3D<Precision> p0, p1, p2, p3;
  GetVertices(p0, p1, p2, p3);
  return new G4Tet("", G4ThreeVector(p0.x(), p0.y(), p0.z()), G4ThreeVector(p1.x(), p1.y(), p1.z()),
                   G4ThreeVector(p2.x(), p2.y(), p2.z()), G4ThreeVector(p3.x(), p3.y(), p3.z()));
}
#endif

#endif // VECCORE_CUDA

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedTet)

#endif // VECCORE_CUDA

} // End namespace vecgeom
