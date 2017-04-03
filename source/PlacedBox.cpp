/// \file PlacedBox.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/PlacedBox.h"
#include "volumes/SpecializedBox.h"
#ifdef VECGEOM_ROOT
#include "TGeoBBox.h"
#endif
#if defined(VECGEOM_USOLIDS) and !defined(VECGEOM_REPLACE_USOLIDS)
#include "UBox.hh"
#endif
#ifdef VECGEOM_GEANT4
#include "G4Box.hh"
#endif

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedBox::PrintType() const
{
  printf("PlacedBox");
}

void PlacedBox::PrintType(std::ostream &s) const
{
  s << "PlacedBox";
}

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedBox::ConvertToUnspecialized() const
{
  return new SimpleBox(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedBox::ConvertToRoot() const
{
  return new TGeoBBox(GetLabel().c_str(), x(), y(), z());
}
#endif

#if defined(VECGEOM_USOLIDS) and !defined(VECGEOM_REPLACE_USOLIDS)
::VUSolid const *PlacedBox::ConvertToUSolids() const
{
  return new UBox(GetLabel(), x(), y(), z());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedBox::ConvertToGeant4() const
{
  return new G4Box("", x(), y(), z());
}
#endif

#endif // VECCORE_CUDA

} // End impl namespace

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedBox)

#endif // VECCORE_CUDA

} // End namespace vecgeom
