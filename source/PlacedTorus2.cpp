/// \file PlacedTorus2.cpp
#include "VecGeom/volumes/PlacedTorus2.h"
#include "VecGeom/volumes/Torus2.h"
#include "VecGeom/volumes/SpecializedTorus2.h"

#ifdef VECGEOM_ROOT
#include "TGeoTorus.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Torus.hh"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void PlacedTorus2::PrintType() const
{
  printf("PlacedTorus2");
}

void PlacedTorus2::PrintType(std::ostream &s) const
{
  s << "PlacedTorus2";
}

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedTorus2::ConvertToUnspecialized() const
{
  return new SimpleTorus2(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedTorus2::ConvertToRoot() const
{
  const UnplacedTorus2 &ut = *(static_cast<UnplacedTorus2 const *>(GetUnplacedVolume()));
  return new TGeoTorus(GetLabel().c_str(), ut.rtor(), ut.rmin(), ut.rmax(), ut.sphi() * kRadToDeg,
                       ut.dphi() * kRadToDeg);
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedTorus2::ConvertToGeant4() const
{
  const UnplacedTorus2 &ut = *(static_cast<UnplacedTorus2 const *>(GetUnplacedVolume()));
  return new G4Torus(GetLabel().c_str(), ut.rmin(), ut.rmax(), ut.rtor(), ut.sphi(), ut.dphi());
}
#endif

#endif // VECCORE_CUDA

} // End impl namespace

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedTorus2)

#endif

} // namespace vecgeom
