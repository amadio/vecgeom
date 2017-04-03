/// \file PlacedTube.cpp
/// \author Georgios Bitzes (georgios.bitzes@cern.ch)

#include "volumes/PlacedTube.h"
#include "volumes/Tube.h"
#include "volumes/SpecializedTube.h"
#include "base/Vector3D.h"

#ifdef VECGEOM_ROOT
#include "TGeoTube.h"
#endif

#ifdef VECGEOM_USOLIDS
#include "UTubs.hh"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Tubs.hh"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedTube::ConvertToUnspecialized() const
{
  return new SimpleTube(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedTube::ConvertToRoot() const
{
  UnplacedTube const *t = static_cast<UnplacedTube const *>(GetUnplacedVolume());
  if (t->dphi() >= 2 * M_PI) return new TGeoTube(GetLabel().c_str(), t->rmin(), t->rmax(), t->z());
  return new TGeoTubeSeg(GetLabel().c_str(), t->rmin(), t->rmax(), t->z(), t->sphi() * (180 / M_PI),
                         (t->sphi() + t->dphi()) * (180 / M_PI));
}
#endif

#if defined(VECGEOM_USOLIDS) && !defined(VECGEOM_REPLACE_USOLIDS)
::VUSolid const *PlacedTube::ConvertToUSolids() const
{
  UnplacedTube const *t = static_cast<UnplacedTube const *>(GetUnplacedVolume());
  return new UTubs(GetLabel().c_str(), t->rmin(), t->rmax(), t->z(), t->sphi(), t->dphi());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedTube::ConvertToGeant4() const
{
  UnplacedTube const *t = static_cast<UnplacedTube const *>(GetUnplacedVolume());
  return new G4Tubs(GetLabel().c_str(), t->rmin(), t->rmax(), t->z(), t->sphi(), t->dphi());
}
#endif

#endif // VECCORE_CUDA

} // End impl namespace

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedTube, TubeTypes::UniversalTube)

#ifndef VECGEOM_NO_SPECIALIZATION
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedTube, TubeTypes::NonHollowTube)
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedTube, TubeTypes::NonHollowTubeWithSmallerThanPiSector)
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedTube, TubeTypes::NonHollowTubeWithBiggerThanPiSector)
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedTube, TubeTypes::NonHollowTubeWithPiSector)

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedTube, TubeTypes::HollowTube)
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedTube, TubeTypes::HollowTubeWithSmallerThanPiSector)
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedTube, TubeTypes::HollowTubeWithBiggerThanPiSector)
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(SpecializedTube, TubeTypes::HollowTubeWithPiSector)
#endif

#endif

} // End global namespace
