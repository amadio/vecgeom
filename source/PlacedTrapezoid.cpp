/// \file PlacedTrapezoid.cpp
/// \author Guilherme Lima (lima at fnal dot gov)

#include "volumes/PlacedTrapezoid.h"
#include "volumes/SpecializedTrapezoid.h"
#ifdef VECGEOM_ROOT
#include "TGeoArb8.h"
#endif
#if defined(VECGEOM_USOLIDS) && !defined(VECGEOM_REPLACE_USOLIDS)
#include "UTrap.hh"
#endif
#ifdef VECGEOM_GEANT4
#include "G4Trap.hh"
#endif

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

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedTrapezoid::ConvertToRoot() const
{
  const UnplacedTrapezoid &ut = *GetUnplacedVolume();
  return new TGeoTrap(GetLabel().c_str(), ut.dz(), ut.theta() * kRadToDeg, ut.phi() * kRadToDeg, ut.dy1(), ut.dx1(),
                      ut.dx2(), ut.alpha1() * kRadToDeg, ut.dy2(), ut.dx3(), ut.dx4(), ut.alpha2() * kRadToDeg);
}
#endif

#if defined(VECGEOM_USOLIDS) && !defined(VECGEOM_REPLACE_USOLIDS)
::VUSolid const *PlacedTrapezoid::ConvertToUSolids() const
{
  const UnplacedTrapezoid &ut = *GetUnplacedVolume();
  //  const TrapezoidStruct<double> &t = GetUnplacedVolume()->GetStruct();
  return new UTrap(GetLabel().c_str(), ut.dz(), ut.theta(), ut.phi(), ut.dy1(), ut.dx1(), ut.dx2(), ut.alpha1(),
                   ut.dy2(), ut.dx3(), ut.dx4(), ut.alpha2());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedTrapezoid::ConvertToGeant4() const
{
  const UnplacedTrapezoid &ut = *GetUnplacedVolume();
  return new G4Trap(GetLabel().c_str(), ut.dz(), ut.theta(), ut.phi(), ut.dy1(), ut.dx1(), ut.dx2(), ut.alpha1(),
                    ut.dy2(), ut.dx3(), ut.dx4(), ut.alpha2());
}
#endif

#endif // VECCORE_CUDA

} // End impl namespace

#ifdef VECCORE_CUDA
VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedTrapezoid)
#endif

} // End namespace vecgeom
