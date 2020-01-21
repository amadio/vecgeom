/*
 * PlacedCutTube.cpp
 *
 *  Created on: 03.11.2016
 *      Author: mgheata
 */
#include "VecGeom/volumes/PlacedCutTube.h"
#include "VecGeom/volumes/SpecializedCutTube.h"
#include "VecGeom/base/Vector3D.h"

#ifdef VECGEOM_ROOT
#include "TGeoTube.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4CutTubs.hh"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedCutTube::ConvertToUnspecialized() const
{
  return new SimpleCutTube(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedCutTube::ConvertToRoot() const
{
  UnplacedCutTube const *t = static_cast<UnplacedCutTube const *>(GetUnplacedVolume());
  return new TGeoCtub(GetLabel().c_str(), t->rmin(), t->rmax(), t->z(), t->sphi() * (180 / M_PI),
                      (t->sphi() + t->dphi()) * (180 / M_PI), t->BottomNormal().x(), t->BottomNormal().y(),
                      t->BottomNormal().z(), t->TopNormal().x(), t->TopNormal().y(), t->TopNormal().z());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedCutTube::ConvertToGeant4() const
{
  UnplacedCutTube const *t = static_cast<UnplacedCutTube const *>(GetUnplacedVolume());
  G4ThreeVector pLowNorm(t->BottomNormal().x(), t->BottomNormal().y(), t->BottomNormal().z());
  G4ThreeVector pHighNorm(t->TopNormal().x(), t->TopNormal().y(), t->TopNormal().z());
  return new G4CutTubs(GetLabel().c_str(), t->rmin(), t->rmax(), t->z(), t->sphi(), t->dphi(), pLowNorm, pHighNorm);
}
#endif

#endif // VECCORE_CUDA

} // End impl namespace

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedCutTube)

#endif // VECCORE_CUDA

} // End global namespace
