/*
 * PlacedGenTrap.cpp
 *
 *  Created on: Aug 3, 2014
 *      Author: swenzel
 */

#include "volumes/GenTrap.h"

#ifdef VECGEOM_ROOT
#include "TGeoArb8.h"
#endif

#ifdef VECGEOM_USOLIDS
#include "UGenericTrap.hh"
#include "UVector2.hh"
#endif

#ifdef VECGEOM_GEANT4
#include "G4GenericTrap.hh"
#include "G4TwoVector.hh"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH

void PlacedGenTrap::PrintType() const { printf("PlacedGenTrap"); }

#ifndef VECGEOM_NVCC

VPlacedVolume const *PlacedGenTrap::ConvertToUnspecialized() const {
  return new SimpleGenTrap(GetLabel().c_str(), logical_volume_, GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedGenTrap::ConvertToRoot() const {
  double vertexarray[16];
  for (int i = 0; i < 8; ++i) {
    vertexarray[2 * i] = GetVertex(i).x();
    vertexarray[2 * i + 1] = GetVertex(i).y();
  }
  return new TGeoArb8(GetLabel().c_str(), GetDZ(), &vertexarray[0]);
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const *PlacedGenTrap::ConvertToUSolids() const {
  std::vector<UVector2> vertices;
  for (int i = 0; i < 8; ++i) {
    vertices.push_back(UVector2(GetVertex(i).x(), GetVertex(i).y()));
  }
  return new UGenericTrap(GetLabel(), GetDZ(), vertices);
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedGenTrap::ConvertToGeant4() const {
  std::vector<G4TwoVector> vertices;
  for (int i = 0; i < 8; ++i) {
    vertices.push_back(G4TwoVector(GetVertex(i).x(), GetVertex(i).y()));
  }
  return new G4GenericTrap(GetLabel(), GetDZ(), vertices);
}
#endif

#endif // VECGEOM_NVCC

} // End impl namespace

#ifdef VECGEOM_NVCC

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedGenTrap)

#endif // VECGEOM_NVCC

} // End namespace vecgeom
