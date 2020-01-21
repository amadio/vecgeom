/// \file PlacedPolyhedron.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "VecGeom/volumes/PlacedPolyhedron.h"

#include "VecGeom/volumes/kernel/GenericKernels.h"
#include "VecGeom/volumes/SpecializedPolyhedron.h"

#ifdef VECGEOM_ROOT
#include "TGeoPgon.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Polyhedra.hh"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
int PlacedPolyhedron::PhiSegmentIndex(Vector3D<Precision> const &point) const
{
  Vector3D<Precision> localPoint = VPlacedVolume::GetTransformation()->Transform(point);
  return PolyhedronImplementation<Polyhedron::EInnerRadii::kGeneric, Polyhedron::EPhiCutout::kGeneric>::FindPhiSegment<
      Precision>(GetUnplacedVolume()->GetStruct(), localPoint);
}

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedPolyhedron::ConvertToUnspecialized() const
{
  return new SimplePolyhedron(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedPolyhedron::ConvertToRoot() const
{

  const int zPlaneCount = GetZSegmentCount() + 1;

  TGeoPgon *pgon = new TGeoPgon(GetLabel().c_str(), GetPhiStart() * kRadToDeg, GetPhiDelta() * kRadToDeg,
                                GetSideCount(), zPlaneCount);

  // Define sections of TGeoPgon. It takes care of the rest internally once the
  // last section is set.
  for (int i = 0; i < zPlaneCount; ++i) {
    pgon->DefineSection(i, GetZPlanes()[i], GetRMin()[i], GetRMax()[i]);
  }

  return pgon;
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedPolyhedron::ConvertToGeant4() const
{

  return new G4Polyhedra(GetLabel().c_str(), GetPhiStart(), GetPhiDelta(), GetSideCount(), GetZSegmentCount() + 1,
                         &GetZPlanes()[0], &GetRMin()[0], &GetRMax()[0]);
}
#endif
#endif // !VECCORE_CUDA

} // End inline namespace

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_4(SpecializedPolyhedron)

#endif

} // End namespace vecgeom
