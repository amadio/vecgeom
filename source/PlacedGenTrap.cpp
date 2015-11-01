/*
 * PlacedGenTrap.cpp
 *
 *  Created on: Aug 3, 2014
 *      Author: swenzel
 */

#include "volumes/GenTrap.h"


#if defined(VECGEOM_BENCHMARK) && defined(VECGEOM_ROOT)
#include "TGeoArb8.h"
#endif

#if defined(VECGEOM_BENCHMARK) && defined(VECGEOM_USOLIDS)
#include "UGenericTrap.hh"
#include "UVector2.hh"
#endif

#if defined(VECGEOM_BENCHMARK) && defined(VECGEOM_GEANT4)
#include "G4GenericTrap.hh"
#include "G4TwoVector.hh"
#endif

namespace VECGEOM_NAMESPACE {

#ifdef VECGEOM_BENCHMARK

VPlacedVolume const* PlacedGenTrap::ConvertToUnspecialized() const {
  return new SimpleGenTrap(GetLabel().c_str(), logical_volume(),
                                  transformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedGenTrap::ConvertToRoot() const {
    double vertexarray[16];
    for(int i=0;i<8;++i)
    {
        vertexarray[2*i]=GetVertex(i).x();
        vertexarray[2*i+1]=GetVertex(i).y();
    }
    return new TGeoArb8(GetLabel().c_str(), GetDZ(), &vertexarray[0]);
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedGenTrap::ConvertToUSolids() const {
  std::vector<UVector2> vertices;
  for(int i=0;i<8;++i) {
    vertices.push_back( UVector2( GetVertex(i).x(), GetVertex(i).y() ) );
  }
  return new UGenericTrap(GetLabel(), GetDZ(), vertices);
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedGenTrap::ConvertToGeant4() const {
  std::vector<G4TwoVector> vertices;
  for(int i=0;i<8;++i) {
    vertices.push_back( G4TwoVector( GetVertex(i).x(), GetVertex(i).y() ) );
  }
  return new G4GenericTrap(GetLabel(), GetDZ(), vertices);
}
#endif

#endif // VECGEOM_BENCHMARK

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void PlacedGenTrap_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr);

VPlacedVolume* PlacedGenTrap::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const gpu_ptr) const {
  PlacedGenTrap_CopyToGpu(logical_volume, transformation, this->id(),
                                 gpu_ptr);
  CudaAssertError();
  return gpu_ptr;
}

VPlacedVolume* PlacedGenTrap::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation) const {
  VPlacedVolume *const gpu_ptr = vecgeom::AllocateOnGpu<PlacedGenTrap>();
  return this->CopyToGpu(logical_volume, transformation, gpu_ptr);
}

#endif // VECGEOM_CUDA_INTERFACE

#ifdef VECGEOM_NVCC

class LogicalVolume;
class Transformation3D;
class VPlacedVolume;

__global__
void PlacedGenTrap_ConstructOnGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::SimpleParallelepiped(
    reinterpret_cast<vecgeom_cuda::LogicalVolume const*>(logical_volume),
    reinterpret_cast<vecgeom_cuda::Transformation3D const*>(transformation),
    NULL,
    id
  );
}

void PlacedGenTrap_CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    const int id, VPlacedVolume *const gpu_ptr) {
  PlacedGenTrap_ConstructOnGpu<<<1, 1>>>(logical_volume, transformation,
                                                id, gpu_ptr);
}

#endif // VECGEOM_NVCC

} // End namespace vecgeom


