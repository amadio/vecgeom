//===-- volumes/PlacedHype.h - Instruction class definition -------*- C++ -*-===//
//
//                     GeantV - VecGeom
//
// This file is distributed under the LGPL
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file volumes/PlacedHype.h 
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)
/// \brief This file contains the declaration of the PlacedHype class
//===----------------------------------------------------------------------===//


#ifndef VECGEOM_VOLUMES_PLACEDHYPE_H_
#define VECGEOM_VOLUMES_PLACEDHYPE_H_

#include "base/Global.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedHype.h"

namespace VECGEOM_NAMESPACE {

class PlacedHype : public VPlacedVolume {

public:

  typedef UnplacedHype UnplacedShape_t;

#ifndef VECGEOM_NVCC

  PlacedHype(char const *const label,
                   LogicalVolume const *const logical_volume,
                   Transformation3D const *const transformation,
                   PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox) {}

  PlacedHype(LogicalVolume const *const logical_volume,
                   Transformation3D const *const transformation,
                   PlacedBox const *const boundingBox)
      : PlacedHype("", logical_volume, transformation, boundingBox) {}

#else

  __device__
  PlacedHype(LogicalVolume const *const logical_volume,
                   Transformation3D const *const transformation,
                   PlacedBox const *const boundingBox, const int id)
      : VPlacedVolume(logical_volume, transformation, boundingBox, id) {}

#endif

    virtual ~PlacedHype() {}
    VECGEOM_CUDA_HEADER_BOTH
    UnplacedHype const* GetUnplacedVolume() const {
        return static_cast<UnplacedHype const *>(
        logical_volume()->unplaced_volume());
    }
    
    VECGEOM_CUDA_HEADER_BOTH
    UnplacedHype * GetUnplacedVolumeNonConst() const {
        return static_cast<UnplacedHype *>(const_cast<VUnplacedVolume *>(
            logical_volume()->unplaced_volume()));
    }
    
    //get
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetRmin() const{ return GetUnplacedVolume()->GetRmin();}
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetRmax() const{ return GetUnplacedVolume()->GetRmax();}
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetRmin2() const{ return GetUnplacedVolume()->GetRmin2();}
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetRmax2() const{ return GetUnplacedVolume()->GetRmax2();}
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetStIn() const{ return GetUnplacedVolume()->GetStIn();}
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetStOut() const{ return GetUnplacedVolume()->GetStOut();}
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTIn() const{ return GetUnplacedVolume()->GetTIn();}
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTOut() const{ return GetUnplacedVolume()->GetTOut();}
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTIn2() const{ return GetUnplacedVolume()->GetTIn2();}
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTOut2() const{ return GetUnplacedVolume()->GetTOut2();}
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetDz() const{ return GetUnplacedVolume()->GetDz();}
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetDz2() const{ return GetUnplacedVolume()->GetDz2();}
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetEndInnerRadius() const{ return GetUnplacedVolume()->GetEndInnerRadius();}
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetEndInnerRadius2() const{ return GetUnplacedVolume()->GetEndInnerRadius2();}
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetEndOuterRadius() const{ return GetUnplacedVolume()->GetEndOuterRadius();}
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetEndOuterRadius2() const{ return GetUnplacedVolume()->GetEndOuterRadius2();}
    

    
#ifdef VECGEOM_BENCHMARK
    virtual VPlacedVolume const* ConvertToUnspecialized() const;
#ifdef VECGEOM_ROOT
    virtual TGeoShape const* ConvertToRoot() const;
#endif
#ifdef VECGEOM_USOLIDS
    virtual ::VUSolid const* ConvertToUSolids() const;
#endif
#endif // VECGEOM_BENCHMARK

#ifdef VECGEOM_CUDA_INTERFACE
    virtual VPlacedVolume* CopyToGpu(LogicalVolume const *const logical_volume,
                                   Transformation3D const *const transformation,
                                   VPlacedVolume *const gpu_ptr) const;
    virtual VPlacedVolume* CopyToGpu(
      LogicalVolume const *const logical_volume,
      Transformation3D const *const transformation) const;
#endif

};

} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDHYPE_H_
