//===-- volumes/UnplacedHype.h - Instruction class definition -------*- C++ -*-===//
//
//                     GeantV - VecGeom
//
// This file is distributed under the LGPL
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file volumes/UnplacedHype.h
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)
/// \brief This file contains the declaration of the UnplacedHype class
///
/// _____________________________________________________________________________
/// A Hype is the solid bounded by the following surfaces:
/// - 2 planes parallel with XY cutting the Z axis at Z=-dz and Z=+dz
/// - the surface of revolution of a parabola described by:
/// z = a*(x*x + y*y) + b
/// The parameters a and b are automatically computed from:
/// - rlo is the radius of the circle of intersection between the
/// parabolic surface and the plane z = -dz
/// - rhi is the radius of the circle of intersection between the
/// parabolic surface and the plane z = +dz
/// -dz = a*rlo^2 + b
/// dz = a*rhi^2 + b      where: rhi>rlo, both >= 0
///
/// note:
/// dd = 1./(rhi^2 - rlo^2);
/// a = 2.*dz*dd;
/// b = - dz * (rlo^2 + rhi^2)*dd;
///
/// in respect with the G4 implementation we have:
/// k1=1/a
/// k2=-b/a
///
/// a=1/k1
/// b=-k2/k1
//===----------------------------------------------------------------------===//

#ifndef VECGEOM_VOLUMES_UNPLACEDHYPE_H_
#define VECGEOM_VOLUMES_UNPLACEDHYPE_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"

namespace VECGEOM_NAMESPACE {

class UnplacedHype : public VUnplacedVolume, AlignedBase {

private:
    Precision fRmin;
    Precision fStIn;
    Precision fRmax;
    Precision fStOut;
    Precision fDz;
    
    //Precomputed Values
    Precision fTIn;     //Tangent of the Inner stereo angle
    Precision fTOut;    //Tangent of the Outer stereo angle
    Precision fTIn2;    //squared value of fTIn
    Precision fTOut2;   //squared value of fTOut
    

    
public:
    
    //constructor
    VECGEOM_CUDA_HEADER_BOTH
    UnplacedHype(const Precision rmin, const Precision stIn, const Precision rmax, const Precision stOut, const Precision dz);
    
    //get
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetRmin() const{ return fRmin;}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetRmax() const{ return fRmax;}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetStIn() const{ return fStIn;}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetStOut() const{ return fStOut;}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTIn() const{ return fTIn;}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTOut() const{ return fTOut;}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTIn2() const{ return fTIn2;}
    
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTOut2() const{ return fTOut2;}
    
    //set
    VECGEOM_CUDA_HEADER_BOTH
    void SetParameters(const Precision rMin, const Precision stIn, const Precision rMax, const Precision stOut, const Precision dz);

    virtual int memory_size() const { return sizeof(*this); }

//__________________________________________________________________
    

    VECGEOM_CUDA_HEADER_BOTH
    virtual void Print() const;
//__________________________________________________________________
    

    virtual void Print(std::ostream &os) const;
    
//__________________________________________________________________
    


    template <TranslationCode transCodeT, RotationCode rotCodeT>
    VECGEOM_CUDA_HEADER_DEVICE
    static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
  virtual VUnplacedVolume* CopyToGpu() const;
  virtual VUnplacedVolume* CopyToGpu(VUnplacedVolume *const gpu_ptr) const;
#endif

private:

  
    //Specialized Volume
  VECGEOM_CUDA_HEADER_DEVICE
  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
      const int id,
#endif
      VPlacedVolume *const placement = NULL) const;

};

} // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDHYPE_H_