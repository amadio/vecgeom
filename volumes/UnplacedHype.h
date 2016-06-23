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
/// Hyperboloid class is defined by 5 parameters
/// A Hype is the solid bounded by the following surfaces:
/// - 2 planes parallel with XY cutting the Z axis at Z=-dz and Z=+dz
///- Inner and outer lateral surfaces. These represent the surfaces
/// described by the revolution of 2 hyperbolas about the Z axis:
/// r^2 - (t*z)^2 = a^2 where:
/// r = distance between hyperbola and Z axis at coordinate z
/// t = tangent of the stereo angle (angle made by hyperbola asimptotic lines and Z axis). t=0 means cylindrical
/// surface.
/// a = distance between hyperbola and Z axis at z=0
//===----------------------------------------------------------------------===//

#ifndef VECGEOM_VOLUMES_UNPLACEDHYPE_H_
#define VECGEOM_VOLUMES_UNPLACEDHYPE_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedHype;)
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedHype);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedHype : public VUnplacedVolume, public AlignedBase {

private:
  Precision fRmin;  // Inner radius
  Precision fStIn;  // Stereo angle for inner surface
  Precision fRmax;  // Outer radius
  Precision fStOut; // Stereo angle for outer surface
  Precision fDz;    // z-coordinate of the cutting planes

  // Precomputed Values
  Precision fTIn;      // Tangent of the Inner stereo angle
  Precision fTOut;     // Tangent of the Outer stereo angle
  Precision fTIn2;     // Squared value of fTIn
  Precision fTOut2;    // Squared value of fTOut
  Precision fTIn2Inv;  // Inverse of fTIn2
  Precision fTOut2Inv; // Inverse of fTOut2
  Precision fRmin2;    // Squared Inner radius
  Precision fRmax2;    // Squared Outer radius
  Precision fDz2;      // Squared z-coordinate

  Precision fEndInnerRadius2; // Squared endcap Inner Radius
  Precision fEndOuterRadius2; // Squared endcap Outer Radius
  Precision fEndInnerRadius;  // Endcap Inner Radius
  Precision fEndOuterRadius;  // Endcap Outer Radius

  Precision fInSqSide; // side of the square inscribed in the inner circle

  // Volume and Surface Area
  Precision fCubicVolume, fSurfaceArea;
  Precision zToleranceLevel;
  Precision innerRadToleranceLevel, outerRadToleranceLevel;

public:
  // constructor
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedHype(const Precision rMin, const Precision rMax, const Precision stIn, const Precision stOut,
               const Precision dz);

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetZToleranceLevel() const { return zToleranceLevel; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetInnerRadToleranceLevel() const { return innerRadToleranceLevel; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetOuterRadToleranceLevel() const { return outerRadToleranceLevel; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetRmin() const { return fRmin; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetRmax() const { return fRmax; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetRmin2() const { return fRmin2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetRmax2() const { return fRmax2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetStIn() const { return fStIn; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetStOut() const { return fStOut; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetTIn() const { return fTIn; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetTOut() const { return fTOut; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetTIn2() const { return fTIn2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetTOut2() const { return fTOut2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetTIn2Inv() const { return fTIn2Inv; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetTOut2Inv() const { return fTOut2Inv; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetDz() const { return fDz; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetDz2() const { return fDz2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetEndInnerRadius() const { return fEndInnerRadius; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetEndInnerRadius2() const { return fEndInnerRadius2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetEndOuterRadius() const { return fEndOuterRadius; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetEndOuterRadius2() const { return fEndOuterRadius2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetInSqSide() const { return fInSqSide; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void SetParameters(const Precision rMin, const Precision rMax, const Precision stIn, const Precision stOut,
                     const Precision dz);

  VECGEOM_CUDA_HEADER_BOTH
  Precision Volume(bool outer);

  VECGEOM_CUDA_HEADER_BOTH
  Precision Area(bool outer);

  VECGEOM_CUDA_HEADER_BOTH
  Precision AreaEndCaps();

  VECGEOM_CUDA_HEADER_BOTH
  void CalcCapacity();

  VECGEOM_CUDA_HEADER_BOTH
  void CalcSurfaceArea();

  VECGEOM_CUDA_HEADER_BOTH
  void DetectConvexity();

  VECGEOM_CUDA_HEADER_BOTH
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Capacity() const { return fCubicVolume; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision SurfaceArea() const { return fSurfaceArea; }

  Vector3D<Precision> GetPointOnSurface() const override;

  // VECGEOM_CUDA_HEADER_BOTH
  std::string GetEntityType() const;

  VECGEOM_CUDA_HEADER_BOTH
  void GetParametersList(int aNumber, Precision *aArray) const;

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedHype *Clone() const;

  VECGEOM_CUDA_HEADER_BOTH
  std::ostream &StreamInfo(std::ostream &os) const;

  VECGEOM_CUDA_HEADER_BOTH
  void ComputeBBox() const;

  VECGEOM_CUDA_HEADER_BOTH
  bool InnerSurfaceExists() const;

  virtual int memory_size() const override { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const override;

  virtual void Print(std::ostream &os) const override;

#ifndef VECGEOM_NVCC
  template <TranslationCode trans_code, RotationCode rot_code>
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               VPlacedVolume *const placement = NULL);

  static VPlacedVolume *CreateSpecializedVolume(LogicalVolume const *const volume,
                                                Transformation3D const *const transformation,
                                                const TranslationCode trans_code, const RotationCode rot_code,
                                                VPlacedVolume *const placement = NULL);

#else

  template <TranslationCode trans_code, RotationCode rot_code>
  __device__
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               const int id, VPlacedVolume *const placement = NULL);

  __device__ static VPlacedVolume *CreateSpecializedVolume(LogicalVolume const *const volume,
                                                           Transformation3D const *const transformation,
                                                           const TranslationCode trans_code,
                                                           const RotationCode rot_code, const int id,
                                                           VPlacedVolume *const placement = NULL);

#endif

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedHype>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

private:
#ifndef VECGEOM_NVCC
  VPlacedVolume *SpecializedVolume(LogicalVolume const *const lvolume, Transformation3D const *const transformation,
                                   const TranslationCode trans_code, const RotationCode rot_code,
                                   VPlacedVolume *const placement = NULL) const override
  {
    return CreateSpecializedVolume(lvolume, transformation, trans_code, rot_code, placement);
  }

#else
  __device__ VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                              Transformation3D const *const transformation,
                                              const TranslationCode trans_code, const RotationCode rot_code,
                                              const int id, VPlacedVolume *const placement = NULL) const override
  {
    return CreateSpecializedVolume(volume, transformation, trans_code, rot_code, id, placement);
  }

#endif
};
}
} // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDHYPE_H_
