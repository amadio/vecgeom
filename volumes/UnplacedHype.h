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

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedHype;);
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
  VECCORE_ATT_HOST_DEVICE
  UnplacedHype(const Precision rMin, const Precision rMax, const Precision stIn, const Precision stOut,
               const Precision dz);

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetZToleranceLevel() const { return zToleranceLevel; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetInnerRadToleranceLevel() const { return innerRadToleranceLevel; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetOuterRadToleranceLevel() const { return outerRadToleranceLevel; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRmin() const { return fRmin; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRmax() const { return fRmax; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRmin2() const { return fRmin2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRmax2() const { return fRmax2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetStIn() const { return fStIn; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetStOut() const { return fStOut; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTIn() const { return fTIn; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTOut() const { return fTOut; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTIn2() const { return fTIn2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTOut2() const { return fTOut2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTIn2Inv() const { return fTIn2Inv; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTOut2Inv() const { return fTOut2Inv; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDz() const { return fDz; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDz2() const { return fDz2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetEndInnerRadius() const { return fEndInnerRadius; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetEndInnerRadius2() const { return fEndInnerRadius2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetEndOuterRadius() const { return fEndOuterRadius; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetEndOuterRadius2() const { return fEndOuterRadius2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetInSqSide() const { return fInSqSide; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetParameters(const Precision rMin, const Precision rMax, const Precision stIn, const Precision stOut,
                     const Precision dz);

  VECCORE_ATT_HOST_DEVICE
  Precision Volume(bool outer);

  VECCORE_ATT_HOST_DEVICE
  Precision Area(bool outer);

  VECCORE_ATT_HOST_DEVICE
  Precision AreaEndCaps();

  VECCORE_ATT_HOST_DEVICE
  void CalcCapacity();

  VECCORE_ATT_HOST_DEVICE
  void CalcSurfaceArea();

  VECCORE_ATT_HOST_DEVICE
  void DetectConvexity();

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision Capacity() const { return fCubicVolume; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision SurfaceArea() const { return fSurfaceArea; }

  Vector3D<Precision> GetPointOnSurface() const override;

  // VECCORE_ATT_HOST_DEVICE
  std::string GetEntityType() const;

  VECCORE_ATT_HOST_DEVICE
  void GetParametersList(int aNumber, Precision *aArray) const;

  VECCORE_ATT_HOST_DEVICE
  UnplacedHype *Clone() const;

  VECCORE_ATT_HOST_DEVICE
  std::ostream &StreamInfo(std::ostream &os) const;

  VECCORE_ATT_HOST_DEVICE
  void ComputeBBox() const;

  VECCORE_ATT_HOST_DEVICE
  bool InnerSurfaceExists() const;

  virtual int MemorySize() const override { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override;

  virtual void Print(std::ostream &os) const override;

#ifndef VECCORE_CUDA
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
#ifndef VECCORE_CUDA
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
