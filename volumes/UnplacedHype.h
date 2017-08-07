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
#include "volumes/HypeStruct.h"
#include "volumes/kernel/HypeImplementation.h"
#include "volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedHype;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedHype);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedHype : public SIMDUnplacedVolumeImplHelper<HypeImplementation>, public AlignedBase {

private:
  HypeStruct<Precision> fHype;

public:
  // constructor
  VECCORE_ATT_HOST_DEVICE
  UnplacedHype(const Precision rMin, const Precision rMax, const Precision stIn, const Precision stOut,
               const Precision dz)
      : fHype(rMin, rMax, stIn, stOut, dz)
  {

    DetectConvexity();
  }

  VECCORE_ATT_HOST_DEVICE
  HypeStruct<double> const &GetStruct() const { return fHype; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetZToleranceLevel() const { return fHype.zToleranceLevel; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetInnerRadToleranceLevel() const { return fHype.innerRadToleranceLevel; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetOuterRadToleranceLevel() const { return fHype.outerRadToleranceLevel; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRmin() const { return fHype.fRmin; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRmax() const { return fHype.fRmax; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRmin2() const { return fHype.fRmin2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRmax2() const { return fHype.fRmax2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetStIn() const { return fHype.fStIn; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetStOut() const { return fHype.fStOut; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTIn() const { return fHype.fTIn; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTOut() const { return fHype.fTOut; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTIn2() const { return fHype.fTIn2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTOut2() const { return fHype.fTOut2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTIn2Inv() const { return fHype.fTIn2Inv; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTOut2Inv() const { return fHype.fTOut2Inv; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDz() const { return fHype.fDz; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDz2() const { return fHype.fDz2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetEndInnerRadius() const { return fHype.fEndInnerRadius; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetEndInnerRadius2() const { return fHype.fEndInnerRadius2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetEndOuterRadius() const { return fHype.fEndOuterRadius; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetEndOuterRadius2() const { return fHype.fEndOuterRadius2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetInSqSide() const { return fHype.fInSqSide; }

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
  Precision Capacity() const { return fHype.fCubicVolume; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision SurfaceArea() const { return fHype.fSurfaceArea; }

  Vector3D<Precision> SamplePointOnSurface() const override;

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

  VECCORE_ATT_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECCORE_CUDA
                                           const int id,
#endif
                                           VPlacedVolume *const placement = NULL) const final;

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedHype>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif
};
}
} // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDHYPE_H_
