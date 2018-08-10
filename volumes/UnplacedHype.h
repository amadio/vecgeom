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
VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(class, SUnplacedHype, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedHype : public VUnplacedVolume {

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
                     const Precision dz)
  {
    fHype.SetParameters(rMin, rMax, stIn, stOut, dz);
    DetectConvexity();
  }

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

  Precision Capacity() const override { return fHype.fCubicVolume; }

  // VECCORE_ATT_HOST_DEVICE
  Precision SurfaceArea() const override { return fHype.fSurfaceArea; }

  template <bool ForInnerSurface>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  Precision GetHypeRadius2(Precision dz) const
  {
    if (ForInnerSurface)
      return GetRmin2() + GetTIn2() * dz * dz;
    else
      return GetRmax2() + GetTOut2() * dz * dz;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool PointOnZSurface(Vector3D<Precision> const &p) const
  {
    return (p.z() > (GetDz() - GetZToleranceLevel())) && (p.z() < (GetDz() + GetZToleranceLevel()));
  }

  template <bool ForInnerSurface>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool PointOnHyperbolicSurface(Vector3D<Precision> const &p) const
  {
    Precision hypeR2    = 0.;
    hypeR2              = GetHypeRadius2<ForInnerSurface>(p.z());
    Precision pointRad2 = p.Perp2();
    return ((pointRad2 > (hypeR2 - GetOuterRadToleranceLevel())) &&
            (pointRad2 < (hypeR2 + GetOuterRadToleranceLevel())));
  }

  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const override
  {

    bool valid = true;

    Precision absZ(std::fabs(p.z()));
    Precision distZ(absZ - GetDz());
    Precision dist2Z(distZ * distZ);

    Precision xR2 = p.Perp2();
    Precision dist2Outer(std::fabs(xR2 - GetHypeRadius2<false>(absZ)));
    Precision dist2Inner(std::fabs(xR2 - GetHypeRadius2<true>(absZ)));

    // EndCap
    if (PointOnZSurface(p) || ((dist2Z < dist2Inner) && (dist2Z < dist2Outer)))
      normal = Vector3D<Precision>(0.0, 0.0, p.z() < 0 ? -1.0 : 1.0);

    // OuterHyperbolic Surface
    if (PointOnHyperbolicSurface<false>(p) || ((dist2Outer < dist2Inner) && (dist2Outer < dist2Z)))
      normal = Vector3D<Precision>(p.x(), p.y(), -p.z() * GetTOut2()).Unit();

    // InnerHyperbolic Surface
    if (PointOnHyperbolicSurface<true>(p) || ((dist2Inner < dist2Outer) && (dist2Inner < dist2Z)))
      normal = Vector3D<Precision>(-p.x(), -p.y(), p.z() * GetTIn2()).Unit();

    return valid;
  }

  Vector3D<Precision> SamplePointOnSurface() const override;

  std::string GetEntityType() const;

  VECCORE_ATT_HOST_DEVICE
  void GetParametersList(int aNumber, Precision *aArray) const;

  VECCORE_ATT_HOST_DEVICE
  UnplacedHype *Clone() const;

  std::ostream &StreamInfo(std::ostream &os) const;

  VECCORE_ATT_HOST_DEVICE
  void ComputeBBox() const;

  VECCORE_ATT_HOST_DEVICE
  bool InnerSurfaceExists() const;

  virtual int MemorySize() const override { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override;

  virtual void Print(std::ostream &os) const override;

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override
  {
    return DevicePtr<cuda::SUnplacedHype<cuda::HypeTypes::UniversalHype>>::SizeOf();
  }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

#ifndef VECCORE_CUDA
#ifdef VECGEOM_ROOT
  TGeoShape const *ConvertToRoot(char const *label) const;
#endif

#ifdef VECGEOM_GEANT4
  G4VSolid const *ConvertToGeant4(char const *label) const;
#endif
#endif
};

template <>
struct Maker<UnplacedHype> {
  template <typename... ArgTypes>
  static UnplacedHype *MakeInstance(const Precision rMin, const Precision rMax, const Precision stIn,
                                    const Precision stOut, const Precision dz);
};

template <typename HypeType = HypeTypes::UniversalHype>
class SUnplacedHype : public SIMDUnplacedVolumeImplHelper<HypeImplementation<HypeType>, UnplacedHype>,
                      public AlignedBase {
public:
  using BaseType_t = SIMDUnplacedVolumeImplHelper<HypeImplementation<HypeType>, UnplacedHype>;
  using BaseType_t::BaseType_t;

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

private:
#ifndef VECCORE_CUDA
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           const TranslationCode trans_code, const RotationCode rot_code,
                                           VPlacedVolume *const placement = NULL) const override
  {
    return VolumeFactory::CreateByTransformation<SUnplacedHype<HypeType>>(volume, transformation, trans_code, rot_code,
                                                                          placement);
  }

#else
  VECCORE_ATT_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           const TranslationCode trans_code, const RotationCode rot_code, const int id,
                                           VPlacedVolume *const placement = NULL) const override
  {
    return VolumeFactory::CreateByTransformation<SUnplacedHype<HypeType>>(volume, transformation, trans_code, rot_code,
                                                                          id, placement);
  }
#endif
};

using GenericUnplacedHype = SUnplacedHype<HypeTypes::UniversalHype>;

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#include "volumes/SpecializedHype.h"

#endif // VECGEOM_VOLUMES_UNPLACEDHYPE_H_
