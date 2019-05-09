/// @file UnplacedEllipticalCone.h
/// @author Raman Sehgal (raman.sehgal@cern.ch), Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDELLIPTICALCONE_H_
#define VECGEOM_VOLUMES_UNPLACEDELLIPTICALCONE_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/EllipticalConeStruct.h" // the pure EllipticalCone struct
#include "volumes/kernel/EllipticalConeImplementation.h"
#include "volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedEllipticalCone;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedEllipticalCone);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedEllipticalCone : public SIMDUnplacedVolumeImplHelper<EllipticalConeImplementation>, public AlignedBase {

private:
  EllipticalConeStruct<Precision> fEllipticalCone;

private:
  void CheckParameters();

  Vector3D<Precision> SamplePointOnLateralSurface() const;

public:
  VECCORE_ATT_HOST_DEVICE
  UnplacedEllipticalCone(Precision a, Precision b, Precision h, Precision zcut);

  VECCORE_ATT_HOST_DEVICE
  EllipticalConeStruct<Precision> const &GetStruct() const { return fEllipticalCone; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSemiAxisX() const { return fEllipticalCone.fDx; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSemiAxisY() const { return fEllipticalCone.fDy; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetZMax() const { return fEllipticalCone.fDz; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetZTopCut() const { return fEllipticalCone.fZCut; }

  void SetParameters(Precision a, Precision b, Precision h, Precision zcut)
  {
    fEllipticalCone.fDx   = a;
    fEllipticalCone.fDy   = b;
    fEllipticalCone.fDz   = h;
    fEllipticalCone.fZCut = zcut;
    CheckParameters();
  };

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  Precision Capacity() const override { return fEllipticalCone.fCubicVolume; }

  Precision SurfaceArea() const override { return fEllipticalCone.fSurfaceArea; }

  Vector3D<Precision> SamplePointOnSurface() const override;

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const override
  {
    bool valid;
    normal = EllipticalConeImplementation::NormalKernel(fEllipticalCone, p, valid);
    return valid;
  }

  std::string GetEntityType() const { return "EllipticalCone"; }

  std::ostream &StreamInfo(std::ostream &os) const;

public:
  virtual int MemorySize() const final { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override;

  virtual void Print(std::ostream &os) const override;

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedEllipticalCone>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

#ifndef VECCORE_CUDA
  // this is the function called from the VolumeFactory
  // this may be specific to the shape
  template <TranslationCode trans_code, RotationCode rot_code>
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               VPlacedVolume *const placement = NULL);

  VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume, Transformation3D const *const transformation,
                                   const TranslationCode trans_code, const RotationCode rot_code,
                                   VPlacedVolume *const placement) const override;
#else
  template <TranslationCode trans_code, RotationCode rot_code>
  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               const int id, VPlacedVolume *const placement = NULL);
  VECCORE_ATT_DEVICE VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                                      Transformation3D const *const transformation,
                                                      const TranslationCode trans_code, const RotationCode rot_code,
                                                      const int id, VPlacedVolume *const placement) const override;

#endif
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
