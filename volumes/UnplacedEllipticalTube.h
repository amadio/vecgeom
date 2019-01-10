/// @file UnplacedEllipticalTube.h
/// @author Raman Sehgal (raman.sehgal@cern.ch), Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDELLIPTICALTUBE_H_
#define VECGEOM_VOLUMES_UNPLACEDELLIPTICALTUBE_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/EllipticalTubeStruct.h" // the pure EllipticalTube struct
#include "volumes/kernel/EllipticalTubeImplementation.h"
#include "volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedEllipticalTube;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedEllipticalTube);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedEllipticalTube : public SIMDUnplacedVolumeImplHelper<EllipticalTubeImplementation>, public AlignedBase {

private:
  EllipticalTubeStruct<Precision> fEllipticalTube;

  void CheckParameters();

public:
  VECCORE_ATT_HOST_DEVICE
  UnplacedEllipticalTube(Precision dx, Precision dy, Precision dz);

  VECCORE_ATT_HOST_DEVICE
  EllipticalTubeStruct<Precision> const &GetStruct() const { return fEllipticalTube; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDx() const { return fEllipticalTube.fDx; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDy() const { return fEllipticalTube.fDy; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDz() const { return fEllipticalTube.fDz; }

  void SetParameters(Precision dx, Precision dy, Precision dz)
  {
    fEllipticalTube.fDx = dx;
    fEllipticalTube.fDy = dy;
    fEllipticalTube.fDz = dz;
    CheckParameters();
  };
  void SetDx(Precision dx)
  {
    fEllipticalTube.fDx = dx;
    CheckParameters();
  };
  void SetDy(Precision dy)
  {
    fEllipticalTube.fDy = dy;
    CheckParameters();
  };
  void SetDz(Precision dz)
  {
    fEllipticalTube.fDz = dz;
    CheckParameters();
  };

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  Precision Capacity() const override { return fEllipticalTube.fCubicVolume; }

  Precision SurfaceArea() const override { return fEllipticalTube.fSurfaceArea; }

  Vector3D<Precision> SamplePointOnSurface() const override;

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const override
  {
    bool valid;
    normal = EllipticalTubeImplementation::NormalKernel(fEllipticalTube, p, valid);
    return valid;
  }

  std::string GetEntityType() const { return "EllipticalTube"; }

  std::ostream &StreamInfo(std::ostream &os) const;

public:
  virtual int MemorySize() const final { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override;

  virtual void Print(std::ostream &os) const override;

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedEllipticalTube>::SizeOf(); }
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
