#ifndef VECGEOM_VOLUMES_UNPLACED_SEXTRUVOLUME_H_
#define VECGEOM_VOLUMES_UNPLACED_SEXTRUVOLUME_H_

#include "base/AlignedBase.h"
#include "base/Global.h"
#include "base/Vector3D.h"
#include "volumes/PolygonalShell.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/kernel/SExtruImplementation.h"
#include "volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedSExtruVolume;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedSExtruVolume);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedSExtruVolume : public LoopUnplacedVolumeImplHelper<SExtruImplementation>, public AlignedBase {

private:
  PolygonalShell fPolyShell;

public:
  VECCORE_ATT_HOST_DEVICE
  UnplacedSExtruVolume(int nvertices, double *x, double *y, Precision lowerz, Precision upperz)
      : fPolyShell(nvertices, x, y, lowerz, upperz)
  {
  }

  VECCORE_ATT_HOST_DEVICE
  UnplacedSExtruVolume(UnplacedSExtruVolume const &other) : fPolyShell(other.fPolyShell) {}

  VECCORE_ATT_HOST_DEVICE
  PolygonalShell const &GetStruct() const { return fPolyShell; }

  Precision Capacity() const override { return fPolyShell.fPolygon.Area() * (fPolyShell.fUpperZ - fPolyShell.fLowerZ); }

  // VECCORE_ATT_HOST_DEVICE
  Precision SurfaceArea() const override { return fPolyShell.SurfaceArea() + 2. * fPolyShell.fPolygon.Area(); }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override { fPolyShell.Extent(aMin, aMax); }

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const override
  {
    bool valid(false);
    normal = SExtruImplementation::NormalKernel(fPolyShell, p, valid);
    return valid;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override;

  virtual void Print(std::ostream &os) const override;

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedSExtruVolume>::SizeOf(); }
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
