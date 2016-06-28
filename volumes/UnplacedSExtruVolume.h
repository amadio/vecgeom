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
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedSExtruVolume(int nvertices, double *x, double *y, Precision lowerz, Precision upperz)
      : fPolyShell(nvertices, x, y, lowerz, upperz)
  {
  }

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedSExtruVolume(UnplacedSExtruVolume const &other) : fPolyShell(other.fPolyShell) {}

  VECGEOM_CUDA_HEADER_BOTH
  PolygonalShell const &GetStruct() const { return fPolyShell; }

  VECGEOM_CUDA_HEADER_BOTH
  Precision Capacity() const /*override*/
  {
    return fPolyShell.fPolygon.Area() * (fPolyShell.fUpperZ - fPolyShell.fLowerZ);
  }

  VECGEOM_CUDA_HEADER_BOTH
  Precision SurfaceArea() const /*override*/ { return fPolyShell.SurfaceArea() + 2. * fPolyShell.fPolygon.Area(); }

  VECGEOM_CUDA_HEADER_BOTH
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override { fPolyShell.Extent(aMin, aMax); }

  Vector3D<Precision> GetPointOnSurface() const override;

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const override
  {
    bool valid(false);
    normal = SExtruImplementation::NormalKernel(fPolyShell, p, valid);
    return valid;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const override;

  virtual void Print(std::ostream &os) const override;

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedSExtruVolume>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

#ifndef VECGEOM_NVCC
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
  __device__
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               const int id, VPlacedVolume *const placement = NULL);
  __device__ VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                              Transformation3D const *const transformation,
                                              const TranslationCode trans_code, const RotationCode rot_code,
                                              const int id, VPlacedVolume *const placement) const override;

#endif
};
}
} // End global namespace

#endif
