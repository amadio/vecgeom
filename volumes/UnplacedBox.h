#ifndef VECGEOM_VOLUMES_UNPLACEDBOX_H_
#define VECGEOM_VOLUMES_UNPLACEDBOX_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/BoxStruct.h" // the pure box struct
#include "volumes/kernel/BoxImplementation.h"
#include "volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedBox;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedBox);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedBox : public SIMDUnplacedVolumeImplHelper<BoxImplementation>, public AlignedBase {

private:
  BoxStruct<double> fBox;

public:
  using Kernel = BoxImplementation;
  UnplacedBox(Vector3D<Precision> const &dim) : fBox(dim) {}
  UnplacedBox(char const *, Vector3D<Precision> const &dim) : fBox(dim) {}

  VECCORE_ATT_HOST_DEVICE
  UnplacedBox(const Precision dx, const Precision dy, const Precision dz) : fBox(dx, dy, dz)
  {
    fGlobalConvexity = true;
  }
  UnplacedBox(char const *, const Precision dx, const Precision dy, const Precision dz) : fBox(dx, dy, dz)
  {
    fGlobalConvexity = true;
  }

  VECCORE_ATT_HOST_DEVICE
  UnplacedBox(UnplacedBox const &other) : fBox(other.fBox) {}

  VECCORE_ATT_HOST_DEVICE
  BoxStruct<double> const &GetStruct() const { return fBox; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<Precision> const &dimensions() const { return fBox.fDimensions; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision x() const { return dimensions().x(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision y() const { return dimensions().y(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision z() const { return dimensions().z(); }

  void SetX(double xx) { fBox.fDimensions[0] = xx; }
  void SetY(double yy) { fBox.fDimensions[1] = yy; }
  void SetZ(double zz) { fBox.fDimensions[2] = zz; }

  Precision Capacity() const override { return 8.0 * x() * y() * z(); }

  // VECCORE_ATT_HOST_DEVICE
  Precision SurfaceArea() const override
  {
    // factor 8 because x(),... are half-lengths
    return 8.0 * (x() * y() + y() * z() + x() * z());
  }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    // Returns the full 3D cartesian extent of the volume
    aMin = -fBox.fDimensions;
    aMax = fBox.fDimensions;
  }

  Vector3D<Precision> SamplePointOnSurface() const override;

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const override
  {
    bool valid;
    normal = BoxImplementation::NormalKernel(fBox, p, valid);
    return valid;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override;

  virtual void Print(std::ostream &os) const override;

#ifndef VECCORE_CUDA
  virtual SolidMesh *CreateMesh3D(Transformation3D const &trans, size_t nSegments) const override;
#endif

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedBox>::SizeOf(); }
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
