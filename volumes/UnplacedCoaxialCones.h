/// @file UnplacedCoaxialCones.h
/// @author Raman Sehgal (raman.sehgal@cern.ch), Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDCOAXIALCONES_H_
#define VECGEOM_VOLUMES_UNPLACEDCOAXIALCONES_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/CoaxialConesStruct.h" // the pure CoaxialCones struct
#include "volumes/kernel/CoaxialConesImplementation.h"
#include "volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedCoaxialCones;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedCoaxialCones);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedCoaxialCones : public SIMDUnplacedVolumeImplHelper<CoaxialConesImplementation>, public AlignedBase {

private:
  CoaxialConesStruct<Precision> fCoaxialCones;

  void CheckParameters();

public:
  /*
   * All the required Parametric CoaxialCones Constructor
   *
   */
  VECCORE_ATT_HOST_DEVICE
  UnplacedCoaxialCones();


  VECCORE_ATT_HOST_DEVICE
  CoaxialConesStruct<Precision> const &GetStruct() const { return fCoaxialCones; }

  /*
   * Check Parameter function if required
   *
   */

/*
 * Required Getters and Setters
 *
 */

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  Precision Capacity() const override { return fCoaxialCones.fCubicVolume; }

  Precision SurfaceArea() const override { return fCoaxialCones.fSurfaceArea; }

  Vector3D<Precision> SamplePointOnSurface() const override;

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const override
  {
    bool valid;
    normal = CoaxialConesImplementation::NormalKernel(fCoaxialCones, p, valid);
    return valid;
  }

  std::string GetEntityType() const { return "CoaxialCones"; }

  std::ostream &StreamInfo(std::ostream &os) const;

public:
  virtual int MemorySize() const final { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override;

  virtual void Print(std::ostream &os) const override;

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedCoaxialCones>::SizeOf(); }
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
