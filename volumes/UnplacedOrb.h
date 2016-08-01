/// \file UnplacedOrb.h
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDORB_H_
#define VECGEOM_VOLUMES_UNPLACEDORB_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/OrbStruct.h" // the pure Orb struct
#include "volumes/kernel/OrbImplementation.h"
#include "volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedOrb;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedOrb);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedOrb : public SIMDUnplacedVolumeImplHelper<OrbImplementation>, public AlignedBase {

private:
  OrbStruct<double> fOrb;

  // Caching the Volume and SurfaceArea
  Precision fCubicVolume, fSurfaceArea;

  Precision fEpsilon, fRTolerance;

public:
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedOrb();

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedOrb(const Precision r);

  VECGEOM_CUDA_HEADER_BOTH
  void SetRadialTolerance();

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetRadialTolerance() const { return fRTolerance; }

  VECGEOM_CUDA_HEADER_BOTH
  OrbStruct<double> const &GetStruct() const { return fOrb; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision GetRadius() const { return fOrb.fR; }

  VECGEOM_CUDA_HEADER_BOTH
  // VECGEOM_FORCE_INLINE
  void SetRadius(Precision r);

  VECGEOM_CUDA_HEADER_BOTH
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision Capacity() const { return fCubicVolume; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision SurfaceArea() const { return fSurfaceArea; }

  virtual Vector3D<Precision> GetPointOnSurface() const;

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const override
  {
    bool valid;
    normal = OrbImplementation::NormalKernel(fOrb, p, valid);
    return valid;
  }

  std::string GetEntityType() const;

#if defined(VECGEOM_USOLIDS)
  VECGEOM_CUDA_HEADER_BOTH
  void GetParametersList(int aNumber, double *aArray) const;

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedOrb *Clone() const;

  std::ostream &StreamInfo(std::ostream &os) const;
#endif

public:
  virtual int memory_size() const final { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const override;

  virtual void Print(std::ostream &os) const override;

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedOrb>::SizeOf(); }
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
