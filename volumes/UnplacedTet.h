/// \file UnplacedTet.h
/// \author Raman Sehgal (raman.sehgal@cern.ch), Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDTET_H_
#define VECGEOM_VOLUMES_UNPLACEDTET_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/TetStruct.h" // the pure Tet struct
#include "volumes/kernel/TetImplementation.h"
#include "volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedTet;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedTet);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedTet : public SIMDUnplacedVolumeImplHelper<TetImplementation>, public AlignedBase {

private:
  TetStruct<double> fTet;

public:
  VECCORE_ATT_HOST_DEVICE
  UnplacedTet();

  VECCORE_ATT_HOST_DEVICE
  UnplacedTet(const Vector3D<Precision> &p0, const Vector3D<Precision> &p1, const Vector3D<Precision> &p2,
              const Vector3D<Precision> &p3);

  VECCORE_ATT_HOST_DEVICE
  UnplacedTet(const Precision p0[], const Precision p1[], const Precision p2[], const Precision p3[])
      : fTet(p0, p1, p2, p3)
  {
    fGlobalConvexity = true;
  }

  VECCORE_ATT_HOST_DEVICE
  TetStruct<double> const &GetStruct() const { return fTet; }

  // All the Required Getters and Setters

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void GetVertices(Vector3D<Precision> &p0, Vector3D<Precision> &p1, Vector3D<Precision> &p2,
                   Vector3D<Precision> &p3) const
  {
    p0 = fTet.fVertex[0];
    p1 = fTet.fVertex[1];
    p2 = fTet.fVertex[2];
    p3 = fTet.fVertex[3];
  }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  Precision Capacity() const override { return fTet.fCubicVolume; }

  Precision SurfaceArea() const override { return fTet.fSurfaceArea; }

  Vector3D<Precision> SamplePointOnSurface() const override;

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const override
  {
    bool valid;
    normal = TetImplementation::NormalKernel(fTet, p, valid);
    return valid;
  }

  std::string GetEntityType() const;

  VECCORE_ATT_HOST_DEVICE
  void GetParametersList(int aNumber, double *aArray) const;

  std::ostream &StreamInfo(std::ostream &os) const;

public:
  virtual int MemorySize() const final { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override;

  virtual void Print(std::ostream &os) const override;

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedTet>::SizeOf(); }
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
}
} // End global namespace

#endif
