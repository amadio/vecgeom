/// \file UnplacedTet.h
/// \author Raman Sehgal (raman.sehgal@cern.ch)

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
  UnplacedTet(const Vector3D<Precision> anchor, const Vector3D<Precision> p2, const Vector3D<Precision> p3,
              const Vector3D<Precision> p4);

  VECCORE_ATT_HOST_DEVICE
  UnplacedTet(const Precision verticesx[], const Precision verticesy[], const Precision verticesz[])
      : fTet(verticesx, verticesy, verticesz)
  {
    fGlobalConvexity = true;
  }

  VECCORE_ATT_HOST_DEVICE
  TetStruct<double> const &GetStruct() const { return fTet; }

  // All the Required Getters and Setters

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<Precision> GetAnchor() const { return fTet.fVertex[0]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<Precision> GetP2() const { return fTet.fVertex[1]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<Precision> GetP3() const { return fTet.fVertex[2]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<Precision> GetP4() const { return fTet.fVertex[3]; }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  Precision Capacity() const override { return fTet.fCubicVolume; }

  // VECCORE_ATT_HOST_DEVICE
  Precision SurfaceArea() const override { return fTet.fSurfaceArea; }

  Vector3D<Precision> SamplePointOnSurface() const override;

  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &norm) const override;

  VECCORE_ATT_HOST_DEVICE
  void ApproxSurfaceNormal(Vector3D<Precision> const &p, Vector3D<Precision> &norm) const;

  std::string GetEntityType() const;

  VECCORE_ATT_HOST_DEVICE
  void GetParametersList(int aNumber, double *aArray) const;

  VECCORE_ATT_HOST_DEVICE
  UnplacedTet *Clone() const;

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
