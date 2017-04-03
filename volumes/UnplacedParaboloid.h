///===-- volumes/UnplacedParaboloid.h - Instruction class definition -------*- C++ -*-===//
///
/// \file volumes/UnplacedParaboloid.h
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)
/// \brief This file contains the declaration of the UnplacedParaboloid class
///
/// _____________________________________________________________________________
/// A paraboloid is the solid bounded by the following surfaces:
/// - 2 planes parallel with XY cutting the Z axis at Z=-dz and Z=+dz
/// - the surface of revolution of a parabola described by:
/// z = a*(x*x + y*y) + b
/// The parameters a and b are automatically computed from:
/// - rlo is the radius of the circle of intersection between the
/// parabolic surface and the plane z = -dz
/// - rhi is the radius of the circle of intersection between the
/// parabolic surface and the plane z = +dz
/// -dz = a*rlo^2 + b
/// dz = a*rhi^2 + b      where: rhi>rlo, both >= 0
///
/// note:
/// dd = 1./(rhi^2 - rlo^2);
/// a = 2.*dz*dd;
/// b = - dz * (rlo^2 + rhi^2)*dd;
///
/// in respect with the G4 implementation we have:
/// k1=1/a
/// k2=-b/a
///
/// a=1/k1
/// b=-k2/k1
//===----------------------------------------------------------------------===//
///
/// revision + moving to new backend structure : Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDPARABOLOID_H_
#define VECGEOM_VOLUMES_UNPLACEDPARABOLOID_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/ParaboloidStruct.h" // the pure Paraboloid struct
#include "volumes/kernel/ParaboloidImplementation.h"
#include "volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedParaboloid;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedParaboloid);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedParaboloid : public SIMDUnplacedVolumeImplHelper<ParaboloidImplementation>, public AlignedBase {

private:
  ParaboloidStruct<double> fParaboloid;

  // Varibale to store Cached values of Volume and SurfaceArea
  Precision fCubicVolume, fSurfaceArea;

public:
  VECCORE_ATT_HOST_DEVICE
  UnplacedParaboloid();

  VECCORE_ATT_HOST_DEVICE
  UnplacedParaboloid(const Precision rlo, const Precision rhi, const Precision dz);

  VECCORE_ATT_HOST_DEVICE
  ParaboloidStruct<double> const &GetStruct() const { return fParaboloid; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRlo() const { return fParaboloid.fRlo; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRhi() const { return fParaboloid.fRhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDz() const { return fParaboloid.fDz; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetA() const { return fParaboloid.fA; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetB() const { return fParaboloid.fB; }

  VECCORE_ATT_HOST_DEVICE
  // VECGEOM_FORCE_INLINE
  void SetRlo(Precision val)
  {
    fParaboloid.SetRlo(val);
    CalcCapacity();
    CalcSurfaceArea();
  }

  VECCORE_ATT_HOST_DEVICE
  void SetRhi(Precision val)
  {
    fParaboloid.SetRhi(val);
    CalcCapacity();
    CalcSurfaceArea();
  }

  VECCORE_ATT_HOST_DEVICE
  void SetDz(Precision val)
  {
    fParaboloid.SetDz(val);
    CalcCapacity();
    CalcSurfaceArea();
  }

  VECCORE_ATT_HOST_DEVICE
  void SetRloAndRhiAndDz(Precision rlo, Precision rhi, Precision dz)
  {
    fParaboloid.SetRloAndRhiAndDz(rlo, rhi, dz);
    CalcCapacity();
    CalcSurfaceArea();
  }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  VECCORE_ATT_HOST_DEVICE
  void CalcCapacity();

  VECCORE_ATT_HOST_DEVICE
  void CalcSurfaceArea();

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision Capacity() const { return fCubicVolume; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision SurfaceArea() const { return fSurfaceArea; }

  virtual Vector3D<Precision> GetPointOnSurface() const override;

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const override
  {
    bool valid = false;
    normal     = ParaboloidImplementation::NormalKernel(fParaboloid, p, valid);
    return valid;
  }

  std::string GetEntityType() const;

#if defined(VECGEOM_USOLIDS)
  VECCORE_ATT_HOST_DEVICE
  void GetParametersList(int aNumber, double *aArray) const;

  VECCORE_ATT_HOST_DEVICE
  UnplacedParaboloid *Clone() const;

  std::ostream &StreamInfo(std::ostream &os) const;
#endif

public:
  virtual int MemorySize() const final { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override;

  virtual void Print(std::ostream &os) const override;

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedParaboloid>::SizeOf(); }
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
