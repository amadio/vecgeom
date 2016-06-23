/// @file UnplacedBox.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#pragma once

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/BoxStruct.h" // the pure box struct
#include "volumes/kernel/BoxImplementation_Evolution.h"
#include "volumes/UnplacedVolumeImplHelper.h"
namespace vecgeom {
namespace evolution {

// VECGEOM_DEVICE_FORWARD_DECLARE( class UnplacedBox_Evolution; )
// VECGEOM_DEVICE_DECLARE_CONV( UnplacedBox_Evolution )

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedBox : public SIMDUnplacedVolumeImplHelper<BoxImplementation>, public AlignedBase {

private:
  BoxStruct<double> fBox;

public:
  BoxStruct<double> const &GetStruct() const { return fBox; }

  UnplacedBox(Vector3D<Precision> const &dim) : fBox(dim) {}
  UnplacedBox(char const *, Vector3D<Precision> const &dim) : fBox(dim) {}

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedBox(const Precision dx, const Precision dy, const Precision dz) : fBox(dx, dy, dz) {}
  UnplacedBox(char const *, const Precision dx, const Precision dy, const Precision dz) : fBox(dx, dy, dz) {}

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedBox(UnplacedBox const &other) : fBox(other.fBox) {}

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedBox>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
#endif

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Precision> const &dimensions() const { return fBox.fDimensions; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision x() const { return dimensions().x(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision y() const { return dimensions().y(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision z() const { return dimensions().z(); }

#if !defined(VECGEOM_NVCC)
  Precision Capacity() { return 8.0 * x() * y() * z(); }

  VECGEOM_INLINE
  Precision SurfaceArea() const
  {
    // factor 8 because dimensions_ are half-lengths
    return 8.0 * (x() * y() + y() * z() + x() * z());
  }

  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    // Returns the full 3D cartesian extent of the solid.
    aMin = -fBox.fDimensions;
    aMax = fBox.fDimensions;
  }

  Vector3D<Precision> GetPointOnSurface() const override { return Vector3D<Precision>(0, 0, 0); }

  virtual bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const override
  {
    assert(false && "not yet implemented\n");
    // normal = BoxImplementation<VecCore::VecGeomBackend::Scalar<double>>::NormalKernel(fBox, p, valid);
    return false;
  }

  virtual std::string GetEntityType() const { return "Box"; }
#endif // !VECGEOM_NVCC

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const override{};

  virtual void Print(std::ostream &os) const override{};

#ifndef VECGEOM_NVCC
  // this is the function called from the VolumeFactory
  // this may be specific to the shape
  template <TranslationCode trans_code, RotationCode rot_code>
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               VPlacedVolume *const placement = NULL);
#else
  template <TranslationCode trans_code, RotationCode rot_code>
  __device__
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               const int id, VPlacedVolume *const placement = NULL);
#endif
};
}
}
} // End global namespace
