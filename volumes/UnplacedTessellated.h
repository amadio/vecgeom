/// @file UnplacedTessellated.h
/// @author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDTESSELLATED_H_
#define VECGEOM_VOLUMES_UNPLACEDTESSELLATED_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"
#include "TessellatedStruct.h"
#include "volumes/kernel/TessellatedImplementation.h"
#include "volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedTessellated;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedTessellated);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedTessellated : public SIMDUnplacedVolumeImplHelper<TessellatedImplementation>, public AlignedBase {
private:
  mutable TessellatedStruct<double> fTessellation; ///< Structure with tessellation parameters

public:
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTessellated() : fTesselation() { fGlobalConvexity = false; }

  VECGEOM_CUDA_HEADER_BOTH
  TessellatedStruct<double> const &GetStruct() const { return fTessellation; }

  virtual int memory_size() const final { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const { fTessellation.Extent(aMin, aMax); }

#ifndef VECGEOM_NVCC
  // Computes capacity of the shape in [length^3]
  Precision Capacity() const;

  Precision SurfaceArea() const;

  int ChooseSurface() const;

  Vector3D<Precision> GetPointOnSurface() const;

  VECGEOM_CUDA_HEADER_BOTH
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const;

#endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const final;

  std::string GetEntityType() const { return "Tesselated"; }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECGEOM_CUDA_HEADER_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedTessellated>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
#endif

#if defined(VECGEOM_USOLIDS)
  std::ostream &StreamInfo(std::ostream &os) const;
#endif

private:
  virtual void Print(std::ostream &os) const final;

  VECGEOM_CUDA_HEADER_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
                                           const int id,
#endif
                                           VPlacedVolume *const placement = NULL) const final;
};
}
} // end global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDTESSELLATED_H_
