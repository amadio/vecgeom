#ifndef UNPLACEDBOOLEANVOLUME_H_
#define UNPLACEDBOOLEANVOLUME_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/PlacedVolume.h"

enum BooleanOperation { kUnion, kIntersection, kSubtraction };

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedBooleanVolume;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedBooleanVolume);

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * A class representing a simple UNPLACED boolean volume A-B
 * It takes two template arguments:
 * 1.: the mother (or left) volume A in unplaced form
 * 2.: the subtraction (or right) volume B in placed form;
 * the placement is with respect to the left volume
 *
 *
 *
 * will be a boolean solid where two boxes are subtracted
 * and B is only translated (not rotated) with respect to A
 *
 */
class UnplacedBooleanVolume : public VUnplacedVolume, public AlignedBase {

public:
  VPlacedVolume const *fLeftVolume;
  VPlacedVolume const *fRightVolume;
  BooleanOperation const fOp;

public:
  // need a constructor
  VECCORE_ATT_HOST_DEVICE
  UnplacedBooleanVolume(BooleanOperation op, VPlacedVolume const *left, VPlacedVolume const *right)
      : fLeftVolume(left), fRightVolume(right), fOp(op)
  {
    fGlobalConvexity = false;
#ifndef VECCORE_CUDA
    if (fLeftVolume->IsAssembly() || fRightVolume->IsAssembly()) {
      throw std::runtime_error("Trying to make boolean out of assembly which is not supported\n");
    }
#endif
  }

  virtual int MemorySize() const { return sizeof(*this); }

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedBooleanVolume>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
#endif

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  BooleanOperation GetOp() const { return fOp; }

  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const;

#if !defined(VECCORE_CUDA)
  VECGEOM_FORCE_INLINE
  Precision Capacity() const
  {
    // TBDONE -- need some sampling
    return 0.;
  }

  VECGEOM_FORCE_INLINE
  Precision SurfaceArea() const
  {
    // TBDONE -- need some sampling
    return 0.;
  }
#endif // !VECCORE_CUDA

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const;

  Vector3D<Precision> SamplePointOnSurface() const
  {
    // TBDONE
    return Vector3D<Precision>();
  }

  std::string GetEntityType() const { return "BooleanVolume"; }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const {};

  virtual void Print(std::ostream & /*os*/) const {};

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

  VPlacedVolume const *GetLeft() const { return fLeftVolume; }
  VPlacedVolume const *GetRight() const { return fRightVolume; }

private:
  VECCORE_ATT_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECCORE_CUDA
                                           const int id,
#endif
                                           VPlacedVolume *const placement = NULL) const;

  void SetLeft(VPlacedVolume const *pvol) { fLeftVolume = pvol; }
  void SetRight(VPlacedVolume const *pvol) { fRightVolume = pvol; }

  friend class GeoManager;
}; // End class

} // End impl namespace

} // End global namespace

#endif /* UNPLACEDBOOLEANVOLUME_H_ */
