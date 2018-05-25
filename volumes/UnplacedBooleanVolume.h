#ifndef UNPLACEDBOOLEANVOLUME_H_
#define UNPLACEDBOOLEANVOLUME_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/PlacedVolume.h"
#include "volumes/BooleanStruct.h"
#include "volumes/kernel/BooleanImplementation.h"
#include "volumes/UnplacedVolumeImplHelper.h"
#ifndef VECCORE_CUDA
#include "volumes/UnplacedMultiUnion.h"
#endif

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_1v(class, UnplacedBooleanVolume, BooleanOperation, Arg1);

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * A class representing a simple UNPLACED boolean volume A-B
 * It takes two template arguments:
 * 1.: the mother (or left) volume A in unplaced form
 * 2.: the (or right) volume B in placed form, acting on A with a boolean operation;
 * the placement is with respect to the left volume
 *
 *
 *
 * will be a boolean solid where two boxes are subtracted
 * and B is only translated (not rotated) with respect to A
 *
 */
template <BooleanOperation Op>
class UnplacedBooleanVolume : public LoopUnplacedVolumeImplHelper<BooleanImplementation<Op>>, public AlignedBase {

public:
  BooleanStruct fBoolean;
  using LoopUnplacedVolumeImplHelper<BooleanImplementation<Op>>::fGlobalConvexity;

  // the constructor
  VECCORE_ATT_HOST_DEVICE
  UnplacedBooleanVolume(BooleanOperation op, VPlacedVolume const *left, VPlacedVolume const *right)
      : fBoolean(op, left, right)
  {
    fGlobalConvexity = false;
#ifndef VECCORE_CUDA
    if (fBoolean.fLeftVolume->IsAssembly() || fBoolean.fRightVolume->IsAssembly()) {
      throw std::runtime_error("Trying to make boolean out of assembly which is not supported\n");
    }
#endif
  }

  virtual int MemorySize() const override { return sizeof(*this); }

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedBooleanVolume<Op>>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  BooleanOperation GetOp() const { return fBoolean.fOp; }

  VECCORE_ATT_HOST_DEVICE
  BooleanStruct const &GetStruct() const { return fBoolean; }

  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override;

  Precision Capacity() const override
  {
    if (fBoolean.fCapacity < 0.) {
      fBoolean.fCapacity = VUnplacedVolume::EstimateCapacity(1000000);
    }
    return fBoolean.fCapacity;
  }

  Precision SurfaceArea() const override
  {
    if (fBoolean.fSurfaceArea < 0.) {
      fBoolean.fSurfaceArea = VUnplacedVolume::EstimateSurfaceArea(1000000);
    }
    return fBoolean.fSurfaceArea;
  }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override;

  Vector3D<Precision> SamplePointOnSurface() const override
  {
    // TBDONE
    return Vector3D<Precision>();
  }

  std::string GetEntityType() const { return "BooleanVolume"; }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override{};

  virtual void Print(std::ostream & /*os*/) const override{};

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

  VPlacedVolume const *GetLeft() const { return fBoolean.fLeftVolume; }
  VPlacedVolume const *GetRight() const { return fBoolean.fRightVolume; }

  /** @brief Count number of Boolean nodes of different types under this Boolean volume */
  size_t CountNodes(size_t &nunion, size_t &nintersection, size_t &nsubtraction) const
  {
    switch (fBoolean.fOp) {
    case kUnion:
      nunion++;
      break;
    case kIntersection:
      nintersection++;
      break;
    case kSubtraction:
      nsubtraction++;
      break;
    };
    UnplacedBooleanVolume const *left =
        dynamic_cast<UnplacedBooleanVolume const *>(fBoolean.fLeftVolume->GetUnplacedVolume());
    if (left) left->CountNodes(nunion, nintersection, nsubtraction);
    UnplacedBooleanVolume const *right =
        dynamic_cast<UnplacedBooleanVolume const *>(fBoolean.fRightVolume->GetUnplacedVolume());
    if (right) right->CountNodes(nunion, nintersection, nsubtraction);
    return (nunion + nintersection + nsubtraction);
  }

#ifndef VECCORE_CUDA
  /** @brief Flatten the Boolean node structure and create multiple union volumes if possible */
  UnplacedMultiUnion *Flatten(size_t min_unions = 1, Transformation3D const *trbase = nullptr,
                              UnplacedMultiUnion *munion = nullptr)
  {
    size_t nunion{0}, nintersection{0}, nsubtraction{0};
    CountNodes(nunion, nintersection, nsubtraction);
    if (nunion < min_unions) {
      // If a multi-union is being built-up, add this volume
      if (munion) munion->AddNode(this, *trbase);
      return nullptr;
    }
    VUnplacedVolume *vol;
    if (fBoolean.fOp == kUnion) {
      bool creator        = munion == nullptr;
      if (!munion) munion = new UnplacedMultiUnion();
      Transformation3D transform;

      // Compute left transformation
      transform = (trbase) ? *trbase : Transformation3D();
      transform.MultiplyFromRight(*fBoolean.fLeftVolume->GetTransformation());
      vol            = (VUnplacedVolume *)fBoolean.fLeftVolume->GetUnplacedVolume();
      auto left_bool = dynamic_cast<UnplacedBooleanVolume *>(vol);
      if (left_bool)
        left_bool->Flatten(min_unions, &transform, munion);
      else
        munion->AddNode(vol, transform);

      // Compute right transformation
      transform = (trbase) ? *trbase : Transformation3D();
      transform.MultiplyFromRight(*fBoolean.fRightVolume->GetTransformation());
      vol             = (VUnplacedVolume *)fBoolean.fRightVolume->GetUnplacedVolume();
      auto right_bool = dynamic_cast<UnplacedBooleanVolume *>(vol);
      if (right_bool)
        right_bool->Flatten(min_unions, &transform, munion);
      else
        munion->AddNode(vol, transform);

      if (creator) {
        munion->Close();
        return munion;
      }
      return nullptr;
    }

    // Analyze branches in case of subtraction or intersection
    vol            = (VUnplacedVolume *)fBoolean.fLeftVolume->GetUnplacedVolume();
    auto left_bool = dynamic_cast<UnplacedBooleanVolume *>(vol);
    if (left_bool) {
      auto left_new = left_bool->Flatten(min_unions);
      if (left_new) {
        // Replace existing left volume with the new one
        auto lvol            = new LogicalVolume(left_new);
        auto pvol            = lvol->Place(fBoolean.fLeftVolume->GetTransformation());
        fBoolean.fLeftVolume = pvol;
      }
    }
    vol             = (VUnplacedVolume *)fBoolean.fRightVolume->GetUnplacedVolume();
    auto right_bool = dynamic_cast<UnplacedBooleanVolume *>(vol);
    if (right_bool) {
      auto right_new = right_bool->Flatten(min_unions);
      if (right_new) {
        // Replace existing right volume with the new one
        auto lvol             = new LogicalVolume(right_new);
        auto pvol             = lvol->Place(fBoolean.fRightVolume->GetTransformation());
        fBoolean.fRightVolume = pvol;
      }
    }
    if (munion) munion->AddNode(this, *trbase);
    return nullptr;
  }
#endif

private:
  VECCORE_ATT_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECCORE_CUDA
                                           const int id,
#endif
                                           VPlacedVolume *const placement = NULL) const override;

  void SetLeft(VPlacedVolume const *pvol) { fBoolean.fLeftVolume = pvol; }
  void SetRight(VPlacedVolume const *pvol) { fBoolean.fRightVolume = pvol; }

  friend class GeoManager;
}; // End class
} // End impl namespace

} // End global namespace

#endif /* UNPLACEDBOOLEANVOLUME_H_ */
