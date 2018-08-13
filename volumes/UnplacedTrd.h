/// @file UnplacedTrd.h
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDTRD_H_
#define VECGEOM_VOLUMES_UNPLACEDTRD_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"
#include "TrdStruct.h"
#include "volumes/kernel/TrdImplementation.h"
#include "volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedTrd;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedTrd);
VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(class, SUnplacedTrd, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedTrd : public VUnplacedVolume {
private:
  TrdStruct<double> fTrd; ///< Structure with trapezoid parameters

public:
  VECCORE_ATT_HOST_DEVICE
  UnplacedTrd() : fTrd() { fGlobalConvexity = true; }

  VECCORE_ATT_HOST_DEVICE
  UnplacedTrd(const Precision x, const Precision y, const Precision z) : fTrd(x, y, z) { fGlobalConvexity = true; }

  // special case Trd1 when dY1 == dY2
  VECCORE_ATT_HOST_DEVICE
  UnplacedTrd(const Precision x1, const Precision x2, const Precision y1, const Precision z) : fTrd(x1, x2, y1, z)
  {
    fGlobalConvexity = true;
  }

  // general case
  VECCORE_ATT_HOST_DEVICE
  UnplacedTrd(const Precision x1, const Precision x2, const Precision y1, const Precision y2, const Precision z)
      : fTrd(x1, x2, y1, y2, z)
  {
    fGlobalConvexity = true;
  }

  VECCORE_ATT_HOST_DEVICE
  TrdStruct<double> const &GetStruct() const { return fTrd; }

  VECCORE_ATT_HOST_DEVICE
  void SetAllParameters(Precision x1, Precision x2, Precision y1, Precision y2, Precision z)
  {
    fTrd.SetAllParameters(x1, x2, y1, y2, z);
  }

  VECCORE_ATT_HOST_DEVICE
  void SetXHalfLength1(Precision arg)
  {
    fTrd.fDX1 = arg;
    fTrd.CalculateCached();
  }
  VECCORE_ATT_HOST_DEVICE
  void SetXHalfLength2(Precision arg)
  {
    fTrd.fDX2 = arg;
    fTrd.CalculateCached();
  }
  VECCORE_ATT_HOST_DEVICE
  void SetYHalfLength1(Precision arg)
  {
    fTrd.fDY1 = arg;
    fTrd.CalculateCached();
  }
  VECCORE_ATT_HOST_DEVICE
  void SetYHalfLength2(Precision arg)
  {
    fTrd.fDY2 = arg;
    fTrd.CalculateCached();
  }
  VECCORE_ATT_HOST_DEVICE
  void SetZHalfLength(Precision arg)
  {
    fTrd.fDZ = arg;
    fTrd.CalculateCached();
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dx1() const { return fTrd.fDX1; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dx2() const { return fTrd.fDX2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dy1() const { return fTrd.fDY1; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dy2() const { return fTrd.fDY2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dz() const { return fTrd.fDZ; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision x2minusx1() const { return fTrd.fX2minusX1; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision y2minusy1() const { return fTrd.fY2minusY1; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision halfx1plusx2() const { return fTrd.fHalfX1plusX2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision halfy1plusy2() const { return fTrd.fHalfY1plusY2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision fx() const { return fTrd.fFx; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision fy() const { return fTrd.fFy; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision calfx() const { return fTrd.fCalfX; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision calfy() const { return fTrd.fCalfY; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision ToleranceX() const { return fTrd.fToleranceX; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision ToleranceY() const { return fTrd.fToleranceY; }

  // virtual int MemorySize() const final { return sizeof(*this); }
  virtual int MemorySize() const override { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    aMin = Vector3D<Precision>(-Max(fTrd.fDX1, fTrd.fDX2), -Max(fTrd.fDY1, fTrd.fDY2), -fTrd.fDZ);
    aMax = Vector3D<Precision>(Max(fTrd.fDX1, fTrd.fDX2), Max(fTrd.fDY1, fTrd.fDY2), fTrd.fDZ);
  }

  // Computes capacity of the shape in [length^3]
  Precision Capacity() const override;

  Precision SurfaceArea() const override;

  Vector3D<Precision> SamplePointOnSurface() const override;

  Precision GetPlusXArea() const
  { //  Area in +x direction
    return 2 * fTrd.fDZ * (fTrd.fDY1 + fTrd.fDY2) * fTrd.fSecxz;
  }

  Precision GetMinusXArea() const
  { // Area in -x direction
    return GetPlusXArea();
  }

  Precision GetPlusYArea() const
  { // Area in +y direction
    return 2 * fTrd.fDZ * (fTrd.fDX1 + fTrd.fDX2) * fTrd.fSecyz;
  }

  Precision GetMinusYArea() const
  { // Area in -y direction
    return GetPlusYArea();
  }

  Precision GetPlusZArea() const
  { // Area in +Z
    return 4 * fTrd.fDX2 * fTrd.fDY2;
  }

  Precision GetMinusZArea() const
  { // Area in -Z
    return 4 * fTrd.fDX1 * fTrd.fDY1;
  }

  int ChooseSurface() const;

#ifndef VECCORE_CUDA
  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override;
#endif

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const final;

  virtual void Print(std::ostream &os) const final;

  std::string GetEntityType() const { return "Trd"; }

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override
  {
    return DevicePtr<cuda::SUnplacedTrd<cuda::TrdTypes::UniversalTrd>>::SizeOf();
  }

  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

  std::ostream &StreamInfo(std::ostream &os) const;

#ifndef VECCORE_CUDA
#ifdef VECGEOM_ROOT
  TGeoShape const *ConvertToRoot(char const *label) const;
#endif

#ifdef VECGEOM_GEANT4
  G4VSolid const *ConvertToGeant4(char const *label) const;
#endif
#endif
};

template <>
struct Maker<UnplacedTrd> {
  template <typename... ArgTypes>
  static UnplacedTrd *MakeInstance(const Precision x1, const Precision x2, const Precision y1, const Precision y2,
                                   const Precision z);
  template <typename... ArgTypes>
  static UnplacedTrd *MakeInstance(const Precision x1, const Precision x2, const Precision y1, const Precision z);
};

template <typename TrdType = TrdTypes::UniversalTrd>
class SUnplacedTrd : public SIMDUnplacedVolumeImplHelper<TrdImplementation<TrdType>, UnplacedTrd>, public AlignedBase {
public:
  using Kernel     = TrdImplementation<TrdType>;
  using BaseType_t = SIMDUnplacedVolumeImplHelper<TrdImplementation<TrdType>, UnplacedTrd>;
  using BaseType_t::BaseType_t;

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifndef VECCORE_CUDA
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           const TranslationCode trans_code, const RotationCode rot_code,
                                           VPlacedVolume *const placement = NULL) const override
  {
    return VolumeFactory::CreateByTransformation<SUnplacedTrd<TrdType>>(volume, transformation, trans_code, rot_code,
                                                                        placement);
  }

#else
  VECCORE_ATT_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           const TranslationCode trans_code, const RotationCode rot_code, const int id,
                                           VPlacedVolume *const placement = NULL) const override
  {
    return VolumeFactory::CreateByTransformation<SUnplacedTrd<TrdType>>(volume, transformation, trans_code, rot_code,
                                                                        id, placement);
  }
#endif
};

using GenericUnplacedTrd = SUnplacedTrd<TrdTypes::UniversalTrd>;

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#include "volumes/SpecializedTrd.h"

#endif // VECGEOM_VOLUMES_UNPLACEDTRD_H_
