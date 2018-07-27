/// @file PlacedTrd.h
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDTRD_H_
#define VECGEOM_VOLUMES_PLACEDTRD_H_

#include "base/Global.h"
#ifndef VECCORE_CUDA
#include "base/RNG.h"
#include <cmath>
#endif
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/kernel/TrdImplementation.h"
#include "volumes/PlacedVolImplHelper.h"
#include "volumes/UnplacedTrd.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedTrd;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedTrd);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedTrd : public VPlacedVolume {

public:
  using VPlacedVolume::VPlacedVolume;
#ifndef VECCORE_CUDA
  // constructor inheritance;
  PlacedTrd(char const *const label, LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation, vecgeom::PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logicalVolume, transformation, boundingBox)
  {
  }

  PlacedTrd(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
            vecgeom::PlacedBox const *const boundingBox)
      : PlacedTrd("", logicalVolume, transformation, boundingBox)
  {
  }
#else
  VECCORE_ATT_DEVICE PlacedTrd(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                               PlacedBox const *const boundingBox, const int id)
      : VPlacedVolume(logicalVolume, transformation, boundingBox, id)
  {
  }
#endif

  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedTrd() {}

  VECCORE_ATT_HOST_DEVICE
  UnplacedTrd const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedTrd const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dx1() const { return GetUnplacedVolume()->dx1(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dx2() const { return GetUnplacedVolume()->dx2(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dy1() const { return GetUnplacedVolume()->dy1(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dy2() const { return GetUnplacedVolume()->dy2(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dz() const { return GetUnplacedVolume()->dz(); }

  Precision GetXHalfLength1() const { return GetUnplacedVolume()->dx1(); }
  Precision GetXHalfLength2() const { return GetUnplacedVolume()->dx2(); }
  Precision GetYHalfLength1() const { return GetUnplacedVolume()->dy1(); }
  Precision GetYHalfLength2() const { return GetUnplacedVolume()->dy2(); }
  Precision GetZHalfLength() const { return GetUnplacedVolume()->dz(); }
  void SetXHalfLength1(Precision arg) { const_cast<UnplacedTrd *>(GetUnplacedVolume())->SetXHalfLength1(arg); }
  void SetXHalfLength2(Precision arg) { const_cast<UnplacedTrd *>(GetUnplacedVolume())->SetXHalfLength2(arg); }
  void SetYHalfLength1(Precision arg) { const_cast<UnplacedTrd *>(GetUnplacedVolume())->SetYHalfLength1(arg); }
  void SetYHalfLength2(Precision arg) { const_cast<UnplacedTrd *>(GetUnplacedVolume())->SetYHalfLength2(arg); }
  void SetZHalfLength(Precision arg) { const_cast<UnplacedTrd *>(GetUnplacedVolume())->SetZHalfLength(arg); }
  void SetAllParameters(Precision x1, Precision x2, Precision y1, Precision y2, Precision z)
  {
    const_cast<UnplacedTrd *>(GetUnplacedVolume())->SetAllParameters(x1, x2, y1, y2, z);
  }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

#ifndef VECCORE_CUDA
  virtual Precision Capacity() override { return GetUnplacedVolume()->Capacity(); }

  virtual Vector3D<Precision> SamplePointOnSurface() const override
  {
    return GetUnplacedVolume()->SamplePointOnSurface();
  }

  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override
  {
    return GetUnplacedVolume()->Normal(point, normal);
  }

  /** @brief Memory size in bytes */
  VECGEOM_FORCE_INLINE
  virtual int MemorySize() const override { return sizeof(*this); }

  virtual VPlacedVolume const *ConvertToUnspecialized() const override;

#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override;
#endif
#ifdef VECGEOM_GEANT4
  G4VSolid const *ConvertToGeant4() const override;
#endif
#endif // VECCORE_CUDA
};

template <typename UnplacedTrd_t>
class SPlacedTrd : public PlacedVolumeImplHelper<UnplacedTrd_t, PlacedTrd> {
  using Base = PlacedVolumeImplHelper<UnplacedTrd_t, PlacedTrd>;

public:
  typedef UnplacedTrd UnplacedShape_t;
  using Base::Base;
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDTUBE_H_
