/// \file PlacedHype.h
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDHYPE_H_
#define VECGEOM_VOLUMES_PLACEDHYPE_H_

#include "VecGeom/base/Global.h"
#ifndef VECCORE_CUDA
#include "VecGeom/base/RNG.h"
#include <cmath>
#endif
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/kernel/HypeImplementation.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedHype.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedHype;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedHype);
inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedHype : public VPlacedVolume {

public:
  using VPlacedVolume::VPlacedVolume;
#ifndef VECCORE_CUDA

  PlacedHype(char const *const label, LogicalVolume const *const logicalVolume,
             Transformation3D const *const transformation)
      : VPlacedVolume(label, logicalVolume, transformation)
  {
  }

  PlacedHype(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation)
      : PlacedHype("", logicalVolume, transformation)
  {
  }
#else
  VECCORE_ATT_DEVICE PlacedHype(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                                const int id, const int copy_no, const int child_id)
      : VPlacedVolume(logicalVolume, transformation, id, copy_no, child_id)
  {
  }
#endif

  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedHype() {}

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  UnplacedHype const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedHype const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

  VECCORE_ATT_HOST_DEVICE
  virtual void PrintType() const override;
  virtual void PrintType(std::ostream &s) const override;

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRmin() const { return GetUnplacedVolume()->GetRmin(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRmax() const { return GetUnplacedVolume()->GetRmax(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRmin2() const { return GetUnplacedVolume()->GetRmin2(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRmax2() const { return GetUnplacedVolume()->GetRmax2(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetStIn() const { return GetUnplacedVolume()->GetStIn(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetStOut() const { return GetUnplacedVolume()->GetStOut(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTIn() const { return GetUnplacedVolume()->GetTIn(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTOut() const { return GetUnplacedVolume()->GetTOut(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTIn2() const { return GetUnplacedVolume()->GetTIn2(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTOut2() const { return GetUnplacedVolume()->GetTOut2(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTIn2Inv() const { return GetUnplacedVolume()->GetTIn2Inv(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTOut2Inv() const { return GetUnplacedVolume()->GetTOut2Inv(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDz() const { return GetUnplacedVolume()->GetDz(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDz2() const { return GetUnplacedVolume()->GetDz2(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetEndInnerRadius() const { return GetUnplacedVolume()->GetEndInnerRadius(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetEndInnerRadius2() const { return GetUnplacedVolume()->GetEndInnerRadius2(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetEndOuterRadius() const { return GetUnplacedVolume()->GetEndOuterRadius(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetEndOuterRadius2() const { return GetUnplacedVolume()->GetEndOuterRadius2(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetInSqSide() const { return GetUnplacedVolume()->GetInSqSide(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetZToleranceLevel() const { return GetUnplacedVolume()->GetZToleranceLevel(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetInnerRadToleranceLevel() const { return GetUnplacedVolume()->GetInnerRadToleranceLevel(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetOuterRadToleranceLevel() const { return GetUnplacedVolume()->GetOuterRadToleranceLevel(); }

  template <bool ForInnerSurface>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Precision GetHypeRadius2(Precision dz) const
  {
    return GetUnplacedVolume()->GetHypeRadius2<ForInnerSurface>(dz);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool PointOnZSurface(Vector3D<Precision> const &p) const { return GetUnplacedVolume()->PointOnZSurface(p); }

  template <bool ForInnerSurface>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool PointOnHyperbolicSurface(Vector3D<Precision> const &p) const
  {
    return GetUnplacedVolume()->PointOnHyperbolicSurface<ForInnerSurface>(p);
  }

  Precision SurfaceArea() const override { return GetUnplacedVolume()->SurfaceArea(); }

  //  VECCORE_ATT_HOST_DEVICE
  //  VECGEOM_FORCE_INLINE
  //  void ComputeBBox() const { return GetUnplacedVolume()->ComputeBBox();}

  VECGEOM_FORCE_INLINE
  virtual int MemorySize() const override { return sizeof(*this); }

#ifndef VECCORE_CUDA
  virtual VPlacedVolume const *ConvertToUnspecialized() const override;

#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override { return GetUnplacedVolume()->ConvertToRoot(GetName()); }
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const *ConvertToGeant4() const override { return GetUnplacedVolume()->ConvertToGeant4(GetName()); }
#endif

#endif // VECCORE_CUDA
};

template <typename UnplacedHype_t>
class SPlacedHype : public PlacedVolumeImplHelper<UnplacedHype_t, PlacedHype> {
  using Base = PlacedVolumeImplHelper<UnplacedHype_t, PlacedHype>;

public:
  typedef UnplacedHype UnplacedShape_t;
  using Base::Base;
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDHYPE_H_
