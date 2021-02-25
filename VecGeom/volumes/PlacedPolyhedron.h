/// \file PlacedPolyhedron.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDPOLYHEDRON_H_
#define VECGEOM_VOLUMES_PLACEDPOLYHEDRON_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/kernel/PolyhedronImplementation.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedPolyhedron.h"

class UPolyhedraHistorical;
struct PolyhedraSideRZ {   // Avoid clash in class name UPolyhedraSideRZ
  vecgeom::Precision r, z; // start of vector
};

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedPolyhedron;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedPolyhedron);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedPolyhedron : public PlacedVolumeImplHelper<UnplacedPolyhedron, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedPolyhedron, VPlacedVolume>;

public:
  typedef UnplacedPolyhedron UnplacedShape_t;

#ifndef VECCORE_CUDA
  // constructor inheritance;
  using Base::Base;
  PlacedPolyhedron(char const *const label, LogicalVolume const *const logicalVolume,
                   Transformation3D const *const transformation)
      : Base(label, logicalVolume, transformation)
  {
  }

  PlacedPolyhedron(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation)
      : PlacedPolyhedron("", logicalVolume, transformation)
  {
  }
#else
  VECCORE_ATT_DEVICE PlacedPolyhedron(LogicalVolume const *const logicalVolume,
                                      Transformation3D const *const transformation, const int id, const int copy_no,
                                      const int child_id)
      : Base(logicalVolume, transformation, id, copy_no, child_id)
  {
  }
#endif

  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedPolyhedron() {}

  // Accessors

  VECCORE_ATT_HOST_DEVICE
  UnplacedPolyhedron const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedPolyhedron const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int GetSideCount() const { return GetUnplacedVolume()->GetSideCount(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int GetZSegmentCount() const { return GetUnplacedVolume()->GetZSegmentCount(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool HasInnerRadii() const { return GetUnplacedVolume()->HasInnerRadii(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool HasPhiCutout() const { return GetUnplacedVolume()->HasPhiCutout(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  ZSegment const &GetZSegment(int index) const { return GetUnplacedVolume()->GetZSegment(index); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Array<ZSegment> const &GetZSegments() const { return GetUnplacedVolume()->GetZSegments(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Array<Precision> const &GetZPlanes() const { return GetUnplacedVolume()->GetZPlanes(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Array<Precision> const &GetRMin() const { return GetUnplacedVolume()->GetRMin(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Array<Precision> const &GetRMax() const { return GetUnplacedVolume()->GetRMax(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<Precision> GetPhiSection(int i) const { return GetUnplacedVolume()->GetPhiSection(i); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  SOA3D<Precision> const &GetPhiSections() const { return GetUnplacedVolume()->GetPhiSections(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetPhiStart() const { return GetUnplacedVolume()->GetPhiStart(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetPhiEnd() const { return GetUnplacedVolume()->GetPhiEnd(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetPhiDelta() const { return GetUnplacedVolume()->GetPhiDelta(); }

  VECCORE_ATT_HOST_DEVICE
  int PhiSegmentIndex(Vector3D<Precision> const &point) const;

  bool IsOpen() const { return (GetUnplacedVolume()->GetPhiDelta() < kTwoPi); }
  bool IsGeneric() const { return false; }
  Precision GetStartPhi() const { return GetUnplacedVolume()->GetPhiStart(); }
  Precision GetEndPhi() const { return GetUnplacedVolume()->GetPhiEnd(); }
  int GetNumSide() const { return GetUnplacedVolume()->GetSideCount(); }
  int GetNumRZCorner() const { return GetZPlanes().size(); }

  // CUDA specific

  virtual int MemorySize() const override { return sizeof(*this); }

// Comparison specific
#ifndef VECCORE_CUDA
  virtual VPlacedVolume const *ConvertToUnspecialized() const override;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override;
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const *ConvertToGeant4() const override;
#endif
#endif // VECCORE_CUDA
};

} // namespace VECGEOM_IMPL_NAMESPACE

} // namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDPOLYHEDRON_H_
