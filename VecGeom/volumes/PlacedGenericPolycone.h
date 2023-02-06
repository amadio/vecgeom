/// @file PlacedGenericPolycone.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDGENERICPOLYCONE_H_
#define VECGEOM_VOLUMES_PLACEDGENERICPOLYCONE_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/kernel/GenericPolyconeImplementation.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedGenericPolycone.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedGenericPolycone;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedGenericPolycone);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedGenericPolycone : public PlacedVolumeImplHelper<UnplacedGenericPolycone, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedGenericPolycone, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  // constructor inheritance;
  using Base::Base;
  PlacedGenericPolycone(char const *const label, LogicalVolume const *const logicalVolume,
                        Transformation3D const *const transformation)
      : Base(label, logicalVolume, transformation)
  {
  }

  PlacedGenericPolycone(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation)
      : PlacedGenericPolycone("", logicalVolume, transformation)
  {
  }
#else
  VECCORE_ATT_DEVICE PlacedGenericPolycone(LogicalVolume const *const logicalVolume,
                                           Transformation3D const *const transformation, const int id,
                                           const int copy_no, const int child_id)
      : Base(logicalVolume, transformation, id, copy_no, child_id)
  {
  }
#endif
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedGenericPolycone() {}

  /*
   * Put the Required Getters and Setters here
   */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSPhi() const { return GetUnplacedVolume()->GetSPhi(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDPhi() const { return GetUnplacedVolume()->GetDPhi(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int GetNumRz() const { return GetUnplacedVolume()->GetNumRz(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector<Precision> GetR() const { return GetUnplacedVolume()->GetR(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector<Precision> GetZ() const { return GetUnplacedVolume()->GetZ(); }

  VECCORE_ATT_HOST_DEVICE
  virtual void PrintType() const override;
  virtual void PrintType(std::ostream &os) const override;

// Comparison specific
#ifndef VECCORE_CUDA
  virtual VPlacedVolume const *ConvertToUnspecialized() const override;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override
  {
    return nullptr; // There is no corresponding TGeo shape
  }
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const *ConvertToGeant4() const override;
#endif
#endif // VECCORE_CUDA
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
