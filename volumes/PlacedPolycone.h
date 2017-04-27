/*
 * PlacedPolycone.h
 *
 *  Created on: Dec 8, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_PLACEDPOLYCONE_H_
#define VECGEOM_VOLUMES_PLACEDPOLYCONE_H_

#include "base/Global.h"
#include "backend/Backend.h"

#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"

#include "volumes/UnplacedPolycone.h"

class UPolyconeHistorical;

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedPolycone;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedPolycone);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedPolycone : public VPlacedVolume {

public:
  typedef UnplacedPolycone UnplacedShape_t;

#ifndef VECCORE_CUDA
  PlacedPolycone(char const *const label, LogicalVolume const *const logical_volume,
                 Transformation3D const *const transformation, PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox)
  {
  }

  PlacedPolycone(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                 PlacedBox const *const boundingBox)
      : PlacedPolycone("", logical_volume, transformation, boundingBox)
  {
  }

#else
  __device__ PlacedPolycone(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox, const int id)
      : VPlacedVolume(logical_volume, transformation, boundingBox, id)
  {
  }
#endif

  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedPolycone() {}

  VECCORE_ATT_HOST_DEVICE
  UnplacedPolycone const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedPolycone const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

  bool IsOpen() const { return (GetUnplacedVolume()->GetDeltaPhi() < kTwoPi); }
  Precision GetStartPhi() const { return GetUnplacedVolume()->GetStartPhi(); }
  Precision GetEndPhi() const { return GetUnplacedVolume()->GetEndPhi(); }
  int GetNumRZCorner() const { return 2 * (int)(GetUnplacedVolume()->GetNz()); } // in USolids nCorners = 2*nPlanes

  PolyconeHistorical *GetOriginalParameters() const
  {
	  return GetUnplacedVolume()->fOriginal_parameters;
  //  assert(false && "*** Method PlacedPolycone::GetOriginalParameters() has been deprecated.\n");
  //  return NULL;
  }
  void Reset()
  {
    assert(
        false &&
        "*** Method PlacedPolycone::Reset() has been deprecated, no 'originalParameters' to be used for reInit().\n");
  }

#if defined(VECGEOM_USOLIDS)
  //  VECCORE_ATT_HOST_DEVICE
  std::ostream &StreamInfo(std::ostream &os) const override { return GetUnplacedVolume()->StreamInfo(os); }
#endif

#ifndef VECCORE_CUDA
  virtual VPlacedVolume const *ConvertToUnspecialized() const override;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override;
#endif
#if defined(VECGEOM_USOLIDS) && !defined(VECGEOM_REPLACE_USOLIDS)
  virtual ::VUSolid const *ConvertToUSolids() const override;
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const *ConvertToGeant4() const override;
#endif

  virtual Precision Capacity() override { return GetUnplacedVolume()->Capacity(); }

  virtual void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

#if defined(VECGEOM_USOLIDS)
  std::string GetEntityType() const override { return GetUnplacedVolume()->GetEntityType(); }
#endif

  // virtual
  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override
  {
    return GetUnplacedVolume()->Normal(point, normal);
  }
  // bool valid;
  // BoxImplementation<translation::kIdentity, rotation::kIdentity>::NormalKernel<kScalar>(
  //*GetUnplacedVolume(),
  // point,
  // normal, valid);
  // return valid;
  //}

  virtual Vector3D<Precision> GetPointOnSurface() const override { return GetUnplacedVolume()->GetPointOnSurface(); }

  virtual double SurfaceArea() override { return GetUnplacedVolume()->SurfaceArea(); }
#endif

}; // end of class

} // end inline namespace

} // end global namespace

#endif /* VECGEOM_VOLUMES_PLACEDPOLYCONE_H_ */
