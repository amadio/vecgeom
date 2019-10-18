#pragma once

#include "base/Global.h"
#include "base/Vector3D.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

class LogicalVolume;
class VPlacedVolume;
class NavigationState;

//! Pure virtual base class for LevelLocators

//! Pure abstract base class for LevelLocators
//! which are classes providing functions
//! to locate a track in a logical volume; Via the virtual
//! interface it is possible to write custom locators which are specialized for
//! a given logical volume and its content
class VLevelLocator {

public:
  /**
   * Function which takes a logical volume and a local point in the reference frame of the logical volume
   * and which determines in which daughter ( or the logical volume ) itself the given point is located
   *
   *
   * @param  lvol is a logical volume
   * @param  localpoint is a point in the coordinate frame of the logical volume and should be contained within it
   * @param  daughterpvol is the placed volume in which the localpoint is contained (result of the computation)
   * @param  daughterlocalpoint is the local point in the next pvol (result of the computation)
   * @return true of point is in a daughter; false otherwise
   */
  VECCORE_ATT_HOST_DEVICE
  virtual bool LevelLocate(LogicalVolume const * /*lvol*/, Vector3D<Precision> const & /*localpoint*/,
                           VPlacedVolume const *& /*pvol*/, Vector3D<Precision> & /*daughterlocalpoint*/) const = 0;

  /**
   * Function which takes a logical volume and a local point in the reference frame of the logical volume
   * and which determines in which daughter ( or the logical volume ) itself the given point is located
   *
   *
   * @param  lvol is a logical volume
   * @param  localpoint is a point in the coordinate frame of the logical volume and should be contained within it
   * @param  outstate is a navigationstate which gets modified to point to the correct volume within this level (result
   * of the computation)
   * @param  daughterlocalpoint is the local point in the next pvol (result of the computation)
   * @return true of point is in a daughter; false otherwise
   */
  VECCORE_ATT_HOST_DEVICE
  virtual bool LevelLocate(LogicalVolume const * /*lvol*/, Vector3D<Precision> const & /*localpoint*/,
                           NavigationState & /*outstate*/, Vector3D<Precision> & /*daughterlocalpoint*/) const = 0;

  /**
   * Function which takes a logical volume and a local point in the reference frame of the logical volume
   * and which determines in which daughter ( or the logical volume ) itself the given point is located
   *
   *
   * @param  lvol is a logical volume
   * @param  pvol a physical volume to be excluded
   * @param  localpoint is a point in the coordinate frame of the logical volume and should be contained within it
   * @param  daughterpvol is the placed volume in which the localpoint is contained (result of the computation)
   * @param  daughterlocalpoint is the local point in the next pvol (result of the computation)
   * @return true of point is in a daughter; false otherwise
   */
  VECCORE_ATT_HOST_DEVICE
  virtual bool LevelLocateExclVol(LogicalVolume const * /*lvol*/, VPlacedVolume const * /*pvol excl*/,
                                  Vector3D<Precision> const & /*localpoint*/, VPlacedVolume const *& /*pvol*/,
                                  Vector3D<Precision> & /*daughterlocalpoint*/) const = 0;

  /**
   * Function which takes a logical volume and a ray (local point + local direction) in the reference frame of the
   * logical volume and which determines in which daughter ( or the logical volume ) itself the given point is located.
   * This version resembles the logic done in Geant4.
   *
   * @param  lvol is a logical volume
   * @param  pvol a physical volume to be excluded
   * @param  localpoint is a point in the coordinate frame of the logical volume and should be contained within it
   * @param  localdir is a direction in the coordinate frame of the logical volume
   * @param  daughterpvol is the placed volume in which the localpoint is contained (result of the computation)
   * @param  daughterlocalpoint is the local point in the next pvol (result of the computation)
   * @return true if point is in a daughter; false otherwise
   */
  VECCORE_ATT_HOST_DEVICE
  virtual bool LevelLocateExclVol(LogicalVolume const * /*lvol*/, VPlacedVolume const * /*pvol excl*/,
                                  Vector3D<Precision> const & /*localpoint*/, Vector3D<Precision> const & /*localdir*/,
                                  VPlacedVolume const *& /*pvol*/,
                                  Vector3D<Precision> & /*daughterlocalpoint*/) const = 0;

  virtual std::string GetName() const = 0;

  virtual ~VLevelLocator() {}

}; // end class declaration
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
