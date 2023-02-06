// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of the placed Ellipsoid volume
/// @file volumes/PlacedEllipsoid.h
/// @author Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_PLACEDELLIPSOID_H_
#define VECGEOM_VOLUMES_PLACEDELLIPSOID_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/kernel/EllipsoidImplementation.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedEllipsoid.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedEllipsoid;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedEllipsoid);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Class for the positioned ellipsoid volume
class PlacedEllipsoid : public PlacedVolumeImplHelper<UnplacedEllipsoid, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedEllipsoid, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  using Base::Base;

  /// Constructor
  /// @param label Name of logical volume
  /// @param logicalVolume The logical volume to be positioned
  /// @param transformation The positioning transformation
  PlacedEllipsoid(char const *const label, LogicalVolume const *const logicalVolume,
                  Transformation3D const *const transformation)
      : Base(label, logicalVolume, transformation)
  {
  }

  /// Constructor
  /// @param logicalVolume The logical volume to be positioned
  /// @param transformation The positioning transformation.
  PlacedEllipsoid(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation)
      : PlacedEllipsoid("", logicalVolume, transformation)
  {
  }
#else
  /// CUDA version of constructor
  VECCORE_ATT_DEVICE PlacedEllipsoid(LogicalVolume const *const logicalVolume,
                                     Transformation3D const *const transformation, const int id, const int copy_no,
                                     const int child_id)
      : Base(logicalVolume, transformation, id, copy_no, child_id)
  {
  }
#endif
  /// Destructor
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedEllipsoid() {}

  /// Getter for x semi-axis
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDx() const { return GetUnplacedVolume()->GetDx(); }

  /// Getter for y semi-axis
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDy() const { return GetUnplacedVolume()->GetDy(); }

  /// Getter for z semi-axis
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDz() const { return GetUnplacedVolume()->GetDz(); }

  /// Getter for bottom cut in local z, return -dz if the cut is not set
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetZBottomCut() const { return GetUnplacedVolume()->GetZBottomCut(); }

  /// Getter for top cut in local z, return +dz if the cut is not set
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetZTopCut() const { return GetUnplacedVolume()->GetZTopCut(); }

  /// Set ellipsoid dimensions
  /// @param dx Length of x semi-axis
  /// @param dy Length of y semi-axis
  /// @param dy Length of z semi-axis
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetSemiAxes(Precision dx, Precision dy, Precision dz)
  {
    const_cast<UnplacedEllipsoid *>(GetUnplacedVolume())->SetSemiAxes(dx, dy, dz);
  }

  /// Set cuts in z
  /// @param zBottomCut Bottom cut in local z, shape lies above this plane
  /// @param zTopCut Top cut in local z, shape lies below this plane
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetZCuts(Precision zBottomCut, Precision zTopCut)
  {
    const_cast<UnplacedEllipsoid *>(GetUnplacedVolume())->SetZCuts(zBottomCut, zTopCut);
  }

  VECCORE_ATT_HOST_DEVICE
  virtual void PrintType() const override;
  virtual void PrintType(std::ostream &os) const override;

// Comparison specific
#ifndef VECCORE_CUDA
  virtual VPlacedVolume const *ConvertToUnspecialized() const override;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override
  {
    return nullptr; // There is no suitable TGeo shape 2019.06.17
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
