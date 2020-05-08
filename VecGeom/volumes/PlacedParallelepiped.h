// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of the placed parallelepiped volume.
/// @file volumes/PlacedParallelepiped.h
/// @author Johannes de Fine Licht, Mihaela Gheata

#ifndef VECGEOM_VOLUMES_PLACEDPARALLELEPIPED_H_
#define VECGEOM_VOLUMES_PLACEDPARALLELEPIPED_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/kernel/ParallelepipedImplementation.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedParallelepiped.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedParallelepiped;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedParallelepiped);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Class for the positioned parallelepiped volume
class PlacedParallelepiped : public PlacedVolumeImplHelper<UnplacedParallelepiped, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedParallelepiped, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  using Base::Base;
  /// Constructor
  /// @param label Name of logical volume.
  /// @param logical_volume The logical volume to be positioned.
  /// @param transformation The positioning transformation.
  /// @param boundingBox Pointer to bounding box (may be null); To be deprecated
  PlacedParallelepiped(char const *const label, LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation, vecgeom::PlacedBox const *const boundingBox)
      : Base(label, logical_volume, transformation, boundingBox)
  {
  }

  /// Constructor
  /// @param logical_volume The logical volume to be positioned.
  /// @param transformation The positioning transformation.
  /// @param boundingBox Pointer to bounding box (may be null); To be deprecated
  PlacedParallelepiped(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                       vecgeom::PlacedBox const *const boundingBox)
      : PlacedParallelepiped("", logical_volume, transformation, boundingBox)
  {
  }

#else
  /// CUDA version of constructor
  VECCORE_ATT_DEVICE PlacedParallelepiped(LogicalVolume const *const logical_volume,
                                          Transformation3D const *const transformation,
                                          PlacedBox const *const boundingBox, const int id, const int copy_no,
                                          const int child_id)
      : Base(logical_volume, transformation, boundingBox, id, copy_no, child_id)
  {
  }

#endif
  /// Destructor
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedParallelepiped() {}

  /// Getter for unplaced volume
  VECCORE_ATT_HOST_DEVICE
  UnplacedParallelepiped const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedParallelepiped const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

  /// Accessor for dimensions
  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> const &GetDimensions() const { return GetUnplacedVolume()->GetDimensions(); }

  /// Accessor for x dimension
  VECCORE_ATT_HOST_DEVICE
  Precision GetX() const { return GetUnplacedVolume()->GetX(); }

  /// Accessor for y dimension
  VECCORE_ATT_HOST_DEVICE
  Precision GetY() const { return GetUnplacedVolume()->GetY(); }

  /// Accessor for z dimension
  VECCORE_ATT_HOST_DEVICE
  Precision GetZ() const { return GetUnplacedVolume()->GetZ(); }

  /// Accessor for alpha angle
  VECCORE_ATT_HOST_DEVICE
  Precision GetAlpha() const { return GetUnplacedVolume()->GetAlpha(); }

  /// Accessor for polar angle
  VECCORE_ATT_HOST_DEVICE
  Precision GetTheta() const { return GetUnplacedVolume()->GetTheta(); }

  /// Accessor for azimuthal angle
  VECCORE_ATT_HOST_DEVICE
  Precision GetPhi() const { return GetUnplacedVolume()->GetPhi(); }

  /// Returns tan(alpha)
  VECCORE_ATT_HOST_DEVICE
  Precision GetTanAlpha() const { return GetUnplacedVolume()->GetTanAlpha(); }

  /// Returns tan(alpha)*sin(phi)
  VECCORE_ATT_HOST_DEVICE
  Precision GetTanThetaSinPhi() const { return GetUnplacedVolume()->GetTanThetaSinPhi(); }

  /// Returns tan(alpha)*cos(phi)
  VECCORE_ATT_HOST_DEVICE
  Precision GetTanThetaCosPhi() const { return GetUnplacedVolume()->GetTanThetaCosPhi(); }

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

#endif // VECGEOM_VOLUMES_PLACEDPARALLELEPIPED_H_
