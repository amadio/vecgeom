// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// This file contains the declaration of the PlacedParaboloid class
/// @file volumes/PlacedParaboloid.h
/// @author Marilena Bandieramonte

#ifndef VECGEOM_VOLUMES_PLACEDPARABOLOID_H_
#define VECGEOM_VOLUMES_PLACEDPARABOLOID_H_

#include "base/Global.h"

#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/kernel/ParaboloidImplementation.h"
#include "volumes/PlacedVolImplHelper.h"
#include "volumes/UnplacedParaboloid.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedParaboloid;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedParaboloid);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Class for the positioned paraboloid volume
class PlacedParaboloid : public PlacedVolumeImplHelper<UnplacedParaboloid, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedParaboloid, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  // constructor inheritance;
  using Base::Base;

  /// Constructor
  /// @param label Name of logical volume.
  /// @param logicalVolume The logical volume to be positioned.
  /// @param transformation The positioning transformation.
  /// @param boundingBox Pointer to bounding box (may be null); To be deprecated
  PlacedParaboloid(char const *const label, LogicalVolume const *const logicalVolume,
                   Transformation3D const *const transformation, vecgeom::PlacedBox const *const boundingBox)
      : Base(label, logicalVolume, transformation, boundingBox)
  {
  }

  /// Constructor
  /// @param logicalVolume The logical volume to be positioned.
  /// @param transformation The positioning transformation.
  /// @param boundingBox Pointer to bounding box (may be null); To be deprecated
  PlacedParaboloid(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                   vecgeom::PlacedBox const *const boundingBox)
      : PlacedParaboloid("", logicalVolume, transformation, boundingBox)
  {
  }
#else
  /// CUDA version of constructor
  VECCORE_ATT_DEVICE PlacedParaboloid(LogicalVolume const *const logicalVolume,
                                      Transformation3D const *const transformation, PlacedBox const *const boundingBox,
                                      const int id)
      : Base(logicalVolume, transformation, boundingBox, id)
  {
  }
#endif
  /// Destructor
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedParaboloid() {}

  /// Getter for the raduis of the circle at z = -dz
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRlo() const { return GetUnplacedVolume()->GetRlo(); }

  /// Getter for the raduis of the circle at z = +dz
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRhi() const { return GetUnplacedVolume()->GetRhi(); }

  /// Getter for the half size in z
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDz() const { return GetUnplacedVolume()->GetDz(); }

  /// Sets the raduis of the circle at z = -dz
  VECCORE_ATT_HOST_DEVICE
  void SetRlo(Precision arg) { const_cast<UnplacedParaboloid *>(GetUnplacedVolume())->SetRlo(arg); }

  /// Sets the raduis of the circle at z = +dz
  VECCORE_ATT_HOST_DEVICE
  void SetRhi(Precision arg) { const_cast<UnplacedParaboloid *>(GetUnplacedVolume())->SetRhi(arg); }

  /// Sets the half size in z
  VECCORE_ATT_HOST_DEVICE
  void SetDz(Precision arg) { const_cast<UnplacedParaboloid *>(GetUnplacedVolume())->SetDz(arg); }

  VECCORE_ATT_HOST_DEVICE
  virtual void PrintType() const override;
  virtual void PrintType(std::ostream &os) const override;

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

#endif
