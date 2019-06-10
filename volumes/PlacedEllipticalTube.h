// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of the Placed Elliptical Tube volume
/// @file volumes/PlacedEllipticalTube.h
/// @author Raman Sehgal, Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_PLACEDELLIPTICALTUBE_H_
#define VECGEOM_VOLUMES_PLACEDELLIPTICALTUBE_H_

#include "base/Global.h"

#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/kernel/EllipticalTubeImplementation.h"
#include "volumes/PlacedVolImplHelper.h"
#include "volumes/UnplacedEllipticalTube.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedEllipticalTube;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedEllipticalTube);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Class for the positioned elliptical tube volume
class PlacedEllipticalTube : public PlacedVolumeImplHelper<UnplacedEllipticalTube, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedEllipticalTube, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  using Base::Base;

  /// Constructor
  /// @param label Name of logical volume
  /// @param logicalVolume The logical volume to be positioned
  /// @param transformation The positioning transformation
  /// @param boundingBox Pointer to bounding box (may be null); To be deprecated
  PlacedEllipticalTube(char const *const label, LogicalVolume const *const logicalVolume,
                       Transformation3D const *const transformation, vecgeom::PlacedBox const *const boundingBox)
      : Base(label, logicalVolume, transformation, boundingBox)
  {
  }

  /// Constructor
  /// @param logicalVolume The logical volume to be positioned
  /// @param transformation The positioning transformation.
  /// @param boundingBox Pointer to bounding box (may be null); To be deprecated
  PlacedEllipticalTube(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                       vecgeom::PlacedBox const *const boundingBox)
      : PlacedEllipticalTube("", logicalVolume, transformation, boundingBox)
  {
  }
#else
  /// CUDA version of constructor
  VECCORE_ATT_DEVICE PlacedEllipticalTube(LogicalVolume const *const logicalVolume,
                                          Transformation3D const *const transformation,
                                          PlacedBox const *const boundingBox, const int id)
      : Base(logicalVolume, transformation, boundingBox, id)
  {
  }
#endif
  /// Destructor
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedEllipticalTube() {}

  /// Getter for x semi-axis
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDx() const { return GetUnplacedVolume()->GetDx(); }

  /// Getter for y semi-axis
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDy() const { return GetUnplacedVolume()->GetDy(); }

  /// Getter for the half length in z
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDz() const { return GetUnplacedVolume()->GetDz(); }

  /// Setter for the elliptical tube parameters
  /// @param dx Length of x semi-axis
  /// @param dy Length of y semi-axis
  /// @param dz Half length in z
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetParameters(Precision dx, Precision dy, Precision dz)
  {
    const_cast<UnplacedEllipticalTube *>(GetUnplacedVolume())->SetParameters(dx, dy, dz);
  };

  /// Set x semi-axis
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetDx(Precision dx) { const_cast<UnplacedEllipticalTube *>(GetUnplacedVolume())->SetDx(dx); };

  /// Set y semi-axis
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetDy(Precision dy) { const_cast<UnplacedEllipticalTube *>(GetUnplacedVolume())->SetDy(dy); };

  /// Set half length in z
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetDz(Precision dz) { const_cast<UnplacedEllipticalTube *>(GetUnplacedVolume())->SetDz(dz); };

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
