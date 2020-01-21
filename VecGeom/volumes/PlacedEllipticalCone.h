// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of the Placed Elliptical Cone volume.
/// @file volumes/PlacedEllipticalCone.h
/// @author Raman Sehgal and Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_PLACEDELLIPTICALCONE_H_
#define VECGEOM_VOLUMES_PLACEDELLIPTICALCONE_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/kernel/EllipticalConeImplementation.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedEllipticalCone.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedEllipticalCone;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedEllipticalCone);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Class for the positioned elliptical cone volume
class PlacedEllipticalCone : public PlacedVolumeImplHelper<UnplacedEllipticalCone, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedEllipticalCone, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  using Base::Base;

  /// Constructor
  /// @param label Name of logical volume
  /// @param logicalVolume The logical volume to be positioned
  /// @param transformation The positioning transformation
  /// @param boundingBox Pointer to bounding box (may be null); To be deprecated
  PlacedEllipticalCone(char const *const label, LogicalVolume const *const logicalVolume,
                       Transformation3D const *const transformation, vecgeom::PlacedBox const *const boundingBox)
      : Base(label, logicalVolume, transformation, boundingBox)
  {
  }

  /// Constructor
  /// @param logicalVolume The logical volume to be positioned
  /// @param transformation The positioning transformation.
  /// @param boundingBox Pointer to bounding box (may be null); To be deprecated
  PlacedEllipticalCone(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                       vecgeom::PlacedBox const *const boundingBox)
      : PlacedEllipticalCone("", logicalVolume, transformation, boundingBox)
  {
  }
#else
  /// CUDA version of constructor
  VECCORE_ATT_DEVICE PlacedEllipticalCone(LogicalVolume const *const logicalVolume,
                                          Transformation3D const *const transformation,
                                          PlacedBox const *const boundingBox, const int id)
      : Base(logicalVolume, transformation, boundingBox, id)
  {
  }
#endif
  /// Destructor
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedEllipticalCone() {}

  /// Getter for the parameter that specifies inclination of the conical surface in x
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSemiAxisX() const { return GetUnplacedVolume()->GetSemiAxisX(); }

  /// Getter for the parameter that specifies inclination of the conical surface in y
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSemiAxisY() const { return GetUnplacedVolume()->GetSemiAxisY(); }

  /// Getter for the height
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetZMax() const { return GetUnplacedVolume()->GetZMax(); }

  /// Getter for the cut in z
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetZTopCut() const { return GetUnplacedVolume()->GetZTopCut(); }

  /// Setter for the elliptical cone parameters
  /// @param a Inclination of the conical surface in x
  /// @param b Inclination of the conical surface in y
  /// @param h Height
  /// @param zcut cut in z
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetParameters(Precision a, Precision b, Precision h, Precision zcut)
  {
    const_cast<UnplacedEllipticalCone *>(GetUnplacedVolume())->SetParameters(a, b, h, zcut);
  };

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
