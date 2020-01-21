// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \brief Declaration of the placed Orb volume
/// \file volumes/PlacedOrb.h
/// \author Raman Sehgal

/// 2015: initial version (Raman Sehgal)
/// 2016: cleanup; move to PlacedVolImplHelper (Raman Sehgal)

#ifndef VECGEOM_VOLUMES_PLACEDORB_H_
#define VECGEOM_VOLUMES_PLACEDORB_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/kernel/OrbImplementation.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedOrb.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedOrb;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedOrb);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedOrb : public PlacedVolumeImplHelper<UnplacedOrb, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedOrb, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  /// Constructor
  /// @param label Name of logical volume
  /// @param logicalVolume The logical volume to be positioned
  /// @param transformation The positioning transformation
  /// @param boundingBox Pointer to bounding box (may be null); To be deprecated
  using Base::Base;
  PlacedOrb(char const *const label, LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation, vecgeom::PlacedBox const *const boundingBox)
      : Base(label, logicalVolume, transformation, boundingBox)
  {
  }

  /// Constructor
  /// @param logicalVolume The logical volume to be positioned
  /// @param transformation The positioning transformation.
  /// @param boundingBox Pointer to bounding box (may be null); To be deprecated
  PlacedOrb(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
            vecgeom::PlacedBox const *const boundingBox)
      : PlacedOrb("", logicalVolume, transformation, boundingBox)
  {
  }
#else
  /// CUDA version of constructor
  VECCORE_ATT_DEVICE PlacedOrb(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                               PlacedBox const *const boundingOrb, const int id)
      : Base(logicalVolume, transformation, boundingOrb, id)
  {
  }
#endif

  /// Destructor
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedOrb() {}

  /// Getter for Radius
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRadius() const { return GetUnplacedVolume()->GetRadius(); }

  /// Setter for Radius
  VECCORE_ATT_HOST_DEVICE
  void SetRadius(Precision arg) { const_cast<UnplacedOrb *>(GetUnplacedVolume())->SetRadius(arg); }

  VECCORE_ATT_HOST_DEVICE
  virtual void PrintType() const override;
  virtual void PrintType(std::ostream &os) const override;

// Comparison specific
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

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
