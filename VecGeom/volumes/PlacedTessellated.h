// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \brief Declaration of the data structure for the tessellated shape.
/// \file volumes/PlacedTessellated.h
/// \author Mihaela Gheata (CERN/ISS)

#ifndef VECGEOM_VOLUMES_PLACEDTESSELLATED_H_
#define VECGEOM_VOLUMES_PLACEDTESSELLATED_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/kernel/TessellatedImplementation.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedTessellated.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedTessellated;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedTessellated);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Class representing a placed tessellated volume.
class PlacedTessellated : public PlacedVolumeImplHelper<UnplacedTessellated, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedTessellated, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  // constructor inheritance;
  using Base::Base;

  /// Constructor
  /// @param label Name of logical volume
  /// @param logicalVolume The logical volume to be positioned
  /// @param transformation The positioning transformation
  /// @param boundingBox Pointer to bounding box (may be null); To be deprecated
  PlacedTessellated(char const *const label, LogicalVolume const *const logicalVolume,
                    Transformation3D const *const transformation, vecgeom::PlacedBox const *const boundingBox)
      : Base(label, logicalVolume, transformation, boundingBox)
  {
  }

  /// Constructor
  /// @param logicalVolume The logical volume to be positioned
  /// @param transformation The positioning transformation.
  /// @param boundingBox Pointer to bounding box (may be null); To be deprecated
  PlacedTessellated(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                    vecgeom::PlacedBox const *const boundingBox)
      : PlacedTessellated("", logicalVolume, transformation, boundingBox)
  {
  }
#else
  /// CUDA version of constructor
  __device__ PlacedTessellated(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                               PlacedBox const *const boundingBox, const int id)
      : Base(logicalVolume, transformation, boundingBox, id)
  {
  }
#endif

  /// Destructor
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedTessellated() {}

  /// Getter for the UnplacedTessellated
  VECCORE_ATT_HOST_DEVICE
  UnplacedTessellated const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedTessellated const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

  /// Computation of the extent.
  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

#ifndef VECCORE_CUDA
  /// Computation of the capacity.
  virtual Precision Capacity() override { return GetUnplacedVolume()->Capacity(); }

  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override
  {
    return GetUnplacedVolume()->Normal(point, normal);
  }

  /** @brief Memory size in bytes */
  VECGEOM_FORCE_INLINE
  virtual int MemorySize() const override { return sizeof(*this); }

  virtual VPlacedVolume const *ConvertToUnspecialized() const override;

#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override;
#endif
#ifdef VECGEOM_GEANT4
  G4VSolid const *ConvertToGeant4() const override;
#endif
#endif // VECCORE_CUDA
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDTUBE_H_
