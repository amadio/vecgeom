// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of the placed tetrahedron volume
/// @file volumes/PlacedTet.h
/// @author Raman Sehgal, Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_PLACEDTET_H_
#define VECGEOM_VOLUMES_PLACEDTET_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/kernel/TetImplementation.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedTet.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedTet;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedTet);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Class for the positioned tetrahedron volume
class PlacedTet : public PlacedVolumeImplHelper<UnplacedTet, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedTet, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  // constructor inheritance;
  using Base::Base;

  /// Constructor
  /// @param label Name of logical volume
  /// @param logicalVolume The logical volume to be positioned
  /// @param transformation The positioning transformation
  /// @param boundingBox Pointer to bounding box (may be null); To be deprecated
  PlacedTet(char const *const label, LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation, vecgeom::PlacedBox const *const boundingBox)
      : Base(label, logicalVolume, transformation, boundingBox)
  {
  }

  /// Constructor
  /// @param logicalVolume The logical volume to be positioned
  /// @param transformation The positioning transformation.
  /// @param boundingBox Pointer to bounding box (may be null); To be deprecated
  PlacedTet(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
            vecgeom::PlacedBox const *const boundingBox)
      : PlacedTet("", logicalVolume, transformation, boundingBox)
  {
  }
#else
  /// CUDA version of constructor
  VECCORE_ATT_DEVICE PlacedTet(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                               PlacedBox const *const boundingTet, const int id)
      : Base(logicalVolume, transformation, boundingTet, id)
  {
  }
#endif
  /// Destructor
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedTet() {}

  /// Getter for the tetrahedron vertices
  /// @param [out] p0 Point given as 3D vector
  /// @param [out] p1 Point given as 3D vector
  /// @param [out] p2 Point given as 3D vector
  /// @param [out] p3 Point given as 3D vector
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void GetVertices(Vector3D<Precision> &p0, Vector3D<Precision> &p1, Vector3D<Precision> &p2,
                   Vector3D<Precision> &p3) const
  {
    GetUnplacedVolume()->GetVertices(p0, p1, p2, p3);
  }

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
