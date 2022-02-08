// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of the placed Trd volume
/// @file volumes/PlacedTrd.h
/// @author Georgios Bitzes

#ifndef VECGEOM_VOLUMES_PLACEDTRD_H_
#define VECGEOM_VOLUMES_PLACEDTRD_H_

#include "VecGeom/base/Global.h"
#ifndef VECCORE_CUDA
#include "VecGeom/base/RNG.h"
#include <cmath>
#endif
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/kernel/TrdImplementation.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedTrd.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedTrd;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedTrd);

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Class for the positioned Trd volume
class PlacedTrd : public VPlacedVolume {

public:
  using VPlacedVolume::VPlacedVolume;
#ifndef VECCORE_CUDA
  /// Constructor
  /// @param label Name of logical volume.
  /// @param logicalVolume The logical volume to be positioned.
  /// @param transformation The positioning transformation.
  PlacedTrd(char const *const label, LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation)
      : VPlacedVolume(label, logicalVolume, transformation)
  {
  }

  /// Constructor
  /// @param logicalVolume The logical volume to be positioned.
  /// @param transformation The positioning transformation.
  PlacedTrd(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation)
      : PlacedTrd("", logicalVolume, transformation)
  {
  }
#else
  /// CUDA version of constructor
  VECCORE_ATT_DEVICE PlacedTrd(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                               const int id, const int copy_no, const int child_id)
      : VPlacedVolume(logicalVolume, transformation, id, copy_no, child_id)
  {
  }
#endif

  /// Destructor
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedTrd() {}

  /// Getter for unplaced volume
  VECCORE_ATT_HOST_DEVICE
  UnplacedTrd const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedTrd const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

  VECCORE_ATT_HOST_DEVICE
  virtual void PrintType() const override;
  virtual void PrintType(std::ostream &s) const override;

  /// Getter for half-length along x at -dz
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dx1() const { return GetUnplacedVolume()->dx1(); }

  /// Getter for half-length along x at +dz
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dx2() const { return GetUnplacedVolume()->dx2(); }

  /// Getter for half-length along y at -dz
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dy1() const { return GetUnplacedVolume()->dy1(); }

  /// Getter for half-length along y at +dz
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dy2() const { return GetUnplacedVolume()->dy2(); }

  /// Getter for half-length along z
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision dz() const { return GetUnplacedVolume()->dz(); }

  /// Getter for half-length along x at -dz
  Precision GetXHalfLength1() const { return GetUnplacedVolume()->dx1(); }

  /// Getter for half-length along x at +dz
  Precision GetXHalfLength2() const { return GetUnplacedVolume()->dx2(); }

  /// Getter for half-length along y at -dz
  Precision GetYHalfLength1() const { return GetUnplacedVolume()->dy1(); }

  /// Getter for half-length along y at +dz
  Precision GetYHalfLength2() const { return GetUnplacedVolume()->dy2(); }

  /// Getter for half-length along z
  Precision GetZHalfLength() const { return GetUnplacedVolume()->dz(); }

  /// Setter for half-length along x at -dz
  void SetXHalfLength1(Precision arg) { const_cast<UnplacedTrd *>(GetUnplacedVolume())->SetXHalfLength1(arg); }

  /// Setter for half-length along x at +dz
  void SetXHalfLength2(Precision arg) { const_cast<UnplacedTrd *>(GetUnplacedVolume())->SetXHalfLength2(arg); }

  /// Setter for half-length along y at -dz
  void SetYHalfLength1(Precision arg) { const_cast<UnplacedTrd *>(GetUnplacedVolume())->SetYHalfLength1(arg); }

  /// Setter for half-length along y at +dz
  void SetYHalfLength2(Precision arg) { const_cast<UnplacedTrd *>(GetUnplacedVolume())->SetYHalfLength2(arg); }

  /// Setter for half-length along z
  void SetZHalfLength(Precision arg) { const_cast<UnplacedTrd *>(GetUnplacedVolume())->SetZHalfLength(arg); }

  /// Setter for all parameters
  /// @param x1 Half-length along x at the surface positioned at -dz
  /// @param x2 Half-length along x at the surface positioned at +dz
  /// @param y1 Half-length along y at the surface positioned at -dz
  /// @param y2 Half-length along y at the surface positioned at +dz
  /// @param z Half-length along z axis
  void SetAllParameters(Precision x1, Precision x2, Precision y1, Precision y2, Precision z)
  {
    const_cast<UnplacedTrd *>(GetUnplacedVolume())->SetAllParameters(x1, x2, y1, y2, z);
  }

#ifndef VECCORE_CUDA
  /// Returns memory size in bytes
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

template <typename UnplacedTrd_t>
class SPlacedTrd : public PlacedVolumeImplHelper<UnplacedTrd_t, PlacedTrd> {
  using Base = PlacedVolumeImplHelper<UnplacedTrd_t, PlacedTrd>;

public:
  typedef UnplacedTrd UnplacedShape_t;
  using Base::Base;
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDTUBE_H_
