//===-- volumes/PlacedParaboloid.h - Instruction class definition -------*- C++ -*-===//
///
/// \file volumes/PlacedParaboloid.h
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)
/// \brief This file contains the declaration of the PlacedParaboloid class
//===----------------------------------------------------------------------===//
///
/// revision + moving to new backend structure : Raman Sehgal (raman.sehgal@cern.ch)

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

class PlacedParaboloid : public PlacedVolumeImplHelper<UnplacedParaboloid, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedParaboloid, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  // constructor inheritance;
  using Base::Base;
  PlacedParaboloid(char const *const label, LogicalVolume const *const logicalVolume,
                   Transformation3D const *const transformation, vecgeom::PlacedBox const *const boundingBox)
      : Base(label, logicalVolume, transformation, boundingBox)
  {
  }

  PlacedParaboloid(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                   vecgeom::PlacedBox const *const boundingBox)
      : PlacedParaboloid("", logicalVolume, transformation, boundingBox)
  {
  }
#else
  VECCORE_ATT_DEVICE PlacedParaboloid(LogicalVolume const *const logicalVolume,
                                      Transformation3D const *const transformation, PlacedBox const *const boundingBox,
                                      const int id)
      : Base(logicalVolume, transformation, boundingBox, id)
  {
  }
#endif
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedParaboloid() {}

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRlo() const { return GetUnplacedVolume()->GetRlo(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRhi() const { return GetUnplacedVolume()->GetRhi(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDz() const { return GetUnplacedVolume()->GetDz(); }

  VECCORE_ATT_HOST_DEVICE
  void SetRlo(Precision arg) { const_cast<UnplacedParaboloid *>(GetUnplacedVolume())->SetRlo(arg); }

  VECCORE_ATT_HOST_DEVICE
  void SetRhi(Precision arg) { const_cast<UnplacedParaboloid *>(GetUnplacedVolume())->SetRhi(arg); }

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

} // end inline namespace
} // End global namespace

#endif
