/// @file PlacedEllipticalTube.h
/// @author Raman Sehgal (raman.sehgal@cern.ch), Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

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

class PlacedEllipticalTube : public PlacedVolumeImplHelper<UnplacedEllipticalTube, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedEllipticalTube, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA
  // constructor inheritance;
  using Base::Base;
  PlacedEllipticalTube(char const *const label, LogicalVolume const *const logicalVolume,
                       Transformation3D const *const transformation,
                       vecgeom::PlacedBox const *const boundingEllipticalTube)
      : Base(label, logicalVolume, transformation, boundingEllipticalTube)
  {
  }

  PlacedEllipticalTube(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                       vecgeom::PlacedBox const *const boundingEllipticalTube)
      : PlacedEllipticalTube("", logicalVolume, transformation, boundingEllipticalTube)
  {
  }
#else
  VECCORE_ATT_DEVICE PlacedEllipticalTube(LogicalVolume const *const logicalVolume,
                                          Transformation3D const *const transformation,
                                          PlacedBox const *const boundingEllipticalTube, const int id)
      : Base(logicalVolume, transformation, boundingEllipticalTube, id)
  {
  }
#endif
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedEllipticalTube() {}

  /*Vector3D<Precision> GetSomething() const { return GetUnplacedVolume()->GetSomething(); }*/
  // Getters and Setters if required
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDx() const { return GetUnplacedVolume()->GetDx(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDy() const { return GetUnplacedVolume()->GetDy(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDz() const { return GetUnplacedVolume()->GetDz(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetParameters(Precision dx, Precision dy, Precision dz)
  {
    const_cast<UnplacedEllipticalTube *>(GetUnplacedVolume())->SetParameters(dx, dy, dz);
  };

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetDx(Precision dx) { const_cast<UnplacedEllipticalTube *>(GetUnplacedVolume())->SetDx(dx); };

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetDy(Precision dy) { const_cast<UnplacedEllipticalTube *>(GetUnplacedVolume())->SetDy(dy); };

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
