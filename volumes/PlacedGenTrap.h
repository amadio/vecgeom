/*
 * PlacedGenTrap.h
 *
 *  Created on: Aug 3, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_PLACEDGENTRAP_H_
#define VECGEOM_VOLUMES_PLACEDGENTRAP_H_


#include "base/Global.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedGenTrap.h"
#include "volumes/kernel/GenTrapImplementation.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class PlacedGenTrap; )
VECGEOM_DEVICE_DECLARE_CONV( PlacedGenTrap )

inline namespace VECGEOM_IMPL_NAMESPACE {

class Transformation3D;

class PlacedGenTrap : public VPlacedVolume {

public:

#ifndef VECGEOM_NVCC

  PlacedGenTrap(char const *const label,
            LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation,
            PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logicalVolume, transformation, boundingBox) {}

  PlacedGenTrap(LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation,
            PlacedBox const *const boundingBox)
      : PlacedGenTrap("", logicalVolume, transformation, boundingBox) {}

#else

  __device__
  PlacedGenTrap(LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation,
            PlacedBox const *const boundingBox,
            const int id)
      : VPlacedVolume(logicalVolume, transformation, boundingBox, id) {}

#endif
  VECGEOM_CUDA_HEADER_BOTH
  virtual ~PlacedGenTrap() {}

  // Accessors

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedGenTrap const* GetUnplacedVolume() const {
    return static_cast<UnplacedGenTrap const *>(
        GetLogicalVolume()->GetUnplacedVolume());
  }

  // method giving access to members of unplaced ...
  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> const & GetVertex(int i) const
  {
      return GetUnplacedVolume()->GetVertex(i);
  }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDZ() const { return GetUnplacedVolume()->GetDZ(); }

#if !defined(VECGEOM_NVCC)
  virtual Precision Capacity() override {
      return GetUnplacedVolume()->volume();
  }

  virtual
  void Extent(Vector3D<Precision> & aMin, Vector3D<Precision> & aMax) const override
  {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

  virtual
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal ) const override
  {
//    return false;

    bool valid;
    GenTrapImplementation<translation::kIdentity, rotation::kIdentity>::NormalKernel<kScalar>(
              *GetUnplacedVolume(),
              point,
              normal, valid);
    return valid;

  }

  virtual
  Vector3D<Precision> GetPointOnSurface() const override {
    return GetUnplacedVolume()->GetPointOnSurface();
  }

  virtual double SurfaceArea() override {
     return GetUnplacedVolume()->SurfaceArea();
  }

#if defined(VECGEOM_USOLIDS)
  virtual std::string GetEntityType() const override { return GetUnplacedVolume()->GetEntityType() ;}
#endif
#endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const override;

  // CUDA specific

  VECGEOM_INLINE
  virtual int memory_size() const override { return sizeof(*this); }

  // Comparison specific

#if defined(VECGEOM_USOLIDS)
  std::ostream& StreamInfo(std::ostream &os) const override {
    return GetUnplacedVolume()->StreamInfo(os);
  }
#endif

#ifndef VECGEOM_NVCC
  virtual VPlacedVolume const* ConvertToUnspecialized() const override;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const* ConvertToRoot() const override;
#endif
#if defined(VECGEOM_USOLIDS) && !defined(VECGEOM_REPLACE_USOLIDS)
  virtual ::VUSolid const* ConvertToUSolids() const override;
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const* ConvertToGeant4() const override;
#endif

#endif // VECGEOM_NVCC

};

} } // end global namespace


#endif /* VECGEOM_VOLUMES_PLACEDGENTRAP_H_ */
