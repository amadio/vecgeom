/// \file PlacedGenTrap.h
/// \author: swenzel
/// Created on: Aug 3, 2014
///  Modified and completed: mihaela.gheata@cern.ch

#ifndef VECGEOM_VOLUMES_PLACEDGENTRAP_H_
#define VECGEOM_VOLUMES_PLACEDGENTRAP_H_

#include "base/Global.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedGenTrap.h"
#include "volumes/kernel/GenTrapImplementation.h"
#include "volumes/PlacedVolImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedGenTrap;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedGenTrap);

inline namespace VECGEOM_IMPL_NAMESPACE {

class Transformation3D;

class PlacedGenTrap : public PlacedVolumeImplHelper<UnplacedGenTrap, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedGenTrap, VPlacedVolume>;

public:
#ifndef VECGEOM_NVCC

  /** @brief PlacedGenTrap constructor
  * @param label Name of the object
  * @param logicalVolume Logical volume
  * @param transformation Transformation matrix (local to mother)
  * @param halfzheight The half-height of the GenTrap
  * @param boundingBox Bounding box
  */
  // constructor inheritance;
  using Base::Base;
  PlacedGenTrap(char const *const label, LogicalVolume const *const logicalVolume,
                Transformation3D const *const transformation, vecgeom::PlacedBox const *const boundingBox)
      : Base(label, logicalVolume, transformation, boundingBox)
  {
  }

  /** @brief PlacedGenTrap constructor
  * @param logicalVolume Logical volume
  * @param transformation Transformation matrix (local to mother)
  * @param halfzheight The half-height of the GenTrap
  * @param boundingBox Bounding box
  */
  PlacedGenTrap(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                vecgeom::PlacedBox const *const boundingBox)
      : PlacedGenTrap("", logicalVolume, transformation, boundingBox)
  {
  }

#else

  /** @brief PlacedGenTrap device constructor
  * @param logicalVolume Logical volume
  * @param transformation Transformation matrix (local to mother)
  * @param halfzheight The half-height of the GenTrap
  * @param boundingBox Bounding box
  */
  __device__ PlacedGenTrap(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation,
                           PlacedBox const *const boundingBox, const int id)
      : Base(logicalVolume, transformation, boundingBox, id)
  {
  }

#endif
  /** @brief PlacedGenTrap destructor */
  VECGEOM_CUDA_HEADER_BOTH
  virtual ~PlacedGenTrap() {}

  /** @brief Getter for unplaced volume */
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedGenTrap const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedGenTrap const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

  /** @brief Getter for one of the 8 vertices in Vector3D<Precision> form */
  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> const &GetVertex(int i) const { return GetUnplacedVolume()->GetVertex(i); }

  /** @brief Getter for the half-height */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDZ() const { return GetUnplacedVolume()->GetDZ(); }

#if !defined(VECGEOM_NVCC)
  /** @brief Interface method for computing capacity */
  virtual Precision Capacity() override { return GetUnplacedVolume()->volume(); }

  /** @brief Computes the extent on X/Y/Z of the trapezoid */
  virtual void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

  /** @brief Shortcut for computing the normal */
  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override
  {
    return GetUnplacedVolume()->Normal(point, normal);
  }

  /** @brief Generates randomly a point on the surface of the trapezoid */
  virtual Vector3D<Precision> GetPointOnSurface() const override { return GetUnplacedVolume()->GetPointOnSurface(); }

  /** @brief Implementation of surface area computation */
  virtual double SurfaceArea() override { return GetUnplacedVolume()->SurfaceArea(); }

#if defined(VECGEOM_USOLIDS)
  /** @brief Get type name */
  virtual std::string GetEntityType() const override { return GetUnplacedVolume()->GetEntityType(); }
#endif
#endif

  /** @brief Print type name */
  VECGEOM_CUDA_HEADER_BOTH
  void PrintType() const override;

  /** @brief Print type name */
  void PrintType(std::ostream &os) const override;

  // CUDA specific

  /** @brief Memory size in bytes */
  VECGEOM_FORCE_INLINE
  virtual int MemorySize() const override { return sizeof(*this); }

// Comparison specific

#if defined(VECGEOM_USOLIDS)
  /** @brief Stream trapezoid information in the USolids style */
  std::ostream &StreamInfo(std::ostream &os) const override { return GetUnplacedVolume()->StreamInfo(os); }
#endif

#ifndef VECGEOM_NVCC
  /** @brief Convert to unspecialized placement */
  virtual VPlacedVolume const *ConvertToUnspecialized() const override;
#ifdef VECGEOM_ROOT
  /** @brief Convert to ROOT shape */
  virtual TGeoShape const *ConvertToRoot() const override;
#endif
#if defined(VECGEOM_USOLIDS) && !defined(VECGEOM_REPLACE_USOLIDS)
  /** @brief Convert to USolids solid */
  virtual ::VUSolid const *ConvertToUSolids() const override;
#endif
#ifdef VECGEOM_GEANT4
  /** @brief Convert to Geant4 solid */
  virtual G4VSolid const *ConvertToGeant4() const override;
#endif

#endif // VECGEOM_NVCC
};
}
} // end global namespace

#endif /* VECGEOM_VOLUMES_PLACEDGENTRAP_H_ */
