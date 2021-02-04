/// \file PlacedGenTrap.h
/// \author: swenzel
/// Created on: Aug 3, 2014
///  Modified and completed: mihaela.gheata@cern.ch

#ifndef VECGEOM_VOLUMES_PLACEDGENTRAP_H_
#define VECGEOM_VOLUMES_PLACEDGENTRAP_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedGenTrap.h"
#include "VecGeom/volumes/kernel/GenTrapImplementation.h"
#include "VecGeom/volumes/PlacedVolImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedGenTrap;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedGenTrap);

inline namespace VECGEOM_IMPL_NAMESPACE {

class Transformation3D;

class PlacedGenTrap : public PlacedVolumeImplHelper<UnplacedGenTrap, VPlacedVolume> {
  using Base = PlacedVolumeImplHelper<UnplacedGenTrap, VPlacedVolume>;

public:
#ifndef VECCORE_CUDA

  /** @brief PlacedGenTrap constructor
   * @param label Name of the object
   * @param logicalVolume Logical volume
   * @param transformation Transformation matrix (local to mother)
   * @param halfzheight The half-height of the GenTrap
   */
  // constructor inheritance;
  using Base::Base;
  PlacedGenTrap(char const *const label, LogicalVolume const *const logicalVolume,
                Transformation3D const *const transformation)
      : Base(label, logicalVolume, transformation)
  {
  }

  /** @brief PlacedGenTrap constructor
   * @param logicalVolume Logical volume
   * @param transformation Transformation matrix (local to mother)
   * @param halfzheight The half-height of the GenTrap
   */
  PlacedGenTrap(LogicalVolume const *const logicalVolume, Transformation3D const *const transformation)
      : PlacedGenTrap("", logicalVolume, transformation)
  {
  }

#else

  /** @brief PlacedGenTrap device constructor
   * @param logicalVolume Logical volume
   * @param transformation Transformation matrix (local to mother)
   * @param halfzheight The half-height of the GenTrap
   */
  VECCORE_ATT_DEVICE PlacedGenTrap(LogicalVolume const *const logicalVolume,
                                   Transformation3D const *const transformation, const int id, const int copy_no,
                                   const int child_id)
      : Base(logicalVolume, transformation, id, copy_no, child_id)
  {
  }

#endif
  /** @brief PlacedGenTrap destructor */
  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedGenTrap() {}

  /** @brief Getter for unplaced volume */
  VECCORE_ATT_HOST_DEVICE
  UnplacedGenTrap const *GetUnplacedVolume() const
  {
    return static_cast<UnplacedGenTrap const *>(GetLogicalVolume()->GetUnplacedVolume());
  }

  /** @brief Getter for one of the 8 vertices in Vector3D<Precision> form */
  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> const &GetVertex(int i) const { return GetUnplacedVolume()->GetVertex(i); }

  /** @brief Getter for the half-height */
  VECCORE_ATT_HOST_DEVICE
  Precision GetDZ() const { return GetUnplacedVolume()->GetDZ(); }

#if !defined(VECCORE_CUDA)
  /** @brief Interface method for computing capacity */
  virtual Precision Capacity() override { return GetUnplacedVolume()->volume(); }

  /** @brief Computes the extent on X/Y/Z of the trapezoid */
  virtual void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

  /** @brief Shortcut for computing the normal */
  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override
  {
    return GetUnplacedVolume()->Normal(point, normal);
  }
#endif

  /** @brief Print type name */
  VECCORE_ATT_HOST_DEVICE
  void PrintType() const override;

  /** @brief Print type name */
  void PrintType(std::ostream &os) const override;

  // CUDA specific

  /** @brief Memory size in bytes */
  VECGEOM_FORCE_INLINE
  virtual int MemorySize() const override { return sizeof(*this); }

  // Comparison specific

#ifndef VECCORE_CUDA
  /** @brief Convert to unspecialized placement */
  virtual VPlacedVolume const *ConvertToUnspecialized() const override;
#ifdef VECGEOM_ROOT
  /** @brief Convert to ROOT shape */
  virtual TGeoShape const *ConvertToRoot() const override;
#endif
#ifdef VECGEOM_GEANT4
  /** @brief Convert to Geant4 solid */
  virtual G4VSolid const *ConvertToGeant4() const override;
#endif

#endif // VECCORE_CUDA
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* VECGEOM_VOLUMES_PLACEDGENTRAP_H_ */
