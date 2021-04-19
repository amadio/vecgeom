// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \brief Declaration of the logical volume.
/// \file LogicalVolume.h
/// \author created by Johannes de Fine Licht, Sandro Wenzel (CERN)

#ifndef VECGEOM_VOLUMES_LOGICALVOLUME_H_
#define VECGEOM_VOLUMES_LOGICALVOLUME_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector.h"
#include "VecGeom/volumes/UnplacedVolume.h"

#include <string>
#include <list>
#include <set>
#include <typeindex>
#include <typeinfo>

class TGeoShape;

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class LogicalVolume;);
VECGEOM_DEVICE_FORWARD_DECLARE(class VPlacedVolume;);
VECGEOM_DEVICE_FORWARD_DECLARE(class VLevelLocator;);
VECGEOM_DEVICE_FORWARD_DECLARE(class VSafetyEstimator;);
VECGEOM_DEVICE_FORWARD_DECLARE(class VNavigator;);

VECGEOM_DEVICE_DECLARE_CONV(class, LogicalVolume);

inline namespace VECGEOM_IMPL_NAMESPACE {

class Region; // forward declaration for a class which
              // a user needs to implement in his code

class VLevelLocator;
class VSafetyEstimator;
class VNavigator;
typedef VPlacedVolume const *Daughter;
class GeoManager;

/**
 * \brief Class responsible for storing the list of daughter volumes
 *        of an unplaced volume and various associated properties.
 *
 *  A logical volume adds various properties to unplaced volumes, notably
 *  the list of daughter volumes it contains. As such a logical volume
 *  describes an un-positioned sub-tree of the geometry hierarchy.
 *  The class also holds various pointers to structures associated with
 *  unplaced volumes, such as those used by simulation engines for tracking.
 *  This class follows largely the class existing in Geant4: G4LogicalVolume.
 */
class LogicalVolume {
  friend class GeoManager;

private:
  /// Pointer to concrete unplaced volume/shape
  VUnplacedVolume const *fUnplacedVolume;

  int fId; ///< global id of logical volume object

  std::string *fLabel; ///< name of logical volume

  static int gIdCount; ///< a static class counter

  //--- The following pointers are used mainly by simulation packages ---
  // Consider enabling them only when needed

  /// A pointer member to register arbitrary objects with logical volumes.
  /// Included for the moment to model UserExtension like in a TGeoVolume.
  void *fUserExtensionPtr;

  void *fMaterialPtr; ///< Pointer to some user material class (used by Geant-V)

  void *fMaterialCutsPtr; ///< Pointer to some user cuts (used by Geant-V)

  void *fBasketManagerPtr; ///< A specific pointer used by Geant-V

  Region *fRegion = nullptr; ///< Pointer to a region object (following the Geant4 Region concept)

  //--- The following pointers are used by VecGeom itself ---

  VLevelLocator const *fLevelLocator;       ///< A locator class for this logical volume.
  VSafetyEstimator const *fSafetyEstimator; ///< A safety estimator class for this logical volume.
  VNavigator const *fNavigator;             ///< Pointer to attached VecGeom navigator.

  /// The container of daughter (placed) volumes which are placed inside this logical volume
  Vector<Daughter> *fDaughters;

  using CudaDaughter_t = cuda::VPlacedVolume const *;
  friend class CudaManager;

  // possibility to change pointer of daughter volumes ( can be used by GeoManager )
  //  void SetDaughter(unsigned int i, VPlacedVolume const *pvol);

public:
#ifndef VECCORE_CUDA
  /// Standard constructor taking a name and an unplaced volume
  LogicalVolume(char const *const label, VUnplacedVolume const *const unplaced_vol);

  /// Standard constructor taking an unplaced volume
  LogicalVolume(VUnplacedVolume const *const unplaced_vol) : LogicalVolume("", unplaced_vol) {}

  /// copy operators deleted
  LogicalVolume(LogicalVolume const &other) = delete;
  LogicalVolume *operator=(LogicalVolume const &other) = delete;

#else
  VECCORE_ATT_DEVICE
  LogicalVolume(VUnplacedVolume const *const unplaced_vol, int id, Vector<Daughter> *GetDaughter);
#endif

  ~LogicalVolume();

  /// Returns the unplaced volume
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  VUnplacedVolume const *GetUnplacedVolume() const { return fUnplacedVolume; }

  // will be deprecated in favour of better encapsulation of internal storage
  /// Returns the list of daughter volumes
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector<Daughter> const &GetDaughters() const { return *fDaughters; }

  // will be deprecated in favour of better encapsulation of internal storage
  /// Returns pointer to the list of daughter volumes
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector<Daughter> const *GetDaughtersp() const { return fDaughters; }

  /// Returns pointer to the list of daughter volumes
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector<Daughter> *GetDaughtersp() { return fDaughters; }

  /// Returns user request for caching transformations
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsReqCaching() const { return false; }

  //  VECCORE_ATT_HOST_DEVICE
  //  VECGEOM_FORCE_INLINE
  //  VPlacedVolume const* GetDaughter(unsigned int i) const { return daughters_->operator[](i); }
  //
  //  VECCORE_ATT_HOST_DEVICE
  //  VECGEOM_FORCE_INLINE
  //  unsigned int GetNDaughters() const { return daughters_->size(); }

  /// Returns the total number of placeds volume contained in this logical volume AND below.
  size_t GetNTotal() const;

  /// Returns the user extension pointer.
  VECGEOM_FORCE_INLINE
  void *GetUserExtensionPtr() const { return fUserExtensionPtr; }

  /// Returns the material extension pointer.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void *GetMaterialPtr() const { return fMaterialPtr; }

  /// Returns the cuts extension pointer.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void *GetMaterialCutsPtr() const { return fMaterialCutsPtr; }

  /// Returns the basked manager pointer (Geant-V specific).
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void *GetBasketManagerPtr() const { return fBasketManagerPtr; }

  /// Returns the level locator (used by VecGeom navigation).
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  VLevelLocator const *GetLevelLocator() const { return fLevelLocator; }

  /// Sets the level locator (used by VecGeom navigation).
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetLevelLocator(VLevelLocator const *locator) { fLevelLocator = locator; }

  /// Returns the safety estimator (used by VecGeom navigation).
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  VSafetyEstimator const *GetSafetyEstimator() const { return fSafetyEstimator; }

  /// Sets the safety estimator (used by VecGeom navigation).
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetSafetyEstimator(VSafetyEstimator const *est) { fSafetyEstimator = est; }

  /// Returns the navigator for this logical volume (used by VecGeom navigation).
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  VNavigator const *GetNavigator() const { return fNavigator; }

  /// Sets the navigator for this logical volume (used by VecGeom navigation).
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetNavigator(VNavigator const *n) { fNavigator = n; }

  /// Returns the id of this logical volume.
  int id() const { return fId; }

  /// Returns the name of this logical volume.
  const char *GetName() const { return fLabel->c_str(); }
  /// Returns the name of this logical volume as string.
  std::string const &GetLabel() const { return *fLabel; }

  /// Sets the name/label of this logical volume.
  void SetLabel(char const *const label)
  {
    if (fLabel) delete fLabel;
    fLabel = new std::string(label);
  }

  /// Set user extension pointer of this logical volume.
  VECGEOM_FORCE_INLINE
  void SetUserExtensionPtr(void *userpointer) { fUserExtensionPtr = userpointer; }

  /// Set the material pointer of this logical volume.
  VECGEOM_FORCE_INLINE
  void SetMaterialPtr(void *matpointer) { fMaterialPtr = matpointer; }

  /// Set the cuts pointer of this logical volume.
  VECGEOM_FORCE_INLINE
  void SetMaterialCutsPtr(void *matcutpointer) { fMaterialCutsPtr = matcutpointer; }

  /// Set the basket manager pointer of this logical volume (Geant-V specific).
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  void SetBasketManagerPtr(void *basketpointer) { fBasketManagerPtr = basketpointer; }

  /// Print the daughter information of this logical volume with a given indentation.
  VECCORE_ATT_HOST_DEVICE
  void Print(const int indent = 0) const;

  /// Print the daughter information of this logical volume and of all containing daughters
  /// up to a certain depth
  VECCORE_ATT_HOST_DEVICE
  void PrintContent(const int depth = 0) const;

  /// Produce a labeled placed volume out of this logical volume by using a given transformation.
  VPlacedVolume *Place(char const *const label, Transformation3D const *const transformation) const;

  /// Produce a placed volume out of this logical volume by using a given transformation.
  VPlacedVolume *Place(Transformation3D const *const transformation) const;

  /// Produce a labeled placed volume out of this logical volume with identity transformation.
  VPlacedVolume *Place(char const *const label) const;

  /// Produce a trivially placed volume out of this logical volume with identity transformation.
  VPlacedVolume *Place() const;

  /// Adds/places a named daughter volume inside this logical volume.
  /// @param label Name of placed daughter volume.
  /// @param volume The logical volume to be added to the list of daughter volumes.
  /// @param transformation The transformation used to place the daughter.
  VPlacedVolume const *PlaceDaughter(char const *const label, LogicalVolume *const volume,
                                     Transformation3D const *const transformation);

  /// Adds/places a daughter volume inside this logical volume.
  /// @param volume The logical volume to be added to the list of daughter volumes.
  /// @param transformation The transformation used to place the daughter.
  VPlacedVolume const *PlaceDaughter(LogicalVolume *const volume, Transformation3D const *const transformation);

  /// Adds/places an already existing placed daughter volume
  void PlaceDaughter(VPlacedVolume *const placed);

  /// Returns true of at least one of the daughters is an assembly.
  bool ContainsAssembly() const;

  friend std::ostream &operator<<(std::ostream &os, LogicalVolume const &vol);

  /// Returns the region object associated to this logical volume.
  Region *GetRegion() { return fRegion; }

  /** Sets the Region pointer for this logical Volume.
   * @param region The Region object that  will be associated to this logical volume.
   * @param pushdown If pushdown=true this region will be applied to all logical volumes below (in the geom hierarchy)
   *
   * Set region will override any existing region. The user has to make sure to call in the right order.
   */
  void SetRegion(Region *region, bool pushdown = true);

#ifdef VECGEOM_CUDA_INTERFACE
  DevicePtr<cuda::LogicalVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const unplaced_vol, int id,
                                           DevicePtr<cuda::Vector<CudaDaughter_t>> GetDaughter) const;
  DevicePtr<cuda::LogicalVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const unplaced_vol, int id,
                                           DevicePtr<cuda::Vector<CudaDaughter_t>> GetDaughter,
                                           DevicePtr<cuda::LogicalVolume> const gpu_ptr) const;
#endif

private:
  std::set<LogicalVolume *> GetSetOfDaughterLogicalVolumes() const;

}; // End class

inline void LogicalVolume::SetRegion(Region *region, bool pushdown)
{
  fRegion = region;
  if (pushdown) {
    for (auto &lv : GetSetOfDaughterLogicalVolumes()) {
      lv->SetRegion(region, true);
    }
  }
}

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_LOGICALVOLUME_H_
