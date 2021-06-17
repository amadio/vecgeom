// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \file PlacedVolume.h
/// \author created by Johannes de Fine Licht, Sandro Wenzel (CERN)

#ifndef VECGEOM_VOLUMES_PLACEDVOLUME_H_
#define VECGEOM_VOLUMES_PLACEDVOLUME_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include <string>

#ifdef VECGEOM_GEANT4
#include <G4VSolid.hh>
#endif

using Real_v = vecgeom::VectorBackend::Real_v;

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class VPlacedVolume;);
VECGEOM_DEVICE_DECLARE_CONV(class, VPlacedVolume);
#ifndef VECCORE_CUDA
template <>
struct kCudaType<const cxx::VPlacedVolume *> {
  using type_t = const cuda::VPlacedVolume *;
};
#endif

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedBox;
class GeoManager;
template <typename T>
class SOA3D;

/*!
 * \brief A placed volume is a positioned logical volume.
 *
 * Special features of VecGeom are that:
 *
 *   1. Placed volumes offer
 *      similar geometry interfaces (distance, etc) as unplaced volumes for user convenience.
 *   2. We can have sub-types of placed volumes, specialized according
 *      to underlying unplaced volume type as well as per categories
 *      of placing transformations.
 *
 * Given a logical volume and a transformation, factory methods
 * will generate the most suitable sub-type of a placed volume so that
 * geometry APIs, such as VPlacedVolume::DistanceToIn(position, direction)
 * are optimized as much as possible. Some CMake flags allow to select how
 * far this is done or not.
 */
class VPlacedVolume {
  friend class GeoManager;

private:
  unsigned int id_;  ///< Integer id
  int copy_no_ = 0;  ///< Copy number for the physical volume, used by transport
  int ichild_  = -1; ///< Index in the mother volume list;

  // Use a pointer so the string won't be constructed on the GPU
  std::string *label_;            ///< Label/name of placed volume
  static unsigned int g_id_count; ///< Static instance counter

protected:
  LogicalVolume const *logical_volume_; ///< Pointer to positioned logical volume
#ifdef VECGEOM_INPLACE_TRANSFORMATIONS
  Transformation3D fTransformation; ///< The positioning transformation
#else
  Transformation3D const *fTransformation; ///< The positioning transformation
#endif

#ifndef VECCORE_CUDA

  /** Constructor
   * \param label Name of logical volume.
   * \param logical_vol The logical volume to be positioned.
   * \param transform The positioning transformation.
   */
  VPlacedVolume(char const *const label, LogicalVolume const *const logical_vol,
                Transformation3D const *const transform);

  /** Constructor
   * \param logical_vol The logical volume to be positioned.
   * \param transform The positioning transformation.
   */
  VPlacedVolume(LogicalVolume const *const logical_vol, Transformation3D const *const transform)
      : VPlacedVolume("", logical_vol, transform)
  {
  }

#else
  /// CUDA version of constructor
  VECCORE_ATT_DEVICE VPlacedVolume(LogicalVolume const *const logical_vol, Transformation3D const *const transformation,
                                   unsigned int id, int copy_no, int ichild)
#ifdef VECGEOM_INPLACE_TRANSFORMATIONS
      : logical_volume_(logical_vol), fTransformation(*transformation), id_(id), copy_no_(copy_no), ichild_(ichild),
        label_(NULL)
  {
  }
#else
      : logical_volume_(logical_vol), fTransformation(transformation), id_(id), copy_no_(copy_no), ichild_(ichild),
        label_(NULL)
  {
  }
#endif
#endif

  VECGEOM_FORCE_INLINE
  void SetChildId(int index) { ichild_ = index; }

public:
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume(VPlacedVolume const &);
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume *operator=(VPlacedVolume const &);

  VECCORE_ATT_HOST_DEVICE
  virtual ~VPlacedVolume();

  /// Returns integer index associated to this volume.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  unsigned int id() const { return id_; }

  /// Returns copy number.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int GetChildId() const { return ichild_; }

  /// LogicalVolume::PlaceDaughters is a friend that can set the child index
  friend void LogicalVolume::PlaceDaughter(VPlacedVolume *const placed);

  /// Returns copy number.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int GetCopyNo() const { return copy_no_; }

  /// Returns value of static instance counter
  static unsigned int GetIdCount() { return g_id_count; }

  /// Returns name/label.
  std::string const &GetLabel() const { return *label_; }

  /// Returns underlying logical volume.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  LogicalVolume const *GetLogicalVolume() const { return logical_volume_; }

  /// Returns daughter container of logical volume.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector<Daughter> const &GetDaughters() const { return logical_volume_->GetDaughters(); }

  /// Finds the index of a given daughter having its pointer (linear complexity)
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int IndexOf(Daughter daughter) const
  {
    int id = 0;
    for (auto d : logical_volume_->GetDaughters()) {
      if (d == daughter) return id;
      id++;
    }
    return -1;
  }

  /// Returns name/label.
  VECCORE_ATT_HOST
  VECGEOM_FORCE_INLINE
  const char *GetName() const { return (*label_).c_str(); }

  /// Returns unplaced volume encapsulated in the logical volume.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  VUnplacedVolume const *GetUnplacedVolume() const { return logical_volume_->GetUnplacedVolume(); }

  /// Returns if underlying unplaced volume is an assembly.
  VECCORE_ATT_HOST_DEVICE
  bool IsAssembly() const { return GetUnplacedVolume()->IsAssembly(); }

  /// Returns underlying transformation.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Transformation3D const *GetTransformation() const
  {
#ifdef VECGEOM_INPLACE_TRANSFORMATIONS
    return &fTransformation;
#else
    return fTransformation;
#endif
  }
#ifndef VECCORE_CUDA
  SolidMesh *CreateMesh3D(size_t nFaces) const
  {
    return GetUnplacedVolume()->CreateMesh3D(*this->GetTransformation(), nFaces);
  }
#endif
  /// Sets logical volume.
  VECCORE_ATT_HOST_DEVICE
  void SetLogicalVolume(LogicalVolume const *const logical_vol) { logical_volume_ = logical_vol; }

  /// Sets transformation.
  VECCORE_ATT_HOST_DEVICE
  void SetTransformation(Transformation3D const *const transform)
  {
#ifdef VECGEOM_INPLACE_TRANSFORMATIONS
    fTransformation = *transform;
#else
    fTransformation = transform;
#endif
  }

  /// Sets name/label.
  void set_label(char const *label)
  {
    if (label_) delete label_;
    label_ = new std::string(label);
  }

  /// Sets copy number.
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetCopyNo(int copy_no) { copy_no_ = copy_no; }

  friend std::ostream &operator<<(std::ostream &os, VPlacedVolume const &vol);

  /// Returns in-memory size in bytes of deriving objects (used to copy to GPU).
  virtual int MemorySize() const = 0;

  /// Print info about placed volume.
  VECCORE_ATT_HOST_DEVICE
  virtual void Print(const int indent = 0) const;

  /// Print info about placed volume.
  VECCORE_ATT_HOST_DEVICE
  virtual void PrintType() const = 0;

  // some functions allowing for some very basic "introspection"

  /// Print the actual volume type to an outstream
  virtual void PrintType(std::ostream &os) const = 0;
  /// Print the implementation struct of this volume to an outstream
  virtual void PrintImplementationType(std::ostream &os) const = 0;
  /// Print the unplaced type to an outstream
  virtual void PrintUnplacedType(std::ostream &os) const = 0;

  /// Returns translation code/enumeration of the placed volume
  virtual int GetTransCode() const { return translation::kGeneric; }

  /// Returns rotation code/enumeration of the placed volume
  virtual int GetRotCode() const { return rotation::kGeneric; }

  /// Recursively prints contained volumes to standard output.
  VECCORE_ATT_HOST_DEVICE
  void PrintContent(const int depth = 0) const;

  // Geometry functionality like in unplaced volume but taking the placement
  // into account.

  /*!
   * Returns whether a space point is contained or not in the placed volume.
   * This is similar to the functionality in VUnplacedVolume but taking into account
   * the positioning of the shape due to the placement.
   */
  VECCORE_ATT_HOST_DEVICE
  virtual bool Contains(Vector3D<Precision> const &point) const = 0;

  virtual void Contains(SOA3D<Precision> const &point, bool *const output) const = 0;

  /*!
   * Returns whether a space point is contained or not in the placed volume.
   * Also returns the transformed position.
   *
   * \param point A given space point.
   * \param localPoint The point in the natural reference frame of the shape.
   *
   * This is similar to the functionality in VUnplacedVolume but taking into account
   * the positioning of the shape due to the placement.
   */
  VECCORE_ATT_HOST_DEVICE
  virtual bool Contains(Vector3D<Precision> const &point, Vector3D<Precision> &localPoint) const = 0;

  /// Direct dispatch to Contains of underlying unplaced volume without coordinate/placement transformation.
  VECCORE_ATT_HOST_DEVICE
  virtual bool UnplacedContains(Vector3D<Precision> const &localPoint) const = 0;

  /**
   * Like similar function in VUnplacedVolume but taking into account
   * the positioning of the shape due to the placement.
   */
  VECCORE_ATT_HOST_DEVICE
  virtual EnumInside Inside(Vector3D<Precision> const &point) const = 0;

  virtual void Inside(SOA3D<Precision> const &point, Inside_t *const output) const = 0;

  /**
   * Like similar function in VUnplacedVolume but taking into account
   * the positioning of the shape due to the placement.
   */
  VECCORE_ATT_HOST_DEVICE
  virtual Precision DistanceToIn(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                                 const Precision step_max = kInfLength) const = 0;

  /**
   * Like similar function in VUnplacedVolume but taking into account
   * the positioning of the shape due to the placement.
   */
  // if we have any SIMD backend, we offer a SIMD interface
  virtual Real_v DistanceToInVec(Vector3D<Real_v> const &position, Vector3D<Real_v> const &direction,
                                 const Real_v step_max = kInfLength) const = 0;

  /**
   * Like similar function in VUnplacedVolume but taking into account
   * the positioning of the shape due to the placement.
   */
  template <typename T>
  VECGEOM_FORCE_INLINE
  T DistanceToIn(Vector3D<T> const &position, Vector3D<T> const &direction, const T step_max = T(kInfLength)) const
  {
    return DistanceToInVec(position, direction, step_max);
  }

  /**
   * Like similar function in VUnplacedVolume but taking into account
   * the positioning of the shape due to the placement.
   */
  virtual void DistanceToIn(SOA3D<Precision> const &position, SOA3D<Precision> const &direction,
                            Precision const *const step_max, Precision *const output) const = 0;

  /**
   * Like similar function in VUnplacedVolume. Here position and direction are supposed to be
   * in the frame of the placed volume!
   */
  VECCORE_ATT_HOST_DEVICE
  virtual Precision DistanceToOut(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                                  Precision const step_max = kInfLength) const = 0;

  /**
   * Like similar function in VUnplacedVolume. Here position and direction are supposed to be
   * in the frame of the placed volume!
   */
  virtual Real_v DistanceToOutVec(Vector3D<Real_v> const &position, Vector3D<Real_v> const &direction,
                                  Real_v const step_max = kInfLength) const = 0;

  /**
   * Like similar function in VUnplacedVolume. Here position and direction are supposed to be
   * in the frame of the placed volume!
   */
  template <typename T>
  VECGEOM_FORCE_INLINE
  T DistanceToOut(Vector3D<T> const &position, Vector3D<T> const &direction, const T step_max = T(kInfLength)) const
  {
    return DistanceToOutVec(position, direction, step_max);
  }

  /** A "placed" version of the DistanceToOut function; here
   * the point and direction are first of all transformed into the reference frame of the
   * shape. So given a position and direction in the reference frame in which the placed volume
   * is positioned, we transform everything into the coordinate system of the placed volume and
   * calculate DistanceToOut from there.
   */
  VECCORE_ATT_HOST_DEVICE
  virtual Precision PlacedDistanceToOut(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                                        Precision const step_max = kInfLength) const = 0;

  /// to be deprecated
  virtual void DistanceToOut(SOA3D<Precision> const &position, SOA3D<Precision> const &direction,
                             Precision const *const step_max, Precision *const output) const = 0;

  /// to be deprecated
  virtual void DistanceToOut(SOA3D<Precision> const &position, SOA3D<Precision> const &direction,
                             Precision const *const step_max, Precision *const output,
                             int *const nextnodeindex) const = 0;

  /**
   * Like similar function in VUnplacedVolume but taking into account
   * the positioning of the shape due to the placement.
   */
  VECCORE_ATT_HOST_DEVICE
  virtual Precision SafetyToIn(Vector3D<Precision> const &position) const = 0;

  /**
   * Like similar function in VUnplacedVolume but taking into account
   * the positioning of the shape due to the placement.
   */
  virtual Real_v SafetyToInVec(Vector3D<Real_v> const &position) const = 0;

  /**
   * Like similar function in VUnplacedVolume but taking into account
   * the positioning of the shape due to the placement.
   */
  template <typename T>
  VECGEOM_FORCE_INLINE
  T SafetyToIn(Vector3D<T> const &p) const
  {
    return SafetyToInVec(p);
  }

  /// to be deprecated
  virtual void SafetyToIn(SOA3D<Precision> const &position, Precision *const safeties) const = 0;

  /// to be deprecated
  virtual void SafetyToInMinimize(SOA3D<Precision> const &points, Precision *const safeties) const = 0;

  /**
   * Like similar function in VUnplacedVolume. Here position is supposed to be
   * in the frame of the placed volume.
   */
  VECCORE_ATT_HOST_DEVICE
  virtual Precision SafetyToOut(Vector3D<Precision> const &position) const = 0;

  /**
   * Like similar function in VUnplacedVolume. Here position is supposed to be
   * in the frame of the placed volume.
   */
  virtual Real_v SafetyToOutVec(Vector3D<Real_v> const &position) const = 0;

  /// to be deprecated
  virtual void SafetyToOut(SOA3D<Precision> const &position, Precision *const safeties) const = 0;

  template <typename T>
  VECGEOM_FORCE_INLINE
  T SafetyToOut(Vector3D<T> const &p) const
  {
    return SafetyToOutVec(p);
  }

  /// Simple forward to capacity on VUnplacedVolume
  virtual Precision Capacity();

  /// Get Extent of VUnplacedVolume, then apply transformation and recalculate
  VECCORE_ATT_HOST_DEVICE
  virtual void Extent(Vector3D<Precision> & /* min */, Vector3D<Precision> & /* max */) const;

  /// Get Normal of VUnplacedVolume, then apply transformation
  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const & /*point*/, Vector3D<Precision> & /*normal*/) const;

  /// Like SurfaceArea on VUnplacedVolume
  virtual Precision SurfaceArea() const = 0;

public:
#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const                                                                       = 0;
  virtual DevicePtr<cuda::VPlacedVolume> CopyToGpu(DevicePtr<cuda::LogicalVolume> const logical_volume,
                                                   DevicePtr<cuda::Transformation3D> const transform,
                                                   DevicePtr<cuda::VPlacedVolume> const gpu_ptr) const      = 0;
  virtual DevicePtr<cuda::VPlacedVolume> CopyToGpu(DevicePtr<cuda::LogicalVolume> const logical_volume,
                                                   DevicePtr<cuda::Transformation3D> const transform) const = 0;
  /**
   * Copy many instances of this class to the GPU.
   * \param host_volumes Host volumes to be copied. These should all be of the same type as the class that this function is called with.
   * \param logical_volumes GPU addresses of the logical volumes corresponding to the placed volumes.
   * \param transforms GPU addresses of the transformations corresponding to the placed volumes.
   * \param in_gpu_ptrs GPU addresses where the GPU instances of the host volumes should be placed.
   * \note This requires an explicit template instantiation of ConstructManyOnGpu<ThisClass_t>().
   * \see VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL
   */
  virtual void CopyManyToGpu(std::vector<VPlacedVolume const *> const & host_volumes,
                             std::vector<DevicePtr<cuda::LogicalVolume>> const & logical_volumes,
                             std::vector<DevicePtr<cuda::Transformation3D>> const & transforms,
                             std::vector<DevicePtr<cuda::VPlacedVolume>> const & in_gpu_ptrs) const = 0;

  template <typename Derived>
  DevicePtr<cuda::VPlacedVolume> CopyToGpuImpl(DevicePtr<cuda::LogicalVolume> const logical_volume,
                                               DevicePtr<cuda::Transformation3D> const transform,
                                               DevicePtr<cuda::VPlacedVolume> const in_gpu_ptr) const
  {
    DevicePtr<CudaType_t<Derived>> gpu_ptr(in_gpu_ptr);
    gpu_ptr.Construct(logical_volume, transform, nullptr, this->id(), this->GetCopyNo(), this->GetChildId());
    CudaAssertError();
    // Need to go via the void* because the regular c++ compilation
    // does not actually see the declaration for the cuda version
    // (and thus can not determine the inheritance).
    return DevicePtr<cuda::VPlacedVolume>((void *)gpu_ptr);
  }
  template <typename Derived>
  DevicePtr<cuda::VPlacedVolume> CopyToGpuImpl(DevicePtr<cuda::LogicalVolume> const logical_volume,
                                               DevicePtr<cuda::Transformation3D> const transform) const
  {
    DevicePtr<CudaType_t<Derived>> gpu_ptr;
    gpu_ptr.Allocate();
    return this->CopyToGpuImpl<Derived>(logical_volume, transform, DevicePtr<cuda::VPlacedVolume>((void *)gpu_ptr));
  }

#endif

#ifndef VECCORE_CUDA
  /// A conversion function creating a generic unspecialized instance of the placed volume
  virtual VPlacedVolume const *ConvertToUnspecialized() const = 0;
#ifdef VECGEOM_ROOT
  /// A conversion function to a TGeoShape (when TGeo support is available)
  virtual TGeoShape const *ConvertToRoot() const = 0;
#endif
#ifdef VECGEOM_GEANT4
  /// A conversion function to a Geant4 G4VSolid (when G4 support is available)
  virtual G4VSolid const *ConvertToGeant4() const;
#endif
#endif // VECCORE_CUDA
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#ifdef VECCORE_CUDA

/**
 * Trigger template instantiations of DevicePtr<Type>::SizeOf() and DevicePtr<Type>::Construct.
 */
#define VECGEOM_DEVICE_INST_PLACED_VOLUME(PlacedVol)                                                       \
  namespace cxx {                                                                                          \
  template size_t DevicePtr<cuda::PlacedVol>::SizeOf();                                                    \
  template void DevicePtr<cuda::PlacedVol>::Construct(DevicePtr<cuda::LogicalVolume> const logical_volume, \
                                                      DevicePtr<cuda::Transformation3D> const transform,   \
                                                      const unsigned int id, const int copy_no,            \
                                                      const int child_id) const;                           \
  }

/**
 * Trigger template instantiations of DevicePtr<Type>::SizeOf() and DevicePtr<Type>::Construct.
 */
#define VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol, Extra)                                                  \
  namespace cxx {                                                                                                 \
  template size_t DevicePtr<cuda::PlacedVol, Extra>::SizeOf();                                                    \
  template void DevicePtr<cuda::PlacedVol, Extra>::Construct(DevicePtr<cuda::LogicalVolume> const logical_volume, \
                                                             DevicePtr<cuda::Transformation3D> const transform,   \
                                                             const unsigned int id, const int copy_no,            \
                                                             const int child_id) const;                           \
  template void ConstructManyOnGpu<cuda::PlacedVol, Extra>(                                                       \
      std::size_t nElement, DevicePtr<cuda::VPlacedVolume> const * gpu_ptrs,                                      \
      DevicePtr<cuda::LogicalVolume> const * logical, DevicePtr<cuda::Transformation3D> const * trafo,            \
      decltype(std::declval<VPlacedVolume>().id()) const * ids,                                                   \
      decltype(std::declval<VPlacedVolume>().GetCopyNo()) const * copyNos,                                        \
      decltype(std::declval<VPlacedVolume>().GetChildId()) const * childIds);                                     \
  }

#if defined(VECGEOM_NO_SPECIALIZATION) || !defined(VECGEOM_CUDA_VOLUME_SPECIALIZATION)

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT(PlacedVol, trans) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<trans, rotation::kGeneric>)

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(PlacedVol) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT(PlacedVol, translation::kGeneric)

#else // VECGEOM_NO_SPECIALIZATION

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT(PlacedVol, trans)             \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<trans, rotation::kGeneric>)  \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<trans, rotation::kDiagonal>) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<trans, rotation::kIdentity>) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<trans, 0x046>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<trans, 0x054>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<trans, 0x062>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<trans, 0x076>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<trans, 0x0a1>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<trans, 0x0ad>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<trans, 0x0dc>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<trans, 0x0e3>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<trans, 0x10a>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<trans, 0x11b>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<trans, 0x155>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<trans, 0x16a>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<trans, 0x18e>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<trans, 0x1b1>)

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(PlacedVol)                  \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT(PlacedVol, translation::kGeneric) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT(PlacedVol, translation::kIdentity)

#endif // VECGEOM_NO_SPECIALIZATION

#if defined(VECGEOM_NO_SPECIALIZATION) || !defined(VECGEOM_CUDA_VOLUME_SPECIALIZATION)

#define VECGEOM_DEVICE_INST_PLACED_POLYHEDRON_ALL_CUTOUT(PlacedVol, radii) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<radii, Polyhedron::EPhiCutout::kGeneric>)

#define VECGEOM_DEVICE_INST_PLACED_POLYHEDRON_ALLSPEC(PlacedVol) \
  VECGEOM_DEVICE_INST_PLACED_POLYHEDRON_ALL_CUTOUT(PlacedVol, Polyhedron::EInnerRadii::kGeneric)

#else // VECGEOM_NO_SPECIALIZATION

#define VECGEOM_DEVICE_INST_PLACED_POLYHEDRON_ALL_CUTOUT(PlacedVol, radii)                   \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<radii, Polyhedron::EPhiCutout::kGeneric>) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<radii, Polyhedron::EPhiCutout::kFalse>)   \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<radii, Polyhedron::EPhiCutout::kTrue>)    \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol<radii, Polyhedron::EPhiCutout::kLarge>)

#define VECGEOM_DEVICE_INST_PLACED_POLYHEDRON_ALLSPEC(PlacedVol)                                 \
  VECGEOM_DEVICE_INST_PLACED_POLYHEDRON_ALL_CUTOUT(PlacedVol, Polyhedron::EInnerRadii::kGeneric) \
  VECGEOM_DEVICE_INST_PLACED_POLYHEDRON_ALL_CUTOUT(PlacedVol, Polyhedron::EInnerRadii::kFalse)   \
  VECGEOM_DEVICE_INST_PLACED_POLYHEDRON_ALL_CUTOUT(PlacedVol, Polyhedron::EInnerRadii::kTrue)

#endif // VECGEOM_NO_SPECIALIZATION

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3(PlacedVol, Extra, Type)                                      \
  namespace cxx {                                                                                             \
  template size_t DevicePtr<cuda::PlacedVol, Extra, cuda::Type>::SizeOf();                                    \
  template void DevicePtr<cuda::PlacedVol, Extra, cuda::Type>::Construct(                                     \
      DevicePtr<cuda::LogicalVolume> const logical_volume, DevicePtr<cuda::Transformation3D> const transform, \
      const unsigned int id, const int copy_no, const int child_id) const;                                    \
  template void ConstructManyOnGpu<cuda::PlacedVol, Extra, cuda::Type>(                                       \
      std::size_t nElement, DevicePtr<cuda::VPlacedVolume> const * gpu_ptrs,                                  \
      DevicePtr<cuda::LogicalVolume> const * logical, DevicePtr<cuda::Transformation3D> const * trafo,        \
      decltype(std::declval<VPlacedVolume>().id()) const * ids,                                               \
      decltype(std::declval<VPlacedVolume>().GetCopyNo()) const * copyNos,                                    \
      decltype(std::declval<VPlacedVolume>().GetChildId()) const * childIds);                                 \
  }

#if defined(VECGEOM_NO_SPECIALIZATION) || !defined(VECGEOM_CUDA_VOLUME_SPECIALIZATION)

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_3(PlacedVol, trans, Type) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3(PlacedVol<trans, rotation::kGeneric, Type>)
#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(PlacedVol, Type) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_3(PlacedVol, translation::kGeneric, Type)

#else // VECGEOM_NO_SPECIALIZATION

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_3(PlacedVol, trans, Type)             \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3(PlacedVol<trans, rotation::kGeneric, Type>)  \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3(PlacedVol<trans, rotation::kDiagonal, Type>) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3(PlacedVol<trans, rotation::kIdentity, Type>) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3(PlacedVol<trans, 0x046, Type>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3(PlacedVol<trans, 0x054, Type>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3(PlacedVol<trans, 0x062, Type>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3(PlacedVol<trans, 0x076, Type>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3(PlacedVol<trans, 0x0a1, Type>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3(PlacedVol<trans, 0x0ad, Type>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3(PlacedVol<trans, 0x0dc, Type>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3(PlacedVol<trans, 0x0e3, Type>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3(PlacedVol<trans, 0x10a, Type>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3(PlacedVol<trans, 0x11b, Type>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3(PlacedVol<trans, 0x155, Type>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3(PlacedVol<trans, 0x16a, Type>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3(PlacedVol<trans, 0x18e, Type>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_3(PlacedVol<trans, 0x1b1, Type>)

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_3(PlacedVol, Type)                  \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_3(PlacedVol, translation::kGeneric, Type) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_3(PlacedVol, translation::kIdentity, Type)

#endif // VECGEOM_NO_SPECIALIZATION

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_4(PlacedVol, trans, radii, phi)                                \
  namespace cxx {                                                                                             \
  template size_t DevicePtr<cuda::PlacedVol, trans, radii, phi>::SizeOf();                                    \
  template void DevicePtr<cuda::PlacedVol, trans, radii, phi>::Construct(                                     \
      DevicePtr<cuda::LogicalVolume> const logical_volume, DevicePtr<cuda::Transformation3D> const transform, \
      const unsigned int id, const int copy_no, const int child_id) const;                                    \
  template void ConstructManyOnGpu<cuda::PlacedVol, trans, radii, phi>(                                       \
      std::size_t nElement, DevicePtr<cuda::VPlacedVolume> const * gpu_ptrs,                                  \
      DevicePtr<cuda::LogicalVolume> const * logical, DevicePtr<cuda::Transformation3D> const * trafo,        \
      decltype(std::declval<VPlacedVolume>().id()) const * ids,                                               \
      decltype(std::declval<VPlacedVolume>().GetCopyNo()) const * copyNos,                                    \
      decltype(std::declval<VPlacedVolume>().GetChildId()) const * childIds);                                 \
  }

#if defined(VECGEOM_NO_SPECIALIZATION) || !defined(VECGEOM_CUDA_VOLUME_SPECIALIZATION)

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_4(PlacedVol, trans, radii, phi) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_4(PlacedVol<trans, rotation::kGeneric, radii, phi>)
#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_4(PlacedVol)                                                     \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_4(PlacedVol, translation::kGeneric, Polyhedron::EInnerRadii::kGeneric, \
                                              Polyhedron::EPhiCutout::kGeneric)

#else

// Really we should be enumerating the option
#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_4(PlacedVol, trans, radii, phi)             \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_4(PlacedVol<trans, rotation::kGeneric, radii, phi>)  \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_4(PlacedVol<trans, rotation::kDiagonal, radii, phi>) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_4(PlacedVol<trans, rotation::kIdentity, radii, phi>) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_4(PlacedVol<trans, 0x046, radii, phi>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_4(PlacedVol<trans, 0x054, radii, phi>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_4(PlacedVol<trans, 0x062, radii, phi>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_4(PlacedVol<trans, 0x076, radii, phi>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_4(PlacedVol<trans, 0x0a1, radii, phi>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_4(PlacedVol<trans, 0x0ad, radii, phi>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_4(PlacedVol<trans, 0x0dc, radii, phi>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_4(PlacedVol<trans, 0x0e3, radii, phi>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_4(PlacedVol<trans, 0x10a, radii, phi>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_4(PlacedVol<trans, 0x11b, radii, phi>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_4(PlacedVol<trans, 0x155, radii, phi>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_4(PlacedVol<trans, 0x16a, radii, phi>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_4(PlacedVol<trans, 0x18e, radii, phi>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_4(PlacedVol<trans, 0x1b1, radii, phi>)

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_TRANS_4(PlacedVol, radii, phi)                \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_4(PlacedVol, translation::kGeneric, radii, phi) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_4(PlacedVol, translation::kIdentity, radii, phi)

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_RADII_4(PlacedVol, phi)                              \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_TRANS_4(PlacedVol, Polyhedron::EInnerRadii::kFalse, phi)   \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_TRANS_4(PlacedVol, Polyhedron::EInnerRadii::kGeneric, phi) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_TRANS_4(PlacedVol, Polyhedron::EInnerRadii::kTrue, phi)

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_4(PlacedVol)                               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_RADII_4(PlacedVol, Polyhedron::EPhiCutout::kFalse)   \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_RADII_4(PlacedVol, Polyhedron::EPhiCutout::kGeneric) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_RADII_4(PlacedVol, Polyhedron::EPhiCutout::kTrue)    \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_RADII_4(PlacedVol, Polyhedron::EPhiCutout::kLarge)

#endif

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN(PlacedVol, trans, rot)                                          \
  namespace cxx {                                                                                                      \
  template size_t DevicePtr<cuda::PlacedVol, trans, rot>::SizeOf();                                                    \
  template void DevicePtr<cuda::PlacedVol, trans, rot>::Construct(DevicePtr<cuda::LogicalVolume> const logical_volume, \
                                                                  DevicePtr<cuda::Transformation3D> const transform,   \
                                                                  const unsigned int id, const int copy_no,            \
                                                                  const int child_id) const;                           \
  template void ConstructManyOnGpu<cuda::PlacedVol, trans, rot>(                                                       \
      std::size_t nElement, DevicePtr<cuda::VPlacedVolume> const * gpu_ptrs,                                           \
      DevicePtr<cuda::LogicalVolume> const * logical, DevicePtr<cuda::Transformation3D> const * trafo,                 \
      decltype(std::declval<VPlacedVolume>().id()) const * ids,                                                        \
      decltype(std::declval<VPlacedVolume>().GetCopyNo()) const * copyNos,                                             \
      decltype(std::declval<VPlacedVolume>().GetChildId()) const * childIds);                                          \
  }

#if defined(VECGEOM_NO_SPECIALIZATION) || !defined(VECGEOM_CUDA_VOLUME_SPECIALIZATION)

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_BOOLEAN(PlacedVol, Op, trans) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN(PlacedVol<Op, trans, rotation::kGeneric>)

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_BOOLEAN(PlacedVol, Op) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_BOOLEAN(PlacedVol, Op, translation::kGeneric)

#else // VECGEOM_NO_SPECIALIZATION

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_BOOLEAN(PlacedVol, Op, trans)             \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN(PlacedVol<Op, trans, rotation::kGeneric>)  \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN(PlacedVol<Op, trans, rotation::kDiagonal>) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN(PlacedVol<Op, trans, rotation::kIdentity>) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN(PlacedVol<Op, trans, 0x046>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN(PlacedVol<Op, trans, 0x054>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN(PlacedVol<Op, trans, 0x062>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN(PlacedVol<Op, trans, 0x076>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN(PlacedVol<Op, trans, 0x0a1>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN(PlacedVol<Op, trans, 0x0ad>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN(PlacedVol<Op, trans, 0x0dc>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN(PlacedVol<Op, trans, 0x0e3>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN(PlacedVol<Op, trans, 0x10a>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN(PlacedVol<Op, trans, 0x11b>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN(PlacedVol<Op, trans, 0x155>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN(PlacedVol<Op, trans, 0x16a>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN(PlacedVol<Op, trans, 0x18e>)               \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL_BOOLEAN(PlacedVol<Op, trans, 0x1b1>)

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC_BOOLEAN(PlacedVol, Op)                  \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_BOOLEAN(PlacedVol, Op, translation::kGeneric) \
  VECGEOM_DEVICE_INST_PLACED_VOLUME_ALL_ROT_BOOLEAN(PlacedVol, Op, translation::kIdentity)

#endif // VECGEOM_NO_SPECIALIZATION

#endif

#endif // VECGEOM_VOLUMES_PLACEDVOLUME_H_
