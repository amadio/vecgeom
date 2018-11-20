/// \file placed_volume.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDVOLUME_H_
#define VECGEOM_VOLUMES_PLACEDVOLUME_H_

#include "base/Global.h"
#include "volumes/LogicalVolume.h"
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

class VPlacedVolume {
  friend class GeoManager;

private:
  unsigned int id_;
  // Use a pointer so the string won't be constructed on the GPU
  std::string *label_;
  static unsigned int g_id_count;

protected:
  LogicalVolume const *logical_volume_;
#ifdef VECGEOM_INPLACE_TRANSFORMATIONS
  Transformation3D fTransformation;
#else
  Transformation3D const *fTransformation;
#endif
  PlacedBox const *bounding_box_;

#ifndef VECCORE_CUDA

  VPlacedVolume(char const *const label, LogicalVolume const *const logical_vol,
                Transformation3D const *const transform, PlacedBox const *const boundingbox);

  VPlacedVolume(LogicalVolume const *const logical_vol, Transformation3D const *const transform,
                PlacedBox const *const boundingbox)
      : VPlacedVolume("", logical_vol, transform, boundingbox)
  {
  }

#else

  VECCORE_ATT_DEVICE VPlacedVolume(LogicalVolume const *const logical_vol, Transformation3D const *const transformation,
                                   PlacedBox const *const boundingbox, unsigned int id)
#ifdef VECGEOM_INPLACE_TRANSFORMATIONS
      : logical_volume_(logical_vol), fTransformation(*transformation), bounding_box_(boundingbox), id_(id),
        label_(NULL)
  {
  }
#else
      : logical_volume_(logical_vol), fTransformation(transformation), bounding_box_(boundingbox), id_(id), label_(NULL)
  {
  }
#endif
#endif

public:
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume(VPlacedVolume const &);
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume *operator=(VPlacedVolume const &);

  VECCORE_ATT_HOST_DEVICE
  virtual ~VPlacedVolume();

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  unsigned int id() const { return id_; }

  static unsigned int GetIdCount() { return g_id_count; }

  std::string const &GetLabel() const { return *label_; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  PlacedBox const *bounding_box() const { return bounding_box_; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  LogicalVolume const *GetLogicalVolume() const { return logical_volume_; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector<Daughter> const &GetDaughters() const { return logical_volume_->GetDaughters(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  const char *GetName() const { return (*label_).c_str(); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  VUnplacedVolume const *GetUnplacedVolume() const { return logical_volume_->GetUnplacedVolume(); }

  bool IsAssembly() const { return GetUnplacedVolume()->IsAssembly(); }

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

  VECCORE_ATT_HOST_DEVICE
  void SetLogicalVolume(LogicalVolume const *const logical_vol) { logical_volume_ = logical_vol; }

  VECCORE_ATT_HOST_DEVICE
  void SetTransformation(Transformation3D const *const transform)
  {
#ifdef VECGEOM_INPLACE_TRANSFORMATIONS
    fTransformation = *transform;
#else
    fTransformation = transform;
#endif
  }

  void set_label(char const *label)
  {
    if (label_) delete label_;
    label_ = new std::string(label);
  }

  friend std::ostream &operator<<(std::ostream &os, VPlacedVolume const &vol);

  virtual int MemorySize() const = 0;

  VECCORE_ATT_HOST_DEVICE
  virtual void Print(const int indent = 0) const;

  VECCORE_ATT_HOST_DEVICE
  virtual void PrintType() const = 0;

  // some functions allowing for some very basic "introspection"
  // print the volume type to an outstream
  virtual void PrintType(std::ostream &os) const = 0;
  // print the implemtation struct of this volume to an outstream
  virtual void PrintImplementationType(std::ostream &os) const = 0;
  // print the unplaced type to an outstream
  virtual void PrintUnplacedType(std::ostream &os) const = 0;

  // returns translation and rotation code of the placed volume
  virtual int GetTransCode() const { return translation::kGeneric; }
  virtual int GetRotCode() const { return rotation::kGeneric; }

  /// Recursively prints contained volumes to standard output.
  VECCORE_ATT_HOST_DEVICE
  void PrintContent(const int depth = 0) const;

  // Geometry functionality

  VECCORE_ATT_HOST_DEVICE
  virtual bool Contains(Vector3D<Precision> const &point) const = 0;

  virtual void Contains(SOA3D<Precision> const &point, bool *const output) const = 0;

  /// \return The input point transformed to the local reference frame.
  VECCORE_ATT_HOST_DEVICE
  virtual bool Contains(Vector3D<Precision> const &point, Vector3D<Precision> &localPoint) const = 0;

  /// \param localPoint Point in the local reference frame of the volume.
  VECCORE_ATT_HOST_DEVICE
  virtual bool UnplacedContains(Vector3D<Precision> const &localPoint) const = 0;

  VECCORE_ATT_HOST_DEVICE
  virtual EnumInside Inside(Vector3D<Precision> const &point) const = 0;

  virtual void Inside(SOA3D<Precision> const &point, Inside_t *const output) const = 0;

  VECCORE_ATT_HOST_DEVICE
  virtual Precision DistanceToIn(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                                 const Precision step_max = kInfLength) const = 0;

  // if we have any SIMD backend, we offer a SIMD interface
  virtual Real_v DistanceToInVec(Vector3D<Real_v> const &position, Vector3D<Real_v> const &direction,
                                 const Real_v step_max = kInfLength) const = 0;

  template <typename T>
  VECGEOM_FORCE_INLINE
  T DistanceToIn(Vector3D<T> const &position, Vector3D<T> const &direction, const T step_max = T(kInfLength)) const
  {
    return DistanceToInVec(position, direction, step_max);
  }

  virtual void DistanceToIn(SOA3D<Precision> const &position, SOA3D<Precision> const &direction,
                            Precision const *const step_max, Precision *const output) const = 0;

  // to be deprecated
  virtual void DistanceToInMinimize(SOA3D<Precision> const &position, SOA3D<Precision> const &direction,
                                    int daughterindex, Precision *const output, int *const nextnodeids) const = 0;

  VECCORE_ATT_HOST_DEVICE
  virtual Precision DistanceToOut(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                                  Precision const step_max = kInfLength) const = 0;

  // define this interface in case we don't have the Scalar interface

  virtual Real_v DistanceToOutVec(Vector3D<Real_v> const &position, Vector3D<Real_v> const &direction,
                                  Real_v const step_max = kInfLength) const = 0;

  template <typename T>
  VECGEOM_FORCE_INLINE
  T DistanceToOut(Vector3D<T> const &position, Vector3D<T> const &direction, const T step_max = T(kInfLength)) const
  {
    return DistanceToOutVec(position, direction, step_max);
  }

  // a "placed" version of the distancetoout function; here
  // the point and direction are first of all transformed into the reference frame of the
  // callee. The normal DistanceToOut method does not do this
  VECCORE_ATT_HOST_DEVICE
  virtual Precision PlacedDistanceToOut(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                                        Precision const step_max = kInfLength) const = 0;

  virtual void DistanceToOut(SOA3D<Precision> const &position, SOA3D<Precision> const &direction,
                             Precision const *const step_max, Precision *const output) const = 0;

  virtual void DistanceToOut(SOA3D<Precision> const &position, SOA3D<Precision> const &direction,
                             Precision const *const step_max, Precision *const output,
                             int *const nextnodeindex) const = 0;

  VECCORE_ATT_HOST_DEVICE
  virtual Precision SafetyToIn(Vector3D<Precision> const &position) const = 0;

  virtual Real_v SafetyToInVec(Vector3D<Real_v> const &position) const = 0;

  template <typename T>
  VECGEOM_FORCE_INLINE
  T SafetyToIn(Vector3D<T> const &p) const
  {
    return SafetyToInVec(p);
  }

  virtual void SafetyToIn(SOA3D<Precision> const &position, Precision *const safeties) const = 0;

  // to be deprecated
  virtual void SafetyToInMinimize(SOA3D<Precision> const &points, Precision *const safeties) const = 0;

  VECCORE_ATT_HOST_DEVICE
  virtual Precision SafetyToOut(Vector3D<Precision> const &position) const = 0;

  virtual Real_v SafetyToOutVec(Vector3D<Real_v> const &position) const = 0;

  virtual void SafetyToOut(SOA3D<Precision> const &position, Precision *const safeties) const = 0;

  template <typename T>
  VECGEOM_FORCE_INLINE
  T SafetyToOut(Vector3D<T> const &p) const
  {
    return SafetyToOutVec(p);
  }

  // to be deprecated
  virtual void SafetyToOutMinimize(SOA3D<Precision> const &points, Precision *const safeties) const = 0;

  /// \brief Return the cubic volume of the shape
  // It is currently not a const function since some shapes might cache this value, if it is expensive to calculate.
  virtual Precision Capacity();

  VECCORE_ATT_HOST_DEVICE
  virtual void Extent(Vector3D<Precision> & /* min */, Vector3D<Precision> & /* max */) const;

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const & /*point*/, Vector3D<Precision> & /*normal*/) const;

  virtual Precision SurfaceArea() const = 0;

public:
#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const                                                                       = 0;
  virtual DevicePtr<cuda::VPlacedVolume> CopyToGpu(DevicePtr<cuda::LogicalVolume> const logical_volume,
                                                   DevicePtr<cuda::Transformation3D> const transform,
                                                   DevicePtr<cuda::VPlacedVolume> const gpu_ptr) const      = 0;
  virtual DevicePtr<cuda::VPlacedVolume> CopyToGpu(DevicePtr<cuda::LogicalVolume> const logical_volume,
                                                   DevicePtr<cuda::Transformation3D> const transform) const = 0;

  template <typename Derived>
  DevicePtr<cuda::VPlacedVolume> CopyToGpuImpl(DevicePtr<cuda::LogicalVolume> const logical_volume,
                                               DevicePtr<cuda::Transformation3D> const transform,
                                               DevicePtr<cuda::VPlacedVolume> const in_gpu_ptr) const
  {
    DevicePtr<CudaType_t<Derived>> gpu_ptr(in_gpu_ptr);
    gpu_ptr.Construct(logical_volume, transform, nullptr, this->id());
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
  virtual VPlacedVolume const *ConvertToUnspecialized() const = 0;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const = 0;
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const *ConvertToGeant4() const = 0;
#endif
#endif // VECCORE_CUDA
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#ifdef VECCORE_CUDA

#define VECGEOM_DEVICE_INST_PLACED_VOLUME(PlacedVol)                                                       \
  namespace cxx {                                                                                          \
  template size_t DevicePtr<cuda::PlacedVol>::SizeOf();                                                    \
  template void DevicePtr<cuda::PlacedVol>::Construct(DevicePtr<cuda::LogicalVolume> const logical_volume, \
                                                      DevicePtr<cuda::Transformation3D> const transform,   \
                                                      DevicePtr<cuda::PlacedBox> const boundingBox,        \
                                                      const unsigned int id) const;                        \
  }

#define VECGEOM_DEVICE_INST_PLACED_VOLUME_IMPL(PlacedVol, Extra)                                                  \
  namespace cxx {                                                                                                 \
  template size_t DevicePtr<cuda::PlacedVol, Extra>::SizeOf();                                                    \
  template void DevicePtr<cuda::PlacedVol, Extra>::Construct(DevicePtr<cuda::LogicalVolume> const logical_volume, \
                                                             DevicePtr<cuda::Transformation3D> const transform,   \
                                                             DevicePtr<cuda::PlacedBox> const boundingBox,        \
                                                             const unsigned int id) const;                        \
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
      DevicePtr<cuda::PlacedBox> const boundingBox, const unsigned int id) const;                             \
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
      DevicePtr<cuda::PlacedBox> const boundingBox, const unsigned int id) const;                             \
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
                                                                  DevicePtr<cuda::PlacedBox> const boundingBox,        \
                                                                  const unsigned int id) const;                        \
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
