#ifndef VECGEOM_VOLUMES_UNPLACEDVOLUME_H_
#define VECGEOM_VOLUMES_UNPLACEDVOLUME_H_

#include "base/Global.h"
#include "base/Transformation3D.h"
#include "base/SOA3D.h"
#include <string>
#include <ostream>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class VUnplacedVolume; )
VECGEOM_DEVICE_DECLARE_CONV(class,VUnplacedVolume)

inline namespace VECGEOM_IMPL_NAMESPACE {

class LogicalVolume;
class VPlacedVolume;

// The abstract interface class for unplaced volumes
class VUnplacedVolume {

private:

  friend class CudaManager;

protected:
  bool fGlobalConvexity;

public:
  // alias for the globally selected VectorType
  using Real_v = vecgeom::VectorBackend::Real_v;

  VECGEOM_CUDA_HEADER_BOTH
  virtual ~VUnplacedVolume() {}

  // ---------------- Contains --------------------------------------------------------------------

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Contains(Vector3D<Precision> const &p) const /* = 0 */;

  VECGEOM_CUDA_HEADER_BOTH
  virtual EnumInside Inside(Vector3D<Precision> const &p) const /* = 0 */;


  // ---------------- DistanceToOut functions -----------------------------------------------------

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToOut(Vector3D<Precision> const &p, Vector3D<Precision> const &d,
                                  Precision step_max = kInfinity) const /* = 0 */;

  // the USolid/GEANT4-like interface for DistanceToOut (returning also exiting normal)
  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToOut(Vector3D<Precision> const &p, Vector3D<Precision> const &d,
                                  Vector3D<Precision> &normal, bool &convex, Precision step_max = kInfinity) const /* = 0 */;

  // an explicit SIMD interface
  VECGEOM_CUDA_HEADER_BOTH
  virtual Real_v DistanceToOutVec(Vector3D<Real_v> const &p, Vector3D<Real_v> const &d, Real_v const &step_max) const /* = 0 */;

  // a helper tramponline to dispatch to SafetyToInVec if type is not scalar
  template <typename T> VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE T DistanceToOut(Vector3D<T> const &p, Vector3D<T> const &d,
                                                                                T const &step_max) const {
    return DistanceToOutVec(p,d,step_max);
  }

  // the container/basket interface (possibly to be deprecated)
  virtual void DistanceToOut(SOA3D<Precision> const &points, SOA3D<Precision> const &directions,
                             Precision const *const step_max, Precision *const output) const /* = 0 */;

  // ---------------- SafetyToOut functions -----------------------------------------------------

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision SafetyToOut(Vector3D<Precision> const &p) const /* = 0 */;

  // an explicit SIMD interface
  virtual Real_v SafetyToOutVec(Vector3D<Real_v> const &p) const /* = 0 */;

  // the tramponline to dispatch to SafetyToOutVec if type is not scalar
  template <typename T> VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE T SafetyToOut(Vector3D<T> const &p) const {
    return SafetyToOutVec(p);
  }

  // the container/basket interface (possibly to be deprecated)
  virtual void SafetyToOut(SOA3D<Precision> const &points, Precision *const output) const /* = 0*/;

  // ---------------- DistanceToIn functions -----------------------------------------------------

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToIn(Vector3D<Precision> const &position,
                                 Vector3D<Precision> const &direction,
                                 const Precision step_max = kInfinity) const /* = 0 */;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Real_v DistanceToInVec(Vector3D<Real_v> const &position,
                                 Vector3D<Real_v> const &direction,
                                 const Real_v &step_max = Real_v(kInfinity)) const /* = 0 */;

  // the tramponline to dispatch to SafetyToInVec if type is not scalar
  // the T = Precision this template will not instantiate as the compiler finds another matching function
  template <typename T>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE T DistanceToIn(Vector3D<T> const &p, Vector3D<T> const &d, T const &step_max)
  {
    return DistanceToInVec(p, d, step_max);
  }

  // ---------------- SafetyToIn functions -------------------------------------------------------

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision SafetyToIn(Vector3D<Precision> const &position) const /* = 0 */;

  // explicit SIMD interface
  virtual Real_v SafetyToInVec(Vector3D<Real_v> const &p) const /* = 0 */;

  // the tramponline to dispatch to SafetyToInVec if type is not scalar
  template <typename T> VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE T SafetyToIn(Vector3D<T> const &p) const {
    return SafetyToInVec(p);
  }


  // ---------------- Normal ---------------------------------------------------------------------

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal ) const /* = 0 */;


  // ---------------- GetPointOnSurface ----------------------------------------------------------
  virtual Vector3D<Precision> GetPointOnSurface() const /* = 0 */;


  // ----------------- Extent --------------------------------------------------------------------
  VECGEOM_CUDA_HEADER_BOTH
  virtual void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const /* = 0 */;

  /** Function to detect whether a volume is globally convex or not.
   *  Return a boolean, true if volume is convex, otherwise false.
   *
   *  Default safe value for all the shapes is set to false.
   */
  VECGEOM_CUDA_HEADER_BOTH
  bool IsConvex() const { return fGlobalConvexity; }


  /**
   * Uses the virtual print method.
   * \sa print(std::ostream &ps)
   */
  friend std::ostream& operator<<(std::ostream& os, VUnplacedVolume const &vol);

  /**
   * Should return the size of bytes of the deriving class. Necessary for
   * copying to the GPU.
   */
  virtual int memory_size() const = 0;


  /**
   * Constructs the deriving class on the GPU and returns a pointer to GPU
   * memory where the object has been instantiated.
   */
#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const = 0;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const =0;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const =0;

  template <typename Derived, typename... ArgsTypes>
  DevicePtr<cuda::VUnplacedVolume> CopyToGpuImpl(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr,
                                                 ArgsTypes... params) const
  {
     DevicePtr<CudaType_t<Derived> > gpu_ptr(in_gpu_ptr);
     gpu_ptr.Construct(params...);
     CudaAssertError();
     // Need to go via the void* because the regular c++ compilation
     // does not actually see the declaration for the cuda version
     // (and thus can not determine the inheritance).
     return DevicePtr<cuda::VUnplacedVolume>((void*)gpu_ptr);
  }
  template <typename Derived>
  DevicePtr<cuda::VUnplacedVolume> CopyToGpuImpl() const
  {
     DevicePtr<CudaType_t<Derived> > gpu_ptr;
     gpu_ptr.Allocate();
     return this->CopyToGpu(DevicePtr<cuda::VUnplacedVolume>((void*)gpu_ptr));
  }

#endif

  /**
   * Virtual print to accommodate outstreams.
   */
  virtual void Print(std::ostream &os) const =0;

  /**
   * C-style printing for CUDA purposes.
   */
  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const =0;

  // Is not static because a virtual function must be called to initialize
  // specialized volume as the shape of the deriving class.
  VPlacedVolume* PlaceVolume(
      char const *const label,
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      VPlacedVolume *const placement = NULL) const;

  VPlacedVolume* PlaceVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      VPlacedVolume *const placement = NULL) const;

private:

#ifndef VECGEOM_NVCC

  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      VPlacedVolume *const placement = NULL) const = 0;

#else
  __device__
  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      const int id, VPlacedVolume *const placement = NULL) const = 0;

#endif

};


std::ostream& operator<<(std::ostream &os, VUnplacedVolume const &vol);

} // End inline namespace

} // End global namespace


#endif // VECGEOM_VOLUMES_UNPLACEDVOLUME_H_
