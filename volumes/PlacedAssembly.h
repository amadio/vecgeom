// LICENSING INFORMATION TBD

#ifndef VECGEOM_PLACEDASSEMBLY_H
#define VECGEOM_PLACEDASSEMBLY_H

#include "volumes/UnplacedAssembly.h"
#include "volumes/PlacedVolume.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class PlacedAssembly;);
VECGEOM_DEVICE_DECLARE_CONV(class, PlacedAssembly);

inline namespace VECGEOM_IMPL_NAMESPACE {

class NavigationState;

// placed version of an assembly
// simple and unspecialized implementation
class PlacedAssembly : public VPlacedVolume {

private:
public:
#ifndef VECCORE_CUDA
  VECCORE_ATT_HOST_DEVICE
  PlacedAssembly(char const *const label, LogicalVolume const *const logicalVolume,
                 Transformation3D const *const transformation)
      : VPlacedVolume(label, logicalVolume, transformation, nullptr)
  {
  } // the constructor
#else
  VECCORE_ATT_DEVICE PlacedAssembly(char const *const label, LogicalVolume const *const logical_volume,
                                    Transformation3D const *const transformation, PlacedBox const *const boundingBox,
                                    const int id)
      : VPlacedVolume(logical_volume, transformation, nullptr, id)
  {
  }
#endif

  VECCORE_ATT_HOST_DEVICE
  virtual ~PlacedAssembly();

  // the VPlacedVolume Interfaces -----
  virtual int MemorySize() const override { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void PrintType() const override { printf("PlacedAssembly"); }

  virtual void PrintType(std::ostream &s) const override { s << "PlacedAssembly"; }

  virtual void PrintImplementationType(std::ostream &) const override
  {
#ifndef VECCORE_CUDA
    throw std::runtime_error("unimplemented function called");
#endif
  }
  virtual void PrintUnplacedType(std::ostream &) const override
  {
#ifndef VECCORE_CUDA
    throw std::runtime_error("unimplemented function called");
#endif
  }

  VECCORE_ATT_HOST_DEVICE
  virtual void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override
  {
    static_cast<UnplacedAssembly const *>(GetUnplacedVolume())->UnplacedAssembly::Extent(aMin, aMax);
  }

  VECCORE_ATT_HOST_DEVICE
  virtual bool Contains(Vector3D<Precision> const &p) const override
  {
    return static_cast<UnplacedAssembly const *>(GetUnplacedVolume())
        ->UnplacedAssembly::Contains(GetTransformation()->Transform(p));
  }

  VECCORE_ATT_HOST_DEVICE
  virtual bool Contains(Vector3D<Precision> const &point, Vector3D<Precision> &localPoint) const override
  {
    localPoint = GetTransformation()->Transform(point);
    return static_cast<UnplacedAssembly const *>(GetUnplacedVolume())->UnplacedAssembly::Contains(localPoint);
  }

  virtual void Contains(SOA3D<Precision> const & /*points*/, bool *const /*output*/) const override
  {
    throw std::runtime_error("unimplemented function called");
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual bool UnplacedContains(Vector3D<Precision> const &point) const override
  {
    return static_cast<UnplacedAssembly const *>(GetUnplacedVolume())->UnplacedAssembly::Contains(point);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual EnumInside Inside(Vector3D<Precision> const & /*point*/) const override
  {
#ifndef VECCORE_CUDA
    throw std::runtime_error("unimplemented function called");
#endif
    return vecgeom::kOutside; // dummy return
  }

  virtual void Inside(SOA3D<Precision> const & /*points*/, Inside_t *const /*output*/) const override
  {
    throw std::runtime_error("unimplemented function called");
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual Precision DistanceToIn(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                                 const Precision step_max = kInfLength) const override
  {
    return static_cast<UnplacedAssembly const *>(GetUnplacedVolume())
        ->UnplacedAssembly::DistanceToIn(GetTransformation()->Transform(position),
                                         GetTransformation()->TransformDirection(direction), step_max);
  }

  virtual void DistanceToIn(SOA3D<Precision> const & /*position*/, SOA3D<Precision> const & /*direction*/,
                            Precision const *const /*stepMax*/, Precision *const /*output*/) const override
  {
    throw std::runtime_error("unimplemented function called");
  }

  virtual void DistanceToInMinimize(SOA3D<Precision> const & /*position*/, SOA3D<Precision> const & /*direction*/,
                                    int /*daughterindex*/, Precision *const /*output*/,
                                    int *const /*nextnodeids*/) const override
  {
    throw std::runtime_error("unimplemented function called");
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual Precision DistanceToOut(Vector3D<Precision> const & /*position*/, Vector3D<Precision> const & /*direction*/,
                                  Precision const /*stepMax*/) const override
  {
#ifndef VECCORE_CUDA
    throw std::runtime_error("unimplemented function called");
#endif
    return -1.; // dummy return
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision PlacedDistanceToOut(Vector3D<Precision> const & /*position*/,
                                        Vector3D<Precision> const & /*direction*/,
                                        Precision const /*stepMax*/) const override
  {
#ifndef VECCORE_CUDA
    throw std::runtime_error("unimplemented function called");
#endif
    return -1.;
  }

  virtual void DistanceToOut(SOA3D<Precision> const & /*position*/, SOA3D<Precision> const & /*direction*/,
                             Precision const *const /*step_max*/, Precision *const /*output*/) const override
  {
    throw std::runtime_error("unimplemented function called");
  }

  virtual void DistanceToOut(SOA3D<Precision> const & /*position*/, SOA3D<Precision> const & /*direction*/,
                             Precision const *const /*step_max*/, Precision *const /*output*/,
                             int *const /*nextnodeindex*/) const override
  {
    throw std::runtime_error("unimplemented function called");
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual Precision SafetyToOut(Vector3D<Precision> const &position) const override
  {
    return GetUnplacedVolume()->SafetyToOut(position);
  }

  virtual void SafetyToOut(SOA3D<Precision> const & /*position*/, Precision *const /*safeties*/) const override
  {
    throw std::runtime_error("unimplemented function called");
  }

  virtual void SafetyToOutMinimize(SOA3D<Precision> const & /*position*/, Precision *const /*safeties*/) const override
  {
    throw std::runtime_error("unimplemented function called");
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual Precision SafetyToIn(Vector3D<Precision> const &position) const override
  {
    return static_cast<UnplacedAssembly const *>(GetUnplacedVolume())
        ->UnplacedAssembly::SafetyToIn(GetTransformation()->Transform(position));
  }

  virtual void SafetyToIn(SOA3D<Precision> const & /*position*/, Precision *const /*safeties*/) const override
  {
    throw std::runtime_error("unimplemented function called");
  }

  virtual void SafetyToInMinimize(SOA3D<Precision> const & /*position*/, Precision *const /*safeties*/) const override
  {
    throw std::runtime_error("unimplemented function called");
  }

  // the SIMD vector interfaces (not implemented)
  virtual Real_v DistanceToInVec(Vector3D<Real_v> const & /*position*/, Vector3D<Real_v> const & /*direction*/,
                                 Real_v const /*step_max*/ = kInfLength) const override
  {
#ifndef VECCORE_CUDA
    throw std::runtime_error("unimplemented function called");
#endif
    return Real_v(-1.);
  }

  virtual Real_v DistanceToOutVec(Vector3D<Real_v> const & /*position*/, Vector3D<Real_v> const & /*direction*/,
                                  Real_v const /*step_max*/ = kInfLength) const override
  {
#ifndef VECCORE_CUDA
    throw std::runtime_error("unimplemented function called");
#endif
    return Real_v(-1.);
  }

  virtual Real_v SafetyToInVec(Vector3D<Real_v> const & /*position*/) const override
  {
#ifndef VECCORE_CUDA
    throw std::runtime_error("unimplemented function called");
#endif
    return Real_v(-1.);
  }

  virtual Real_v SafetyToOutVec(Vector3D<Real_v> const & /*position*/) const override
  {
#ifndef VECCORE_CUDA
    throw std::runtime_error("unimplemented function called");
#endif
    return Real_v(-1.);
  }

  Precision Capacity() override { return static_cast<UnplacedAssembly const *>(GetUnplacedVolume())->Capacity(); }

  Precision SurfaceArea() const override
  {
    return static_cast<UnplacedAssembly const *>(GetUnplacedVolume())->SurfaceArea();
  }

#ifndef VECCORE_CUDA
  virtual VPlacedVolume const *ConvertToUnspecialized() const override { return this; }
#ifdef VECGEOM_ROOT
  virtual TGeoShape const *ConvertToRoot() const override { throw std::runtime_error("unimplemented function called"); }
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const *ConvertToGeant4() const override
  {
    throw std::runtime_error("unimplemented function called");
  }
#endif
#endif

#ifdef VECGEOM_CUDA_INTERFACE
  // TBD properly
  virtual size_t DeviceSizeOf() const override { return 0; /*DevicePtr<cuda::PlacedAssembly>::SizeOf();*/ }
  virtual DevicePtr<cuda::VPlacedVolume> CopyToGpu(DevicePtr<cuda::LogicalVolume> const /*logical_volume*/,
                                                   DevicePtr<cuda::Transformation3D> const /*transform*/,
                                                   DevicePtr<cuda::VPlacedVolume> const /*gpu_ptr*/) const override
  {
    return DevicePtr<cuda::VPlacedVolume>(nullptr);
  }
  virtual DevicePtr<cuda::VPlacedVolume> CopyToGpu(DevicePtr<cuda::LogicalVolume> const /*logical_volume*/,
                                                   DevicePtr<cuda::Transformation3D> const /*transform*/) const override
  {
    return DevicePtr<cuda::VPlacedVolume>(nullptr);
  }
#endif

  // specific PlacedAssembly Interfaces ---------

  // an extended contains functions needed for navigation
  // if this function returns true it modifies the navigation state to point to the first non-assembly volume
  // the point is contained in
  // this function is not part of the generic UnplacedVolume interface but we could consider doing so
  bool Contains(Vector3D<Precision> const &p, Vector3D<Precision> &lp, NavigationState &state) const
  {
    state.Push(this);
    // call unplaced variant with transformed point
    auto indaughter = static_cast<UnplacedAssembly const *>(GetUnplacedVolume())
                          ->UnplacedAssembly::Contains(GetTransformation()->Transform(p), lp, state);
    if (!indaughter) state.Pop();
    return indaughter;
  }

}; // end class
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // PLACEDASSEMBLY_H
