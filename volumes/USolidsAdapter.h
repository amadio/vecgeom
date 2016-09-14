// USolids Adapter.

#ifndef VECGEOM_VOLUMES_USOLIDSADAPTER_H
#define VECGEOM_VOLUMES_USOLIDSADAPTER_H

#include "base/Global.h"
#include "VUSolid.hh"

#include "base/Vector3D.h"
#include "UVector3.hh"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

/// \brief USolids compatibility signatures.
///
/// These do not necessarily provide all the return values promised by the
/// interface, so use volumes in this way with caution.
template <class UnplacedVolume_t>
class USolidsAdapter : public VUSolid, protected UnplacedVolume_t {
  // using protected inheritance since the Adapter is supposed to be a VUSolid
  // "implemented-in-terms-of" the VecGeom UnplacedVolume_t
  // the choice of protected vs private is due to the fact that we want to propagate
  // functions further down in the inheritance hierarchy

protected:
  // using
  using UnplacedVolume_t::DistanceToOut;
  using UnplacedVolume_t::DistanceToIn;

public:
  // bring in some members from base (to avoid name hiding)

  // VecGeom volumes have special delete/new ("AlignedBase") and we need to make
  // these functions public again
  using UnplacedVolume_t::operator delete;
  using UnplacedVolume_t::operator new;

  USolidsAdapter(const std::string &name) : VUSolid(name) {}
  USolidsAdapter() : VUSolid() {}

  template <typename... T>
  USolidsAdapter(const std::string &name, const T &... params) : VUSolid(name), UnplacedVolume_t(params...)
  {
  }

  virtual ~USolidsAdapter() {}

  double DistanceToOut(Vector3D<double> const &position, Vector3D<double> const &direction,
                       Precision stepMax = kInfLength) const override
  {

    double output = UnplacedVolume_t::DistanceToOut(position, direction, stepMax);
    // apply USolids convention: convert negative values to zero
    if (output < kHalfTolerance) output = 0.;
    return output;
  }

  EnumInside Inside(const UVector3 &aPoint) const override { return UnplacedVolume_t::Inside(aPoint); }

  // these function names are specific to USolids but can be reimplemented in terms of
  // other interfaces:

  double SafetyFromOutside(Vector3D<double> const &point, bool accurate = false) const override
  {
    (void)accurate; // fix 'unused variable' warning
    double output = UnplacedVolume_t::SafetyToIn(point);
    // apply USolids convention: convert negative values to zero
    if (output < kHalfTolerance) output = 0.;
    return output;
  }

  double SafetyFromInside(Vector3D<double> const &point, bool accurate = false) const override
  {
    (void)accurate; // fix 'unused variable' warning
    double output = UnplacedVolume_t::SafetyToOut(point);
    // apply USolids convention: convert negative values to zero
    MaskedAssign(output < kHalfTolerance, 0., &output);
    return output;
  }

  // the following function is somewhat USolid specific
  double DistanceToOut(Vector3D<double> const &point, Vector3D<double> const &direction, Vector3D<double> &normal,
                       bool &convex, double stepMax = kInfLength) const override
  {
    double d                  = UnplacedVolume_t::DistanceToOut(point, direction, stepMax);
    Vector3D<double> hitpoint = point + d * direction;
    UnplacedVolume_t::Normal(hitpoint, normal);

    convex = UnplacedVolume_t::IsConvex();
    // apply USolids distance conventions
    if (d < kHalfTolerance) d = 0.;
    return d;
  }

  double DistanceToIn(Vector3D<double> const &position, Vector3D<double> const &direction,
                      const double step_max = kInfLength) const override
  {
    auto d = UnplacedVolume_t::DistanceToIn(position, direction, step_max);
    // apply USolids distance conventions
    if (d < kHalfTolerance) d = 0.;
    return d;
  }

  VECGEOM_CUDA_HEADER_BOTH
  bool Normal(const UVector3 &aPoint, UVector3 &aNormal) const override
  {
    return UnplacedVolume_t::Normal(aPoint, aNormal);
  }

  void Extent(UVector3 &aMin, UVector3 &aMax) const override { return UnplacedVolume_t::Extent(aMin, aMax); }

  // UGeometryType  GetEntityType() const override {
  //   return UnplacedVolume_t::GetEntityType();
  // }

  //  like  CubicVolume()
  double Capacity() override { return UnplacedVolume_t::Capacity(); }

  double SurfaceArea() override { return UnplacedVolume_t::SurfaceArea(); }

  UVector3 GetPointOnSurface() const override { return UnplacedVolume_t::GetPointOnSurface(); }
};

} // inline namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_USOLIDSADAPTER_H
