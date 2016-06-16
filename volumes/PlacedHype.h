/// \file PlacedHype.h
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_PLACEDHYPE_H_
#define VECGEOM_VOLUMES_PLACEDHYPE_H_

#include "base/Global.h"
#include "backend/Backend.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/UnplacedHype.h"
#include "volumes/kernel/HypeImplementation.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class PlacedHype; )
VECGEOM_DEVICE_DECLARE_CONV( class, PlacedHype );
inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedHype : public VPlacedVolume {

public:

  typedef UnplacedHype UnplacedShape_t;

#ifndef VECGEOM_NVCC

  PlacedHype(char const *const label,
                       LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox) {}

  PlacedHype(LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox)
      : PlacedHype("", logical_volume, transformation, boundingBox) {}

#else

  __device__
  PlacedHype(LogicalVolume const *const logical_volume,
                       Transformation3D const *const transformation,
                       PlacedBox const *const boundingBox, const int id)
      : VPlacedVolume(logical_volume, transformation, boundingBox, id) {}

#endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual ~PlacedHype() {}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  UnplacedHype const* GetUnplacedVolume() const {
    return static_cast<UnplacedHype const *>(
        GetLogicalVolume()->GetUnplacedVolume());
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetRmin() const { return GetUnplacedVolume()->GetRmin(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetRmax() const { return GetUnplacedVolume()->GetRmax(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetRmin2() const { return GetUnplacedVolume()->GetRmin2(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetRmax2() const { return GetUnplacedVolume()->GetRmax2(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetStIn() const { return GetUnplacedVolume()->GetStIn(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetStOut() const { return GetUnplacedVolume()->GetStOut(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetTIn() const { return GetUnplacedVolume()->GetTIn(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetTOut() const { return GetUnplacedVolume()->GetTOut(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetTIn2() const { return GetUnplacedVolume()->GetTIn2(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetTOut2() const { return GetUnplacedVolume()->GetTOut2(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetTIn2Inv() const { return GetUnplacedVolume()->GetTIn2Inv(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetTOut2Inv() const { return GetUnplacedVolume()->GetTOut2Inv(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetDz() const { return GetUnplacedVolume()->GetDz(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetDz2() const { return GetUnplacedVolume()->GetDz2(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetEndInnerRadius() const { return GetUnplacedVolume()->GetEndInnerRadius(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetEndInnerRadius2() const { return GetUnplacedVolume()->GetEndInnerRadius2(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetEndOuterRadius() const { return GetUnplacedVolume()->GetEndOuterRadius(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetEndOuterRadius2() const { return GetUnplacedVolume()->GetEndOuterRadius2(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetInSqSide() const { return GetUnplacedVolume()->GetInSqSide(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetZToleranceLevel() const { return GetUnplacedVolume()->GetZToleranceLevel(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetInnerRadToleranceLevel() const { return GetUnplacedVolume()->GetInnerRadToleranceLevel(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetOuterRadToleranceLevel() const { return GetUnplacedVolume()->GetOuterRadToleranceLevel(); }

  template <bool ForInnerSurface>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE
  Precision GetHypeRadius2(Precision dz) const {
    if (ForInnerSurface)
      return GetRmin2() + GetTIn2() * dz * dz;
    else
      return GetRmax2() + GetTOut2() * dz * dz;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool PointOnZSurface(Vector3D<Precision> const &p) const {
    return (p.z() > (GetDz() - GetZToleranceLevel())) && (p.z() < (GetDz() + GetZToleranceLevel()));
  }

  template <bool ForInnerSurface>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE bool PointOnHyperbolicSurface(Vector3D<Precision> const &p) const {
    Precision hypeR2 = 0.;
    hypeR2 = GetHypeRadius2<ForInnerSurface>(p.z());
    Precision pointRad2 = p.Perp2();
    return ((pointRad2 > (hypeR2 - GetOuterRadToleranceLevel())) &&
            (pointRad2 < (hypeR2 + GetOuterRadToleranceLevel())));
  }

  bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const override {

    bool valid = true;

    Precision absZ(std::fabs(p.z()));
    Precision distZ(absZ - GetDz());
    Precision dist2Z(distZ * distZ);

    Precision xR2 = p.Perp2();
    Precision dist2Outer(std::fabs(xR2 - GetHypeRadius2<false>(absZ)));
    Precision dist2Inner(std::fabs(xR2 - GetHypeRadius2<true>(absZ)));

    // EndCap
    if (PointOnZSurface(p) || ((dist2Z < dist2Inner) && (dist2Z < dist2Outer)))
      normal = Vector3D<Precision>(0.0, 0.0, p.z() < 0 ? -1.0 : 1.0);

    // OuterHyperbolic Surface
    if (PointOnHyperbolicSurface<false>(p) || ((dist2Outer < dist2Inner) && (dist2Outer < dist2Z)))
      normal = Vector3D<Precision>(p.x(), p.y(), -p.z() * GetTOut2()).Unit();

    // InnerHyperbolic Surface
    if (PointOnHyperbolicSurface<true>(p) || ((dist2Inner < dist2Outer) && (dist2Inner < dist2Z)))
      normal = Vector3D<Precision>(-p.x(), -p.y(), p.z() * GetTIn2()).Unit();

    return valid;
  }

  Precision Capacity() override { return GetUnplacedVolume()->Capacity(); }

  Precision SurfaceArea() override  { return GetUnplacedVolume()->SurfaceArea(); }

  VECGEOM_INLINE
#if defined(VECGEOM_USOLIDS)
  std::string GetEntityType() const override { return GetUnplacedVolume()->GetEntityType(); }

  void GetParametersList(int aNumber, double *aArray) const override {
    return GetUnplacedVolume()->GetParametersList(aNumber, aArray);
  }
#endif

  VECGEOM_CUDA_HEADER_BOTH
  void Extent( Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override { return GetUnplacedVolume()->Extent(aMin,aMax);}

  Vector3D<Precision>  GetPointOnSurface() const override { return GetUnplacedVolume()->GetPointOnSurface();}

//  VECGEOM_CUDA_HEADER_BOTH
//  VECGEOM_INLINE
//  void ComputeBBox() const { return GetUnplacedVolume()->ComputeBBox();}

  VECGEOM_INLINE
  virtual int memory_size() const override { return sizeof(*this); }

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

} } // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDHYPE_H_
