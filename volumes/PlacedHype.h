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
VECGEOM_DEVICE_DECLARE_CONV( PlacedHype );
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

//GetFunctions
//_____________________________________________________________

//get
	/*
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetAOut() const{ return GetUnplacedVolume()->GetAOut();}

	VECGEOM_CUDA_HEADER_BOTH
    Precision GetAIn() const{ return GetUnplacedVolume()->GetAIn();}
	*/
    VECGEOM_CUDA_HEADER_BOTH
    Precision GetRmin() const{ return GetUnplacedVolume()->GetRmin();}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetRmax() const{ return GetUnplacedVolume()->GetRmax();}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetRmin2() const{ return GetUnplacedVolume()->GetRmin2();}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetRmax2() const{ return GetUnplacedVolume()->GetRmax2();}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetStIn() const{ return GetUnplacedVolume()->GetStIn();}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetStOut() const{ return GetUnplacedVolume()->GetStOut();}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTIn() const{ return GetUnplacedVolume()->GetTIn();}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTOut() const{ return GetUnplacedVolume()->GetTOut();}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTIn2() const{ return GetUnplacedVolume()->GetTIn2();}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTOut2() const{ return GetUnplacedVolume()->GetTOut2();}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTIn2Inv() const{ return GetUnplacedVolume()->GetTIn2Inv();}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetTOut2Inv() const{ return GetUnplacedVolume()->GetTOut2Inv();}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetDz() const{ return GetUnplacedVolume()->GetDz();}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetDz2() const{ return GetUnplacedVolume()->GetDz2();}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetEndInnerRadius() const{ return GetUnplacedVolume()->GetEndInnerRadius();}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetEndInnerRadius2() const{ return GetUnplacedVolume()->GetEndInnerRadius2();}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetEndOuterRadius() const{ return GetUnplacedVolume()->GetEndOuterRadius();}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetEndOuterRadius2() const{ return GetUnplacedVolume()->GetEndOuterRadius2();}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetInSqSide() const{ return GetUnplacedVolume()->GetInSqSide();}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetZToleranceLevel() const{ return GetUnplacedVolume()->GetZToleranceLevel();}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetInnerRadToleranceLevel() const{ return GetUnplacedVolume()->GetInnerRadToleranceLevel();}

    VECGEOM_CUDA_HEADER_BOTH
    Precision GetOuterRadToleranceLevel() const{ return GetUnplacedVolume()->GetOuterRadToleranceLevel();}


//_____________________________________________________________
//template <bool inner>
//Old Definition
/*
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Precision GetHypeRadius2(bool inner,Precision dz) const
{
if(inner)
	return GetRmin2()+GetTIn2()*dz*dz;
else
	return GetRmax2()+GetTOut2()*dz*dz;

}

*/

//New Definition
template <bool ForInnerSurface>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
Precision GetHypeRadius2(Precision dz) const
{
if(ForInnerSurface)
	return GetRmin2()+GetTIn2()*dz*dz;
else
	return GetRmax2()+GetTOut2()*dz*dz;

}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool PointOnZSurface(Vector3D<Precision> const &p) const
  {
    return (p.z() > (GetDz()-GetZToleranceLevel())) && (p.z() < (GetDz()+GetZToleranceLevel()));
  }

//If this function is used then below two definitions can be removed.
//Currently using this one.
template <bool ForInnerSurface>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool PointOnHyperbolicSurface(Vector3D<Precision> const &p) const
  {
      Precision hypeR2 = 0.;
      hypeR2 = GetHypeRadius2<ForInnerSurface>(p.z());
      Precision pointRad2 = p.Perp2();
      return ((pointRad2 > (hypeR2 - GetOuterRadToleranceLevel())) && (pointRad2 < (hypeR2 + GetOuterRadToleranceLevel())));
  }


//Below two definitions are not in use, still kept it for the reference.

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool PointOnOuterHyperbolicSurface(Vector3D<Precision> const &p) const
  {
      Precision hypeR2 = GetRmax()*GetRmax() + GetTOut2()*p.z()*p.z();
      Precision pointRad2 = p.Perp2();
      return ((pointRad2 > (hypeR2 - GetOuterRadToleranceLevel())) && (pointRad2 < (hypeR2 + GetOuterRadToleranceLevel())));
  }


VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
bool PointOnInnerHyperbolicSurface(Vector3D<Precision> const &p) const
  {
      Precision hypeR2 = GetRmin()*GetRmin() + GetTIn2()*p.z()*p.z();
      Precision pointRad2 = p.Perp2();
      return ((pointRad2 > (hypeR2 - GetOuterRadToleranceLevel())) && (pointRad2 < (hypeR2 + GetOuterRadToleranceLevel())));
  }


//New Definition of Normal
 bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal ) const
  {

  bool valid=true;

  Precision absZ(std::fabs(p.z()));
  Precision distZ(absZ - GetDz());
  Precision dist2Z(distZ*distZ);

  //Precision xR2( p.x()*p.x()+p.y()*p.y() );
  Precision xR2 = p.Perp2();
  Precision dist2Outer( std::fabs(xR2 - GetHypeRadius2<false>(absZ)) );
  Precision dist2Inner( std::fabs(xR2 - GetHypeRadius2<true>(absZ)) );

  //EndCap
  if(PointOnZSurface(p) || ( (dist2Z < dist2Inner) && (dist2Z < dist2Outer) ))
    normal = Vector3D<Precision>( 0.0, 0.0, p.z() < 0 ? -1.0 : 1.0 );

  //OuterHyperbolic Surface
  if(PointOnHyperbolicSurface<false>(p) ||  ( (dist2Outer < dist2Inner) && (dist2Outer < dist2Z) ) )
    normal = Vector3D<Precision>( p.x(), p.y(), -p.z()*GetTOut2() ).Unit();

  //InnerHyperbolic Surface
  if(PointOnHyperbolicSurface<true>(p) ||  ( (dist2Inner < dist2Outer) && (dist2Inner < dist2Z) ) )
    normal = Vector3D<Precision>( -p.x(), -p.y(), p.z()*GetTIn2() ).Unit();

  return valid;

  }

//Old Definition
/*
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
 bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal ) const
  {
//return GetUnplacedVolume()->Normal(p,normal);

      bool valid=true;
      //HypeImplementation<translation::kIdentity, rotation::kIdentity>::NormalKernel<kScalar>(
        //      *GetUnplacedVolume(),
          //    point,
            //  normal, valid);
      //return valid;

Precision absZ(std::fabs(p.z()));
  Precision distZ(absZ - GetDz());
  Precision dist2Z(distZ*distZ);

  Precision xR2( p.x()*p.x()+p.y()*p.y() );
  Precision dist2Outer( std::fabs(xR2 - GetHypeRadius2(false,absZ)) );

  bool done=false;

  if (GetUnplacedVolume()->InnerSurfaceExists())
  {
    //
    // Has inner surface: is this closest?
    //
    Precision dist2Inner( std::fabs(xR2 - GetHypeRadius2(true,absZ)) );
    if (dist2Inner < dist2Z && dist2Inner < dist2Outer && !done)
	{
      normal = Vector3D<Precision>( -p.x(), -p.y(), p.z()*GetTIn2() ).Unit();
	  done = true;
	}
  }

  //
  // Do the "endcaps" win?
  //
  if (dist2Z < dist2Outer && !done)
	{
    //normal = Vector3D<Precision>( 0.0, 0.0, p.z() < 0 ? 1.0 : -1.0 );
	normal = Vector3D<Precision>( 0.0, 0.0, p.z() < 0 ? -1.0 : 1.0 );
	done = true;
	}


  //
  // Outer surface wins
  //
  //else
  if(!done)
  normal = Vector3D<Precision>( p.x(), p.y(), -p.z()*GetTOut2() ).Unit();

  return valid;

  }
*/

  Precision Capacity() override { return GetUnplacedVolume()->Capacity(); }


  Precision SurfaceArea() override  { return GetUnplacedVolume()->SurfaceArea(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  std::string GetEntityType() const { return GetUnplacedVolume()->GetEntityType() ;}


  void Extent( Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override { return GetUnplacedVolume()->Extent(aMin,aMax);}

  void GetParametersList(int aNumber, double *aArray) const { return GetUnplacedVolume()->GetParametersList(aNumber, aArray);}

  Vector3D<Precision>  GetPointOnSurface() const { return GetUnplacedVolume()->GetPointOnSurface();}


  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void ComputeBBox() const { return GetUnplacedVolume()->ComputeBBox();}


#ifndef VECGEOM_NVCC
  virtual VPlacedVolume const* ConvertToUnspecialized() const;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const* ConvertToRoot() const;
#endif
#ifdef VECGEOM_USOLIDS
  virtual ::VUSolid const* ConvertToUSolids() const;
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const* ConvertToGeant4() const;
#endif
#endif // VECGEOM_NVCC

};

} } // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDHYPE_H_
