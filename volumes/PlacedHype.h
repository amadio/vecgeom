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

  virtual ~PlacedHype() {}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  UnplacedHype const* GetUnplacedVolume() const {
    return static_cast<UnplacedHype const *>(
        GetLogicalVolume()->unplaced_volume());
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


//_____________________________________________________________  
//template <bool inner>
VECGEOM_CUDA_HEADER_BOTH  
VECGEOM_INLINE
Precision GetHypeRadius2(bool inner,Precision dz) const
{
if(inner)
	return GetRmin2()+GetTIn2()*dz*dz;
else
	return GetRmax2()+GetTOut2()*dz*dz;

}


VECGEOM_CUDA_HEADER_BOTH  
VECGEOM_INLINE
 bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal ) const
  {
//return GetUnplacedVolume()->Normal(p,normal);

      bool valid;
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

VECGEOM_CUDA_HEADER_BOTH  
VECGEOM_INLINE
Precision Capacity() const { return GetUnplacedVolume()->Capacity(); }


  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision SurfaceArea() const  { return GetUnplacedVolume()->SurfaceArea(); }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  std::string GetEntityType() const { return GetUnplacedVolume()->GetEntityType() ;}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Extent( Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const { return GetUnplacedVolume()->Extent(aMin,aMax);}
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void GetParametersList(int aNumber, double *aArray) const { return GetUnplacedVolume()->GetParametersList(aNumber, aArray);} 
  
  Vector3D<Precision>  GetPointOnSurface() const { return GetUnplacedVolume()->GetPointOnSurface();}
 
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void ComputeBBox() const { return GetUnplacedVolume()->ComputeBBox();}
  
  /*
  VECGEOM_CUDA_HEADER_BOTH
  bool Normal(Vector3D<Precision> const & point, Vector3D<Precision> & normal ) const
  {
      bool valid;
      HypeImplementation<translation::kIdentity, rotation::kIdentity>::NormalKernel<kScalar>(
              *GetUnplacedVolume(),
              point,
              normal, valid);
      return valid;
  }
  */


#ifdef VECGEOM_BENCHMARK
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
#endif // VECGEOM_BENCHMARK

#ifdef VECGEOM_CUDA_INTERFACE
  virtual VPlacedVolume* CopyToGpu(LogicalVolume const *const logical_volume,
                                   Transformation3D const *const transformation,
                                   VPlacedVolume *const gpu_ptr) const;
  virtual VPlacedVolume* CopyToGpu(
      LogicalVolume const *const logical_volume,
      Transformation3D const *const transformation) const;
#endif

};

} } // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDHYPE_H_
