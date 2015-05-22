//===-- kernel/HypeImplementation.h - Instruction class definition -------*- C++ -*-===//
//
//                     GeantV - VecGeom
//
// This file is distributed under the LGPL
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)
/// \brief This file implements the Hype shape
///


#ifndef VECGEOM_VOLUMES_KERNEL_HYPEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_HYPEIMPLEMENTATION_H_

#include "base/Global.h"
#include <iomanip>

#include "base/Transformation3D.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/UnplacedHype.h"
#include "volumes/kernel/shapetypes/HypeTypes.h"

//different SafetyToIn implementations
//#define ACCURATE_BB
#define ACCURATE_BC
//#define ROOTLIKE


//namespace ParaboloidUtilities
//{
//    template <class Backend>
//    VECGEOM_INLINE
//    VECGEOM_CUDA_HEADER_BOTH
//    void DistToHyperboloidSurface(
//                                 UnplacedHype const &unplaced,
//                                 Vector3D<typename Backend::precision_v> const &point,
//                                 Vector3D<typename Backend::precision_v> const &direction,
//                                 typename Backend::precision_v &distance/*,
//                                                                         typename Backend::bool_v in*/)
//    {
//        return;
//    }
//}


namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(HypeImplementation, TranslationCode, translation::kGeneric, RotationCode, rotation::kGeneric)

inline namespace VECGEOM_IMPL_NAMESPACE { 

class PlacedHype;
class UnplacedHype;

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct HypeImplementation {
    
  static const int transC = transCodeT;
  static const int rotC   = rotCodeT;


    
using PlacedShape_t = PlacedHype;
using UnplacedShape_t = UnplacedHype;

VECGEOM_CUDA_HEADER_BOTH
static void PrintType() {
   printf("SpecializedHype<%i, %i>", transCodeT, rotCodeT);
}
        
        template<typename Backend>
        VECGEOM_INLINE
        VECGEOM_CUDA_HEADER_BOTH
        static void UnplacedContains(
                                     UnplacedHype const &hype,
                                     Vector3D<typename Backend::precision_v> const &localPoint,
                                     typename Backend::bool_v &inside);
        
        
        template <typename Backend>
        VECGEOM_INLINE
        VECGEOM_CUDA_HEADER_BOTH
        static void Contains(
                             UnplacedHype const &unplaced,
                             Transformation3D const &transformation,
                             Vector3D<typename Backend::precision_v> const &point,
                             Vector3D<typename Backend::precision_v> &localPoint,
                             typename Backend::bool_v &inside);
        
        template <typename Backend>
        VECGEOM_INLINE
        VECGEOM_CUDA_HEADER_BOTH
        static void Inside(
                           UnplacedHype const &unplaced,
                           Transformation3D const &transformation,
                           Vector3D<typename Backend::precision_v> const &point,
                           typename Backend::inside_v &inside);
        
        template <typename Backend, bool ForInside>
        VECGEOM_INLINE
        VECGEOM_CUDA_HEADER_BOTH
        static void GenericKernelForContainsAndInside(UnplacedHype const &unplaced,
                                                      Vector3D<typename Backend::precision_v> const &,
                                                      typename Backend::bool_v &completelyoutside,
                                                      typename Backend::bool_v &completelyinside);
        
        template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static void DistanceToIn(
                                 UnplacedHype const &unplaced,
                                 Transformation3D const &transformation,
                                 Vector3D<typename Backend::precision_v> const &point,
                                 Vector3D<typename Backend::precision_v> const &direction,
                                 typename Backend::precision_v const &stepMax,
                                 typename Backend::precision_v &distance);
        
        template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static void DistanceToOut(
                                  UnplacedHype const &unplaced,
                                  Vector3D<typename Backend::precision_v> const &point,
                                  Vector3D<typename Backend::precision_v> const &direction,
                                  typename Backend::precision_v const &stepMax,
                                  typename Backend::precision_v &distance);
        
        template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static void SafetyToIn(UnplacedHype const &unplaced,
                               Transformation3D const &transformation,
                               Vector3D<typename Backend::precision_v> const &point,
                               typename Backend::precision_v &safety);
        
        template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static void SafetyToOut(UnplacedHype const &unplaced,
                                Vector3D<typename Backend::precision_v> const &point,
                                typename Backend::precision_v &safety);
        
        template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static void ContainsKernel(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
                                   typename Backend::bool_v &inside);

		template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static void IsSurfacePoint(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
                                   typename Backend::bool_v &surface);
        
        template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static void InsideKernel(
                                 UnplacedHype const &unplaced,
                                 Vector3D<typename Backend::precision_v> const &point,
                                 typename Backend::inside_v &inside);
        
        template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static void DistanceToInKernel(
                                       UnplacedHype const &unplaced,
                                       Vector3D<typename Backend::precision_v> const &point,
                                       Vector3D<typename Backend::precision_v> const &direction,
                                       typename Backend::precision_v const &stepMax,
                                       typename Backend::precision_v &distance);
        
        template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static void DistanceToOutKernel(
                                        UnplacedHype const &unplaced,
                                        Vector3D<typename Backend::precision_v> const &point,
                                        Vector3D<typename Backend::precision_v> const &direction,
                                        typename Backend::precision_v const &stepMax,
                                        typename Backend::precision_v &distance);

		template <class Backend,bool ForInner>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static void DistToHype(
                                        UnplacedHype const &unplaced,
                                        Vector3D<typename Backend::precision_v> const &point,
                                        Vector3D<typename Backend::precision_v> const &direction,
                                        
                                        typename Backend::precision_v &distance);
        
        template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static void SafetyToInKernel(
                                     UnplacedHype const &unplaced,
                                     Vector3D<typename Backend::precision_v> const &point,
                                     typename Backend::precision_v & safety);
        
        template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static void SafetyToOutKernel(
                                      UnplacedHype const &unplaced,
                                      Vector3D<typename Backend::precision_v> const &point,
                                      typename Backend::precision_v &safety);

		template <class Backend>
  		VECGEOM_CUDA_HEADER_BOTH
  		VECGEOM_INLINE
  		static void Normal(
       								UnplacedHype const &unplaced,
       								Vector3D<typename Backend::precision_v> const &point,
       								Vector3D<typename Backend::precision_v> &normal,
       								typename Backend::bool_v &valid );//{}
  
  		template <class Backend>
  		VECGEOM_CUDA_HEADER_BOTH
  		VECGEOM_INLINE
  		static void NormalKernel(
       								UnplacedHype const &unplaced,
      								Vector3D<typename Backend::precision_v> const &point,
       								Vector3D<typename Backend::precision_v> &normal,
       								typename Backend::bool_v &valid );//{}
       
		template <typename Backend,bool ForInnerRad>
		VECGEOM_CUDA_HEADER_BOTH
		VECGEOM_INLINE
		static void RadiusHypeSq(UnplacedHype const &unplaced, typename Backend::precision_v z, typename Backend::precision_v &radsq);
		
		template <class Backend>
  		VECGEOM_CUDA_HEADER_BOTH
  		VECGEOM_INLINE
  		static void ApproxDistOutside(typename Backend::precision_v pr, typename Backend::precision_v pz, Precision r0, Precision tanPhi, typename Backend::precision_v &ret){
		
		Precision dbl_min = 2.2250738585072014e-308;
		typedef typename Backend::precision_v Float_t;
		MaskedAssign(tanPhi < dbl_min , pr-r0 ,&ret);
	
		Float_t tan2Phi = tanPhi*tanPhi;
		Float_t z1 = pz;
		Float_t r1= Sqrt(r0*r0 + z1*z1*tan2Phi);
		
		Float_t z2 = (pr*tanPhi + pz)/(1 + tan2Phi);
		Float_t r2 = Sqrt( r0*r0 + z2*z2*tan2Phi );

		Float_t dr = r2-r1;
		Float_t dz = z2-z1;
	
		Float_t len=Sqrt(dr*dr + dz*dz);

		MaskedAssign(len < dbl_min,pr-r1,&dr);
		MaskedAssign(len < dbl_min,pz-z1,&dz);
		MaskedAssign(len < dbl_min,Sqrt(dr*dr + dz*dz),&ret);

		MaskedAssign(!(len < dbl_min),( Abs((pr-r1)*dz - (pz-z1)*dr)/len ),&ret);
	
	}

		template <class Backend>
  		VECGEOM_CUDA_HEADER_BOTH
  		VECGEOM_INLINE
  		static void ApproxDistInside(typename Backend::precision_v pr, typename Backend::precision_v pz, Precision r0, Precision tan2Phi, typename Backend::precision_v &ret){
			Precision dbl_min = 2.2250738585072014e-308;
			typedef typename Backend::precision_v Float_t;			
			MaskedAssign((tan2Phi < dbl_min),r0 - pr,&ret);
			Float_t rh = Sqrt(r0*r0 + pz*pz*tan2Phi );
		    Float_t dr = -rh;
		    Float_t dz = pz*tan2Phi;
   			Float_t len = Sqrt(dr*dr + dz*dz);
			ret = Abs((pr-rh)*dr)/len;
			
	}

		template <class Backend>
  		VECGEOM_CUDA_HEADER_BOTH
  		VECGEOM_INLINE
  		static void InterSectionExist(typename Backend::precision_v a, typename Backend::precision_v b, typename Backend::precision_v c, typename Backend::bool_v &exist){
			
			exist = (b*b - 4*a*c > 0.);
		}
		
    }; // End struct HypeImplementation

template <TranslationCode transCodeT, RotationCode rotCodeT>    
template <typename Backend,bool ForInnerRad>
VECGEOM_CUDA_HEADER_BOTH
void HypeImplementation<transCodeT, rotCodeT>::RadiusHypeSq(UnplacedHype const &unplaced, typename Backend::precision_v z, typename Backend::precision_v &radsq){
   		if(ForInnerRad)
   		radsq = unplaced.GetRmin2() + unplaced.GetTIn2()*z*z;
   		else
   		radsq = unplaced.GetRmax2() + unplaced.GetTOut2()*z*z;
}


template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void HypeImplementation<transCodeT, rotCodeT>::Normal(
       UnplacedHype const &unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid ){
    //std::cout<<"Entered Normal "<<std::endl;
    NormalKernel<Backend>(unplaced, point, normal, valid);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void HypeImplementation<transCodeT, rotCodeT>::NormalKernel(
       UnplacedHype const &unplaced,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid ){

	//std::cout<<"Entered NormalKernel "<<std::endl;
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v      Bool_t;

	Vector3D<Float_t> localPoint = point;
	Float_t absZ = Abs(localPoint.z());
	Float_t distZ = absZ - unplaced.GetDz();
	Float_t dist2Z = distZ*distZ;
	
	Float_t xR2 = localPoint.x()*localPoint.x() + localPoint.y()*localPoint.y();
	Float_t radOSq(0.);
	RadiusHypeSq<Backend,false>(unplaced,localPoint.z(),radOSq);
	Float_t dist2Outer = Abs(xR2 - radOSq);
	//std::cout<<"LocalPoint : "<<localPoint<<std::endl;
	Bool_t done(false);
	
	
	//Inner Surface Wins	
	if(unplaced.InnerSurfaceExists())
	{
		Float_t radISq(0.);
		RadiusHypeSq<Backend,true>(unplaced,localPoint.z(),radISq);
		Float_t dist2Inner = Abs(xR2 - radISq );
		Bool_t cond = (dist2Inner < dist2Z && dist2Inner < dist2Outer);
		MaskedAssign(!done && cond,-localPoint.x(),&normal.x());
		MaskedAssign(!done && cond,-localPoint.y(),&normal.y());
		MaskedAssign(!done && cond,localPoint.z()*unplaced.GetTIn2(),&normal.z());
		normal = normal.Unit();
		done |= cond;
		if(IsFull(done))
			return;
		
	}

	//End Caps wins
	Bool_t condE = (dist2Z < dist2Outer) ;
	Float_t normZ(0.);
	CondAssign(localPoint.z()<0. , -1. ,1. ,&normZ);
	MaskedAssign(!done && condE ,0. , &normal.x());
	MaskedAssign(!done && condE ,0. , &normal.y());
	MaskedAssign(!done && condE ,normZ , &normal.z());
	normal = normal.Unit();
	done |= condE;
	if(IsFull(done))
		return;

	//Outer Surface Wins
	normal = Vector3D<Float_t>(localPoint.x(),localPoint.y(),-localPoint.z()*unplaced.GetTOut2()).Unit();
	

}



    template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::UnplacedContains(
                                                                   UnplacedHype const &hype,
                                                                   Vector3D<typename Backend::precision_v> const &localPoint,
                                                                   typename Backend::bool_v &inside) {
        
        ContainsKernel<Backend>(hype, localPoint, inside);
    }
    
    template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::Contains(
                                                           UnplacedHype const &unplaced,
                                                           Transformation3D const &transformation,
                                                           Vector3D<typename Backend::precision_v> const &point,
                                                           Vector3D<typename Backend::precision_v> &localPoint,
                                                           typename Backend::bool_v &inside) {
        
        localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
        UnplacedContains<Backend>(unplaced, localPoint, inside);
        
    }
    
    template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::Inside(
                                                         UnplacedHype const &unplaced,
                                                         Transformation3D const &transformation,
                                                         Vector3D<typename Backend::precision_v> const &point,
                                                         typename Backend::inside_v &inside) {
        
        InsideKernel<Backend>(unplaced,
                              transformation.Transform<transCodeT, rotCodeT>(point),
                              inside);
        
    }
    
    template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::DistanceToIn(
                                                               UnplacedHype const &unplaced,
                                                               Transformation3D const &transformation,
                                                               Vector3D<typename Backend::precision_v> const &point,
                                                               Vector3D<typename Backend::precision_v> const &direction,
                                                               typename Backend::precision_v const &stepMax,
                                                               typename Backend::precision_v &distance) {
        
        DistanceToInKernel<Backend>(
                                    unplaced,
                                    transformation.Transform<transCodeT, rotCodeT>(point),
                                    transformation.TransformDirection<rotCodeT>(direction),
                                    stepMax,
                                    distance
                                    );
    }
    
    template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::DistanceToOut(
                                                                UnplacedHype const &unplaced,
                                                                Vector3D<typename Backend::precision_v> const &point,
                                                                Vector3D<typename Backend::precision_v> const &direction,
                                                                typename Backend::precision_v const &stepMax,
                                                                typename Backend::precision_v &distance) {
        
        DistanceToOutKernel<Backend>(
                                     unplaced,
                                     point,
                                     direction,
                                     stepMax,
                                     distance
                                     );
    }
    
    template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    void HypeImplementation<transCodeT, rotCodeT>::SafetyToIn(
                                                             UnplacedHype const &unplaced,
                                                             Transformation3D const &transformation,
                                                             Vector3D<typename Backend::precision_v> const &point,
                                                             typename Backend::precision_v &safety) {
        
        SafetyToInKernel<Backend>(
                                  unplaced,
                                  transformation.Transform<transCodeT, rotCodeT>(point),
                                  safety
                                  );
    }
    
    template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    void HypeImplementation<transCodeT, rotCodeT>::SafetyToOut(
                                                              UnplacedHype const &unplaced,
                                                              Vector3D<typename Backend::precision_v> const &point,
                                                              typename Backend::precision_v &safety) {
        
        SafetyToOutKernel<Backend>(
                                   unplaced,
                                   point,
                                   safety
                                   );
    }
    
    template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::ContainsKernel(
                                                                 UnplacedHype const &unplaced,
                                                                 Vector3D<typename Backend::precision_v> const &localPoint,
                                                                 typename Backend::bool_v &inside) {
        
        typedef typename Backend::bool_v Bool_t;
        Bool_t unused;
        Bool_t outside;
        GenericKernelForContainsAndInside<Backend, false>(unplaced,
                                                          localPoint, unused, outside);
        inside=!outside;
    }


	template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::IsSurfacePoint(
                                                                 UnplacedHype const &unplaced,
                                                                 Vector3D<typename Backend::precision_v> const &localPoint,
                                                                 typename Backend::bool_v &surface) {
        
        typedef typename Backend::bool_v Bool_t;
        Bool_t inside;
        Bool_t outside;
        GenericKernelForContainsAndInside<Backend,true>(unplaced,
                                                          localPoint, inside, outside);
        surface=!outside && !inside;
    }
    
    
    template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <typename Backend, bool ForInside>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::GenericKernelForContainsAndInside(
                                                                                    UnplacedHype const &unplaced,
                                                                                    Vector3D<typename Backend::precision_v> const &point,
                                                                                    typename Backend::bool_v &completelyinside,
                                                                                    typename Backend::bool_v &completelyoutside) {
        typedef typename Backend::precision_v Float_t;
        
	Precision fRmax = unplaced.GetRmax();
	Precision fRmin = unplaced.GetRmin();
	Precision fStIn = unplaced.GetStIn();
	Precision fStOut = unplaced.GetStOut();
	Precision fDz = unplaced.GetDz();
	
	//along Z direction
	completelyoutside = Abs(point.z()) > fDz + kSTolerance*10.0;
	if(ForInside)
	{
	  completelyinside = Abs(point.z()) < fDz - kSTolerance*10.0;
	}

	Float_t r = Sqrt(point.x()*point.x()+point.y()*point.y());
	Float_t rOuter=Sqrt(unplaced.GetRmax2()+unplaced.GetTOut2()*point.z()*point.z());
    	Float_t rInner=Sqrt(unplaced.GetRmin2()+unplaced.GetTIn2()*point.z()*point.z());
	completelyoutside |= (r > rOuter + kSTolerance*10.0) || (r < rInner - kSTolerance*10.0);
	if(ForInside)
	{
	  completelyinside &= (r < rOuter - kSTolerance*10.0) && (r > rInner + kSTolerance*10.0);
	}

	return ;
    }
    
    template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::InsideKernel(
                                                               UnplacedHype const &unplaced,
                                                               Vector3D<typename Backend::precision_v> const &point,
                                                               typename Backend::inside_v &inside) {
        
        typedef typename Backend::bool_v      Bool_t;
        Bool_t completelyinside, completelyoutside;
        GenericKernelForContainsAndInside<Backend,true>(
                                                        unplaced, point, completelyinside, completelyoutside);
        inside=EInside::kSurface;
        MaskedAssign(completelyoutside, EInside::kOutside, &inside);
        MaskedAssign(completelyinside, EInside::kInside, &inside);
    }
    
    template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::DistanceToInKernel(
                                                                     UnplacedHype const &unplaced,
                                                                     Vector3D<typename Backend::precision_v> const &point,
                                                                     Vector3D<typename Backend::precision_v> const &direction,
                                                                     typename Backend::precision_v const &stepMax,
                                                                     typename Backend::precision_v &distance) {
        
        typedef typename Backend::precision_v Float_t;
        typedef typename Backend::bool_v      Bool_t;
        Bool_t done(false);
        distance=kInfinity;

		
        
        Float_t absZ=Abs(point.z());
        Float_t absDirZ=Abs(direction.z());
        Float_t rho2 = point.x()*point.x()+point.y()*point.y();
        Float_t point_dot_direction_x = point.x()*direction.x();
        Float_t point_dot_direction_y = point.y()*direction.y();

	Float_t pDotV3D = point.Dot(direction);
	Precision dZ=unplaced.GetDz();
	Bool_t surface(false);
	IsSurfacePoint<Backend>(unplaced,point,surface);

	
	//Intersection with Hyperbolic surface
	Bool_t inside(false);
	ContainsKernel<Backend>(unplaced,point,inside);
		
	Bool_t outerRegion1(false),outerRegion2(false);
	Float_t radO2(0.);
	RadiusHypeSq<Backend,false>(unplaced,point.z(),radO2);
	outerRegion1 = (absZ > unplaced.GetDz()) || (rho2  > radO2);
	Float_t radI2(0.);
	RadiusHypeSq<Backend,true>(unplaced,point.z(),radI2);
	outerRegion2 = (absZ < unplaced.GetDz()) && (rho2  < radI2);
		
        Bool_t checkZ=point.z()*direction.z() > 0; //means that the point is distancing from the volume
        
        //check if the point is above dZ and is distancing in Z
        Bool_t isDistancingInZ= (absZ>unplaced.GetDz() && checkZ);
        done|=isDistancingInZ;
        if (Backend::early_returns && done == Backend::kTrue) return;
        
        //check if the point is outside the bounding cylinder and is distancing in XY
        Bool_t isDistancingInXY=( (rho2>unplaced.GetEndOuterRadius2()) && (point_dot_direction_x>0 && point_dot_direction_y>0) );
        done|=isDistancingInXY;
        if (Backend::early_returns && done == Backend::kTrue) return;
        
        //check if x coordinate is > EndOuterRadius and the point is distancing in X
        Bool_t isDistancingInX=( (Abs(point.x())>unplaced.GetEndOuterRadius()) && (point_dot_direction_x>0) );
        done|=isDistancingInX;
        if (Backend::early_returns && done == Backend::kTrue) return;
        
        //check if y coordinate is > EndOuterRadiusthe point is distancing in Y
        Bool_t isDistancingInY=( (Abs(point.y())>unplaced.GetEndOuterRadius()) && (point_dot_direction_y>0) );
        done|=isDistancingInY;
        if (Backend::early_returns && done == Backend::kTrue) return;
        
        //is hitting from dz or -dz planes
        Float_t distZ = (absZ-unplaced.GetDz())/absDirZ;
        Float_t xHit = point.x()+distZ*direction.x();
        Float_t yHit = point.y()+distZ*direction.y();
        Float_t rhoHit2=xHit*xHit+yHit*yHit;
        
        Bool_t isCrossingAtDz= (absZ>unplaced.GetDz()) && (!checkZ) && (rhoHit2 <=unplaced.GetEndOuterRadius2() && rhoHit2>=unplaced.GetEndInnerRadius2());
        
        MaskedAssign(isCrossingAtDz, distZ, &distance);
        done|=isCrossingAtDz;
        if (Backend::early_returns && done == Backend::kTrue) return;
        
        
        //is hitting from the hyperboloid surface (OUTER or INNER)
        Float_t dirRho2 = direction.x()*direction.x()+direction.y()*direction.y();
        Float_t point_dot_direction_z = point.z()*direction.z();
        Float_t pointz2=point.z()*point.z();
        Float_t dirz2=direction.z()*direction.z();
    
        //SOLUTION FOR OUTER
        //NB: bOut=-B/2 of the second order equation
        //So the solution is: (b +/- Sqrt(b^2-ac))*ainv
        
        Float_t aOut = dirRho2 - unplaced.GetTOut2() * dirz2;
        Float_t bOut = unplaced.GetTOut2()*point_dot_direction_z - point_dot_direction_x - point_dot_direction_y;
        Float_t cOut = rho2 - unplaced.GetTOut2()* pointz2 - unplaced.GetRmax2();
	
        
        Float_t aOutinv = 1./aOut;
        Float_t prodOut = cOut*aOut;
        Float_t deltaOut = bOut*bOut - prodOut;
        Bool_t deltaOutNeg=deltaOut<0;

        MaskedAssign(deltaOutNeg, 0. , &deltaOut);
        deltaOut = Sqrt(deltaOut);
        
        //Float_t distOut=aOutinv*(bOut -deltaOut);
	std::cout<<std::setprecision(15);
	//std::cout<<aOut<< "  :  "<<bOut<<"  :  "<<cOut<<std::endl;

	Float_t distOut(0.);
	MaskedAssign(bOut>0. ,(cOut/(bOut+deltaOut)) , &distOut);
	MaskedAssign(bOut<0. ,(aOutinv*(bOut -deltaOut)) , &distOut);
        Float_t zHitOut1 = point.z()+distOut*direction.z();
        Bool_t isHittingHyperboloidSurfaceOut1 = ( (distOut> 1E20) || (Abs(zHitOut1)<=unplaced.GetDz()) ); //why: dist > 1E20?
		
		
	Float_t solution_Outer=kInfinity;
	MaskedAssign(!deltaOutNeg &&isHittingHyperboloidSurfaceOut1 && distOut>0, distOut, &solution_Outer);
        
        //SOLUTION FOR INNER
	Float_t aIn = dirRho2 - unplaced.GetTIn2() * dirz2;
        Float_t bIn = unplaced.GetTIn2()*point_dot_direction_z - point_dot_direction_x - point_dot_direction_y;
        Float_t cIn = rho2 - unplaced.GetTIn2()* pointz2 - unplaced.GetRmin2();
        Float_t aIninv = 1./aIn;
        
        Float_t prodIn = cIn*aIn;
        Float_t deltaIn = bIn*bIn - prodIn;
    
        Bool_t deltaInNeg=deltaIn<0;
        MaskedAssign(deltaInNeg, 0. , &deltaIn);
        deltaIn = Sqrt(deltaIn);
        
        //Float_t distIn=aIninv*(bIn +deltaIn);
	std::cout<<std::setprecision(15);
	//std::cout<<aIn<< "  :  "<<bIn<<"  :  "<<cIn<<std::endl;

	Float_t distIn(0.);
	MaskedAssign(bIn<0. ,(cIn/(bIn-deltaIn)) , &distIn);
	MaskedAssign(bIn>0. ,(aIninv*(bIn +deltaIn)) , &distIn);

	Float_t zHitIn1 = point.z()+distIn*direction.z();
        Bool_t isHittingHyperboloidSurfaceIn1 = ( (distIn> 1E20) || (Abs(zHitIn1)<=unplaced.GetDz()) ); //why: dist > 1E20?
    
        Float_t solution_Inner=kInfinity;
	MaskedAssign(!deltaInNeg && isHittingHyperboloidSurfaceIn1 && distIn>0, distIn, &solution_Inner);
        
        Float_t solution=Min(solution_Inner, solution_Outer);
        
        done|=(deltaInNeg && deltaOutNeg);
        MaskedAssign(!done, solution, &distance );
		
	Bool_t isPointInside(false);
	ContainsKernel<Backend>(unplaced,point,isPointInside);
	MaskedAssign(isPointInside, kSTolerance , &distance);
		
    }
    

	template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <class Backend,bool ForInner>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::DistToHype(
                                                                      UnplacedHype const &unplaced,
                                                                      Vector3D<typename Backend::precision_v> const &point,
                                                                      Vector3D<typename Backend::precision_v> const &direction,
                                                                      
                                                                      typename Backend::precision_v &distance) {

	typedef typename Backend::precision_v Float_t;
	typedef typename Backend::bool_v      Bool_t;	
	
	Float_t a(0.),b(0.),c(0.);
	Bool_t exist(false),fal(false);
	Precision tanInnerStereo = unplaced.GetTIn();
	Precision tanInnerStereo2 = tanInnerStereo*tanInnerStereo;
	Precision tanOuterStereo = unplaced.GetTOut();
	Precision tanOuterStereo2 = tanOuterStereo*tanOuterStereo;
	Precision fRmin = unplaced.GetRmin();
	Precision fRmin2 = fRmin * fRmin;
	Precision fRmax = unplaced.GetRmax();
	Precision fRmax2 = fRmax * fRmax;
	
	Float_t distInner=kInfinity;
	Float_t distOuter=kInfinity;
	
	
	Bool_t done(false);
	
	if(ForInner)
	{
	a = direction.x() * direction.x() + direction.y() * direction.y() - tanInnerStereo2*direction.z()*direction.z();
	b = 2*(direction.x()*point.x() + direction.y()*point.y() - tanInnerStereo2*direction.z()*point.z());
	c= point.x()*point.x() + point.y()*point.y() - tanInnerStereo2*point.z()*point.z() - fRmin2;
	//std::cout<<"A : "<<a<<"  :: B : "<<b<<"  :: C : "<<c<<std::endl;
	exist = (b*b - 4*a*c > 0.);
	//std::cout<<"Exist : "<<exist<<std::endl;
	MaskedAssign(!done && exist ,((2*c)/(-b - Sqrt(b*b - 4*a*c)) ),&distInner);
	//std::cout<<"DistOuter : "<<distOuter<<std::endl;
	distance = distInner;
	}
	else
	{
	a = direction.x() * direction.x() + direction.y() * direction.y() - tanOuterStereo2*direction.z()*direction.z();
	b = 2*(direction.x()*point.x() + direction.y()*point.y() - tanOuterStereo2*direction.z()*point.z());
	c= point.x()*point.x() + point.y()*point.y() - tanOuterStereo2*point.z()*point.z() - fRmax2;
	//std::cout<<"A : "<<a<<"  :: B : "<<b<<"  :: C : "<<c<<std::endl;
	exist = (b*b - 4*a*c > 0.);
	//std::cout<<"Exist : "<<exist<<std::endl;
	MaskedAssign(!done && exist ,((2*c)/(-b + Sqrt(b*b - 4*a*c)) ),&distOuter);
	//std::cout<<"DistOuter : "<<distOuter<<std::endl;
	distance = distOuter;
	}
return;
}


	template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::DistanceToOutKernel(
                                                                      UnplacedHype const &unplaced,
                                                                      Vector3D<typename Backend::precision_v> const &point,
                                                                      Vector3D<typename Backend::precision_v> const &direction,
                                                                      typename Backend::precision_v const &stepMax,
                                                                      typename Backend::precision_v &distance) {

	typedef typename Backend::precision_v Float_t;
	typedef typename Backend::bool_v      Bool_t;	

	Float_t a(0.),b(0.),c(0.);
	Bool_t exist(false),fal(false);
	Precision tanInnerStereo = unplaced.GetTIn();
	Precision tanInnerStereo2 = tanInnerStereo*tanInnerStereo;
	Precision tanOuterStereo = unplaced.GetTOut();
	Precision tanOuterStereo2 = tanOuterStereo*tanOuterStereo;
	Precision fRmin = unplaced.GetRmin();
	Precision fRmin2 = fRmin * fRmin;
	Precision fRmax = unplaced.GetRmax();
	Precision fRmax2 = fRmax * fRmax;
	
	Float_t distInner=kInfinity;
	Float_t distOuter=kInfinity;
	//Float_t distInner(0.);
	//Float_t distOuter(0.);
	
	Bool_t done(false);
	
	
	//Handling Inner Surface
	if(unplaced.InnerSurfaceExists())
	{
	a = direction.x() * direction.x() + direction.y() * direction.y() - tanInnerStereo2*direction.z()*direction.z();
	b = 2*direction.x()*point.x() + 2*direction.y()*point.y() - 2*tanInnerStereo2*direction.z()*point.z();
	c= point.x()*point.x() + point.y()*point.y() - tanInnerStereo2*point.z()*point.z() - fRmin2;
	//InterSectionExist<Backend>(a,b,c,exist);
	//std::cout<<a<<"  :  "<<b<<"  :  "<<c<<std::endl;
	exist = (b*b - 4*a*c > 0.);
	//std::cout<<"Exist : "<<exist<<std::endl;

	MaskedAssign(!done && exist && b>0. ,( (-b - Sqrt(b*b - 4*a*c))/(2*a) ),&distInner);
	MaskedAssign(!done && exist && b<=0.,((2*c)/(-b + Sqrt(b*b - 4*a*c)) ),&distInner);
	//std::cout<<"DistInner : "<<distInner<<std::endl;
	//done |= exist;
	MaskedAssign(distInner < 0. ,kInfinity, &distInner);
	//MaskedAssign(distInner < 0. ,0., &distInner);
	}

	//Handling Outer surface
	exist = fal;
	a = direction.x() * direction.x() + direction.y() * direction.y() - tanOuterStereo2*direction.z()*direction.z();
	b = 2*direction.x()*point.x() + 2*direction.y()*point.y() - 2*tanOuterStereo2*direction.z()*point.z();
	c= point.x()*point.x() + point.y()*point.y() - tanOuterStereo2*point.z()*point.z() - fRmax2;
	//std::cout<<a<<"  :  "<<b<<"  :  "<<c<<std::endl;
	//InterSectionExist<Backend>(a,b,c,exist);
	exist = (b*b - 4*a*c > 0.);
	//std::cout<<"Exist : "<<exist<<std::endl;
	
	MaskedAssign(!done && exist && b<0.,( (-b + Sqrt(b*b - 4*a*c))/(2*a) ),&distOuter);
	MaskedAssign(!done && exist && b>=0.,((2*c)/(-b - Sqrt(b*b - 4*a*c)) ),&distOuter);
	//std::cout<<"DistOuter : "<<distOuter<<std::endl;


	//done |= exist;
	MaskedAssign(distOuter < 0. ,kInfinity, &distOuter);
	
    //Handling Z surface
	
    Float_t distZ=kInfinity;
    Float_t dirZinv=1/direction.z();
    Bool_t dir_mask= direction.z()<0;
    MaskedAssign(!done && dir_mask, -(unplaced.GetDz() + point.z())*dirZinv, &distZ);
	//done |= dir_mask;

    MaskedAssign(!done && !dir_mask, (unplaced.GetDz() - point.z())*dirZinv, &distZ);
	
	//My Dev
	//Float_t distDirz = Sqrt((point.Mag2()-fRmin2)/tanInnerStereo2);
	//MaskedAssign((direction.x()==0.) && (direction.y()==0.),distDirz,&distZ);


	//done |= !dir_mask;
	MaskedAssign(distZ < 0. , kInfinity, &distZ);
	//MaskedAssign(distZ < 0. , 0., &distZ);
	//std::cout<<"DistInner : "<<distInner<<"  :: DistOuter : "<<distOuter<<std::endl;
	//std::cout<<"DistZ : "<<distZ<<std::endl;
	distance = Min(distInner,distOuter);
	distance = Min(distance,distZ);


	Bool_t isPointInside(false);
	ContainsKernel<Backend>(unplaced,point,isPointInside);
	MaskedAssign(!isPointInside, kSTolerance , &distance);
	}


	
	/*
    template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::DistanceToOutKernel(
                                                                      UnplacedHype const &unplaced,
                                                                      Vector3D<typename Backend::precision_v> const &point,
                                                                      Vector3D<typename Backend::precision_v> const &direction,
                                                                      typename Backend::precision_v const &stepMax,
                                                                      typename Backend::precision_v &distance) {

        
        typedef typename Backend::precision_v Float_t;
        typedef typename Backend::bool_v      Bool_t;
        
        distance=kInfinity;
        
        //Distance to Z surface
        Float_t distZ=kInfinity;
        Float_t dirZinv=1/direction.z();
        Bool_t dir_mask= direction.z()<0;
        MaskedAssign(dir_mask, -(unplaced.GetDz() + point.z())*dirZinv, &distZ);
        MaskedAssign(!dir_mask, (unplaced.GetDz() - point.z())*dirZinv, &distZ);

        //Distance to INNER and OUTER hyperbola surfaces
        Float_t distHypeInner=kInfinity;
        Float_t distHypeOuter=kInfinity;
        
        Float_t absZ=Abs(point.z());
        Float_t absDirZ=Abs(direction.z());
        Float_t rho2 = point.x()*point.x()+point.y()*point.y();
        Float_t dirRho2 = direction.x()*direction.x()+direction.y()*direction.y();
        Float_t point_dot_direction_x = point.x()*direction.x();
        Float_t point_dot_direction_y = point.y()*direction.y();
        Float_t point_dot_direction_z = point.z()*direction.z();
        Float_t pointz2=point.z()*point.z();
        Float_t dirz2=direction.z()*direction.z();
        
        //SOLUTION FOR OUTER
        //NB: bOut=-B/2 of the second order equation
        //So the solution is: (b +/- Sqrt(b^2-ac))*ainv
        
        Float_t aOut = dirRho2 - unplaced.GetTOut2() * dirz2;
        Float_t bOut = unplaced.GetTOut2()*point_dot_direction_z - point_dot_direction_x - point_dot_direction_y;
        Float_t cOut = rho2 - unplaced.GetTOut2()* pointz2 - unplaced.GetRmax2();
        
        Float_t aOutinv = 1./aOut;
        Float_t prodOut = cOut*aOut;
        Float_t deltaOut = bOut*bOut - prodOut;
        
        Bool_t deltaOutNeg=deltaOut<0;
        MaskedAssign(deltaOutNeg, 0. , &deltaOut);
        deltaOut = Sqrt(deltaOut);
        
        Bool_t mask_signOut=(aOutinv<0);
        Float_t signOut=1.;
        MaskedAssign(mask_signOut, -1., &signOut);
        
        Float_t distOut1=aOutinv*(bOut - signOut*deltaOut);
        Float_t distOut2=aOutinv*(bOut + signOut*deltaOut);
        
        MaskedAssign(distOut1>0 && !deltaOutNeg , distOut1, &distHypeOuter);
        MaskedAssign(distOut1<0 && distOut2>0 && !deltaOutNeg, distOut2, &distHypeOuter);
        MaskedAssign(distOut1<0 && distOut2<0 && !deltaOutNeg, kInfinity, &distHypeOuter);
        
        //SOLUTION FOR INNER
        //NB: bOut=-B/2 of the second order equation
        //So the solution is: (b +/- Sqrt(b^2-ac))*ainv
        
        Float_t aIn = dirRho2 - unplaced.GetTIn2() * dirz2;
        Float_t bIn = unplaced.GetTIn2()*point_dot_direction_z - point_dot_direction_x - point_dot_direction_y;
        Float_t cIn = rho2 - unplaced.GetTIn2()* pointz2 - unplaced.GetRmin2();
        Float_t aIninv = 1./aIn;
        
        Float_t prodIn = cIn*aIn;
        Float_t deltaIn = bIn*bIn - prodIn;
        
        Bool_t deltaInNeg=deltaIn<0;
        MaskedAssign(deltaInNeg, 0. , &deltaIn);
        deltaIn = Sqrt(deltaIn);
        
        Bool_t mask_signIn=(aIninv<0);
        Float_t signIn=1.;
        MaskedAssign(mask_signIn, -1., &signIn);
        
        Float_t distIn1=aIninv*(bIn - signIn*deltaIn);
        Float_t distIn2=aIninv*(bIn + signIn*deltaIn);
        
        MaskedAssign(distIn1>0 && !deltaInNeg, distIn1, &distHypeInner);
        MaskedAssign(distIn1<0 && distIn2>0 && !deltaInNeg, distIn2, &distHypeInner);
        MaskedAssign(distIn1<0 && distIn2<0 && !deltaInNeg, kInfinity, &distHypeInner);
        Float_t distHype=Min(distHypeInner, distHypeOuter);
		//std::cout << " distHypeInner : "<<distHypeInner<<"  :: distHypeOuter : "<< distHypeOuter << "  :: distZ : "<<distZ<<std::endl;
        distance=Min(distHype, distZ);
		//std::cout<<"distance : "<<distance<<std::endl;
		//std::cout<<"kSTolerance*10. : "<<kSTolerance*10.<<std::endl;
		//std::cout<<"(distance < kSTolerance*10.) : "<<(distance < kSTolerance*10.)<<std::endl;
		//MaskedAssign((distance < kSTolerance*100.),0.,&distance);
		
		//This block ideally should come at the top
		
		//Bool_t done(false);
	 	//Bool_t inside(false);
	  	//UnplacedContains<Backend>(unplaced,point,inside);
	  	//MaskedAssign(!done && inside,0.,&distance);	
	  	//done |= inside;
		
    }
    */

    template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::SafetyToInKernel(
                                                                   UnplacedHype const &unplaced,
                                                                   Vector3D<typename Backend::precision_v> const &point,
                                                                   typename Backend::precision_v &safety) {
        
        typedef typename Backend::precision_v Float_t;
        typedef typename Backend::bool_v Bool_t;
		
		
	Precision halfTol = 0.5*kSTolerance;
	Float_t absZ = Abs(point.z());
	Float_t r2 = point.x()*point.x() + point.y()*point.y();
	Float_t r =  Sqrt(r2);
	

	Float_t sigz = absZ - unplaced.GetDz();
	Precision endOuterRadius = unplaced.GetEndOuterRadius();
	Precision endInnerRadius = unplaced.GetEndInnerRadius();
	Precision innerRadius = unplaced.GetRmin();
	Precision outerRadius = unplaced.GetRmax();
	Precision tanInnerStereo2 = unplaced.GetTIn2();
	Precision tanOuterStereo2 = unplaced.GetTOut2();
	Precision tanOuterStereo = unplaced.GetTOut();
	Float_t dr = endInnerRadius - r;
	Float_t answer = Sqrt(dr*dr + sigz*sigz);

	Bool_t innerSurfaceExist(unplaced.InnerSurfaceExists());
	

	Bool_t one(false),two(false),three(false);
	one = (r < endOuterRadius);
	two = !(r < endOuterRadius);
	three = innerSurfaceExist;
	

	//Working nicely but slow
	//Not the finalized one. Needs a relook , cosmetic modification and optimization
	Bool_t inside(false);
	Bool_t done(false);
	
	UnplacedContains<Backend>(unplaced,point,inside);
	MaskedAssign(!done && inside,0.,&safety);	
	done |= inside;
	

	MaskedAssign(!done && (r < endOuterRadius) && (sigz > -halfTol) && innerSurfaceExist &&  (r > endInnerRadius) && (sigz < halfTol),0.,&safety);
	done |= (r < endOuterRadius) && (sigz > -halfTol) && innerSurfaceExist &&  (r > endInnerRadius) && (sigz < halfTol);

	MaskedAssign(!done && (r < endOuterRadius) && (sigz > -halfTol) && innerSurfaceExist && (r > endInnerRadius) && !(sigz < halfTol),sigz,&safety);
	done |= (r < endOuterRadius) && (sigz > -halfTol) && innerSurfaceExist &&  (r > endInnerRadius) && !(sigz < halfTol);
	//std::cout<<"3 - : "<<safety<<std::endl;
	//std::cout<<"Done : "<<done<<std::endl;
	
	MaskedAssign(!done &&  (r < endOuterRadius) && (sigz > -halfTol) && innerSurfaceExist &&  (sigz > dr*tanInnerStereo2) && (answer < halfTol),0.,&safety);
	done |= (r < endOuterRadius) && (sigz > -halfTol) && innerSurfaceExist &&  (sigz > dr*tanInnerStereo2) && (answer < halfTol);

	MaskedAssign(!done && (r < endOuterRadius) && (sigz > -halfTol) && innerSurfaceExist && (sigz > dr*tanInnerStereo2) && !(answer < halfTol),answer,&safety);
	done |= (r < endOuterRadius) && (sigz > -halfTol) && innerSurfaceExist && (sigz > dr*tanInnerStereo2) && !(answer < halfTol);
	//std::cout<<"4 - : "<<safety<<std::endl;
	//std::cout<<"Done : "<<done<<std::endl;

	MaskedAssign(!done && (r < endOuterRadius) && (sigz > -halfTol) && !innerSurfaceExist &&  (sigz < halfTol),0.,&safety);
	done |= (r < endOuterRadius) && (sigz > -halfTol) && !innerSurfaceExist &&  (sigz < halfTol);

	MaskedAssign(!done && (r < endOuterRadius) && (sigz > -halfTol) && !innerSurfaceExist && !(sigz < halfTol),sigz,&safety);
	done |= (r < endOuterRadius) && (sigz > -halfTol) && !innerSurfaceExist && !(sigz < halfTol);
	//std::cout<<"5 - : "<<safety<<std::endl;
	//std::cout<<"Done : "<<done<<std::endl;

	dr = r - endOuterRadius;
	answer = Sqrt(dr*dr + sigz*sigz);
	MaskedAssign(!done && !(r < endOuterRadius) && (sigz > -dr*tanOuterStereo2) && (answer < halfTol),0.,&safety);
	done |= !(r < endOuterRadius) && (sigz > -dr*tanOuterStereo2) && (answer < halfTol);

	MaskedAssign(!done && !(r < endOuterRadius) && (sigz > -dr*tanOuterStereo2) && !(answer < halfTol),answer,&safety);
	done |= !(r < endOuterRadius) && (sigz > -dr*tanOuterStereo2) && !(answer < halfTol);
	//std::cout<<"6 - : "<<safety<<std::endl;
	//std::cout<<"Done : "<<done<<std::endl;

	Float_t radSq;
	RadiusHypeSq<Backend,true>(unplaced,point.z(),radSq);
	//std::cout<<"RADSQ : "<<radSq<<std::endl;
	ApproxDistInside<Backend>(r,absZ,innerRadius,tanInnerStereo2,answer);
	//std::cout<<"Answer-1 : "<<answer<<std::endl;
	MaskedAssign(!done && innerSurfaceExist &&  (r2 < radSq + kSTolerance*endInnerRadius) && (answer < halfTol), 0. ,&safety);
	done |= innerSurfaceExist &&  (r2 < radSq + kSTolerance*endInnerRadius) && (answer < halfTol);
	//std::cout<<"Done : "<<done<<std::endl;

	MaskedAssign(!done && innerSurfaceExist &&  (r2 < radSq + kSTolerance*endInnerRadius) && !(answer < halfTol), answer ,&safety);
	done |= innerSurfaceExist &&  (r2 < radSq + kSTolerance*endInnerRadius) && !(answer < halfTol);

	
	ApproxDistOutside<Backend>( r,absZ,outerRadius,tanOuterStereo,answer );
	MaskedAssign( !done && (answer < halfTol) ,0.,&safety);
	done |= (answer < halfTol);

	MaskedAssign( !done && !(answer < halfTol),answer,&safety);
	done |= !(answer < halfTol);
	//std::cout<<"Done : "<<done<<std::endl;
	
	
	/*
        safety=0.;
        Float_t safety_t;
        Float_t absZ= Abs(point.z());
        Float_t safeZ= absZ-unplaced.GetDz();
 
#ifdef ROOTLIKE
        Float_t rsq = point.x()*point.x()+point.y()*point.y();
        Float_t r = Sqrt(rsq);
    
        //Outer
        Float_t rhsqOut=unplaced.GetRmax2()+unplaced.GetTOut2()*point.z()*point.z();
        Float_t rhOut = Sqrt(rhsqOut);
        Float_t drOut = r - rhOut;
        
        Float_t safermax=0.;
        
        Bool_t mask_drOut=(drOut<0);
        MaskedAssign(mask_drOut, -kInfinity, &safermax);
        
        Bool_t mask_fStOut(Abs(unplaced.GetStOut())<kTolerance);
        MaskedAssign(!mask_drOut && mask_fStOut, Abs(drOut), &safermax);
        
        Float_t zHypeSqOut= Sqrt((r*r-unplaced.GetRmax2())*(unplaced.GetTOut2Inv()));
        Float_t mOut=(zHypeSqOut-absZ)/drOut;
    
        Float_t safe = mOut*drOut/Sqrt(1.+mOut*mOut);
        Bool_t doneOuter(mask_fStOut || mask_drOut);
        MaskedAssign(!doneOuter, safe, &safermax);
        Float_t max_safety= Max(safermax, safeZ);
        
        //Check for Inner Threatment -->this should be managed as a specialization
        if(unplaced.GetEndInnerRadius()!=0)
        {
            Float_t safermin=0.;
            Float_t rhsqIn = unplaced.GetRmin2()+unplaced.GetTIn()*point.z()*point.z();
            Float_t rhIn = Sqrt(rhsqIn);
            Float_t drIn = r-rhIn;
            
            Bool_t mask_drIn(drIn>0.);
            MaskedAssign(mask_drIn, -kInfinity, &safermin);
            
            Bool_t mask_fStIn(Abs(unplaced.GetStIn()<kTolerance));
            MaskedAssign(!mask_drIn && mask_fStIn , Abs(drIn), &safermin);
            
            Bool_t mask_fRmin(unplaced.GetRmin()<kTolerance);
            MaskedAssign(! mask_drIn && !mask_fStIn && mask_fRmin, drIn/Sqrt(1.+unplaced.GetTIn2()), &safermin);
            
            Bool_t mask_drMin=Abs(drIn)<kTolerance;
            MaskedAssign(!mask_drIn && !mask_fStIn && !mask_fRmin && mask_drMin, 0., &safermin);
            Bool_t doneInner(mask_drIn || mask_fStIn ||mask_fRmin || mask_drMin );
          
            Float_t zHypeSqIn= Sqrt( (r*r-unplaced.GetRmin2()) *(unplaced.GetTIn2Inv()) );
            Float_t mIn=-rhIn*unplaced.GetTIn2Inv()/absZ;
            
            safe = mIn*drIn/Sqrt(1.+mIn*mIn);
            MaskedAssign(!doneInner, safe, &safermin);
            max_safety= Max(max_safety, safermin);
        }
        safety=max_safety;
      
#endif
        
#ifdef ACCURATE_BB
        //Bounding-Box implementation
        Float_t absX= Abs(point.x());
        Float_t absY= Abs(point.y());
        
        //check if the point is inside the inner-bounding box
        
        //The square inscribed in the inner circle has side=r*sqrt(2)
        Float_t safeX_In=absX-unplaced.GetInSqSide();
        Float_t safeY_In=absY-unplaced.GetInSqSide();
        Bool_t  mask_bcIn= (safeX_In<0) &&(safeY_In<0) && (safeZ>0);
        safety_t=Min(safeX_In, safeY_In);
        safety_t=Min(safety_t, safeZ);
        Bool_t done(mask_bcIn);
        if (Backend::early_returns && done == Backend::kTrue) return;
        
        Float_t safeX_Out= absX-unplaced.GetEndOuterRadius();
        Float_t safeY_Out= absY-unplaced.GetEndOuterRadius();
        Bool_t  mask_bbOut= (safeX_Out>0) || (safeY_Out>0) || (safeZ>0);
        
        safety_t=Max(safeX_Out, safeY_Out);
        safety_t=Max(safeZ, safety_t);
        MaskedAssign(mask_bbOut , safety_t, &safety);
        done|=mask_bbOut;
        if (Backend::early_returns && done == Backend::kTrue) return;
        
        Float_t rsq = point.x()*point.x()+point.y()*point.y();
        Float_t r = Sqrt(rsq);
        
        //Outer
        Float_t rhsqOut=unplaced.GetRmax2()+unplaced.GetTOut2()*point.z()*point.z();
        Float_t rhOut = Sqrt(rhsqOut);
        Float_t drOut = r - rhOut;
        
        Float_t safermax=0.;
        Bool_t mask_drOut=(drOut<0);
        MaskedAssign(mask_drOut, -kInfinity, &safermax);
        
        Bool_t mask_fStOut(Abs(unplaced.GetStOut())<kTolerance);
        MaskedAssign(!mask_drOut && mask_fStOut, Abs(drOut), &safermax);
        
        Float_t zHypeSqOut= Sqrt((r*r-unplaced.GetRmax2())*(unplaced.GetTOut2Inv()));
        Float_t mOut=(zHypeSqOut-absZ)/drOut;
        
        Float_t safe = mOut*drOut/Sqrt(1.+mOut*mOut);
        Bool_t doneOuter(mask_fStOut || mask_drOut);
        MaskedAssign(!doneOuter, safe, &safermax);
        Float_t max_safety= Max(safermax, safeZ);
        
        //Check for Inner Threatment
        if(unplaced.GetEndInnerRadius()!=0)
        {
            Float_t safermin=0.;
            Float_t rhsqIn = unplaced.GetRmin2()+unplaced.GetTIn()*point.z()*point.z();
            Float_t rhIn = Sqrt(rhsqIn);
            Float_t drIn = r-rhIn;
            
            Bool_t mask_drIn(drIn>0.);
            MaskedAssign(mask_drIn, -kInfinity, &safermin);
            
            Bool_t mask_fStIn(Abs(unplaced.GetStIn()<kTolerance));
            MaskedAssign(!mask_drIn && mask_fStIn , Abs(drIn), &safermin);
            
            Bool_t mask_fRmin(unplaced.GetRmin()<kTolerance);
            MaskedAssign(! mask_drIn && !mask_fStIn && mask_fRmin, drIn/Sqrt(1.+unplaced.GetTIn2()), &safermin);
            
            Bool_t mask_drMin=Abs(drIn)<kTolerance;
            MaskedAssign(!mask_drIn && !mask_fStIn && !mask_fRmin && mask_drMin, 0., &safermin);
            Bool_t doneInner(mask_drIn || mask_fStIn ||mask_fRmin || mask_drMin );

            Float_t zHypeSqIn= Sqrt( (r*r-unplaced.GetRmin2()) *(unplaced.GetTIn2Inv()) );
            
            Float_t mIn=-rhIn*unplaced.GetTIn2Inv()/absZ;
            safe = mIn*drIn/Sqrt(1.+mIn*mIn);
            MaskedAssign(!doneInner, safe, &safermin);
            max_safety= Max(max_safety, safermin);
        }
        
        MaskedAssign(!done, max_safety, &safety);
        
#endif

#ifdef ACCURATE_BC
        //Bounding-Cylinder implementation
        Float_t absX= Abs(point.x());
        Float_t absY= Abs(point.y());
        
        //Then calculate accurate value
        Float_t rsq = point.x()*point.x()+point.y()*point.y();
        Float_t r = Sqrt(rsq);
        
        
        //check if the point is inside the inner-bounding cylinder
        Float_t safeRhoIn=unplaced.GetRmin()-r;
        Bool_t  mask_bcIn= (safeRhoIn>0) && (safeZ>0);
        safety_t=Min(safeZ, safeRhoIn);
        Bool_t done(mask_bcIn);
        MaskedAssign(done, safety_t, &safety);
        if (Backend::early_returns && done == Backend::kTrue) return;
        
        //check if the point is outside the outer-bounding cylinder
        Float_t safeRhoOut=r-unplaced.GetEndOuterRadius();
        Bool_t  mask_bcOut= (safeRhoOut>0) || (safeZ>0);
        
        safety_t=Max(safeZ, safeRhoOut);
        MaskedAssign(!done && mask_bcOut, safety_t, &safety);
        done|=mask_bcOut;
        if (Backend::early_returns && done == Backend::kTrue) return;
    
        //Outer
        Float_t rhsqOut=unplaced.GetRmax2()+unplaced.GetTOut2()*point.z()*point.z();
        Float_t rhOut = Sqrt(rhsqOut);
        Float_t drOut = r - rhOut;
        
        Float_t safermax=0.;
        
        Bool_t mask_drOut=(drOut<0);
        MaskedAssign(mask_drOut, -kInfinity, &safermax);
        
        Bool_t mask_fStOut(Abs(unplaced.GetStOut())<kTolerance);
        MaskedAssign(!mask_drOut && mask_fStOut, Abs(drOut), &safermax);
        
        Float_t zHypeSqOut= Sqrt((r*r-unplaced.GetRmax2())*(unplaced.GetTOut2Inv()));
        Float_t mOut=(zHypeSqOut-absZ)/drOut;

        Float_t safe = mOut*drOut/Sqrt(1.+mOut*mOut);
        Bool_t doneOuter(mask_fStOut || mask_drOut);
        MaskedAssign(!doneOuter, safe, &safermax);
        Float_t max_safety= Max(safermax, safeZ);
        
        //Check for Inner Threatment
        if(unplaced.GetEndInnerRadius()!=0)
        {
            Float_t safermin=0.;
            Float_t rhsqIn = unplaced.GetRmin2()+unplaced.GetTIn()*point.z()*point.z();
            Float_t rhIn = Sqrt(rhsqIn);
            Float_t drIn = r-rhIn;
            
            Bool_t mask_drIn(drIn>0.);
            MaskedAssign(mask_drIn, -kInfinity, &safermin);
            Bool_t doneInner(mask_drIn);
            
            //Bool_t mask_fStIn(Abs(unplaced.GetStIn()<kTolerance));
            //MaskedAssign(!doneInner && mask_fStIn , Abs(drIn), &safermin);
            //doneInner|=mask_fStIn;
            
            //Bool_t mask_fRmin(unplaced.GetRmin()<kTolerance);
            //MaskedAssign(!doneInner && mask_fRmin, drIn/Sqrt(1.+unplaced.GetTIn2()), &safermin);
            //doneInner|=mask_fRmin;
            
            //Bool_t mask_drMin=Abs(drIn)<kTolerance;
            //MaskedAssign(!doneInner && mask_drMin , 0., &safermin);
            //doneInner|=mask_fRmin;
            
           
            Float_t zHypeSqIn= Sqrt( (r*r-unplaced.GetRmin2()) * (unplaced.GetTIn2Inv()) );
            
            Float_t mIn=-rhIn*unplaced.GetTIn2Inv()/absZ;
            safe = mIn*drIn/Sqrt(1.+mIn*mIn);
            MaskedAssign(!doneInner, safe, &safermin);
            max_safety= Max(max_safety, safermin);
        }
        
        MaskedAssign(!done, max_safety, &safety);
        
        
#endif
*/


    }
    
    template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::SafetyToOutKernel(
                                                                    UnplacedHype const &unplaced,
                                                                    Vector3D<typename Backend::precision_v> const &point,
                                                                    typename Backend::precision_v &safety){
        
        typedef typename Backend::precision_v Float_t;
        typedef typename Backend::bool_v Bool_t;
        safety=0.;
        
        Float_t absZ= Abs(point.z());
        Float_t safeZ= unplaced.GetDz()-absZ;
        Float_t rsq = point.x()*point.x()+point.y()*point.y();
        Float_t r = Sqrt(rsq);

#ifdef ROOTLIKE

        Float_t safermax;
        //OUTER
        Float_t rhsqOut=unplaced.GetRmax2()+unplaced.GetTOut2()*point.z()*point.z();
        Float_t rhOut = Sqrt(rhsqOut);
        Float_t drOut = r - rhOut;
    
        
        Bool_t mask_fStOut(unplaced.GetStOut()<kTolerance);
        MaskedAssign(mask_fStOut, Abs(drOut), &safermax);
    
        
        Bool_t mask_dr=Abs(drOut)<kTolerance;
        MaskedAssign(!mask_fStOut && mask_dr, 0., &safermax);
        Bool_t doneOuter(mask_fStOut || mask_dr);
        
        
        Float_t mOut= rhOut/(unplaced.GetTOut2()*absZ);
        Float_t saf = -mOut*drOut/Sqrt(1.+mOut*mOut);
        
        MaskedAssign(!doneOuter, saf, &safermax);
        
        safety=Min(safermax, safeZ);
        
        //Check for Inner Threatment
        if(unplaced.GetEndInnerRadius()!=0)
        {
            Float_t safermin=kInfinity;
            Float_t rhsqIn=unplaced.GetRmin2()+unplaced.GetTIn2()*point.z()*point.z();
            Float_t rhIn = Sqrt(rhsqIn);
            Float_t drIn = r - rhIn;
        
            Bool_t mask_fStIn(Abs(unplaced.GetStIn())<kTolerance);
            MaskedAssign(mask_fStIn, Abs(drIn), &safermin);
        
            Bool_t mask_fRmin(unplaced.GetRmin()<kTolerance);
            MaskedAssign(!mask_fStIn && mask_fRmin, drIn/Sqrt(1.+unplaced.GetTIn2()), &safermin);
            
            Bool_t mask_drMin=Abs(drIn)<kTolerance;
            MaskedAssign(!mask_fStIn && !mask_fRmin && mask_drMin, 0., &safermin);
        
            Bool_t doneInner(mask_fStIn || mask_fRmin || mask_drMin);
            Bool_t mask_drIn=(drIn<0);
            
            Float_t zHypeSqIn= Sqrt((r*r-unplaced.GetRmin2())/(unplaced.GetTIn2()));
            
            Float_t mIn;
            MaskedAssign(mask_drIn, -rhIn/(unplaced.GetTIn2()*absZ), &mIn);
            MaskedAssign(!mask_drIn, (zHypeSqIn-absZ)/drIn, &mIn);
            
            Float_t safe = mIn*drIn/Sqrt(1.+mIn*mIn);
    
            MaskedAssign(!doneInner, safe, &safermin);
            safety=Min(safety, safermin);
        }
#endif

        Float_t safeOuter;
        Bool_t mask_TOut(unplaced.GetTOut2()< DBL_MIN);
        
        MaskedAssign(mask_TOut, unplaced.GetRmax()-r, &safeOuter);
        
        // Corresponding position and normal on hyperbolic
        Float_t rh = Sqrt( unplaced.GetRmax2() + absZ*absZ*unplaced.GetTOut2() );
        
        Float_t dr = -rh;
        Float_t dz_mari = absZ*unplaced.GetTOut2();
        Float_t lenOuter = Sqrt(dr*dr + dz_mari*dz_mari);
        // Answer
        MaskedAssign(!mask_TOut, Abs((r-rh)*dr)/lenOuter, &safeOuter);
        
        if(unplaced.GetEndInnerRadius()!=0)
        {
            //INNER
            Float_t safeInner;
        
            Bool_t mask_TIn(unplaced.GetTIn()< DBL_MIN);
            MaskedAssign(mask_TIn, r-unplaced.GetRmin(), &safeInner);

            // First point
        
            Float_t z1 = absZ;
            Float_t r1 = Sqrt( unplaced.GetRmin2() + z1*z1*unplaced.GetTIn2() );
        
            // Second point
      
            Float_t z2 = (r*unplaced.GetTIn() + absZ)/(1 + unplaced.GetTIn2());
            Float_t r2 = Sqrt( unplaced.GetRmin2() + z2*z2*unplaced.GetTIn2() );
        
            // Line between them
      
            Float_t drInner = r2-r1;
            Float_t dzInner = z2-z1;
        
            Float_t lenInner = Sqrt(drInner*drInner + dzInner*dzInner);
            Bool_t mask_len(lenInner < DBL_MIN);
        
            Float_t drInner2 = r-r1;
            Float_t dzInner2 = absZ-z1;

            MaskedAssign(mask_len && !mask_TIn, Sqrt(drInner2*drInner2 + dzInner2*dzInner2), &safeInner);
            MaskedAssign(!mask_len && !mask_TIn, Abs((r-r1)*dzInner - (absZ-z1)*drInner)/lenInner, &safeInner);
        
            Float_t safe=Min(safeZ, safeOuter);
            safe=Min(safe, safeInner);
        
            safety=safe;
			//std::cout<<"PRINTING SAFETY FROM SafetyToOUt : "<<safety<<std::endl;

        }

		Bool_t inside(false);
		UnplacedContains<Backend>(unplaced,point,inside);
		MaskedAssign(!inside || safety<kSTolerance*100.,0.,&safety);

		
    }

    
}} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_HYPEIMPLEMENTATION_H_
