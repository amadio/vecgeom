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
  		static typename Backend::precision_v ApproxDistOutside(typename Backend::precision_v pr, typename Backend::precision_v pz, Precision r0, Precision tanPhi){ //, typename Backend::precision_v &ret){
        //std::cout<<"Entered ApproxDistFromOutside "<<std::endl;
        /*
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
        //ret = Sqrt(dr*dr + dz*dz);
		MaskedAssign(!(len < dbl_min),( Abs((pr-r1)*dz - (pz-z1)*dr)/len ),&ret);
        */




		typedef typename Backend::precision_v Float_t;
        Float_t ret(0.);
        Float_t dbl_min(2.2250738585072014e-308);
        //typedef typename Backend::bool_v Bool_t;
        Float_t r1 = Sqrt(r0*r0 + tanPhi*tanPhi*pz*pz);
        Float_t z1 = pz;

        Float_t r2 = pr;
        Float_t z2 = Sqrt((pr*pr - r0*r0)/(tanPhi*tanPhi));

        Float_t dz = z2-z1;
        Float_t dr = r2-r1;
        Float_t len = Sqrt(dr*dr + dz*dz);
        CondAssign((len < dbl_min) ,(pr-r1), (((pr-r1)*dz)/len ),&ret);
        return ret;


	}

template <class Backend>
  		VECGEOM_CUDA_HEADER_BOTH
  		VECGEOM_INLINE
  		static typename Backend::precision_v ApproxDistInside(typename Backend::precision_v pr, typename Backend::precision_v pz, Precision r0, Precision tan2Phi){ //, typename Backend::precision_v &ret){
			//Precision dbl_min = 2.2250738585072014e-308;
			typedef typename Backend::precision_v Float_t;
			typedef typename Backend::bool_v Bool_t;
			Float_t dbl_min(2.2250738585072014e-308);
			Bool_t done(false);
			Float_t ret(0.);
			Float_t tan2Phi_v(tan2Phi);
			MaskedAssign((tan2Phi_v < dbl_min),r0 - pr,&ret);
			done |= (tan2Phi_v < dbl_min);
			if(IsFull(done)) return ret;

			Float_t rh = Sqrt(r0*r0 + pz*pz*tan2Phi_v );
		    Float_t dr = -rh;
		    Float_t dz = pz*tan2Phi_v;
   			Float_t len = Sqrt(dr*dr + dz*dz);
			//ret = Abs((pr-rh)*dr)/len;
			MaskedAssign(!done ,Abs((pr-rh)*dr)/len , &ret);
			return ret;

	}

		template <class Backend>
  		VECGEOM_CUDA_HEADER_BOTH
  		VECGEOM_INLINE
  		static void InterSectionExist(typename Backend::precision_v a, typename Backend::precision_v b, typename Backend::precision_v c, typename Backend::bool_v &exist){

			exist = (b*b - 4*a*c > 0.);
		}



		template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static void IsPointOnHyperbolicSurface(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
                                   typename Backend::bool_v &surface, typename Backend::bool_v &inner);



		template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static void IsPointOnZSurfacePlane(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
                                   typename Backend::bool_v &surface, typename Backend::bool_v &zSurf);


		template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static void IsInside(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
                                   typename Backend::bool_v &inside, typename Backend::bool_v &inner);



	template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static void GetDirection(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
									Vector3D<typename Backend::precision_v> const &direction,
                                   typename Backend::bool_v &in){


		typedef typename Backend::precision_v Float_t;

		Float_t pz(point.z()), vz(direction.z());
		MaskedAssign(pz<0.,-pz,&pz);
		MaskedAssign(pz<0.,-vz,&vz);
		Precision tanOuterStereo2 = unplaced.GetTOut2();
		in = ( (point.x()*direction.x() + point.y()*direction.y() - pz*tanOuterStereo2*vz) < 0);

		}

	template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static void GetDirectionOuter(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
									Vector3D<typename Backend::precision_v> const &direction,
                                   typename Backend::bool_v &in){


		typedef typename Backend::precision_v Float_t;

		Float_t pz=point.z();
		Float_t vz=direction.z();

		MaskedAssign(pz<0.,-vz,&vz);
		MaskedAssign(pz<0.,-pz,&pz);


		Precision tanOuterStereo2 = unplaced.GetTOut2();
		Float_t pxy = (point.x()*direction.x() + point.y()*direction.y());
		Float_t zc = pz*tanOuterStereo2*vz;
		Float_t res = pxy - zc;
		in = ( (point.x()*direction.x() + point.y()*direction.y() - pz*tanOuterStereo2*vz) < 0);
		}


		template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static void GetDirectionInner(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
									Vector3D<typename Backend::precision_v> const &direction,
                                   typename Backend::bool_v &out){

		typedef typename Backend::precision_v Float_t;

		Float_t pz(point.z()), vz(direction.z());
		MaskedAssign(pz<0.,-vz,&vz);
		MaskedAssign(pz<0.,-pz,&pz);

		Precision tanInnerStereo2 = unplaced.GetTIn2();
		out = ( (point.x()*direction.x() + point.y()*direction.y() - pz*tanInnerStereo2*vz) > 0);
		}


		template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static void IsPointMovingOutside(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
									Vector3D<typename Backend::precision_v> const &direction,
                                   typename Backend::bool_v &out){

		typedef typename Backend::precision_v Float_t;

		Float_t pz = point.z();
		Float_t vz = direction.z();


		MaskedAssign(vz<0.,-pz,&pz);
		MaskedAssign(vz<0.,-vz,&vz);

		Precision tanOuterStereo2 = unplaced.GetTOut2();
		Vector3D<Float_t> normHere(point.x(),point.y(),-point.z()*tanOuterStereo2);

		out = (normHere.Dot(direction) > 0.);

		}



		template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static void GetDistToOutInZ(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
									Vector3D<typename Backend::precision_v> const &direction,
                                   typename Backend::precision_v &distance){

		typedef typename Backend::precision_v Float_t;

		//Float_t fHalfTol(kSTolerance*10.*0.5);
		Float_t pz = point.z();
		Float_t vz = direction.z();


		MaskedAssign(vz<0.,-pz,&pz);
		MaskedAssign(vz<0.,-vz,&vz);



		}

		template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static void IsPointMovingInsideFromZPlane(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
									Vector3D<typename Backend::precision_v> const &direction,
                                   //typename Backend::precision_v &distance){
									typename Backend::bool_v &inZ){

		//std::cout<<"Entered GetDirectionInZ"<<std::endl;
		//typedef typename Backend::bool_v Bool_t;
		typedef typename Backend::precision_v Float_t;

		//Float_t fHalfTol(kSTolerance*10.*0.5);
		Float_t pz(point.z()), vz(direction.z());

		MaskedAssign(pz<0.,-vz,&vz);
		MaskedAssign(pz<0.,-pz,&pz);

		Float_t sigz = pz-unplaced.GetDz();
		//MaskedAssign((sigz > -fHalfTol) && (vz < 0.) && (sigz < fHalfTol),0. , &distance);

		//MaskedAssign((sigz > -kHalfTolerance) && (vz < 0.) && (sigz < kHalfTolerance),0. , &distance);
		inZ = ((sigz > -kHalfTolerance) && (vz < 0.) && (sigz < kHalfTolerance));

		}


		template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static void IsPointMovingOutsideFromZPlane(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
									Vector3D<typename Backend::precision_v> const &direction,
                                   //typename Backend::precision_v &distance){
								typename Backend::bool_v &out){

		//std::cout<<"Entered GetDirectionInZ"<<std::endl;
		//typedef typename Backend::bool_v Bool_t;
		typedef typename Backend::precision_v Float_t;

		//Float_t fHalfTol(kSTolerance*10.*0.5);
		Float_t pz(point.z()), vz(direction.z());


		MaskedAssign(vz<0.,-pz,&pz);
		MaskedAssign(vz<0.,-vz,&vz);

		Float_t sigz = pz-unplaced.GetDz();

		out = ((sigz > -kHalfTolerance) && (vz > 0.) && (sigz < kHalfTolerance));

		}



    }; // End struct HypeImplementation

template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::IsInside(
                                                                 UnplacedHype const &unplaced,
                                                                 Vector3D<typename Backend::precision_v> const &point,
                                                                 typename Backend::bool_v &inside, typename Backend::bool_v &inner) {

        typedef typename Backend::bool_v Bool_t;
		typedef typename Backend::precision_v Float_t;
        Bool_t outside;
        GenericKernelForContainsAndInside<Backend,true>(unplaced,
                                                          point, inside, outside);

		Float_t rho2 = point.x()*point.x() + point.y()*point.y();
		Float_t radI2(0.),radO2;
		RadiusHypeSq<Backend,true>(unplaced,point.z(),radI2);
		RadiusHypeSq<Backend,false>(unplaced,point.z(),radO2);
		//inner = Abs(Sqrt(radI2) - Sqrt(rho2)) < kSTolerance*10.;
		inner = Abs(Sqrt(radI2) - Sqrt(rho2)) < kTolerance;
    }




template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::IsPointOnHyperbolicSurface(
                                                                 UnplacedHype const &unplaced,
                                                                 Vector3D<typename Backend::precision_v> const &point,
                                                                 typename Backend::bool_v &surface, typename Backend::bool_v &inner) {

        typedef typename Backend::bool_v Bool_t;
		typedef typename Backend::precision_v Float_t;
        Bool_t inside;
        Bool_t outside;
        GenericKernelForContainsAndInside<Backend,true>(unplaced,
                                                          point, inside, outside);
        surface=!outside && !inside;
		Float_t rho2 = point.x()*point.x() + point.y()*point.y();
		Float_t radI2(0.),radO2;
		RadiusHypeSq<Backend,true>(unplaced,point.z(),radI2);
		RadiusHypeSq<Backend,false>(unplaced,point.z(),radO2);
		inner = Abs(Sqrt(radI2) - Sqrt(rho2)) < kSTolerance*10.;
    }

/*
template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::IsSurfacePointOuter(
                                                                 UnplacedHype const &unplaced,
                                                                 Vector3D<typename Backend::precision_v> const &point,
                                                                 typename Backend::bool_v &surface, typename Backend::bool_v &inner) {

        typedef typename Backend::bool_v Bool_t;
		typedef typename Backend::precision_v Float_t;
        Bool_t inside;
        Bool_t outside;
        GenericKernelForContainsAndInside<Backend,true>(unplaced,
                                                          point, inside, outside);
        surface=!outside && !inside;
		Float_t rho2 = point.x()*point.x() + point.y()*point.y();
		Float_t radI2(0.),radO2;
		RadiusHypeSq<Backend,true>(unplaced,point.z(),radI2);
		RadiusHypeSq<Backend,false>(unplaced,point.z(),radO2);
		inner = Abs(Sqrt(radI2) - Sqrt(rho2)) < kSTolerance*10.;
    }

*/

template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::IsPointOnZSurfacePlane(
                                                                 UnplacedHype const &unplaced,
                                                                 Vector3D<typename Backend::precision_v> const &point,
                                                                 typename Backend::bool_v &surface, typename Backend::bool_v &zSurf) {

        typedef typename Backend::bool_v Bool_t;
		typedef typename Backend::precision_v Float_t;
        Bool_t inside;
        Bool_t outside;
        GenericKernelForContainsAndInside<Backend,true>(unplaced,
                                                          point, inside, outside);
        surface=!outside && !inside;
		Float_t rho2 = point.x()*point.x() + point.y()*point.y();
		Float_t radI2(0.),radO2;
		RadiusHypeSq<Backend,true>(unplaced,point.z(),radI2);
		RadiusHypeSq<Backend,false>(unplaced,point.z(),radO2);
		//Bool_t inner = Abs(Sqrt(radI2) - Sqrt(rho2)) < kSTolerance*10.;
		//Bool_t outer = Abs(Sqrt(radO2) - Sqrt(rho2)) < kSTolerance*10.;
		Bool_t inner = Abs(Sqrt(radI2) - Sqrt(rho2)) < kTolerance;
		Bool_t outer = Abs(Sqrt(radO2) - Sqrt(rho2)) < kTolerance;
		zSurf = surface && !inner && !outer;

    }


/*
template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::IsSurfacePointZ(
                                                                 UnplacedHype const &unplaced,
                                                                 Vector3D<typename Backend::precision_v> const &point,
                                                                 typename Backend::bool_v &surface, typename Backend::bool_v &zSurf) {

        typedef typename Backend::bool_v Bool_t;
		typedef typename Backend::precision_v Float_t;
        Bool_t inside;
        Bool_t outside;
        GenericKernelForContainsAndInside<Backend,true>(unplaced,
                                                          point, inside, outside);
        surface=!outside && !inside;
		Float_t rho2 = point.x()*point.x() + point.y()*point.y();
		Float_t radI2(0.),radO2;
		RadiusHypeSq<Backend,true>(unplaced,point.z(),radI2);
		RadiusHypeSq<Backend,false>(unplaced,point.z(),radO2);
		//Bool_t inner = Abs(Sqrt(radI2) - Sqrt(rho2)) < kSTolerance*10.;
		//Bool_t outer = Abs(Sqrt(radO2) - Sqrt(rho2)) < kSTolerance*10.;
		Bool_t inner = Abs(Sqrt(radI2) - Sqrt(rho2)) < kTolerance;
		Bool_t outer = Abs(Sqrt(radO2) - Sqrt(rho2)) < kTolerance;
		zSurf = surface && !inner && !outer;

    }
*/


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
    /*
	Precision fDz = unplaced.GetDz();

	//along Z direction
	//completelyoutside = Abs(point.z()) > fDz + kSTolerance*10.0;

	completelyoutside = Abs(point.z()) > fDz + kTolerance;
	if(ForInside)
	{
	 // completelyinside = Abs(point.z()) < fDz - kSTolerance*10.0;
	completelyinside = Abs(point.z()) < fDz - kTolerance;
	}

	Float_t r = Sqrt(point.x()*point.x()+point.y()*point.y());
	Float_t rOuter=Sqrt(unplaced.GetRmax2()+unplaced.GetTOut2()*point.z()*point.z());
    	Float_t rInner=Sqrt(unplaced.GetRmin2()+unplaced.GetTIn2()*point.z()*point.z());
	//completelyoutside |= (r > rOuter + kSTolerance*10.0) || (r < rInner - kSTolerance*10.0);
	completelyoutside |= (r > rOuter + kTolerance) || (r < rInner - kTolerance);
	if(ForInside)
	{
	 // completelyinside &= (r < rOuter - kSTolerance*10.0) && (r > rInner + kSTolerance*10.0);
	 completelyinside &= (r < rOuter - kTolerance) && (r > rInner + kTolerance);
	}

	return ;
	*/

	Precision fDz = unplaced.GetDz();
	Precision zToleranceLevel = unplaced.GetZToleranceLevel();
	Precision innerRadToleranceLevel = unplaced.GetInnerRadToleranceLevel();
	Precision outerRadToleranceLevel = unplaced.GetOuterRadToleranceLevel();

	completelyoutside = Abs(point.z()) > (fDz + zToleranceLevel);
	Float_t r2 = (point.x()*point.x()+point.y()*point.y());
    Float_t oRad2 = (unplaced.GetRmax2()+unplaced.GetTOut2()*point.z()*point.z());

	completelyoutside |= (r2 > oRad2+outerRadToleranceLevel);
	if(ForInside)
	{
	completelyinside = (Abs(point.z()) < (fDz - zToleranceLevel)) && (r2 < oRad2-outerRadToleranceLevel);
	}

	if(unplaced.InnerSurfaceExists())
	{
	Float_t iRad2 = (unplaced.GetRmin2()+unplaced.GetTIn2()*point.z()*point.z());
	completelyoutside |= (r2 < (iRad2-innerRadToleranceLevel));
	completelyinside &= (r2 > (iRad2+innerRadToleranceLevel));
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

		//std::cout<<"---- Entered DistanceToInkernel -----\n";
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
	//Precision dZ=unplaced.GetDz();
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
        //std::cout<<"-----Reached Here -8 -----------------\n";
        //check if the point is outside the bounding cylinder and is distancing in XY
        Bool_t isDistancingInXY=( (rho2>unplaced.GetEndOuterRadius2()) && (point_dot_direction_x>0 && point_dot_direction_y>0) );
        done|=isDistancingInXY;
        if (Backend::early_returns && done == Backend::kTrue) return;

		//std::cout<<"-----Reached Here -7 -----------------\n";
        //check if x coordinate is > EndOuterRadius and the point is distancing in X
        Bool_t isDistancingInX=( (Abs(point.x())>unplaced.GetEndOuterRadius()) && (point_dot_direction_x>0) );
        done|=isDistancingInX;
        if (Backend::early_returns && done == Backend::kTrue) return;

		//std::cout<<"-----Reached Here -6 -----------------\n";
        //check if y coordinate is > EndOuterRadiusthe point is distancing in Y
        Bool_t isDistancingInY=( (Abs(point.y())>unplaced.GetEndOuterRadius()) && (point_dot_direction_y>0) );
        done|=isDistancingInY;
        if (Backend::early_returns && done == Backend::kTrue) return;

        //is hitting from dz or -dz planes
        Float_t distZ = (absZ-unplaced.GetDz())/absDirZ;
        Float_t xHit = point.x()+distZ*direction.x();
        Float_t yHit = point.y()+distZ*direction.y();
        Float_t rhoHit2=xHit*xHit+yHit*yHit;
        //std::cout<<"-----Reached Here -5 -----------------\n";
        Bool_t isCrossingAtDz= (absZ>unplaced.GetDz()) && (!checkZ) && (rhoHit2 <=unplaced.GetEndOuterRadius2() && rhoHit2>=unplaced.GetEndInnerRadius2());

        MaskedAssign(isCrossingAtDz, distZ, &distance);
        done|=isCrossingAtDz;
        //if (Backend::early_returns && done == Backend::kTrue) return;


        //is hitting from the hyperboloid surface (OUTER or INNER)
        Float_t dirRho2 = direction.x()*direction.x()+direction.y()*direction.y();
        Float_t point_dot_direction_z = point.z()*direction.z();
        Float_t pointz2=point.z()*point.z();
        Float_t dirz2=direction.z()*direction.z();
    //std::cout<<"-----Reached Here -4 -----------------\n";
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

    //std::cout<<std::setprecision(15);

	Float_t distOut(0.);
	MaskedAssign(bOut>0. ,(cOut/(bOut+deltaOut)) , &distOut);
	MaskedAssign(bOut<0. ,(aOutinv*(bOut -deltaOut)) , &distOut);
        Float_t zHitOut1 = point.z()+distOut*direction.z();
        Bool_t isHittingHyperboloidSurfaceOut1 = ( (distOut> 1E20) || (Abs(zHitOut1)<=unplaced.GetDz()) ); //why: dist > 1E20?


	Float_t solution_Outer=kInfinity;
	MaskedAssign(!deltaOutNeg &&isHittingHyperboloidSurfaceOut1 && distOut>0, distOut, &solution_Outer);
        //std::cout<<"-----Reached Here -3----------------\n";
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
//std::cout<<"-----Reached Here -2 -----------------\n";
	Float_t zHitIn1 = point.z()+distIn*direction.z();
        Bool_t isHittingHyperboloidSurfaceIn1 = ( (distIn> 1E20) || (Abs(zHitIn1)<=unplaced.GetDz()) ); //why: dist > 1E20?

        Float_t solution_Inner=kInfinity;
	MaskedAssign(!deltaInNeg && isHittingHyperboloidSurfaceIn1 && distIn>0, distIn, &solution_Inner);

        Float_t solution=Min(solution_Inner, solution_Outer);

        done|=(deltaInNeg && deltaOutNeg);
        MaskedAssign(!done, solution, &distance );


	Bool_t isSurfacePoint(false),inner(false),zSurf(false);
	Float_t pDotV3d = point.Dot(direction);
	//std::cout<<"-----Reached Here -1-----------------\n";

	Bool_t in(false),out(false);
	GetDirectionOuter<Backend>(unplaced,point,direction,in);
	GetDirectionInner<Backend>(unplaced,point,direction,out);

	Float_t zSurfDist(kInfinity);

	//std::cout<<"-----Reached Here 0 -----------------\n";

	//IsSurfacePointOuter<Backend>(unplaced,point,isSurfacePoint,inner);
	IsPointOnHyperbolicSurface<Backend>(unplaced,point,isSurfacePoint,inner);
	//IsSurfacePointZ<Backend>(unplaced,point,isSurfacePoint,zSurf);
	IsPointOnZSurfacePlane<Backend>(unplaced,point,isSurfacePoint,zSurf);
	//GetDirectionInZ<Backend>(unplaced,point,direction,zSurfDist);
	Bool_t inZ(false);
	//GetDirectionInZ<Backend>(unplaced,point,direction,inZ);
	IsPointMovingInsideFromZPlane<Backend>(unplaced,point,direction,inZ);

	//std::cout<<"-----Reached Here-----------------\n";
	MaskedAssign(!done && !zSurf && isSurfacePoint && !inner  && in ,0.,&distance );
	done |= (!zSurf && isSurfacePoint && !inner  && in);
	MaskedAssign(!done && !zSurf && isSurfacePoint && inner  && out ,0.,&distance );
	done |= (!zSurf && isSurfacePoint && inner  && out);

	//MaskedAssign(zSurf,zSurfDist,&distance);
	//std::cout<<"ZSurface Point : "<<zSurf<<"  :: Going in : "<<inZ<<std::endl;
	MaskedAssign(zSurf && inZ,0.,&distance);
		//std::cout<<"-----Reached Here End-----------------\n";



	Bool_t isInside(false);
	IsInside<Backend>(unplaced,point,isInside,inner);
	MaskedAssign(isInside, kSTolerance , &distance);
	done |= isInside;


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

	Float_t distInner(kInfinity);
	Float_t distOuter(kInfinity);


	Bool_t done(false);


	//Handling Inner Surface
	if(unplaced.InnerSurfaceExists())
	{
	a = direction.x() * direction.x() + direction.y() * direction.y() - tanInnerStereo2*direction.z()*direction.z();
	b = 2*direction.x()*point.x() + 2*direction.y()*point.y() - 2*tanInnerStereo2*direction.z()*point.z();
	c= point.x()*point.x() + point.y()*point.y() - tanInnerStereo2*point.z()*point.z() - fRmin2;
	exist = (b*b - 4*a*c > 0.);

	MaskedAssign(/*!done && */exist && b>0. ,( (-b - Sqrt(b*b - 4*a*c))/(2*a) ),&distInner);
	MaskedAssign(/*!done && */exist && b<=0.,((2*c)/(-b + Sqrt(b*b - 4*a*c)) ),&distInner);
	MaskedAssign(distInner < 0. ,kInfinity, &distInner);
	}

	//Handling Outer surface
	exist = fal;
	a = direction.x() * direction.x() + direction.y() * direction.y() - tanOuterStereo2*direction.z()*direction.z();
	b = 2*direction.x()*point.x() + 2*direction.y()*point.y() - 2*tanOuterStereo2*direction.z()*point.z();
	c= point.x()*point.x() + point.y()*point.y() - tanOuterStereo2*point.z()*point.z() - fRmax2;
	exist = (b*b - 4*a*c > 0.);
	MaskedAssign(/*!done && */exist && b<0.,( (-b + Sqrt(b*b - 4*a*c))/(2*a) ),&distOuter);
	MaskedAssign(/*!done && */exist && b>=0.,((2*c)/(-b - Sqrt(b*b - 4*a*c)) ),&distOuter);
	MaskedAssign(distOuter < 0. ,kInfinity, &distOuter);

    //Handling Z surface
	Float_t distZ=kInfinity;
    Float_t dirZinv=1/direction.z();
    Bool_t dir_mask= direction.z()<0;
    MaskedAssign(/*!done && */ dir_mask, -(unplaced.GetDz() + point.z())*dirZinv, &distZ);

    MaskedAssign(/*!done && */ !dir_mask, (unplaced.GetDz() - point.z())*dirZinv, &distZ);
	MaskedAssign(distZ < 0. , kInfinity, &distZ);

	distance = Min(distInner,distOuter);
	distance = Min(distance,distZ);

	//Handling Surface Point
	Bool_t isSurfacePoint(false),inner(false);
	Bool_t in(false),out(false),zSurf(false);
	//IsSurfacePointOuter<Backend>(unplaced,point,isSurfacePoint,inner);
	IsPointOnHyperbolicSurface<Backend>(unplaced,point,isSurfacePoint,inner);
	//GetDirectionDistToOutOuter<Backend>(unplaced,point,direction,out);
	IsPointMovingOutside<Backend>(unplaced,point,direction,out);
	//IsSurfacePointZ<Backend>(unplaced,point,isSurfacePoint,zSurf);
	IsPointOnZSurfacePlane<Backend>(unplaced,point,isSurfacePoint,zSurf);
	MaskedAssign(!zSurf && isSurfacePoint && !inner && out , 0. , &distance);

	//std::cout<<"Point on ZSurface : "<<zSurf<<std::endl;
	Float_t DistZ(kInfinity);
	Bool_t outZ(false);
	//GetDirectionOutZ<Backend>(unplaced,point,direction,outZ);//DistZ);
	IsPointMovingOutsideFromZPlane<Backend>(unplaced,point,direction,outZ);//DistZ);

	//std::cout<<"DistZ : "<<DistZ<<std::endl;
	MaskedAssign(zSurf && outZ,0.,&distance);


	Bool_t isPointInside(false);
	ContainsKernel<Backend>(unplaced,point,isPointInside);
	MaskedAssign(!isPointInside && !done, kSTolerance , &distance);
	}



    template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::SafetyToInKernel(
                                                                   UnplacedHype const &unplaced,
                                                                   Vector3D<typename Backend::precision_v> const &point,
                                                                   typename Backend::precision_v &safety) {

        typedef typename Backend::precision_v Float_t;
        typedef typename Backend::bool_v Bool_t;


	//
	Float_t absZ = Abs(point.z());
	Float_t r2 = point.x()*point.x() + point.y()*point.y();
	Float_t r =  Sqrt(r2);



	Precision endOuterRadius = unplaced.GetEndOuterRadius();
	Precision endInnerRadius = unplaced.GetEndInnerRadius();
	Precision innerRadius = unplaced.GetRmin();
	Precision outerRadius = unplaced.GetRmax();
	Precision tanInnerStereo2 = unplaced.GetTIn2();
	Precision tanOuterStereo2 = unplaced.GetTOut2();
	Precision tanOuterStereo = unplaced.GetTOut();
	Precision tanInnerStereo = unplaced.GetTIn();
	//Float_t dr = endInnerRadius - r;
	//Float_t answer = Sqrt(dr*dr + sigz*sigz);

	//Bool_t innerSurfaceExist(unplaced.InnerSurfaceExists());

    //Bool_t inside(false);
	Bool_t done(false);

	//UnplacedContains<Backend>(unplaced,point,inside);
	//MaskedAssign(!done && inside,0.,&safety);
	//done |= inside;


    //-------------------------------------------------------------

    //New Algo
    //Considering Solid Hyperboloid

    //Debuggng
    //std::cout<<"EndInnerRadius : "<<endInnerRadius<<"  :: EndOuterRadius : "<<endOuterRadius<<std::endl;

    //std::cout<<"R : "<<r<<" :: Z : "<<point.z()<<std::endl;
    //

    safety = -kHalfTolerance;

    Bool_t cond(false);
    Float_t sigz = absZ - unplaced.GetDz();
    cond = (sigz > kHalfTolerance) && (r < endOuterRadius) && (r > endInnerRadius);
    MaskedAssign(cond, sigz , &safety);
    done |= cond;
    if(IsFull(done)) return;

    cond = (sigz > kHalfTolerance) && (r >= endOuterRadius) ;
    MaskedAssign(!done && cond,Sqrt( (r-endOuterRadius)*(r-endOuterRadius) + (sigz)*(sigz) ) , &safety );
    done |= cond;
    if(IsFull(done)) return;


    cond = (sigz > kHalfTolerance) && (r <= endInnerRadius);
    MaskedAssign(!done && cond,Sqrt( (r-endInnerRadius)*(r-endInnerRadius) + (sigz)*(sigz) ) , &safety );
    done |= cond;
    if(IsFull(done)) return;

    cond = (r > Sqrt(outerRadius*outerRadius + tanOuterStereo2*absZ*absZ )) && (absZ > 0.) && (absZ < unplaced.GetDz());
    MaskedAssign(!done && cond , ApproxDistOutside<Backend>( r,absZ,outerRadius,tanOuterStereo), &safety);
    done |= cond;

    MaskedAssign(!done && (r < Sqrt(innerRadius*innerRadius + tanInnerStereo2*absZ*absZ )) && (absZ > 0.) && (absZ < unplaced.GetDz()) , ApproxDistInside<Backend>( r,absZ,innerRadius,tanInnerStereo), &safety);

    //std::cout<<"EndInnerRadius : "<<endInnerRadius<<"  :: EndOuterRadius : "<<endOuterRadius<<std::endl;

    //std::cout<<"R : "<<r<<" :: Z : "<<point.z()<<std::endl;
    //std::cout<<":: "<<Sqrt(outerRadius*outerRadius + tanOuterStereo2*absZ*absZ )<<std::endl;
    //-------------------------------------------------------------


    /*
    Precision halfTol = 0.5*kSTolerance;
    Float_t sigz = absZ - unplaced.GetDz();
    safety=0.;
    Bool_t cond1 = (r < endOuterRadius) && (sigz > -halfTol);
    Bool_t cond(false);
    if(unplaced.InnerSurfaceExists())
    {
      cond =  cond1  &&  (r > endInnerRadius) && !(sigz < halfTol);
	MaskedAssign(!done && cond,sigz,&safety);
	done |= cond;
	if(IsFull(done)) return;

    cond = cond1  && (sigz > dr*tanInnerStereo2) && !(answer < halfTol);
    MaskedAssign(!done && cond,answer,&safety);
    done |= cond;
    if(IsFull(done)) return;
    }
    else
    {

    cond = cond1  && !(sigz < halfTol);
    MaskedAssign(!done && cond,sigz,&safety);
    done |= cond;
    if(IsFull(done)) return;
    }

    dr = r - endOuterRadius;
    answer = Sqrt(dr*dr + sigz*sigz);
    cond = !(r < endOuterRadius) && (sigz > -dr*tanOuterStereo2) && !(answer < halfTol);
    MaskedAssign(!done && cond,answer,&safety);
    done |= cond;
    if(IsFull(done)) return;

    answer = ApproxDistOutside<Backend>( r,absZ,outerRadius,tanOuterStereo);
    answer = Abs(answer);

    MaskedAssign( !done && !(answer < halfTol),answer,&safety);
    done |= !(answer < halfTol);

    if(unplaced.InnerSurfaceExists())
        {
            Float_t radi2;
            RadiusHypeSq<Backend,false>(unplaced,absZ,radi2);
            MaskedAssign( !done && (r2 < (radi2+unplaced.GetInnerRadToleranceLevel())),ApproxDistInside<Backend>( r,absZ,innerRadius,tanInnerStereo),&answer);
            MaskedAssign(!done && !(answer < halfTol), answer, &safety);
            done |= !(answer < halfTol) ; //&& (r2 < (radi2+unplaced.GetInnerRadToleranceLevel()));
            if(IsFull(done)) return;
        }

    */

    /*
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

	MaskedAssign(!done && (r < endOuterRadius) && (sigz > -halfTol) && !innerSurfaceExist &&  (sigz < halfTol),0.,&safety);
	done |= (r < endOuterRadius) && (sigz > -halfTol) && !innerSurfaceExist &&  (sigz < halfTol);

	MaskedAssign(!done && (r < endOuterRadius) && (sigz > -halfTol) && !innerSurfaceExist && !(sigz < halfTol),sigz,&safety);
	done |= (r < endOuterRadius) && (sigz > -halfTol) && !innerSurfaceExist && !(sigz < halfTol);

	dr = r - endOuterRadius;
	answer = Sqrt(dr*dr + sigz*sigz);
	MaskedAssign(!done && !(r < endOuterRadius) && (sigz > -dr*tanOuterStereo2) && (answer < halfTol),0.,&safety);
	done |= !(r < endOuterRadius) && (sigz > -dr*tanOuterStereo2) && (answer < halfTol);

	MaskedAssign(!done && !(r < endOuterRadius) && (sigz > -dr*tanOuterStereo2) && !(answer < halfTol),answer,&safety);
	done |= !(r < endOuterRadius) && (sigz > -dr*tanOuterStereo2) && !(answer < halfTol);

	Float_t radSq;
	RadiusHypeSq<Backend,true>(unplaced,point.z(),radSq);
	ApproxDistInside<Backend>(r,absZ,innerRadius,tanInnerStereo2,answer);
	MaskedAssign(!done && innerSurfaceExist &&  (r2 < radSq + kSTolerance*endInnerRadius) && (answer < halfTol), 0. ,&safety);
	done |= innerSurfaceExist &&  (r2 < radSq + kSTolerance*endInnerRadius) && (answer < halfTol);

	MaskedAssign(!done && innerSurfaceExist &&  (r2 < radSq + kSTolerance*endInnerRadius) && !(answer < halfTol), answer ,&safety);
	done |= innerSurfaceExist &&  (r2 < radSq + kSTolerance*endInnerRadius) && !(answer < halfTol);


	ApproxDistOutside<Backend>( r,absZ,outerRadius,tanOuterStereo,answer );
	MaskedAssign( !done && (answer < halfTol) ,0.,&safety);
	done |= (answer < halfTol);

	MaskedAssign( !done && !(answer < halfTol),answer,&safety);
	done |= !(answer < halfTol);
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
            Float_t safeInner(kInfinity);

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
		//MaskedAssign(!inside || safety<kSTolerance*100.,0.,&safety);
		MaskedAssign(!inside || safety<kTolerance*10.,0.,&safety);


    }


}} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_HYPEIMPLEMENTATION_H_
