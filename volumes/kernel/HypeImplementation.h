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
		static typename Backend::precision_v RadiusHypeSq(UnplacedHype const &unplaced, typename Backend::precision_v z);//, typename Backend::precision_v &radsq);


		template <class Backend>
  		VECGEOM_CUDA_HEADER_BOTH
  		VECGEOM_INLINE
  		static typename Backend::precision_v ApproxDistOutside(typename Backend::precision_v pr, typename Backend::precision_v pz, Precision r0, Precision tanPhi);

        template <class Backend>
  		VECGEOM_CUDA_HEADER_BOTH
  		VECGEOM_INLINE
  		static typename Backend::precision_v ApproxDistInside(typename Backend::precision_v pr, typename Backend::precision_v pz, Precision r0, Precision tan2Phi);

        template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static typename Backend::bool_v IsPointOnSurfaceAndMovingOutside(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,Vector3D<typename Backend::precision_v> const &direction);//,


        template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static typename Backend::bool_v IsPointOnInnerSurfaceAndMovingOutside(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,Vector3D<typename Backend::precision_v> const &direction);//,


        template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static typename Backend::bool_v IsPointOnSurfaceAndMovingInside(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,Vector3D<typename Backend::precision_v> const &direction);//,



        template <class Backend,bool ForDistToIn>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static typename Backend::bool_v GetPointOfIntersectionWithZPlane(UnplacedHype const &unplaced,
                                                                     Vector3D<typename Backend::precision_v> const &point,
                                                                     Vector3D<typename Backend::precision_v> const &direction,
                                                                    typename Backend::precision_v &zDist);


        template <class Backend,bool ForDistToIn>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static typename Backend::bool_v GetPointOfIntersectionWithOuterHyperbolicSurface(UnplacedHype const &unplaced,
                                                                     Vector3D<typename Backend::precision_v> const &point,
                                                                     Vector3D<typename Backend::precision_v> const &direction,
                                                                    typename Backend::precision_v &dist);

        template <class Backend,bool ForDistToIn>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static typename Backend::bool_v GetPointOfIntersectionWithInnerHyperbolicSurface(UnplacedHype const &unplaced,
                                                                     Vector3D<typename Backend::precision_v> const &point,
                                                                     Vector3D<typename Backend::precision_v> const &direction,
                                                                    typename Backend::precision_v &dist);


template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static typename Backend::bool_v IsPointMovingInsideOuterSurface(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
									Vector3D<typename Backend::precision_v> const &direction);

template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static typename Backend::bool_v IsPointMovingInsideInnerSurface(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
									Vector3D<typename Backend::precision_v> const &direction);


template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static typename Backend::bool_v IsPointMovingOutsideOuterSurface(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
									Vector3D<typename Backend::precision_v> const &direction);

template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static typename Backend::bool_v IsPointMovingOutsideInnerSurface(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
									Vector3D<typename Backend::precision_v> const &direction);


    template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static typename Backend::bool_v IsPointMovingInsideFromZPlane(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
									Vector3D<typename Backend::precision_v> const &direction);


        template <class Backend>
        VECGEOM_CUDA_HEADER_BOTH
        VECGEOM_INLINE
        static typename Backend::bool_v IsPointMovingOutsideFromZPlane(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
									Vector3D<typename Backend::precision_v> const &direction);



    }; // End struct HypeImplementation


template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::precision_v HypeImplementation<transCodeT, rotCodeT>::ApproxDistOutside(typename Backend::precision_v pr, typename Backend::precision_v pz, Precision r0, Precision tanPhi){ //, typename Backend::precision_v &ret){

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


template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::precision_v HypeImplementation<transCodeT, rotCodeT>::ApproxDistInside(typename Backend::precision_v pr, typename Backend::precision_v pz, Precision r0, Precision tan2Phi){ //, typename Backend::precision_v &ret){

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

			MaskedAssign(!done ,Abs((pr-rh)*dr)/len , &ret);
			return ret;

	}

template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::bool_v HypeImplementation<transCodeT, rotCodeT>::IsPointMovingInsideOuterSurface(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
									Vector3D<typename Backend::precision_v> const &direction)//,
                                   //typename Backend::bool_v &in)
                                   {


		typedef typename Backend::precision_v Float_t;

		Float_t pz=point.z();
		Float_t vz=direction.z();

		MaskedAssign(pz<0.,-vz,&vz);
		MaskedAssign(pz<0.,-pz,&pz);


		Precision tanOuterStereo2 = unplaced.GetTOut2();
		Float_t pxy = (point.x()*direction.x() + point.y()*direction.y());
		//Float_t zc = pz*tanOuterStereo2*vz;
		//Float_t res = pxy - zc;
		return ( (point.x()*direction.x() + point.y()*direction.y() - pz*tanOuterStereo2*vz) < 0);
		}

     template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::bool_v HypeImplementation<transCodeT, rotCodeT>::IsPointMovingInsideInnerSurface(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
									Vector3D<typename Backend::precision_v> const &direction)//,
                                   //typename Backend::bool_v &in)
                                   {


		typedef typename Backend::precision_v Float_t;

		Float_t pz=point.z();
		Float_t vz=direction.z();

		MaskedAssign(pz<0.,-vz,&vz);
		MaskedAssign(pz<0.,-pz,&pz);


		Precision tanInnerStereo2 = unplaced.GetTIn2();
		Float_t pxy = (point.x()*direction.x() + point.y()*direction.y());
		//Float_t zc = pz*tanOuterStereo2*vz;
		//Float_t res = pxy - zc;
		return ( (point.x()*direction.x() + point.y()*direction.y() - pz*tanInnerStereo2*vz) > 0);
		}

    template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::bool_v HypeImplementation<transCodeT, rotCodeT>::IsPointMovingOutsideOuterSurface(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
									Vector3D<typename Backend::precision_v> const &direction)//,
                                   //typename Backend::bool_v &out)
                                   {

        typedef typename Backend::precision_v Float_t;
        typedef typename Backend::bool_v Bool_t;
        Bool_t out(false);

		Float_t pz = point.z();
		Float_t vz = direction.z();


		MaskedAssign(vz<0.,-pz,&pz);
		MaskedAssign(vz<0.,-vz,&vz);

		Precision tanOuterStereo2 = unplaced.GetTOut2();
		Vector3D<Float_t> normHere(point.x(),point.y(),-point.z()*tanOuterStereo2);

		out = (normHere.Dot(direction) > 0.);
        return out;
		}


    template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::bool_v HypeImplementation<transCodeT, rotCodeT>::IsPointMovingOutsideInnerSurface(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
									Vector3D<typename Backend::precision_v> const &direction)//,
                                   //typename Backend::bool_v &out)
                                   {

        typedef typename Backend::precision_v Float_t;
        //typedef typename Backend::bool_v Bool_t;
        //Bool_t out(false);

		Float_t pz = point.z();
		Float_t vz = direction.z();


		MaskedAssign(vz<0.,-pz,&pz);
		MaskedAssign(vz<0.,-vz,&vz);

		//Precision tanInnerStereo2 = unplaced.GetTIn2();
		Vector3D<Float_t> normHere(-point.x(),-point.y(),point.z()*unplaced.GetTIn2());

		//out = (normHere.Dot(direction) > 0.);
        //return out;
        return (normHere.Dot(direction) > 0.);
		}


    template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::bool_v HypeImplementation<transCodeT, rotCodeT>::IsPointMovingInsideFromZPlane(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
									Vector3D<typename Backend::precision_v> const &direction)

									{

		return (point.z()*direction.z() < 0.);

		}

    template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::bool_v HypeImplementation<transCodeT, rotCodeT>::IsPointMovingOutsideFromZPlane(
                                   UnplacedHype const &unplaced,
                                   Vector3D<typename Backend::precision_v> const &point,
									Vector3D<typename Backend::precision_v> const &direction) //,
        						{


        return (point.z()*direction.z() > 0.);
		}

    template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::bool_v HypeImplementation<transCodeT, rotCodeT>::IsPointOnSurfaceAndMovingOutside(
                                                                 UnplacedHype const &unplaced,
                                                                 Vector3D<typename Backend::precision_v> const &point,Vector3D<typename Backend::precision_v> const &direction)//,
                                                                 //typename Backend::bool_v &surface, typename Backend::bool_v &inner,typename Backend::bool_v &outer,typename Backend::bool_v &zSurf)
                                                                 {

        typedef typename Backend::bool_v Bool_t;
		typedef typename Backend::precision_v Float_t;
		Bool_t surface(false),innerHypeSurf(false),outerHypeSurf(false),zSurf(false);
        Bool_t inside(false), outside(false),done(false);

		Float_t rho2 = point.x()*point.x() + point.y()*point.y();
		//Float_t radI2(0.),radO2(0.);
		Float_t radI2 = RadiusHypeSq<Backend,true>(unplaced,point.z());
		Float_t radO2 = RadiusHypeSq<Backend,false>(unplaced,point.z());

		Bool_t out(false);
		zSurf = ((rho2 - unplaced.GetEndOuterRadius2()) < kTolerance) && ((unplaced.GetEndInnerRadius2() - rho2) < kTolerance) && (Abs(Abs(point.z())-unplaced.GetDz())<kTolerance);
		MaskedAssign(zSurf,(point.z()*direction.z() > 0.),&out);
		done |= zSurf;
		if(IsFull(done)) return out;
		MaskedAssign(!zSurf ,( Abs((radO2) - (rho2)) < unplaced.GetOuterRadToleranceLevel()) ,&outerHypeSurf);
		MaskedAssign(!done && !zSurf && outerHypeSurf , IsPointMovingOutsideOuterSurface<Backend>(unplaced,point,direction), &out );
		done |= (!zSurf && outerHypeSurf);
		if(IsFull(done)) return out;


		if(unplaced.InnerSurfaceExists())
        {
            MaskedAssign(!zSurf && !outerHypeSurf,( Abs((radI2) - (rho2)) < unplaced.GetInnerRadToleranceLevel()) ,&innerHypeSurf);
            MaskedAssign(!done && !zSurf && innerHypeSurf , IsPointMovingOutsideInnerSurface<Backend>(unplaced,point,direction), &out );
            done |= (!zSurf && innerHypeSurf);
            if(IsFull(done)) return out;
        }


        return out;


    }

template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::bool_v HypeImplementation<transCodeT, rotCodeT>::IsPointOnInnerSurfaceAndMovingOutside(
                                                                 UnplacedHype const &unplaced,
                                                                 Vector3D<typename Backend::precision_v> const &point,Vector3D<typename Backend::precision_v> const &direction)

                                                                 {


        typedef typename Backend::bool_v Bool_t;
		typedef typename Backend::precision_v Float_t;


		Float_t rho2 = point.x()*point.x() + point.y()*point.y();
		Float_t absZ = Abs(point.z());
		Float_t radI2 = RadiusHypeSq<Backend,true>(unplaced,point.z());

		Bool_t out(false),innerHypeSurf(false);

		if(unplaced.InnerSurfaceExists())
        {
            innerHypeSurf = ( Abs((radI2) - (rho2)) < unplaced.GetInnerRadToleranceLevel()) && (absZ >= 0.) && (absZ < unplaced.GetDz());
            MaskedAssign(innerHypeSurf,IsPointMovingOutsideInnerSurface<Backend>(unplaced,point,direction) ,&out);

        }


        return out;


    }


template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::bool_v HypeImplementation<transCodeT, rotCodeT>::IsPointOnSurfaceAndMovingInside(
                                                                 UnplacedHype const &unplaced,
                                                                 Vector3D<typename Backend::precision_v> const &point,Vector3D<typename Backend::precision_v> const &direction)

                                                                 {

        typedef typename Backend::bool_v Bool_t;
		typedef typename Backend::precision_v Float_t;
		Bool_t surface(false),innerHypeSurf(false),outerHypeSurf(false),zSurf(false);
        Bool_t inside(false), outside(false),done(false);

		Float_t rho2 = point.x()*point.x() + point.y()*point.y();

		Float_t radI2 = RadiusHypeSq<Backend,true>(unplaced,point.z());
		Float_t radO2 = RadiusHypeSq<Backend,false>(unplaced,point.z());

		Bool_t in(false);
		zSurf = ((rho2 - unplaced.GetEndOuterRadius2()) < kTolerance) && ((unplaced.GetEndInnerRadius2() - rho2) < kTolerance) && (Abs(Abs(point.z())-unplaced.GetDz())<kTolerance);
		MaskedAssign(zSurf,(point.z()*direction.z() < 0.),&in);


		done |= zSurf;
		if(IsFull(done)) return in;

		MaskedAssign(!zSurf ,( Abs((radO2) - (rho2)) < unplaced.GetOuterRadToleranceLevel()) ,&outerHypeSurf);
		MaskedAssign(!done && outerHypeSurf , IsPointMovingInsideOuterSurface<Backend>(unplaced,point,direction), &in );

		if(unplaced.InnerSurfaceExists())
        {
            done |= (!zSurf && outerHypeSurf);
            if(IsFull(done)) return in;

            MaskedAssign(!zSurf && !outerHypeSurf,( Abs((radI2) - (rho2)) < unplaced.GetInnerRadToleranceLevel()) ,&innerHypeSurf);
            MaskedAssign(!done && !zSurf && innerHypeSurf , IsPointMovingInsideInnerSurface<Backend>(unplaced,point,direction), &in );
            done |= (!zSurf && innerHypeSurf);
            if(IsFull(done)) return in;

        }

        return in;

    }



template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend,bool ForInnerRad>
VECGEOM_CUDA_HEADER_BOTH
typename Backend::precision_v HypeImplementation<transCodeT, rotCodeT>::RadiusHypeSq(UnplacedHype const &unplaced, typename Backend::precision_v z)//, typename Backend::precision_v &radsq){
{

   		if(ForInnerRad)
            return (unplaced.GetRmin2() + unplaced.GetTIn2()*z*z);
   		else
            return (unplaced.GetRmax2() + unplaced.GetTOut2()*z*z);
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
	//Float_t radOSq(0.);
	Float_t radOSq = RadiusHypeSq<Backend,false>(unplaced,localPoint.z());//,radOSq);
	Float_t dist2Outer = Abs(xR2 - radOSq);
	//std::cout<<"LocalPoint : "<<localPoint<<std::endl;
	Bool_t done(false);


	//Inner Surface Wins
	if(unplaced.InnerSurfaceExists())
	{
		//Float_t radISq(0.);
		Float_t radISq = RadiusHypeSq<Backend,true>(unplaced,localPoint.z());//,radISq);
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
    template <typename Backend, bool ForInside>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::GenericKernelForContainsAndInside(
                                                                                    UnplacedHype const &unplaced,
                                                                                    Vector3D<typename Backend::precision_v> const &point,
                                                                                    typename Backend::bool_v &completelyinside,
                                                                                    typename Backend::bool_v &completelyoutside) {
    typedef typename Backend::precision_v Float_t;
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

        typedef typename Backend::precision_v Float_t;
        typedef typename Backend::bool_v      Bool_t;

        Float_t absZ=Abs(point.z());
        Precision fDz = unplaced.GetDz();
        distance = kInfinity;
        Float_t zDist(kInfinity),dist(kInfinity);
        Float_t r=(point.x()*point.x() + point.y()*point.y());
        Precision innerRadius = unplaced.GetRmin();
        Precision outerRadius = unplaced.GetRmax();
        Precision tanOuterStereo2 = unplaced.GetTOut2();
        Precision tanInnerStereo2 = unplaced.GetTIn2();

        Bool_t done(false);
        Bool_t cond(false);


        //checking whether point hits the Z Surface of hyperboloid
        Bool_t hittingZPlane = GetPointOfIntersectionWithZPlane<Backend,true>(unplaced,point,direction,zDist) ;
        Bool_t isPointAboveOrBelowHypeAndGoingInside = (absZ > fDz) && (point.z()*direction.z() < 0.);
        cond = isPointAboveOrBelowHypeAndGoingInside && hittingZPlane;
        MaskedAssign(!done && cond  , zDist , &distance );
        done|= cond;
        if(IsFull(done)) return;

        //Moving the point to Z Surface
        Vector3D<Float_t> newPt = point + zDist*direction;
        Float_t rp2 = newPt.Perp2();

        //checking whether point hits the outer hyperbolic surface of hyperboloid
        Bool_t hittingOuterSurfaceFromOutsideZRange = isPointAboveOrBelowHypeAndGoingInside && !hittingZPlane && (rp2 >= unplaced.GetEndOuterRadius2());
        Bool_t hittingOuterSurfaceFromWithinZRange = ((r > ((outerRadius*outerRadius + tanOuterStereo2*absZ*absZ )+kHalfTolerance)) ) && (absZ >= 0.) && (absZ <= fDz);

        cond = (hittingOuterSurfaceFromOutsideZRange || hittingOuterSurfaceFromWithinZRange) && GetPointOfIntersectionWithOuterHyperbolicSurface<Backend,true>(unplaced,point,direction,dist);
        MaskedAssign(!done && cond , dist , &distance);

        if(unplaced.InnerSurfaceExists())
        {
        done |= cond;
        if(IsFull(done)) return;
        Bool_t hittingInnerSurfaceFromOutsideZRange =  isPointAboveOrBelowHypeAndGoingInside && !hittingZPlane && (rp2 <= unplaced.GetEndInnerRadius2());
        Bool_t hittingInnerSurfaceFromWithinZRange = (r < ((innerRadius*innerRadius + tanInnerStereo2*absZ*absZ )-kHalfTolerance)) && (absZ >= 0.) && (absZ <= fDz);

        //If it hits inner hyperbolic surface then distance will be the distance to inner hyperbolic surface
        //Or if the point is on the inner Hyperbolic surface but going out then the distance will be the distance to opposite inner hyperbolic surface.
        cond = (hittingInnerSurfaceFromOutsideZRange || hittingInnerSurfaceFromWithinZRange || (IsPointOnInnerSurfaceAndMovingOutside<Backend>(unplaced,point,direction))) && GetPointOfIntersectionWithInnerHyperbolicSurface<Backend,true>(unplaced,point,direction,dist);
        MaskedAssign(!done && cond , dist , &distance);
        done |= cond;
        if(IsFull(done)) return;
        }

        Bool_t surfaceCond = IsPointOnSurfaceAndMovingInside<Backend>(unplaced,point,direction);
        MaskedAssign(!done && surfaceCond , 0. , &distance);//,isSurfacePoint,inner,outer,zSurf),0., &distance);
        done |= surfaceCond;
        if(IsFull(done)) return;

        Bool_t inside(false),outside(false);
        GenericKernelForContainsAndInside<Backend,true>(unplaced,point,inside,outside);
        MaskedAssign(!done && inside,-kTolerance,&distance);


    }



template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <class Backend,bool ForDistToIn>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::bool_v HypeImplementation<transCodeT, rotCodeT>::GetPointOfIntersectionWithZPlane(UnplacedHype const &unplaced,
                                                                     Vector3D<typename Backend::precision_v> const &point,
                                                                     Vector3D<typename Backend::precision_v> const &direction,
                                                                    typename Backend::precision_v &zDist){

                                                                        typedef typename Backend::precision_v Float_t;
                                                                        typedef typename Backend::bool_v Bool_t;
                                                                        Vector3D<Float_t> newPt;
                                                                        Bool_t inside(false),outside(false);
                                                                        if(ForDistToIn)
                                                                        CondAssign(point.z()>0. , (unplaced.GetDz()-point.z())/direction.z() ,  (-unplaced.GetDz()-point.z())/direction.z(), &zDist);
                                                                        else
                                                                        CondAssign(direction.z()>0. , (unplaced.GetDz()-point.z())/direction.z() ,  (-unplaced.GetDz()-point.z())/direction.z(), &zDist);

                                                                        newPt = point + zDist*direction;
                                                                        //Float_t r = (newPt.x()*newPt.x() + newPt.y()*newPt.y());
                                                                        Float_t r2 = newPt.Perp2();
                                                                        if(!unplaced.InnerSurfaceExists())
                                                                            return (r2 < unplaced.GetEndOuterRadius2());
                                                                        else
                                                                            return  ( (r2 < unplaced.GetEndOuterRadius2()) && (r2 > unplaced.GetEndInnerRadius2()));

                                                                    }

template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <class Backend, bool ForDistToIn>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::bool_v HypeImplementation<transCodeT, rotCodeT>::GetPointOfIntersectionWithInnerHyperbolicSurface(UnplacedHype const &unplaced,
                                                                     Vector3D<typename Backend::precision_v> const &point,
                                                                     Vector3D<typename Backend::precision_v> const &direction,
                                                                    typename Backend::precision_v &dist){//,typename Backend::precision_v zDist,bool cbd){

                                                                        typedef typename Backend::precision_v Float_t;
                                                                        typedef typename Backend::bool_v Bool_t;
                                                                        Bool_t exist(false);

                                                                        Vector3D<Float_t> newPt,tempPt;

                                                                        Precision tanInnerStereo2 = unplaced.GetTIn2();
                                                                        Precision fRmin2 = unplaced.GetRmin2();


                                                                        //Float_t a = direction.x() * direction.x() + direction.y() * direction.y() - tanInnerStereo2*direction.z()*direction.z();
                                                                        //Float_t b = 2*direction.x()*point.x() + 2*direction.y()*point.y() - 2*tanInnerStereo2*direction.z()*point.z();
                                                                        //Float_t c= point.x()*point.x() + point.y()*point.y() - tanInnerStereo2*point.z()*point.z() - fRmin2;

                                                                        Float_t a = direction.Perp2() - tanInnerStereo2*direction.z()*direction.z();
                                                                        Float_t b = 2*(direction.x()*point.x() + direction.y()*point.y() - tanInnerStereo2*direction.z()*point.z());
                                                                        Float_t c = point.Perp2() - tanInnerStereo2*point.z()*point.z() - fRmin2;
                                                                        exist = (b*b - 4*a*c > 0.);

                                                                        if(ForDistToIn)
                                                                        {
                                                                        MaskedAssign(exist && b<0.,( (-b + Sqrt(b*b - 4*a*c))/(2*a) ),&dist);
                                                                        MaskedAssign(exist && b>=0.,((2*c)/(-b - Sqrt(b*b - 4*a*c)) ),&dist);
                                                                        }
                                                                        else
                                                                        {
                                                                        MaskedAssign(exist && b>0.,( (-b - Sqrt(b*b - 4*a*c))/(2*a) ),&dist);
                                                                        MaskedAssign(exist && b<=0.,((2*c)/(-b + Sqrt(b*b - 4*a*c)) ),&dist);
                                                                        }
                                                                        MaskedAssign(dist < 0. ,kInfinity, &dist);
                                                                        newPt = point + dist*direction;

                                                                        return (Abs(newPt.z()) <= unplaced.GetDz());

                                                                    }


template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <class Backend,bool ForDistToIn>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::bool_v HypeImplementation<transCodeT, rotCodeT>::GetPointOfIntersectionWithOuterHyperbolicSurface(UnplacedHype const &unplaced,
                                                                     Vector3D<typename Backend::precision_v> const &point,
                                                                     Vector3D<typename Backend::precision_v> const &direction,
                                                                    typename Backend::precision_v &dist){//,typename Backend::precision_v zDist,bool cbd){

                                                                        typedef typename Backend::precision_v Float_t;
                                                                        typedef typename Backend::bool_v Bool_t;
                                                                        Bool_t exist(false);

                                                                        Vector3D<Float_t> newPt,tempPt;

                                                                        Precision tanOuterStereo2 = unplaced.GetTOut2();
                                                                        Precision fRmax2 = unplaced.GetRmax2();

                                                                        //Float_t a = direction.x() * direction.x() + direction.y() * direction.y() - tanOuterStereo2*direction.z()*direction.z();
                                                                        //Float_t b = 2*direction.x()*point.x() + 2*direction.y()*point.y() - 2*tanOuterStereo2*direction.z()*point.z();
                                                                        //Float_t c= point.x()*point.x() + point.y()*point.y() - tanOuterStereo2*point.z()*point.z() - fRmax2;

                                                                        Float_t a = direction.Perp2() - tanOuterStereo2*direction.z()*direction.z();
                                                                        Float_t b = 2*(direction.x()*point.x() + direction.y()*point.y() - tanOuterStereo2*direction.z()*point.z());
                                                                        Float_t c = point.Perp2() - tanOuterStereo2*point.z()*point.z() - fRmax2;
                                                                        exist = (b*b - 4*a*c > 0.);
                                                                        //std::cout<<"Exist : "<<exist<<std::endl;
                                                                        if(ForDistToIn)
                                                                        {
                                                                        MaskedAssign(exist && b>=0.,( (-b - Sqrt(b*b - 4*a*c))/(2*a) ),&dist);
                                                                        MaskedAssign(exist && b<0.,((2*c)/(-b + Sqrt(b*b - 4*a*c)) ),&dist);
                                                                        }
                                                                        else
                                                                        {
                                                                        MaskedAssign(exist && b<0.,( (-b + Sqrt(b*b - 4*a*c))/(2*a) ),&dist);
                                                                        MaskedAssign(exist && b>=0.,((2*c)/(-b - Sqrt(b*b - 4*a*c)) ),&dist);
                                                                        }
                                                                        MaskedAssign(dist < 0. ,kInfinity, &dist);


                                                                        newPt = point + dist*direction;

                                                                        return (Abs(newPt.z()) <= unplaced.GetDz());

                                                                    }

//New Definition
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

        distance = kInfinity;
        Float_t zDist(kInfinity),dist(kInfinity);

        Bool_t done(false);
        Bool_t intersectionExist(false);




        intersectionExist = GetPointOfIntersectionWithZPlane<Backend,false>(unplaced,point,direction,zDist) ;
        MaskedAssign(zDist < 0., kInfinity, &zDist);
        //MaskedAssign(!done && intersectionExist, zDist, &distance);

        intersectionExist = GetPointOfIntersectionWithOuterHyperbolicSurface<Backend,false>(unplaced,point,direction,dist);
        MaskedAssign(dist < 0., kInfinity, &dist);
        distance = Min(zDist,dist);
        //MaskedAssign(!done && intersectionExist && dist<distance , dist , &distance);


        if(unplaced.InnerSurfaceExists())
        {
        intersectionExist = GetPointOfIntersectionWithInnerHyperbolicSurface<Backend,false>(unplaced,point,direction,dist);
        MaskedAssign(dist < 0., kInfinity, &dist);
        distance = Min(distance,dist);
        //MaskedAssign(!done && intersectionExist && dist<distance, dist , &distance);

        }

        Bool_t surfaceCond = IsPointOnSurfaceAndMovingOutside<Backend>(unplaced,point,direction);
        MaskedAssign(!done && surfaceCond , 0. , &distance);//,isSurfacePoint,inner,outer,zSurf),0., &distance);
        done |= surfaceCond;
        if(IsFull(done)) return;

        Bool_t inside(false),outside(false);
        GenericKernelForContainsAndInside<Backend,true>(unplaced,point,inside,outside);
        MaskedAssign(!done && outside,kTolerance,&distance);
        done |= outside;
        if(IsFull(done)) return;

    }


/*
    //Old Definition

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

	Bool_t inside(false), outside(false);
	GenericKernelForContainsAndInside<Backend,true>(unplaced,point,inside,outside);
	MaskedAssign(outside, kTolerance, &distance);
	done |= outside;
	if(IsFull(done)) return;

    Bool_t surfaceCond = IsPointOnSurfaceAndMovingOutside<Backend>(unplaced,point,direction);
    MaskedAssign(!done && surfaceCond , 0. , &distance);//,isSurfacePoint,inner,outer,zSurf),0., &distance);
    done |= surfaceCond;
    if(IsFull(done)) return;

	//Handling Inner Surface
	if(unplaced.InnerSurfaceExists())
	{
	a = direction.x() * direction.x() + direction.y() * direction.y() - tanInnerStereo2*direction.z()*direction.z();
	b = 2*direction.x()*point.x() + 2*direction.y()*point.y() - 2*tanInnerStereo2*direction.z()*point.z();
	c= point.x()*point.x() + point.y()*point.y() - tanInnerStereo2*point.z()*point.z() - fRmin2;
	exist = (b*b - 4*a*c > 0.);

	MaskedAssign(!done && exist && b>0. ,( (-b - Sqrt(b*b - 4*a*c))/(2*a) ),&distInner);
	MaskedAssign(!done && exist && b<=0.,((2*c)/(-b + Sqrt(b*b - 4*a*c)) ),&distInner);
	MaskedAssign(distInner < 0. ,kInfinity, &distInner);
	}

	//Handling Outer surface
	exist = fal;
	a = direction.x() * direction.x() + direction.y() * direction.y() - tanOuterStereo2*direction.z()*direction.z();
	b = 2*direction.x()*point.x() + 2*direction.y()*point.y() - 2*tanOuterStereo2*direction.z()*point.z();
	c= point.x()*point.x() + point.y()*point.y() - tanOuterStereo2*point.z()*point.z() - fRmax2;
	exist = (b*b - 4*a*c > 0.);
	MaskedAssign(!done && exist && b<0.,( (-b + Sqrt(b*b - 4*a*c))/(2*a) ),&distOuter);
	MaskedAssign(!done && exist && b>=0.,((2*c)/(-b - Sqrt(b*b - 4*a*c)) ),&distOuter);
	MaskedAssign(distOuter < 0. ,kInfinity, &distOuter);

    //Handling Z surface
	Float_t distZ=kInfinity;
    Float_t dirZinv=1/direction.z();
    Bool_t dir_mask= direction.z()<0;
    MaskedAssign(!done &&  dir_mask, -(unplaced.GetDz() + point.z())*dirZinv, &distZ);

    MaskedAssign(!done &&  !dir_mask, (unplaced.GetDz() - point.z())*dirZinv, &distZ);
	MaskedAssign(distZ < 0. , kInfinity, &distZ);

	distance = Min(distInner,distOuter);
	distance = Min(distance,distZ);

	//Handling Surface Point

	Bool_t isSurfacePoint(false),inner(false);
	Bool_t outer(false),out(false),zSurf(false);

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


	//
	Float_t absZ = Abs(point.z());
	Float_t r2 = point.Perp2();
	Float_t r =  Sqrt(r2);



	Precision endOuterRadius = unplaced.GetEndOuterRadius();
	Precision endInnerRadius = unplaced.GetEndInnerRadius();
	Precision innerRadius = unplaced.GetRmin();
	Precision outerRadius = unplaced.GetRmax();
	Precision tanInnerStereo2 = unplaced.GetTIn2();
	Precision tanOuterStereo2 = unplaced.GetTOut2();
	Precision tanOuterStereo = unplaced.GetTOut();
	Bool_t done(false);

    //New Simple Algo

    safety = 0.;

    Bool_t cond(false);
    Float_t sigz = absZ - unplaced.GetDz();
    cond = (sigz > kHalfTolerance) && (r < endOuterRadius) && (r > endInnerRadius);
    MaskedAssign(cond, sigz , &safety);
    done |= cond;
    if(IsFull(done)) return;

    cond = (sigz > kHalfTolerance) && (r > endOuterRadius) ;
    MaskedAssign(!done && cond,Sqrt( (r-endOuterRadius)*(r-endOuterRadius) + (sigz)*(sigz) ) , &safety );
    done |= cond;
    if(IsFull(done)) return;


    cond = (sigz > kHalfTolerance) && (r < endInnerRadius);
    MaskedAssign(!done && cond,Sqrt( (r-endInnerRadius)*(r-endInnerRadius) + (sigz)*(sigz) ) , &safety );
    done |= cond;
    if(IsFull(done)) return;

    cond = (r > (Sqrt(outerRadius*outerRadius + tanOuterStereo2*absZ*absZ )+kHalfTolerance)) && (absZ > 0.) && (absZ < unplaced.GetDz());
    MaskedAssign(!done && cond , ApproxDistOutside<Backend>( r,absZ,outerRadius,tanOuterStereo), &safety);
    done |= cond;
    if(IsFull(done)) return;


    MaskedAssign(!done && (r2 < ( (innerRadius*innerRadius + tanInnerStereo2*absZ*absZ )-kHalfTolerance) ) && (absZ > 0.) && (absZ < unplaced.GetDz()) , ApproxDistInside<Backend>( r,absZ,innerRadius,tanInnerStereo2), &safety);

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
        Float_t r=Sqrt(point.x()*point.x() + point.y()*point.y());
        Float_t absZ= Abs(point.z());
        Precision innerRadius = unplaced.GetRmin();
        Precision outerRadius = unplaced.GetRmax();
        Precision tanOuterStereo = unplaced.GetTOut();
        Precision tanOuterStereo2 = tanOuterStereo*tanOuterStereo;
        Precision tanInnerStereo = unplaced.GetTIn();
        Float_t r1=outerRadius*outerRadius + tanOuterStereo*tanOuterStereo*absZ*absZ;



        //New Algo

        Bool_t inside(false),outside(false);
        Bool_t done(false);

        Float_t distZ(0.),distInner(0.),distOuter(0.);
        safety=0.;
        GenericKernelForContainsAndInside<Backend,true>(unplaced,point,inside,outside);
        done |= (!inside);
        if(IsFull(done)) return;

        MaskedAssign(!done && inside, Abs(Abs(point.z())-unplaced.GetDz()) , &distZ );
        //std::cout<<"----- 1 -----"<<std::endl;
        //MaskedAssign(!done && inside, ApproxDistOutside<Backend>( r,absZ,innerRadius,tanInnerStereo), &distInner);
        //MaskedAssign(!done && inside, (r1-r)*cos(unplaced.GetStOut()), &distOuter);
        if(unplaced.InnerSurfaceExists() && unplaced.GetStIn())
        {
            MaskedAssign(!done && inside, ApproxDistOutside<Backend>( r,absZ,innerRadius,tanInnerStereo), &distInner);

          //  std::cout<<"----- 2 -----"<<std::endl;
        }

        if(unplaced.InnerSurfaceExists() && !unplaced.GetStIn())
        {
            MaskedAssign(!done && inside, (r-innerRadius), &distInner);
            //std::cout<<"----- 3 -----"<<std::endl;
        }

         if(!unplaced.InnerSurfaceExists() && !unplaced.GetStIn())
        {
            MaskedAssign(!done && inside, kInfinity, &distInner);
            //std::cout<<"----- 4 -----"<<std::endl;
        }



        MaskedAssign(!done && inside, ApproxDistInside<Backend>( r,absZ,outerRadius,tanOuterStereo2), &distOuter);
        //MaskedAssign(distInner < 0. , 0. , &distInner);
        //MaskedAssign(distOuter < 0. , 0. , &distOuter);
        //std::cout<<"----- 5 -----"<<std::endl;
        //std::cout<<"DistZ : "<<distZ<<"  :: distInner : "<<distInner<<"  :: distOuter : "<<distOuter<<std::endl;
        safety = Min(distInner,distOuter);
        safety = Min(safety,distZ);
        //std::cout<<"SafetTy : "<<safety<<std::endl;



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
        /*
        Float_t safeZ= unplaced.GetDz()-absZ;
        Float_t rsq = point.x()*point.x()+point.y()*point.y();
        Float_t r = Sqrt(rsq);
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
        */
        /*
		Bool_t inside(false);
		UnplacedContains<Backend>(unplaced,point,inside);
		//MaskedAssign(!inside || safety<kSTolerance*100.,0.,&safety);
		MaskedAssign(!inside || safety<kTolerance*10.,0.,&safety);
        */

    }


}} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_HYPEIMPLEMENTATION_H_
