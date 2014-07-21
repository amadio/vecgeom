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


namespace VECGEOM_NAMESPACE {
    
    template <TranslationCode transCodeT, RotationCode rotCodeT>
    struct HypeImplementation {
        
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
        
    }; // End struct HypeImplementation
    
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
        
        //check if points are above or below the solid
        completelyoutside = Abs(point.z()) > MakePlusTolerant<ForInside>( unplaced.GetDz() );
        if (ForInside)
        {
            completelyinside = Abs(point.z()) < MakeMinusTolerant<ForInside>( unplaced.GetDz());
        }
        if (Backend::early_returns) {
            if ( completelyoutside == Backend::kTrue ) {
                return;
            }
        }
        
        //check if points are outside of the outer surface or outside the inner surface
        Float_t r2=point.x()*point.x()+point.y()*point.y();
        //compute r^2 at a given z coordinate, for the outer hyperbolas
        Float_t rOuter2=unplaced.GetRmax2()+unplaced.GetTOut2()*point.z()*point.z();
        //compute r^2 at a given z coordinate, for the inner hyperbolas
        Float_t rInner2=unplaced.GetRmin2()+unplaced.GetTIn2()*point.z()*point.z();
        
        completelyoutside |= (r2 > MakePlusTolerant<ForInside>( rOuter2 )) || (r2 < MakePlusTolerant<ForInside>( rInner2 ));
        if (ForInside)
        {
            completelyinside &= (r2 < MakeMinusTolerant<ForInside>( rOuter2 )) && (r2 > MakeMinusTolerant<ForInside>( rInner2 ));
        }
        return;
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
        
        Float_t distOut=aOutinv*(bOut -deltaOut);
        
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
        
        Float_t distIn=aIninv*(bIn +deltaIn);

        Float_t zHitIn1 = point.z()+distIn*direction.z();
        Bool_t isHittingHyperboloidSurfaceIn1 = ( (distIn> 1E20) || (Abs(zHitIn1)<=unplaced.GetDz()) ); //why: dist > 1E20?
    
        Float_t solution_Inner=kInfinity;
        MaskedAssign(!deltaInNeg && isHittingHyperboloidSurfaceIn1 && distIn>0, distIn, &solution_Inner);
        
        Float_t solution=Min(solution_Inner, solution_Outer);
        
        done|=(deltaInNeg && deltaOutNeg);
        MaskedAssign(!done, solution, &distance );
        
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
        distance=Min(distHype, distZ);
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
        
        Float_t zHypeSqOut= Sqrt((r*r-unplaced.GetRmax2())/(unplaced.GetTOut2()));
        Float_t mOut=(zHypeSqOut-absZ)/drOut;
    
        Float_t safe = mOut*drOut/Sqrt(1.+mOut*mOut);
        MaskedAssign(!mask_fStOut && !mask_drOut, safe, &safermax);
        Float_t max_safety= Max(safermax, safeZ);
        
        ////Check for Inner Threatment -->this should be managed as a specialization
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
          
            Float_t zHypeSqIn= Sqrt( (r*r-unplaced.GetRmin2()) / (unplaced.GetTIn2()) );
            Float_t mIn=rhIn/(unplaced.GetTIn2()*absZ);
            
            safe = -mIn*drIn/Sqrt(1.+mIn*mIn);
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
        Bool_t  mask_bcIn= (safeX_In<0) &&(safeY_In<0)  && (safeZ>0);
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
        
        Float_t zHypeSqOut= Sqrt((r*r-unplaced.GetRmax2())/(unplaced.GetTOut2()));
        Float_t mOut=(zHypeSqOut-absZ)/drOut;
        
        Float_t safe = mOut*drOut/Sqrt(1.+mOut*mOut);
        MaskedAssign(!mask_fStOut && !mask_drOut, safe, &safermax);
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

            Float_t zHypeSqIn= Sqrt( (r*r-unplaced.GetRmin2()) / (unplaced.GetTIn2()) );
            
            Float_t mIn=rhIn/(unplaced.GetTIn2()*absZ);
            safe = -mIn*drIn/Sqrt(1.+mIn*mIn);
            MaskedAssign(!doneInner, safe, &safermin);
            max_safety= Max(max_safety, safermin);
        }
        
        MaskedAssign(!done, max_safety, &safety);
        
#endif

#ifdef ACCURATE_BC
        //Bounding-Cylinder implementation
        Float_t absX= Abs(point.x());
        Float_t absY= Abs(point.y());
        
        Float_t rho=Sqrt(point.x()*point.x()+point.y()*point.y());
    
        //check if the point is inside the inner-bounding cylinder
        Float_t safeRhoIn=unplaced.GetRmin()-rho;
        Bool_t  mask_bcIn= (safeRhoIn>0) && (safeZ>0);
        safety_t=Min(safeZ, safeRhoIn);
        Bool_t done(mask_bcIn);
        if (Backend::early_returns && done == Backend::kTrue) return;
        
        //check if the point is outside the outer-bounding cylinder
        Float_t safeRhoOut=rho-unplaced.GetEndOuterRadius();
        Bool_t  mask_bcOut= (safeRhoOut>0) || (safeZ>0);
        
        safety_t=Max(safeZ, safeRhoOut);
        MaskedAssign(!done && mask_bcOut, safety_t, &safety);
        done|=mask_bcOut;
        if (Backend::early_returns && done == Backend::kTrue) return;
    
        //Then calculate accurate value
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
        
        Float_t zHypeSqOut= Sqrt((r*r-unplaced.GetRmax2())/(unplaced.GetTOut2()));
        Float_t mOut=(zHypeSqOut-absZ)/drOut;

        Float_t safe = mOut*drOut/Sqrt(1.+mOut*mOut);
        MaskedAssign(!mask_fStOut && !mask_drOut, safe, &safermax);
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
            
           
            Float_t zHypeSqIn= Sqrt( (r*r-unplaced.GetRmin2()) / (unplaced.GetTIn2()) );
            
            Float_t mIn=rhIn/(unplaced.GetTIn2()*absZ);
            safe = -mIn*drIn/Sqrt(1.+mIn*mIn);
            MaskedAssign(!doneInner, safe, &safermin);
            max_safety= Max(max_safety, safermin);
        }
        
        MaskedAssign(!done, max_safety, &safety);
        
        
#endif

    }
    
    template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::SafetyToOutKernel(
                                                                    UnplacedHype const &unplaced,
                                                                    Vector3D<typename Backend::precision_v> const &point,
                                                                    typename Backend::precision_v &safety) {
        
        typedef typename Backend::precision_v Float_t;
        typedef typename Backend::bool_v Bool_t;
        
        safety=0.;
    
        Float_t absZ= Abs(point.z());
        Float_t safeZ= unplaced.GetDz()-absZ;
        Float_t safermax;
        
        Float_t rsq = point.x()*point.x()+point.y()*point.y();
        Float_t r = Sqrt(rsq);
        
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
       
    }
    
} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_HYPEIMPLEMENTATION_H_
