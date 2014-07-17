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
        MaskedAssign(distOut1<0 && distOut2<0 && !deltaOutNeg, kInfinity, &distHypeOuter); //infinity or negative number??
        
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
        MaskedAssign(distIn1<0 && distIn2<0 && !deltaInNeg, kInfinity, &distHypeInner); //infinity or negative number??
        
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
        
        
    }
    
    template <TranslationCode transCodeT, RotationCode rotCodeT>
    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH
    void HypeImplementation<transCodeT, rotCodeT>::SafetyToOutKernel(
                                                                    UnplacedHype const &unplaced,
                                                                    Vector3D<typename Backend::precision_v> const &point,
                                                                    typename Backend::precision_v &safety) {
        
       
    }
    
} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_HYPEIMPLEMENTATION_H_
