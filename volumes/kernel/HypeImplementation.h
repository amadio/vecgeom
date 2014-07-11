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
        
        //is above or below the solid
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





//namespace VECGEOM_NAMESPACE {
//    
//    namespace HypeUtilities
//    {
//        template <class Backend>
//        VECGEOM_INLINE
//        VECGEOM_CUDA_HEADER_BOTH
//        void DistToHypeSurface(
//                                 UnplacedHype const &unplaced,
//                                 Vector3D<typename Backend::precision_v> const &point,
//                                 Vector3D<typename Backend::precision_v> const &direction,
//                                 typename Backend::precision_v &distance/*,
//                                 typename Backend::bool_v in*/) {
//                                 }
//    }
//
////HypeImplementation Starts here
//template <TranslationCode transCodeT, RotationCode rotCodeT>
//struct HypeImplementation {
//    
//    
//    //_________________________________________________________________
//    //GENERICKERNEL --> this now must contains the implementation
//    template <typename Backend, bool ForInside>
//    VECGEOM_INLINE
//    VECGEOM_CUDA_HEADER_BOTH
//    static void GenericKernelForContainsAndInside(UnplacedHype const &unplaced,
//                                                  Vector3D<typename Backend::precision_v> const &point,
//                                                  typename Backend::bool_v &completelyinside,
//                                                  typename Backend::bool_v &completelyoutside)
//    {
//        
////        typedef typename Backend::precision_v Float_t;
////        typedef typename Backend::bool_v Bool_t;
////        
////        
////        //is above or below the solid
////        completelyoutside = Abs(point.z()) > MakePlusTolerant<ForInside>( unplaced.GetDz() );
////        if (ForInside)
////        {
////            completelyinside = Abs(point.z()) < MakeMinusTolerant<ForInside>( unplaced.GetDz());
////        }
////        if (Backend::early_returns) {
////            if ( completelyoutside == Backend::kTrue ) {
////                return;
////            }
////        }
////        
////        //check if points are outside of the outer surface or outside the inner surface
////        Float_t r2=point.x()*point.x()+point.y()*point.y();
////        
////        
////        //compute r^2 at a given z coordinate, for the outer hyperbolas
////        Float_t rOuter2=unplaced.GetRmax2()+unplaced.GetTOut2()*point.z()*point.z();
////        //compute r^2 at a given z coordinate, for the inner hyperbolas
////        Float_t rInner2=unplaced.GetRmin2()+unplaced.GetTIn2()*point.z()*point.z();
////        
////        completelyoutside |= (r2 > MakePlusTolerant<ForInside>( rOuter2 )) || (r2 < MakePlusTolerant<ForInside>( rInner2 ));
////        if (ForInside)
////        {
////            completelyinside &= (r2 < MakeMinusTolerant<ForInside>( rOuter2 )) && (r2 > MakeMinusTolerant<ForInside>( rInner2 ));
////        }
////        return;
//    }
//    
//    
//    /*typedef typename Backend::precision_v Float_t;
//     typedef typename Backend::bool_v Bool_t;
//     
//     //is above the solid
//     Float_t absZ=Abs(point.z());
//     Bool_t isAboveOrBelowSolid= (Abs(point.z())>unplaced.GetDz());
//     Bool_t done(isAboveOrBelowSolid);
//     inside=Backend::kFalse;
//     
//     if(Backend::early_returns && done==Backend::kTrue) return;
//     
//     Float_t r2=point.x()*point.x()+point.y()*point.y();
//     
//     //check if points are outside of the outer surface or outside the inner surface
//     
//     //compute r^2 at a given z coordinate, for the outer hyperbolas
//     Float_t rOuter2=unplaced.GetRmax2()+unplaced.GetTOut2()*point.z()*point.z();
//     //compute r^2 at a given z coordinate, for the inner hyperbolas
//     Float_t rInner2=unplaced.GetRmin2()+unplaced.GetTIn2()*point.z()*point.z();
//     
//     Bool_t isOutsideHyperbolicSurface=r2>rOuter2 || r2<rInner2;
//     done|= isOutsideHyperbolicSurface;
//     
//     MaskedAssign(!done, Backend::kTrue, &inside);*/
//    
//    
//    //_________________________________________________________________
//    //CONTAINSKERNEL --> this calls GenericKernel<Backend, false> ( dimensions, localPoint, unused, outside)
//    template <TranslationCode transCodeT, RotationCode rotCodeT>
//    template <typename Backend>
//    VECGEOM_CUDA_HEADER_BOTH
//    void HypeImplementation<transCodeT, rotCodeT>::ContainsKernel(
//                                                                  UnplacedHype const &unplaced,
//                                                                  Vector3D<typename Backend::precision_v> const &localPoint,
//                                                                  typename Backend::bool_v &inside)
//    {
//        
//        typedef typename Backend::bool_v Bool_t;
//        Bool_t unused;
//        Bool_t outside;
//        GenericKernelForContainsAndInside<Backend, false>(unplaced,
//                                                          localPoint, unused, outside);
//        inside=!outside;
//    }
//    
//    
//    
//    //INSIDE KERNEL--> this calls GenericKernel<Backend,true> ( dimensions, localPoint, completelyinside, completelyoutside)
//    //___________________________________________________________________
//    template <TranslationCode transCodeT, RotationCode rotCodeT>
//    template <class Backend>
//    VECGEOM_CUDA_HEADER_BOTH
//    void HypeImplementation<transCodeT, rotCodeT>::InsideKernel(
//                                                                UnplacedHype const &unplaced,
//                                                                Vector3D<typename Backend::precision_v> const &point,
//                                                                typename Backend::inside_v &inside)
//    {
//        
//        typedef typename Backend::bool_v      Bool_t;
//        Bool_t completelyinside, completelyoutside;
//        GenericKernelForContainsAndInside<Backend,true>(unplaced, point, completelyinside, completelyoutside);
//        inside=EInside::kSurface;
//        MaskedAssign(completelyoutside, EInside::kOutside, &inside);
//        MaskedAssign(completelyinside, EInside::kInside, &inside);
//    }
//    
//    
//    
//    //_________________________________________________________________
//    //UNPLACEDINSIDE--> call the InsideKernel
//    /// \brief Inside method that takes account of the surface for an Unplaced Hype
//    template <class Backend>
//    VECGEOM_INLINE
//    VECGEOM_CUDA_HEADER_BOTH
//    static void UnplacedInside(UnplacedHype const &unplaced,
//                               Vector3D<typename Backend::precision_v> point,
//                               typename Backend::int_v &inside)
//    {
//        
//        InsideKernel<Backend>(unplaced, point, inside);
//    }
//    
//    //_________________________________________________________________
//    //UNPLACEDCONTAINS --> call the ContainsKernel
//    /// \brief UnplacedContains (ROOT STYLE): Inside method that does NOT take account of the surface for an Unplaced Hype
//    template <class Backend>
//    VECGEOM_INLINE
//    VECGEOM_CUDA_HEADER_BOTH
//    static void UnplacedContains(UnplacedHype const &unplaced,
//        Vector3D<typename Backend::precision_v> point,
//        typename Backend::bool_v &inside)
//    {
//        
//        ContainsKernel<Backend>(unplaced, point, inside);
//    }
//    
//    //_________________________________________________________________
//    //INSIDE (local point in the signature) --> just call UnplacedInside
//    /// \brief Inside method that takes account of the surface for a Placed Hype
//    template <class Backend>
//    VECGEOM_INLINE
//    VECGEOM_CUDA_HEADER_BOTH
//    static void Inside(UnplacedHype const &unplaced,
//                       Transformation3D const &transformation,
//                       Vector3D<typename Backend::precision_v> const &point,
//                       Vector3D<typename Backend::precision_v> &localPoint,
//                       typename Backend::int_v &inside)
//    {
//        localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
//        UnplacedInside<Backend>(unplaced, localPoint, inside);
//    }
//    
//    //_________________________________________________________________
//    //INSIDE(local point as local variable)--> just call UnplacedInside
//    template <class Backend>
//    VECGEOM_INLINE
//    VECGEOM_CUDA_HEADER_BOTH
//    static void Inside(UnplacedHype const &unplaced,
//                       Transformation3D const &transformation,
//                       Vector3D<typename Backend::precision_v> const &point,
//                       typename Backend::int_v &inside)
//    {
//        
//      Vector3D<typename Backend::precision_v> localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
//      UnplacedInside<Backend>(unplaced, localPoint, inside);
//    }
//
//    //_________________________________________________________________
//    //CONTAINS --> just call UnplacedContains
//    /// \brief Contains: Inside method that does NOT take account of the surface for a Placed Hype
//    template <class Backend>
//    VECGEOM_INLINE
//    VECGEOM_CUDA_HEADER_BOTH
//    static void Contains(UnplacedHype const &unplaced,
//                       Transformation3D const &transformation,
//                       Vector3D<typename Backend::precision_v> const &point,
//                       Vector3D<typename Backend::precision_v> &localPoint,
//                       typename Backend::bool_v &inside)
//    {
//        
//        localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
//        UnplacedContains<Backend>(unplaced, localPoint, inside);
//        
//    }
//   
//    template <class Backend>
//    VECGEOM_INLINE
//    VECGEOM_CUDA_HEADER_BOTH
//    static void DistanceToIn(
//                             UnplacedHype const &unplaced,
//                             Transformation3D const &transformation,
//                             Vector3D<typename Backend::precision_v> const &point,
//                             Vector3D<typename Backend::precision_v> const &direction,
//                             typename Backend::precision_v const &stepMax,
//                             typename Backend::precision_v &distance) {
//        
//        
//        
//    }
//    
//    template <class Backend>
//    VECGEOM_INLINE
//    VECGEOM_CUDA_HEADER_BOTH
//    static void DistanceToOut(
//                              UnplacedHype const &unplaced,
//                              Vector3D<typename Backend::precision_v> point,
//                              Vector3D<typename Backend::precision_v> direction,
//                              typename Backend::precision_v const &stepMax,
//                              typename Backend::precision_v &distance) {
//      
//        
//
//    }
//
//    template<class Backend>
//    VECGEOM_INLINE
//    VECGEOM_CUDA_HEADER_BOTH
//    static void SafetyToIn(UnplacedHype const &unplaced,
//                         Transformation3D const &transformation,
//                         Vector3D<typename Backend::precision_v> const &point,
//                         typename Backend::precision_v &safety) {
//        
//            }
//    
//    template<class Backend>
//    VECGEOM_INLINE
//    VECGEOM_CUDA_HEADER_BOTH
//    static void SafetyToOut(UnplacedHype const &unplaced,
//                          Vector3D<typename Backend::precision_v> point,
//                          typename Backend::precision_v &safety) {
//
//           }
//
//};
//
//} // End global namespace
//
//#endif // VECGEOM_VOLUMES_KERNEL_HYPEIMPLEMENTATION_H_
