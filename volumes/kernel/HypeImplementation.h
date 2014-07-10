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
    
    namespace HypeUtilities
    {
        template <class Backend>
        VECGEOM_INLINE
        VECGEOM_CUDA_HEADER_BOTH
        void DistToHypeSurface(
                                 UnplacedHype const &unplaced,
                                 Vector3D<typename Backend::precision_v> const &point,
                                 Vector3D<typename Backend::precision_v> const &direction,
                                 typename Backend::precision_v &distance/*,
                                 typename Backend::bool_v in*/) {
                                 }
    }

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct HypeImplementation {

    /// \brief Inside method that takes account of the surface for an Unplaced Hype
    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void UnplacedInside(UnplacedHype const &unplaced,
                               Vector3D<typename Backend::precision_v> point,
                               typename Backend::int_v &inside) {
        
            }

    
    /// \brief UnplacedContains (ROOT STYLE): Inside method that does NOT take account of the surface for an Unplaced Hype
    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void UnplacedContains(UnplacedHype const &unplaced,
        Vector3D<typename Backend::precision_v> point,
        typename Backend::bool_v &inside) {
        
        
        
        
    }

    /// \brief Inside method that takes account of the surface for a Placed Hype
    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void Inside(UnplacedHype const &unplaced,
                       Transformation3D const &transformation,
                       Vector3D<typename Backend::precision_v> const &point,
                       Vector3D<typename Backend::precision_v> &localPoint,
                       typename Backend::int_v &inside) {
        localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
        UnplacedInside<Backend>(unplaced, localPoint, inside);
    }
    
    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void Inside(UnplacedHype const &unplaced,
                       Transformation3D const &transformation,
                       Vector3D<typename Backend::precision_v> const &point,
                       typename Backend::int_v &inside) {
        
      Vector3D<typename Backend::precision_v> localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
      UnplacedInside<Backend>(unplaced, localPoint, inside);
    }


    /// \brief Contains: Inside method that does NOT take account of the surface for a Placed Hype
    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void Contains(UnplacedHype const &unplaced,
                       Transformation3D const &transformation,
                       Vector3D<typename Backend::precision_v> const &point,
                       Vector3D<typename Backend::precision_v> &localPoint,
                       typename Backend::bool_v &inside) {
        
        localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
        UnplacedContains<Backend>(unplaced, localPoint, inside);
        
    }
    
    template <typename Backend, bool ForInside>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void GenericKernelForContainsAndInside(Vector3D<Precision> const &,
                                                  Vector3D<typename Backend::precision_v> const &,
                                                  typename Backend::bool_v &,
                                                  typename Backend::bool_v &);
    
    
    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void DistanceToIn(
                             UnplacedHype const &unplaced,
                             Transformation3D const &transformation,
                             Vector3D<typename Backend::precision_v> const &point,
                             Vector3D<typename Backend::precision_v> const &direction,
                             typename Backend::precision_v const &stepMax,
                             typename Backend::precision_v &distance) {
        
        
        
    }
    
    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void DistanceToOut(
                              UnplacedHype const &unplaced,
                              Vector3D<typename Backend::precision_v> point,
                              Vector3D<typename Backend::precision_v> direction,
                              typename Backend::precision_v const &stepMax,
                              typename Backend::precision_v &distance) {
      
        

    }

    template<class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void SafetyToIn(UnplacedHype const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety) {
        
            }
    
    template<class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void SafetyToOut(UnplacedHype const &unplaced,
                          Vector3D<typename Backend::precision_v> point,
                          typename Backend::precision_v &safety) {

           }

};

} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_HYPEIMPLEMENTATION_H_
