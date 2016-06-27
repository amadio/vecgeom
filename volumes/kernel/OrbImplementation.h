
/// @file OrbImplementation.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_ORBIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_ORBIMPLEMENTATION_H_

#include "base/Global.h"

#include "base/Transformation3D.h"
#include "volumes/kernel/BoxImplementation.h"
#include "volumes/UnplacedOrb.h"
#include "base/Vector3D.h"
#include "volumes/kernel/GenericKernels.h"

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(struct, OrbImplementation, TranslationCode, translation::kGeneric, RotationCode,
                                        rotation::kGeneric);

    inline namespace VECGEOM_IMPL_NAMESPACE
{

  class PlacedOrb;
  class UnplacedOrb;

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  struct OrbImplementation {

    static const int transC = transCodeT;
    static const int rotC   = rotCodeT;

    using PlacedShape_t   = PlacedOrb;
    using UnplacedShape_t = UnplacedOrb;

    VECGEOM_CUDA_HEADER_BOTH
    static void PrintType() { printf("SpecializedOrb<%i, %i>", transCodeT, rotCodeT); }

    template <typename Stream>
    static void PrintType(Stream &s)
    {
      s << "SpecializedOrb<" << transCodeT << "," << rotCodeT << ">";
    }

    template <typename Stream>
    static void PrintImplementationType(Stream &s)
    {
      s << "OrbImplementation<" << transCodeT << "," << rotCodeT << ">";
    }

    template <typename Stream>
    static void PrintUnplacedType(Stream &s)
    {
      s << "UnplacedOrb";
    }

    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void UnplacedContains(UnplacedOrb const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                                 typename Backend::bool_v &inside);

    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void Contains(UnplacedOrb const &unplaced, Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         Vector3D<typename Backend::precision_v> &localPoint, typename Backend::bool_v &inside);

    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void Inside(UnplacedOrb const &unplaced, Transformation3D const &transformation,
                       Vector3D<typename Backend::precision_v> const &point, typename Backend::inside_v &inside);

    template <typename Backend, bool ForInside>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void GenericKernelForContainsAndInside(UnplacedOrb const &unplaced,
                                                  Vector3D<typename Backend::precision_v> const &localPoint,
                                                  typename Backend::bool_v &completelyinside,
                                                  typename Backend::bool_v &completelyoutside);

    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void DistanceToIn(UnplacedOrb const &unplaced, Transformation3D const &transformation,
                             Vector3D<typename Backend::precision_v> const &point,
                             Vector3D<typename Backend::precision_v> const &direction,
                             typename Backend::precision_v const &stepMax, typename Backend::precision_v &distance);

    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void DistanceToOut(UnplacedOrb const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                              Vector3D<typename Backend::precision_v> const &direction,
                              typename Backend::precision_v const &stepMax, typename Backend::precision_v &distance);

    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void SafetyToIn(UnplacedOrb const &unplaced, Transformation3D const &transformation,
                           Vector3D<typename Backend::precision_v> const &point, typename Backend::precision_v &safety);

    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void SafetyToOut(UnplacedOrb const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                            typename Backend::precision_v &safety);

    template <typename Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void ContainsKernel(UnplacedOrb const &unplaced, Vector3D<typename Backend::precision_v> const &localPoint,
                               typename Backend::bool_v &inside);

    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void InsideKernel(UnplacedOrb const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                             typename Backend::inside_v &inside);

    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void DistanceToInKernel(UnplacedOrb const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                                   Vector3D<typename Backend::precision_v> const &direction,
                                   typename Backend::precision_v const &stepMax,
                                   typename Backend::precision_v &distance);

    template <class Backend, bool ForDistanceToIn>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static typename Backend::bool_v DetectIntersectionAndCalculateDistance(
        UnplacedOrb const &unplaced, Vector3D<typename Backend::precision_v> const &point,
        Vector3D<typename Backend::precision_v> const &direction, typename Backend::precision_v &distance);

    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void DistanceToOutKernel(UnplacedOrb const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                                    Vector3D<typename Backend::precision_v> const &direction,
                                    typename Backend::precision_v const &stepMax,
                                    typename Backend::precision_v &distance);

    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void SafetyToInKernel(UnplacedOrb const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                                 typename Backend::precision_v &safety);

    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void SafetyToOutKernel(UnplacedOrb const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                                  typename Backend::precision_v &safety);

    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void Normal(UnplacedOrb const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                       Vector3D<typename Backend::precision_v> &normal, typename Backend::bool_v &valid);

    template <class Backend>
    VECGEOM_INLINE
    VECGEOM_CUDA_HEADER_BOTH
    static void NormalKernel(UnplacedOrb const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                             Vector3D<typename Backend::precision_v> &normal, typename Backend::bool_v &valid);
  };

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  void OrbImplementation<transCodeT, rotCodeT>::Normal(
      UnplacedOrb const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &normal, typename Backend::bool_v &valid)
  {

    NormalKernel<Backend>(unplaced, point, normal, valid);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  void OrbImplementation<transCodeT, rotCodeT>::NormalKernel(
      UnplacedOrb const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &normal, typename Backend::bool_v &valid)
  {

    typedef typename Backend::precision_v Float_t;
    Float_t rad2      = point.Mag2();
    Float_t invRadius = Float_t(1.) / Sqrt(rad2);
    normal            = point * invRadius;
    Float_t tolRMaxP  = unplaced.GetfRTolO();
    Float_t tolRMaxM  = unplaced.GetfRTolI();

    // Check radial surface
    valid = ((rad2 <= tolRMaxP * tolRMaxP) && (rad2 >= tolRMaxM * tolRMaxM)); // means we are on surface
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  void OrbImplementation<transCodeT, rotCodeT>::UnplacedContains(UnplacedOrb const &unplaced,
                                                                 Vector3D<typename Backend::precision_v> const &point,
                                                                 typename Backend::bool_v &inside)
  {

    ContainsKernel<Backend>(unplaced, point, inside);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  void OrbImplementation<transCodeT, rotCodeT>::Contains(
      UnplacedOrb const &unplaced, Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point, Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside)
  {

    localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
    UnplacedContains<Backend>(unplaced, localPoint, inside);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  void OrbImplementation<transCodeT, rotCodeT>::Inside(
      UnplacedOrb const &unplaced, Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point, typename Backend::inside_v &inside)
  {

    InsideKernel<Backend>(unplaced, transformation.Transform<transCodeT, rotCodeT>(point), inside);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  void OrbImplementation<transCodeT, rotCodeT>::DistanceToIn(
      UnplacedOrb const &unplaced, Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point, Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax, typename Backend::precision_v &distance)
  {

    DistanceToInKernel<Backend>(unplaced, transformation.Transform<transCodeT, rotCodeT>(point),
                                transformation.TransformDirection<rotCodeT>(direction), stepMax, distance);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  void OrbImplementation<transCodeT, rotCodeT>::DistanceToOut(
      UnplacedOrb const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction, typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance)
  {

    DistanceToOutKernel<Backend>(unplaced, point, direction, stepMax, distance);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <class Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void OrbImplementation<transCodeT, rotCodeT>::SafetyToIn(
      UnplacedOrb const &unplaced, Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point, typename Backend::precision_v &safety)
  {

    SafetyToInKernel<Backend>(unplaced, transformation.Transform<transCodeT, rotCodeT>(point), safety);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <class Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  void OrbImplementation<transCodeT, rotCodeT>::SafetyToOut(UnplacedOrb const &unplaced,
                                                            Vector3D<typename Backend::precision_v> const &point,
                                                            typename Backend::precision_v &safety)
  {

    SafetyToOutKernel<Backend>(unplaced, point, safety);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH
  void OrbImplementation<transCodeT, rotCodeT>::ContainsKernel(
      UnplacedOrb const &unplaced, Vector3D<typename Backend::precision_v> const &localPoint,
      typename Backend::bool_v &inside)
  {

    typedef typename Backend::bool_v Bool_t;
    Bool_t unused;
    Bool_t outside;
    GenericKernelForContainsAndInside<Backend, false>(unplaced, localPoint, unused, outside);
    inside = !outside;
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <typename Backend, bool ForInside>
  VECGEOM_CUDA_HEADER_BOTH
  void OrbImplementation<transCodeT, rotCodeT>::GenericKernelForContainsAndInside(
      UnplacedOrb const &unplaced, Vector3D<typename Backend::precision_v> const &localPoint,
      typename Backend::bool_v &completelyinside, typename Backend::bool_v &completelyoutside)
  {

    typedef typename Backend::precision_v Float_t;
    Precision fR                    = unplaced.GetRadius();
    Float_t rad2                    = localPoint.Mag2();
    Float_t tolR                    = fR - kTolerance;
    if (ForInside) completelyinside = (rad2 <= tolR * tolR);
    tolR                            = fR + kTolerance;
    completelyoutside               = (rad2 >= tolR * tolR);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  void OrbImplementation<transCodeT, rotCodeT>::InsideKernel(UnplacedOrb const &unplaced,
                                                             Vector3D<typename Backend::precision_v> const &point,
                                                             typename Backend::inside_v &inside)
  {

    typedef typename Backend::bool_v Bool_t;
    Bool_t completelyinside, completelyoutside;
    GenericKernelForContainsAndInside<Backend, true>(unplaced, point, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    MaskedAssign(completelyoutside, EInside::kOutside, &inside);
    MaskedAssign(completelyinside, EInside::kInside, &inside);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <class Backend, bool ForDistanceToIn>
  VECGEOM_CUDA_HEADER_BOTH
  typename Backend::bool_v OrbImplementation<transCodeT, rotCodeT>::DetectIntersectionAndCalculateDistance(
      UnplacedOrb const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction, typename Backend::precision_v &distance)
  {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;
    Float_t rad2    = point.Mag2();
    Float_t pDotV3D = point.Dot(direction);
    Precision fR    = unplaced.GetRadius();
    Float_t c       = rad2 - fR * fR;
    Float_t d2      = (pDotV3D * pDotV3D - c);

    Bool_t cond = ((d2 >= 0.) && (pDotV3D <= 0.));
    if (ForDistanceToIn) {
      MaskedAssign(cond, (-pDotV3D - Sqrt(d2)), &distance);
      return cond;
    } else {
      MaskedAssign((d2 >= 0.), (-pDotV3D + Sqrt(d2)), &distance);
      return (d2 >= 0.);
    }
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  void OrbImplementation<transCodeT, rotCodeT>::DistanceToInKernel(
      UnplacedOrb const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction, typename Backend::precision_v const & /*stepMax*/,
      typename Backend::precision_v &distance)
  {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;
    distance             = kInfinity;
    Float_t rad          = point.Mag();
    Bool_t isPointInside = (rad < (unplaced.GetRadius() - kTolerance));
    MaskedAssign(isPointInside, -1., &distance);
    Bool_t done = isPointInside;
    if (IsFull(done)) return;

    Float_t pDotV3D = point.Dot(direction);
    Bool_t isPointOnSurface =
        (rad >= (unplaced.GetRadius() - kTolerance)) && (rad <= (unplaced.GetRadius() + kTolerance));
    Bool_t cond = (isPointOnSurface && (pDotV3D < 0.));
    MaskedAssign(!done && cond, 0., &distance);
    done |= cond;
    if (IsFull(done)) return;
    Float_t dist(kInfinity);
    MaskedAssign(!done && DetectIntersectionAndCalculateDistance<Backend, true>(unplaced, point, direction, dist), dist,
                 &distance);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  void OrbImplementation<transCodeT, rotCodeT>::DistanceToOutKernel(
      UnplacedOrb const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction, typename Backend::precision_v const & /*stepMax*/,
      typename Backend::precision_v &distance)
  {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    distance = kInfinity;

    Float_t rad           = point.Mag();
    Bool_t isPointOutside = (rad > (unplaced.GetRadius() + kTolerance));
    MaskedAssign(isPointOutside, -1., &distance);
    Bool_t done = isPointOutside;
    if (IsFull(done)) return;

    Float_t pDotV3D = point.Dot(direction);
    Bool_t isPointOnSurface =
        (rad >= (unplaced.GetRadius() - kTolerance)) && (rad <= (unplaced.GetRadius() + kTolerance));
    Bool_t cond = (isPointOnSurface && (pDotV3D > 0.));
    MaskedAssign(!done && cond, 0., &distance);
    done |= cond;
    if (IsFull(done)) return;
    Float_t dist(kInfinity);
    MaskedAssign(!done && DetectIntersectionAndCalculateDistance<Backend, false>(unplaced, point, direction, dist),
                 dist, &distance);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  void OrbImplementation<transCodeT, rotCodeT>::SafetyToInKernel(UnplacedOrb const &unplaced,
                                                                 Vector3D<typename Backend::precision_v> const &point,
                                                                 typename Backend::precision_v &safety)
  {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    Float_t rad          = point.Mag();
    safety               = rad - unplaced.GetRadius();
    Bool_t isPointInside = (rad < (unplaced.GetRadius() - kTolerance));
    MaskedAssign(isPointInside, -1., &safety);
    if (IsFull(isPointInside)) return;

    Bool_t isPointOnSurface =
        (rad > (unplaced.GetRadius() - kTolerance)) && (rad < (unplaced.GetRadius() + kTolerance));
    MaskedAssign(isPointOnSurface, 0., &safety);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  void OrbImplementation<transCodeT, rotCodeT>::SafetyToOutKernel(UnplacedOrb const &unplaced,
                                                                  Vector3D<typename Backend::precision_v> const &point,
                                                                  typename Backend::precision_v &safety)
  {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    Float_t rad = point.Mag();
    safety      = unplaced.GetRadius() - rad;

    Bool_t isPointOutside = (rad > (unplaced.GetRadius() + kTolerance));
    MaskedAssign(isPointOutside, -1., &safety);
    if (IsFull(isPointOutside)) return;

    Bool_t isPointOnSurface =
        (rad > (unplaced.GetRadius() - kTolerance)) && (rad < (unplaced.GetRadius() + kTolerance));
    MaskedAssign(isPointOnSurface, 0., &safety);
  }
}
} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_ORBIMPLEMENTATION_H_
