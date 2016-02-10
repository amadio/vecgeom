/*
 * GenTrapImplementation.h
 *
 *  Created on: Aug 2, 2014
 *      Author: swenzel
 *   Review/completion: Nov 4, 2015
 *      Author: mgheata
 */

#ifndef VECGEOM_VOLUMES_KERNEL_GENTRAPIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_GENTRAPIMPLEMENTATION_H_

#include "base/Global.h"
#include "base/Transformation3D.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/kernel/BoxImplementation.h"
#include "volumes/UnplacedGenTrap.h"
#include "backend/Backend.h"
#include <iostream>

namespace vecgeom {
VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(GenTrapImplementation, TranslationCode, translation::kGeneric, RotationCode,
                                        rotation::kGeneric)

    inline namespace VECGEOM_IMPL_NAMESPACE {

  class PlacedGenTrap;

  template <TranslationCode transCodeT, RotationCode rotCodeT> struct GenTrapImplementation {

    static const int transC = transCodeT;
    static const int rotC = rotCodeT;

    using Vertex_t = Vector3D<Precision>;
    using PlacedShape_t = PlacedGenTrap;
    using UnplacedShape_t = UnplacedGenTrap;

    VECGEOM_CUDA_HEADER_BOTH
    static void PrintType() { printf("SpecializedGenTrap<%i, %i>", transCodeT, rotCodeT); }

    template <typename Backend>
    VECGEOM_INLINE VECGEOM_CUDA_HEADER_BOTH static void
    Contains(UnplacedGenTrap const &unplaced, Transformation3D const &transformation,
             Vector3D<typename Backend::precision_v> const &point, Vector3D<typename Backend::precision_v> &localPoint,
             typename Backend::bool_v &inside);

    template <typename Backend>
    VECGEOM_INLINE VECGEOM_CUDA_HEADER_BOTH static void
    Inside(UnplacedGenTrap const &unplaced, Transformation3D const &transformation,
           Vector3D<typename Backend::precision_v> const &point, typename Backend::inside_v &inside);

    template <typename Backend, bool ForInside>
    VECGEOM_INLINE VECGEOM_CUDA_HEADER_BOTH static void
    GenericKernelForContainsAndInside(UnplacedGenTrap const &, Vector3D<typename Backend::precision_v> const &,
                                      typename Backend::bool_v &, typename Backend::bool_v &);

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE static void
    DistanceToIn(UnplacedGenTrap const &unplaced, Transformation3D const &transformation,
                 Vector3D<typename Backend::precision_v> const &point,
                 Vector3D<typename Backend::precision_v> const &direction, typename Backend::precision_v const &stepMax,
                 typename Backend::precision_v &distance);

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE static void
    DistanceToOut(UnplacedGenTrap const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                  Vector3D<typename Backend::precision_v> const &direction,
                  typename Backend::precision_v const &stepMax, typename Backend::precision_v &distance);

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE static void
    SafetyToIn(UnplacedGenTrap const &unplaced, Transformation3D const &transformation,
               Vector3D<typename Backend::precision_v> const &point, typename Backend::precision_v &safety);

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE static void
    SafetyToOut(UnplacedGenTrap const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                typename Backend::precision_v &safety);

    template <typename Backend>
    VECGEOM_INLINE VECGEOM_CUDA_HEADER_BOTH static void
    UnplacedContains(UnplacedGenTrap const &box, Vector3D<typename Backend::precision_v> const &localPoint,
                     typename Backend::bool_v &inside);

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE static void
    ContainsKernel(UnplacedGenTrap const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                   typename Backend::bool_v &inside);

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE static void
    InsideKernel(UnplacedGenTrap const &boxDimensions, Vector3D<typename Backend::precision_v> const &point,
                 typename Backend::inside_v &inside);

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE static void
    DistanceToInKernel(UnplacedGenTrap const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                       Vector3D<typename Backend::precision_v> const &direction,
                       typename Backend::precision_v const &stepMax, typename Backend::precision_v &distance);

    template <class Backend, bool treatNormal>
    VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE static void
    DistanceToOutKernel(UnplacedGenTrap const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                        Vector3D<typename Backend::precision_v> const &direction,
                        typename Backend::precision_v const &stepMax, typename Backend::precision_v &distance);

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE static void
    SafetyToInKernel(UnplacedGenTrap const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                     typename Backend::precision_v &safety);

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE static void
    SafetyToOutKernel(UnplacedGenTrap const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                      typename Backend::precision_v &safety);

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE static void
    NormalKernel(UnplacedGenTrap const &unplaced, Vector3D<typename Backend::precision_v> const &point,
                 Vector3D<typename Backend::precision_v> &normal, typename Backend::bool_v &valid);

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH static void
    GetClosestEdge(Vector3D<typename Backend::precision_v> const &point, typename Backend::precision_v vertexX[4],
                   typename Backend::precision_v vertexY[4], typename Backend::precision_v &iseg,
                   typename Backend::precision_v &fraction);

  }; // End struct GenTrapImplementation

  //********************************
  //**** implementations start here
  //********************************/

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH void GenTrapImplementation<transCodeT, rotCodeT>::Contains(
      UnplacedGenTrap const &unplaced, Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point, Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside) {

    localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
    ContainsKernel<Backend>(unplaced, localPoint, inside);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH void GenTrapImplementation<transCodeT, rotCodeT>::UnplacedContains(
      UnplacedGenTrap const &box, Vector3D<typename Backend::precision_v> const &localPoint,
      typename Backend::bool_v &inside) {

    ContainsKernel<Backend>(box, localPoint, inside);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH void GenTrapImplementation<transCodeT, rotCodeT>::Inside(
      UnplacedGenTrap const &unplaced, Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point, typename Backend::inside_v &inside) {

    InsideKernel<Backend>(unplaced, transformation.Transform<transCodeT, rotCodeT>(point), inside);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH void GenTrapImplementation<transCodeT, rotCodeT>::DistanceToIn(
      UnplacedGenTrap const &unplaced, Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point, Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax, typename Backend::precision_v &distance) {

    DistanceToInKernel<Backend>(unplaced, transformation.Transform<transCodeT, rotCodeT>(point),
                                transformation.TransformDirection<rotCodeT>(direction), stepMax, distance);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH void GenTrapImplementation<transCodeT, rotCodeT>::DistanceToOut(
      UnplacedGenTrap const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction, typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance) {

    DistanceToOutKernel<Backend, false>(unplaced, point, direction, stepMax, distance);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE void GenTrapImplementation<transCodeT, rotCodeT>::SafetyToIn(
      UnplacedGenTrap const &unplaced, Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point, typename Backend::precision_v &safety) {

    SafetyToInKernel<Backend>(unplaced, transformation.Transform<transCodeT, rotCodeT>(point), safety);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE void GenTrapImplementation<transCodeT, rotCodeT>::SafetyToOut(
      UnplacedGenTrap const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v &safety) {

    SafetyToOutKernel<Backend>(unplaced, point, safety);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE void GenTrapImplementation<transCodeT, rotCodeT>::ContainsKernel(
      UnplacedGenTrap const &unplaced, Vector3D<typename Backend::precision_v> const &localPoint,
      typename Backend::bool_v &inside) {

    typedef typename Backend::bool_v Bool_t;
    Bool_t unused;
    Bool_t outside;
    GenericKernelForContainsAndInside<Backend, false>(unplaced, localPoint, unused, outside);
    inside = !outside;
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <typename Backend, bool ForInside>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE void
  GenTrapImplementation<transCodeT, rotCodeT>::GenericKernelForContainsAndInside(
      UnplacedGenTrap const &unplaced, Vector3D<typename Backend::precision_v> const &localPoint,
      typename Backend::bool_v &completelyinside, typename Backend::bool_v &completelyoutside) {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;
    constexpr Precision tolerancesq = 10000. * kTolerance * kTolerance;
    // Add stronger check against the bounding box, which can allow early returns if point is outside.
    // Local point has to be translated in the bbox local frame.
    BoxImplementation<translation::kIdentity, rotation::kIdentity>::GenericKernelForContainsAndInside<Backend,
                                                                                                      ForInside>(
        unplaced.fBBdimensions, localPoint - unplaced.fBBorigin, completelyinside, completelyoutside);
    //  if (Backend::early_returns) {
    if (IsFull(completelyoutside)) {
      return;
    }
    //  }

    //  if (ForInside)  {
    //    completelyinside = Abs(localPoint.z()) < MakeMinusTolerant<ForInside>( unplaced.fDz );
    //  }

    // analyse z
    Float_t cf = unplaced.fHalfInverseDz * (unplaced.fDz - localPoint.z());
    // analyse if x-y coordinates of localPoint are within polygon at z-height

    //  loop over edges connecting points i with i+4
    Float_t vertexX[4];
    Float_t vertexY[4];
    // vectorizes for scalar backend
    for (int i = 0; i < 4; i++) {
      // calculate x-y positions of vertex i at this z-height
      vertexX[i] = unplaced.fVerticesX[i + 4] + cf * unplaced.fConnectingComponentsX[i];
      vertexY[i] = unplaced.fVerticesY[i + 4] + cf * unplaced.fConnectingComponentsY[i];
    }

    // I currently found out that it is beneficial to keep the early return
    // in disfavor of the vectorizing solution; should be reinvestigated on AVX
    for (int i = 0; i < 4; i++) {
      // this is based on the following idea:
      // we decided for each edge whether the point is above or below the
      // 2d line defined by that edge
      // In fact, this calculation is part of the calculation of the distance
      // of localPoint to that line which is a cross product. In this case it is
      // an embedded cross product of 2D vectors in 3D. The resulting vector always points
      // in z-direction whose z-magnitude is directly related to the distance.
      // see, e.g.,  http://geomalgorithms.com/a02-_lines.html
      int j = (i + 1) % 4;
      Float_t DeltaX = vertexX[j] - vertexX[i];
      Float_t DeltaY = vertexY[j] - vertexY[i];
      Float_t cross = (localPoint.x() - vertexX[i]) * DeltaY - (localPoint.y() - vertexY[i]) * DeltaX;
      if (ForInside) {
        Bool_t onsurf = (cross * cross < tolerancesq * (DeltaX * DeltaX + DeltaY * DeltaY));
        completelyoutside |= ((cross < 0.) && (!onsurf));
        completelyinside &= ((cross > 0.) && (!onsurf));
      } else {
        completelyoutside |= (cross < MakeMinusTolerant<ForInside>(0));
      }

      //    if (Backend::early_returns) {
      if (IsFull(completelyoutside)) {
        return;
      }
      //    }
    }
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE void GenTrapImplementation<transCodeT, rotCodeT>::InsideKernel(
      UnplacedGenTrap const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside) {

    typedef typename Backend::bool_v Bool_t;
    Bool_t completelyinside;
    Bool_t completelyoutside;
    GenericKernelForContainsAndInside<Backend, true>(unplaced, point, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    MaskedAssign(completelyoutside, EInside::kOutside, &inside);
    MaskedAssign(completelyinside, EInside::kInside, &inside);
  }

  template <bool IsSIMD, class Backend> struct FillPlaneDataHelper {
    VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    static void FillPlaneData(UnplacedGenTrap const &unplaced, typename Backend::precision_v &cornerx,
                              typename Backend::precision_v &cornery, typename Backend::precision_v &deltax,
                              typename Backend::precision_v &deltay, typename Backend::bool_v const &top,
                              int edgeindex) {

      // no vectorized data lookup for SIMD
      // need to fill the SIMD types individually

      // now we only need to get the number 2 from somewhere
      for (int i = 0; i < kVectorSize; ++i) {
        int index = edgeindex + top[i] * 4;
        deltax[i] = unplaced.fDeltaX[index];
        deltay[i] = unplaced.fDeltaY[index];
        cornerx[i] = unplaced.fVerticesX[index];
        cornery[i] = unplaced.fVerticesY[index];
      }
    }
  };

  // a partial template specialization for nonSIMD cases (scalar, cuda, ... )
  template <class Backend> struct FillPlaneDataHelper<false, Backend> {
    VECGEOM_CUDA_HEADER_BOTH
    static void FillPlaneData(UnplacedGenTrap const &unplaced, typename Backend::precision_v &cornerx,
                              typename Backend::precision_v &cornery, typename Backend::precision_v &deltax,
                              typename Backend::precision_v &deltay, typename Backend::bool_v const &top,
                              int edgeindex) {
      int index = edgeindex + top * 4;
      deltax = unplaced.fDeltaX[index];
      deltay = unplaced.fDeltaY[index];
      cornerx = unplaced.fVerticesX[index];
      cornery = unplaced.fVerticesY[index];
    }
  };

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE
      // optimized "inside" check for top or bottom z-surfaces
      // this is a bit tricky if different tracks check different planes
      // ( for example in case of Backend = Vc when top is mixed )
      // ( this is because vector data lookup is tricky )
      typename Backend::bool_v
      IsInTopOrBottomPolygon(UnplacedGenTrap const &unplaced, typename Backend::precision_v const &pointx,
                             typename Backend::precision_v const &pointy, typename Backend::bool_v top) {
    // stripped down version of the Contains kernel ( not yet shared with that kernel )
    typedef typename Backend::bool_v Bool_t;
    typedef typename Backend::precision_v Float_t;
    // std::cerr << "IsInTopOrBottom: pointx: " << pointx << "  pointy: " << pointy << "  top: " << top << "\n";

    Bool_t completelyoutside(Backend::kFalse);
    for (int i = 0; i < 4; ++i) {
      Float_t deltaX;
      Float_t deltaY;
      Float_t cornerX;
      Float_t cornerY;

      // thats the only place where scalar and vector code diverge
      // IsSIMD misses...replaced with early_returns
      FillPlaneDataHelper<!Backend::early_returns, Backend>::FillPlaneData(unplaced, cornerX, cornerY, deltaX, deltaY,
                                                                           top, i);

      // std::cerr << i << " CORNERS " << cornerX << " " << cornerY << " " << deltaX << " " << deltaY << "\n";

      Float_t cross = (pointx - cornerX) * deltaY;
      cross -= (pointy - cornerY) * deltaX;

      completelyoutside |= cross < MakeMinusTolerant<true>(0.);
      if (Backend::early_returns) {
        if (IsFull(completelyoutside)) {
          return Backend::kFalse;
        }
      }
    }
    return !completelyoutside;
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE void GenTrapImplementation<transCodeT, rotCodeT>::DistanceToInKernel(
      UnplacedGenTrap const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction, typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance) {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

//#define GENTRAPDEB = 1
#ifdef GENTRAPDEB
    std::cerr << "point: " << point << std::endl;
    std::cerr << "direction: " << direction << std::endl;
#endif
    // do a quick boundary box check (Arb8, USolids) is doing this
    // unplaced.GetBBox();
    // actually this could also give us some indication which face is likely to be hit

    // let me see if we can temporarily force the box function to be a function call
    Float_t bbdistance = 0.;
#ifdef GENTRAP_USENEWBB
    typename Backend::int_v planeid(-1);
    Vector3D<Float_t> hitpoint;
    BoxImplementation<translation::kIdentity, rotation::kIdentity>::DistanceToInKernel2<Backend>(
        unplaced.fBBdimensions, point - unplaced.fBBorigin, direction, stepMax, bbdistance, &planeid, &hitpoint);
#else
    BoxImplementation<translation::kIdentity, rotation::kIdentity>::DistanceToInKernel<Backend>(
        unplaced.fBBdimensions, point - unplaced.fBBorigin, direction, stepMax, bbdistance);
#endif

#ifdef GENTRAPDEB
    std::cerr << "BB gave " << bbdistance << "\n";
#endif
    distance = kInfinity;

    // do a check on bbdistance
    // if none of the tracks can hit even the bounding box; just return
    Bool_t done = bbdistance >= kInfinity;
    if (IsFull(done))
      return;
#ifdef GENTRAPDEB
    Float_t x, y, z;
    x = point.x() + bbdistance * direction.x();
    y = point.y() + bbdistance * direction.y();
    z = point.z() + bbdistance * direction.z();
    std::cerr << "prolongated to box:  x " << x << " y " << y << " z " << z << "\n";
#endif

/*some particle could hit z*/
#ifdef GENTRAP_USENEWBB
#pragma message("WITH IMPROVED BB")
    if (!IsEmpty(planeid == 2)) {
      Bool_t top = direction.z() < 0;
      Bool_t hits = IsInTopOrBottomPolygon<Backend>(unplaced, hitpoint.x(), hitpoint.y(), top);
#ifdef GENTRAPDEB
      std::cerr << " top/bottom hit result " << hits << " \n";
#endif
      MaskedAssign(hits, bbdistance, &distance);
      done |= hits;
      if (IsFull(done))
        return;
    }
#else // do this check again
    //#pragma message("WITH OLD BB")
    // IDEA: we don't need to do this if bbox does not hit z planes
    // IDEA2: and if bbbox hits z-plane everything here is already calculated
    //  Float_t snext;
    Float_t zsafety = Abs(point.z()) - unplaced.fDz;
    Bool_t canhitz = zsafety > MakeMinusTolerant<true>(0.);
    canhitz &= point.z() * direction.z() < 0; // coming towards the origin
    canhitz &= !done;
#ifdef GENTRAPDEB
    std::cerr << " canhitz " << canhitz << " \n";
#endif

    if (!IsEmpty(canhitz)) {
      //   std::cerr << "can potentially hit\n";
      // calculate distance to z-plane ( see Box algorithm )
      // check if hit point is inside top or bottom polygon
      Float_t next = zsafety / Abs(direction.z() + kTiny);
#ifdef GENTRAPDEB
      std::cerr << " zdist " << next << "\n";
#endif
      // transport to z-height of planes
      Float_t coord1 = point.x() + next * direction.x();
      Float_t coord2 = point.y() + next * direction.y();
      Bool_t top = direction.z() < 0;
      // Bool_t hits = IsInTopOrBottomPolygon<Backend>(unplaced, hitpoint.x(), hitpoint.y(), top );
      Bool_t hits = IsInTopOrBottomPolygon<Backend>(unplaced, coord1, coord2, top);
      hits &= canhitz;
      MaskedAssign(hits, bbdistance, &distance);
      done |= hits;
#ifdef GENTRAPDEB
      std::cerr << " hit result " << hits << " bbdistance " << distance << "\n";
#endif

      if (IsFull(done))
        return;
    }
#endif

    // now treat lateral surfaces
    Float_t disttoplanes = unplaced.GetShell().DistanceToIn<Backend>(point, direction, done);
#ifdef GENTRAPDEB
    std::cerr << "disttoplanes " << disttoplanes << "\n";
#endif

    MaskedAssign(!done, Min(disttoplanes, distance), &distance);
#ifdef GENTRAPDEB
    std::cerr << distance << "\n";
#endif
    //  std::cerr << std::endl;
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <class Backend, bool treatNormal>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE void GenTrapImplementation<transCodeT, rotCodeT>::DistanceToOutKernel(
      UnplacedGenTrap const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction, typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance) {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    // we should check here the compilation condition
    // that treatNormal=true can only happen when Backend=kScalar
    // TODO: do this with some nice template features

    Bool_t negDirMask = direction.z() < 0;
    Float_t sign = 1.;
    MaskedAssign(negDirMask, -1., &sign);
    //    Float_t invDirZ = 1./direction.z();
    // this construct costs one multiplication more
    Float_t distmin = (sign * unplaced.fDz - point.z()) / direction.z();

    Float_t distplane = unplaced.GetShell().DistanceToOut<Backend>(point, direction);
    distance = Min(distmin, distplane);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE void GenTrapImplementation<transCodeT, rotCodeT>::SafetyToInKernel(
      UnplacedGenTrap const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v &safety) {

    typedef typename Backend::bool_v Boolean_t;

    Boolean_t inside;
    // Check if all points are outside bounding box
    BoxImplementation<translation::kIdentity, rotation::kIdentity>::ContainsKernel<Backend>(
        unplaced.fBBdimensions, point - unplaced.fBBorigin, inside);
    if (IsEmpty(inside)) {
      // All points outside, so compute safety using the bounding box
      // This is not optimal if top and bottom faces are not on top of each other
      BoxImplementation<translation::kIdentity, rotation::kIdentity>::SafetyToInKernel<Backend>(
          unplaced.fBBdimensions, point - unplaced.fBBorigin, safety);
      return;
    }

    // Do Z
    safety = Abs(point[2]) - unplaced.GetDZ();
    safety = unplaced.GetShell().SafetyToIn<Backend>(point, safety);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH void GenTrapImplementation<transCodeT, rotCodeT>::SafetyToOutKernel(
      UnplacedGenTrap const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v &safety) {

    // Do Z
    safety = unplaced.GetDZ() - Abs(point[2]);
    safety = unplaced.GetShell().SafetyToOut<Backend>(point, safety);
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH void GenTrapImplementation<transCodeT, rotCodeT>::NormalKernel(
      UnplacedGenTrap const &unplaced, Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &normal, typename Backend::bool_v &valid) {

    // Computes the normal on a surface and returns it as a unit vector
    //   In case a point is further than tolerance_normal from a surface, set validNormal=false
    //   Must return a valid vector. (even if the point is not on the surface.)
    //
    //   On an edge or corner, provide an average normal of all facets within tolerance
    // NOTE: the tolerance value used in here is not yet the global surface
    //     tolerance - we will have to revise this value - TODO
    // this version does not yet consider the case when we are not on the surface

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::int_v Int_t;
    typedef typename Backend::bool_v Bool_t;
    valid = Backend::kTrue;
    normal.Set(0., 0., 0.);
    // Do bottom and top faces
    Float_t safz = Abs(unplaced.GetDZ() - Abs(point.z()));
    Bool_t onZ = (safz < 10. * kTolerance);
    MaskedAssign(onZ && (point.z() > 0), 1., &normal[2]);
    MaskedAssign(onZ && (point.z() < 0), -1., &normal[2]);

    //    if (Backend::early_returns) {
    if (IsFull(onZ)) {
      return;
    }
    //    }
    //  Float_t done = onZ;
    // Get the closest edge (point should be on this edge within tolerance)
    Float_t cf = unplaced.fHalfInverseDz * (unplaced.fDz - point.z());
    Float_t vertexX[4];
    Float_t vertexY[4];
    for (int i = 0; i < 4; i++) {
      // calculate x-y positions of vertex i at this z-height
      vertexX[i] = unplaced.fVerticesX[i + 4] + cf * unplaced.fConnectingComponentsX[i];
      vertexY[i] = unplaced.fVerticesY[i + 4] + cf * unplaced.fConnectingComponentsY[i];
    }
    Float_t seg;
    Float_t frac;
    GetClosestEdge<Backend>(point, vertexX, vertexY, seg, frac);
    MaskedAssign(frac < 0., 0., &frac);
    Int_t iseg = seg;
    if (unplaced.IsPlanar()) {
      // Normals for the planar case are pre-computed
      Vertex_t const *normals = unplaced.GetShell().GetNormals();
      normal = normals[iseg];
      return;
    }
    Int_t jseg = (iseg + 1) % 4;
    Float_t x0 = vertexX[iseg];
    Float_t y0 = vertexY[iseg];
    Float_t x2 = vertexX[jseg];
    Float_t y2 = vertexY[jseg];
    x0 += frac * (x2 - x0);
    y0 += frac * (y2 - y0);
    Float_t x1 = unplaced.fVerticesX[iseg + 4];
    Float_t y1 = unplaced.fVerticesY[iseg + 4];
    x1 += frac * (unplaced.fVerticesX[jseg + 4] - x1);
    y1 += frac * (unplaced.fVerticesY[jseg + 4] - y1);
    Float_t ax = x1 - x0;
    Float_t ay = y1 - y0;
    Float_t az = unplaced.GetDZ() - point.z();
    Float_t bx = x2 - x0;
    Float_t by = y2 - y0;
    Float_t bz = 0.;
    // Cross product of the vector given by the section segment (that contains the
    // point) at z=point[2] and the vector connecting the point projection to its
    // correspondent on the top edge.
    normal.Set(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx);
    normal.Normalize();
  }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH void GenTrapImplementation<transCodeT, rotCodeT>::GetClosestEdge(
      Vector3D<typename Backend::precision_v> const &point, typename Backend::precision_v vertexX[4],
      typename Backend::precision_v vertexY[4], typename Backend::precision_v &iseg,
      typename Backend::precision_v &fraction) {
    /// Get index of the edge of the quadrilater represented by vert closest to point.
    /// If [P1,P2] is the closest segment and P is the point, the function returns the fraction of the
    /// projection of (P1P) over (P1P2). If projection of P is not in range [P1,P2] return -1.
    typedef typename Backend::precision_v Float_t;
    //  typedef typename Backend::int_v Int_t;
    typedef typename Backend::bool_v Bool_t;
    iseg = 0.;
    //  Float_t p1X, p1Y, p2X, p2Y;
    Float_t lsq, dx, dy, dpx, dpy, u;
    fraction = -1.;
    Float_t safe = kInfinity;
    Float_t ssq = kInfinity;
    for (int i = 0; i < 4; ++i) {
      int j = (i + 1) % 4;
      dx = vertexX[j] - vertexX[i];
      dy = vertexY[j] - vertexY[i];
      dpx = point.x() - vertexX[i];
      dpy = point.y() - vertexY[i];
      lsq = dx * dx + dy * dy;
      // Current segment collapsed to a point
      Bool_t collapsed = lsq < kTolerance;
      if (!IsEmpty(collapsed)) {
        MaskedAssign(lsq < kTolerance, dpx * dpx + dpy * dpy, &ssq);
        // Missing a masked assign allowing to perform multiple assignments...
        MaskedAssign(ssq < safe, (Precision)i, &iseg);
        MaskedAssign(ssq < safe, -1., &fraction);
        MaskedAssign(ssq < safe, ssq, &safe);
        if (IsFull(collapsed))
          continue;
      }
      // Projection fraction
      u = (dpx * dx + dpy * dy) / (lsq + kTiny);
      MaskedAssign(u > 1 && !collapsed, point.x() - vertexX[j], &dpx);
      MaskedAssign(u > 1 && !collapsed, point.y() - vertexY[j], &dpy);
      MaskedAssign(u >= 0 && u <= 1 && !collapsed, dpx - u * dx, &dpx);
      MaskedAssign(u >= 0 && u <= 1 && !collapsed, dpy - u * dy, &dpy);
      MaskedAssign((u > 1 || u < 0) && !collapsed, -1., &u);
      ssq = dpx * dpx + dpy * dpy;
      MaskedAssign(ssq < safe, (Precision)i, &iseg);
      MaskedAssign(ssq < safe, u, &fraction);
      MaskedAssign(ssq < safe, ssq, &safe);
    }
  }

  //*****************************
  //**** Implementations end here
  //*****************************
}
} // End global namespace

#endif /* GENTRAPIMPLEMENTATION_H_ */
