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
#include "volumes/PlacedGenTrap.h"
#include "backend/Backend.h"
#include <iostream>

namespace vecgeom {
VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v(GenTrapImplementation, TranslationCode, translation::kGeneric, RotationCode, rotation::kGeneric)

inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
struct GenTrapImplementation {

  static const int transC = transCodeT;
  static const int rotC   = rotCodeT;

  using PlacedShape_t = PlacedGenTrap;
  using UnplacedShape_t = UnplacedGenTrap;

  VECGEOM_CUDA_HEADER_BOTH
  static void PrintType() {
     printf("SpecializedGenTrap<%i, %i>", transCodeT, rotCodeT);
  }
  template<typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void UnplacedContains(
      UnplacedGenTrap const &box,
      Vector3D<typename Backend::precision_v> const &localPoint,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Contains(
      UnplacedGenTrap const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> &localPoint,
      typename Backend::bool_v &inside);

  template <typename Backend>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(
      UnplacedGenTrap const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  template <typename Backend, bool ForInside>
  VECGEOM_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void GenericKernelForContainsAndInside(UnplacedGenTrap const &,
          Vector3D<typename Backend::precision_v> const &,
          typename Backend::bool_v &,
          typename Backend::bool_v &);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToIn(
      UnplacedGenTrap const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedGenTrap const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToIn(UnplacedGenTrap const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOut(UnplacedGenTrap const &unplaced,
                          Vector3D<typename Backend::precision_v> const &point,
                          typename Backend::precision_v &safety);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void ContainsKernel(
      UnplacedGenTrap const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::bool_v &inside);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void InsideKernel(
      UnplacedGenTrap const &boxDimensions,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::inside_v &inside);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToInKernel(
      UnplacedGenTrap const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance);

  template <class Backend, bool treatNormal>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOutKernel(
    UnplacedGenTrap const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToInKernel(
      UnplacedGenTrap const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v & safety);

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void SafetyToOutKernel(
      UnplacedGenTrap const &unplaced,
      Vector3D<typename Backend::precision_v> const &point,
      typename Backend::precision_v &safety);

}; // End struct GenTrapImplementation

//********************************
//**** implementations start here
//********************************/

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void GenTrapImplementation<transCodeT, rotCodeT>::UnplacedContains(
    UnplacedGenTrap const &box,
    Vector3D<typename Backend::precision_v> const &localPoint,
    typename Backend::bool_v &inside) {

    ContainsKernel<Backend>(box, localPoint, inside);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <typename Backend>
VECGEOM_CUDA_HEADER_BOTH
void GenTrapImplementation<transCodeT, rotCodeT>::Contains(
    UnplacedGenTrap const &unplaced,
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
void GenTrapImplementation<transCodeT, rotCodeT>::Inside(
    UnplacedGenTrap const &unplaced,
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
void GenTrapImplementation<transCodeT, rotCodeT>::DistanceToIn(
    UnplacedGenTrap const &unplaced,
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
void GenTrapImplementation<transCodeT, rotCodeT>::DistanceToOut(
    UnplacedGenTrap const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  DistanceToOutKernel<Backend, false>(
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
void GenTrapImplementation<transCodeT, rotCodeT>::SafetyToIn(
    UnplacedGenTrap const &unplaced,
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
void GenTrapImplementation<transCodeT, rotCodeT>::SafetyToOut(
    UnplacedGenTrap const &unplaced,
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
VECGEOM_INLINE
void GenTrapImplementation<transCodeT, rotCodeT>::ContainsKernel(
  UnplacedGenTrap const &unplaced,
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
VECGEOM_INLINE
void GenTrapImplementation<transCodeT, rotCodeT>::GenericKernelForContainsAndInside(
  UnplacedGenTrap const &unplaced,
  Vector3D<typename Backend::precision_v> const &localPoint,
  typename Backend::bool_v &completelyinside,
  typename Backend::bool_v &completelyoutside) {

  typedef typename Backend::precision_v Float_t;
  // Add stronger check against the bounding box, which can allow early returns if point is outside. 
  // Local point has to be translated in the bbox local frame.
//  completelyoutside = Abs(localPoint.z()) > MakePlusTolerant<ForInside>( unplaced.fDz );
//  completelyinside = Backend::kFalse;
//  completelyoutside = Backend::kFalse;
  BoxImplementation<translation::kIdentity, rotation::kIdentity>::GenericKernelForContainsAndInside<Backend, ForInside>(
      unplaced.fBoundingBox.dimensions(),
      localPoint-unplaced.fBoundingBoxOrig,
      completelyinside, 
      completelyoutside);
//  if (Backend::early_returns) {
    if ( IsFull(completelyoutside) ) {
      return;
    }
//  }

//  if (ForInside)  {
//    completelyinside = Abs(localPoint.z()) < MakeMinusTolerant<ForInside>( unplaced.fDz );
//  }

  // analyse z
  Float_t  cf = unplaced.fHalfInverseDz * (unplaced.fDz - localPoint.z());
  // analyse if x-y coordinates of localPoint are within polygon at z-height

  //  loop over edges connecting points i with i+4
  Float_t vertexX[4];
  Float_t vertexY[4];
  // vectorizes for scalar backend
  for (int  i = 0; i < 4; i++) {
    // calculate x-y positions of vertex i at this z-height
    vertexX[i] = unplaced.fVerticesX[i + 4] + cf * unplaced.fConnectingComponentsX[i];
    vertexY[i] = unplaced.fVerticesY[i + 4] + cf * unplaced.fConnectingComponentsY[i];
  }

    // this does not vectorize now
    // TODO: study if we get better by storing Delta and scaling it in z
    // ( because then it should vectorize but the cost is to have multiplications )
   // for (int  i = 0; i < 4; i++)
   // {
    //
   // }

  Float_t cross[4];
  Float_t localPointX = localPoint.x();
  Float_t localPointY = localPoint.y();
  // I currently found out that it is beneficial to keep the early return
  // in disfavor of the vectorizing solution; should be reinvestigated on AVX
  for (int  i = 0; i < 4; i++) {
    // this is based on the following idea:
    // we decided for each edge whether the point is above or below the
    // 2d line defined by that edge
    // In fact, this calculation is part of the calculation of the distance
    // of localPoint to that line which is a cross product. In this case it is
    // an embedded cross product of 2D vectors in 3D. The resulting vector always points
    // in z-direction whose z-magnitude is directly related to the distance.
    // see, e.g.,  http://geomalgorithms.com/a02-_lines.html
    int  j = (i + 1) % 4;
    Float_t  DeltaX = vertexX[j]-vertexX[i];
    Float_t  DeltaY = vertexY[j]-vertexY[i];
    cross[i]  = ( localPointX - vertexX[i] ) * DeltaY;
    cross[i] -= ( localPointY - vertexY[i] ) * DeltaX;

    completelyoutside |= (cross[i] < MakeMinusTolerant<ForInside>( 0 ));
    if (ForInside)
      completelyinside  &= (cross[i] > MakePlusTolerant<ForInside>( 0 ));

//    if (Backend::early_returns) {
      if ( IsFull(completelyoutside) ) {
        return;
      }
//    }
  }

}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void GenTrapImplementation<transCodeT, rotCodeT>::InsideKernel(
    UnplacedGenTrap const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::inside_v &inside) {

  typedef typename Backend::bool_v  Bool_t;
  Bool_t completelyinside;
  Bool_t completelyoutside;
  GenericKernelForContainsAndInside<Backend,true>(
      unplaced, point, completelyinside, completelyoutside);
  inside=EInside::kSurface;
  MaskedAssign(completelyoutside, EInside::kOutside, &inside);
  MaskedAssign(completelyinside, EInside::kInside, &inside);
}


template<bool IsSIMD, class Backend>
struct FillPlaneDataHelper
{
VECGEOM_INLINE
static void FillPlaneData(UnplacedGenTrap const &unplaced,
                         typename Backend::precision_v & cornerx,
                         typename Backend::precision_v & cornery,
                         typename Backend::precision_v & deltax,
                         typename Backend::precision_v & deltay,
                         typename Backend::bool_v const & top, int edgeindex ) {
  
    // no vectorized data lookup for SIMD
    // need to fill the SIMD types individually

    // now we only need to get the number 2 from somewhere
    for(int i=0; i< kVectorSize; ++i) {
         int index = edgeindex+top[i]*4;
         deltax[i] = unplaced.fDeltaX[index];
         deltay[i] = unplaced.fDeltaY[index];
         cornerx[i] = unplaced.fVerticesX[index];
         cornery[i] = unplaced.fVerticesY[index];
    }
  }
};

// a partial template specialization for nonSIMD cases (scalar, cuda, ... )
template<class Backend>
struct FillPlaneDataHelper<false, Backend>
{
static void FillPlaneData(UnplacedGenTrap const &unplaced,
                          typename Backend::precision_v & cornerx,
                          typename Backend::precision_v & cornery,
                          typename Backend::precision_v & deltax,
                          typename Backend::precision_v & deltay,
                          typename Backend::bool_v const & top, int edgeindex ) {
    int index = edgeindex+top*4;
    deltax = unplaced.fDeltaX[index];
    deltay = unplaced.fDeltaY[index];
    cornerx = unplaced.fVerticesX[index];
    cornery = unplaced.fVerticesY[index];
}
};

template<class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
// optimized "inside" check for top or bottom z-surfaces
// this is a bit tricky if different tracks check different planes
// ( for example in case of Backend = Vc when top is mixed )
// ( this is because vector data lookup is tricky )
typename Backend::bool_v IsInTopOrBottomPolygon( UnplacedGenTrap const& unplaced,
        typename Backend::precision_v const & pointx,
        typename Backend::precision_v const & pointy,
        typename Backend::bool_v top )
{
    // stripped down version of the Contains kernel ( not yet shared with that kernel )
    typedef typename Backend::bool_v  Bool_t;
    typedef typename Backend::precision_v  Float_t;
    // std::cerr << "IsInTopOrBottom: pointx: " << pointx << "  pointy: " << pointy << "  top: " << top << "\n";

    Bool_t completelyoutside( Backend::kFalse );
    for (int  i=0; i<4; ++i) {
        Float_t deltaX;
        Float_t deltaY;
        Float_t cornerX;
        Float_t cornerY;

        // thats the only place where scalar and vector code diverge
	// IsSIMD misses...replaced with early_returns
        FillPlaneDataHelper<!Backend::early_returns, Backend>::FillPlaneData(
                unplaced, cornerX, cornerY, deltaX, deltaY, top, i );

        // std::cerr << i << " CORNERS " << cornerX << " " << cornerY << " " << deltaX << " " << deltaY << "\n";

        Float_t cross  = (pointx - cornerX) * deltaY;
        cross -= (pointy - cornerY) * deltaX;

       completelyoutside |= cross < MakeMinusTolerant<true>( 0. );
        if (Backend::early_returns) {
         if ( IsFull ( completelyoutside ) ) {
                  return Backend::kFalse;
         }
       }
    }
   return ! completelyoutside;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void GenTrapImplementation<transCodeT, rotCodeT>::DistanceToInKernel(
    UnplacedGenTrap const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v      Bool_t;

//#define GENTRAPDEB = 1
#ifdef GENTRAPDEB
  std::cerr << "point: " << point << std::endl;
  std::cerr << "direction: " << direction << std::endl;
#endif  
  //do a quick boundary box check (Arb8, USolids) is doing this
  //unplaced.GetBBox();
  //actually this could also give us some indication which face is likely to be hit

// let me see if we can temporarily force the box function to be a function call
  Float_t bbdistance = 0.;
#ifdef GENTRAP_USENEWBB
  typename Backend::int_v planeid(-1);
  Vector3D<Float_t> hitpoint;
  BoxImplementation<translation::kIdentity, rotation::kIdentity>::DistanceToInKernel2<Backend>(
      unplaced.fBoundingBox.dimensions(),
      point-unplaced.fBoundingBoxOrig,
      direction,
      stepMax,
      bbdistance,
      &planeid, &hitpoint );
#else
  BoxImplementation<translation::kIdentity, rotation::kIdentity>::DistanceToInKernel<Backend>(
        unplaced.fBoundingBox.dimensions(),
        point-unplaced.fBoundingBoxOrig,
        direction,
        stepMax,
        bbdistance );
#endif

#ifdef GENTRAPDEB
   std::cerr << "BB gave " << bbdistance << "\n";
#endif
   distance = kInfinity;

  // do a check on bbdistance
  // if none of the tracks can hit even the bounding box; just return
  Bool_t done = bbdistance >= kInfinity;
  if( IsFull ( done ) ) return;
#ifdef GENTRAPDEB
  Float_t x,y,z;
  x=point.x()+bbdistance*direction.x();
  y=point.y()+bbdistance*direction.y();
  z=point.z()+bbdistance*direction.z();
  std::cerr << "prolongated to box:  x " << x << " y " << y << " z " << z << "\n";
#endif

  /*some particle could hit z*/
 #ifdef GENTRAP_USENEWBB
#pragma message("WITH IMPROVED BB")
  if( ! IsEmpty (planeid == 2 ) ){
      Bool_t top = direction.z() < 0;
      Bool_t hits = IsInTopOrBottomPolygon<Backend>(unplaced, hitpoint.x(), hitpoint.y(), top );
      #ifdef GENTRAPDEB
      std::cerr << " top/bottom hit result " << hits << " \n";
      #endif
      MaskedAssign(hits, bbdistance, &distance);
      done |= hits;
      if( IsFull(done) ) return;
  }
#else // do this check again
//#pragma message("WITH OLD BB")
  // IDEA: we don't need to do this if bbox does not hit z planes
  // IDEA2: and if bbbox hits z-plane everything here is already calculated
//  Float_t snext;
  Float_t zsafety = Abs(point.z()) - unplaced.fDz;
  Bool_t canhitz =  zsafety > MakeMinusTolerant<true>(0.);
  canhitz        &= point.z()*direction.z() < 0; // coming towards the origin
  canhitz        &= !done;
#ifdef GENTRAPDEB
      std::cerr << " canhitz " << canhitz << " \n";
#endif      

  if( ! IsEmpty (canhitz)  ){
   //   std::cerr << "can potentially hit\n";
   // calculate distance to z-plane ( see Box algorithm )
   // check if hit point is inside top or bottom polygon
      Float_t next = zsafety / Abs(direction.z() + kTiny);
#ifdef GENTRAPDEB
      std::cerr << " zdist " << next << "\n";
#endif
      // transport to z-height of planes
      Float_t coord1 = point.x() + next*direction.x();
      Float_t coord2 = point.y() + next*direction.y();
      Bool_t top = direction.z() < 0;
     // Bool_t hits = IsInTopOrBottomPolygon<Backend>(unplaced, hitpoint.x(), hitpoint.y(), top );
      Bool_t hits = IsInTopOrBottomPolygon<Backend>(unplaced, coord1, coord2, top );
      hits &= canhitz;
      MaskedAssign(hits, bbdistance, &distance);
      done |= hits;
      #ifdef GENTRAPDEB
      std::cerr << " hit result " << hits << " bbdistance " << distance << "\n";
      #endif

      if( IsFull(done) ) return;
  }
#endif

  // now treat lateral surfaces
  Float_t disttoplanes = unplaced.GetShell().DistanceToIn<Backend>(
          point, direction, done );
#ifdef GENTRAPDEB
  std::cerr << "disttoplanes " << disttoplanes << "\n";
#endif

  MaskedAssign( ! done, Min(disttoplanes, distance), &distance );
#ifdef GENTRAPDEB
  std::cerr << distance << "\n";
#endif
//  std::cerr << std::endl;
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend, bool treatNormal>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void GenTrapImplementation<transCodeT, rotCodeT>::DistanceToOutKernel(
    UnplacedGenTrap const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax,
    typename Backend::precision_v &distance) {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    // we should check here the compilation condition
    // that treatNormal=true can only happen when Backend=kScalar
    // TODO: do this with some nice template features

    Bool_t negDirMask = direction.z() < 0;
    Float_t sign = 1.;
    MaskedAssign( negDirMask, -Backend::kOne, &sign);
//    Float_t invDirZ = 1./direction.z();
    // this construct costs one multiplication more
    Float_t distmin = ( sign*unplaced.fDz - point.z() ) / direction.z();

    Float_t distplane = unplaced.GetShell().DistanceToOut<Backend>( point, direction );
    distance = Min( distmin, distplane );

}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void GenTrapImplementation<transCodeT, rotCodeT>::SafetyToInKernel(
    UnplacedGenTrap const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  typedef typename Backend::bool_v Boolean_t;
  
  Boolean_t inside;  
  // Check if all points are outside bounding box
  BoxImplementation<translation::kIdentity, rotation::kIdentity>::ContainsKernel<Backend>(
      unplaced.fBoundingBox.dimensions(),
      point-unplaced.fBoundingBoxOrig,
      inside);
  if (IsEmpty( inside )) {
    // All points outside, so compute safety using the bounding box
    // This is not optimal if top and bottom faces are not on top of each other
    BoxImplementation<translation::kIdentity, rotation::kIdentity>::SafetyToInKernel<Backend>(
      unplaced.fBoundingBox.dimensions(),
      point-unplaced.fBoundingBoxOrig,
      safety);
    return;  
  }
    
  // Do Z
  safety = Abs(point[2]) - unplaced.GetDZ();
  safety = unplaced.GetShell().SafetyToIn<Backend>( point, safety );
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void GenTrapImplementation<transCodeT, rotCodeT>::SafetyToOutKernel(
    UnplacedGenTrap const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  // Do Z
   safety = unplaced.GetDZ() - Abs(point[2]);
   safety = unplaced.GetShell().SafetyToOut<Backend>( point, safety );
}


//*****************************
//**** Implementations end here
//*****************************

} } // End global namespace

#endif /* GENTRAPIMPLEMENTATION_H_ */
