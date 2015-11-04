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
//********************************

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
  BoxImplementation<translation::kIdentity, rotation::kIdentity>::GenericKernelForContainsAndInside<Backend, false>(
      unplaced.fBoundingBox.dimensions(),
      localPoint-unplaced.fBoundingBoxOrig,
      completelyinside, 
      completelyoutside);
  if (Backend::early_returns) {
    if ( completelyoutside == Backend::kTrue ) {
      return;
    }
  }

  if (ForInside)  {
    completelyinside = Abs(localPoint.z()) < MakeMinusTolerant<ForInside>( unplaced.fDz );
  }

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

    completelyoutside |= cross[i] < MakeMinusTolerant<ForInside>( 0 );
    if (Backend::early_returns) {
      if ( completelyoutside == Backend::kTrue ) {
        return;
      }
    }
  }

  for (int  i = 0; i < 4; i++) {
    completelyoutside |= cross[i] < 0;
    if (Backend::early_returns) {
      if ( completelyoutside == Backend::kTrue ) {
        return;
      }
    }
    completelyinside  &= cross[i] > 0;
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
  Bool_t completelyinside, completelyoutside;
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
    for(int i=0; i< 2; ++i) {
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

      //  std::cerr << i << " CORNERS " << cornerX << " " << cornerY << " " << deltaX << " " << deltaY << "\n";

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

  //do a quick boundary box check (Arb8, USolids) is doing this
  //unplaced.GetBBox();
  //actually this could also give us some indication which face is likely to be hit

// let me see if we can temporarily force the box function to be a function call
  Float_t bbdistance;
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

  //#define GENTRAPDEB = 1
#ifdef GENTRAPDEB
   std::cerr << "BB gave " << bbdistance << "\n";
#endif
   distance = kInfinity;

  // do a check on bbdistance
  // if none of the tracks can hit even the bounding box; just return
  Bool_t done = bbdistance >= kInfinity;
  if( IsFull ( done ) ) return;
#ifdef GENTRAPDEB
  std::cerr << " bbdistance " << bbdistance << "\n";
  Float_t x,y,z;
  x=point.x()+bbdistance*direction.x();
  y=point.y()+bbdistance*direction.y();
  z=point.z()+bbdistance*direction.z();
  std::cerr << "x " << x << " y " << y << " z " << z << "\n";
#endif



  /*some particle could hit z*/
 #ifdef GENTRAP_USENEWBB
#pragma message("WITH IMPROVED BB")
  if( ! IsEmpty (planeid == 2 ) ){
      Bool_t top = direction.z() < 0;
      Bool_t hits = IsInTopOrBottomPolygon<Backend>(unplaced, hitpoint.x(), hitpoint.y(), top );
      #ifdef GENTRAPDEB
      std::cerr << " hit result " << hits << " \n";
      #endif
      MaskedAssign(hits, bbdistance, &distance);
      done |= hits;
      if( IsFull(done) ) return;
  }
#else // do this check again
#pragma message("WITH OLD BB")
  // IDEA: we don't need to do this if bbox does not hit z planes
  // IDEA2: and if bbbox hits z-plane everything here is already calculated
  Float_t snext;
  Float_t zsafety = Abs(point.z()) - unplaced.fDz;
  Bool_t canhitz =  zsafety > MakeMinusTolerant<true>(0.);
  canhitz        &= point.z()*direction.z() < 0; // coming towards the origin

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

      #ifdef GENTRAPDEB
      std::cerr << " hit result " << hits << " \n";
      #endif
      MaskedAssign(hits, bbdistance, &distance);
      done |= hits;

      if( IsFull(done) ) return;
  }
#endif

  // now treat lateral surfaces
  Float_t disttoplanes = unplaced.GetShell().DistanceToIn<Backend>(
          point, direction, done );
 // std::cerr << disttoplanes << "\n";

  MaskedAssign( ! done, Min(disttoplanes, distance), &distance );
 // std::cerr << distance << "\n";
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
    Float_t invDirZ = 1./direction.z();
    // this construct costs one multiplication more
    Float_t distmin = ( sign*unplaced.fDz - point.z() ) / direction.z();

    Float_t distplane = unplaced.GetShell().DistanceToOut<Backend>( point, direction );
    distance = Min( distmin, distplane );

    // to be done
//    if (direction.z() < 0)
//      {
//        distmin = (-unplaced.fDz - point.z()) / direction.z();
//        if( treatNormal ){
//          // side = topZ ;
//          // aNormalVector.Set(0, 0, -1);
//        }
//      }
//    else
//      {
//        if (direction.z() > 0)
//        {
//          distmin = (unplaced.fDz - point.z()) / direction.z();
//          if ( treatNormal )
//          {
//            side = kPZ;
//            aNormalVector.Set(0, 0, 1);
//          }
//          else // this else statement can be avoided by applying "tiny" trick
//          {
//            distmin = UUtils::Infinity();
//          }
//      }

//        analysing code from TGeoArb8
//        // Computes distance to plane ipl :
//        // ipl=0 : points 0,4,1,5
//        // ipl=1 : points 1,5,2,6
//        // ipl=2 : points 2,6,3,7
//        // ipl=3 : points 3,7,0,4
//           Double_t xa,xb,xc,xd;
//           Double_t ya,yb,yc,yd;
//           Double_t eps = 10.*TGeoShape::Tolerance();
//           Double_t norm[3];
//           Double_t dirp[3];
//           Double_t ndotd = 0;
//           Int_t j = (ipl+1)%4;
//           xa=fXY[ipl][0];
//           ya=fXY[ipl][1];
//           xb=fXY[ipl+4][0];
//           yb=fXY[ipl+4][1];
//           xc=fXY[j][0];
//           yc=fXY[j][1];
//           xd=fXY[4+j][0];
//           yd=fXY[4+j][1];
//           Double_t dz2 =0.5/fDz;
//           Double_t tx1 =dz2*(xb-xa);
//           Double_t ty1 =dz2*(yb-ya);
//           Double_t tx2 =dz2*(xd-xc);
//           Double_t ty2 =dz2*(yd-yc);
//           Double_t dzp =fDz+point[2];
//           Double_t xs1 =xa+tx1*dzp;
//           Double_t ys1 =ya+ty1*dzp;
//           Double_t xs2 =xc+tx2*dzp;
//           Double_t ys2 =yc+ty2*dzp;
//           Double_t dxs =xs2-xs1;
//           Double_t dys =ys2-ys1;
//           Double_t dtx =tx2-tx1;
//           Double_t dty =ty2-ty1;
//           Double_t a=(dtx*dir[1]-dty*dir[0]+(tx1*ty2-tx2*ty1)*dir[2])*dir[2];
//           Double_t signa = TMath::Sign(1.,a);
//           Double_t b=dxs*dir[1]-dys*dir[0]+(dtx*point[1]-dty*point[0]+ty2*xs1-ty1*xs2
//                      +tx1*ys2-tx2*ys1)*dir[2];
//           Double_t c=dxs*point[1]-dys*point[0]+xs1*ys2-xs2*ys1;
//           Double_t s=TGeoShape::Big();
//           Double_t x1,x2,y1,y2,xp,yp,zi;
//           if (TMath::Abs(a)<eps) {
//              // Surface is planar
//              if (TMath::Abs(b)<eps) return TGeoShape::Big(); // Track parallel to surface
//              s=-c/b;
//              if (TMath::Abs(s)<1.E-6 && TMath::Abs(TMath::Abs(point[2])-fDz)>eps) {
//                 memcpy(dirp,dir,3*sizeof(Double_t));
//                 dirp[0] = -3;
//                 // Compute normal pointing outside
//                 ((TGeoArb8*)this)->ComputeNormal(point,dirp,norm);
//                 ndotd = dir[0]*norm[0]+dir[1]*norm[1]+dir[2]*norm[2];
//                 if (!in) ndotd*=-1.;
//                 if (ndotd>0) {
//                    s = TMath::Max(0.,s);
//                    zi = (point[0]-xs1)*(point[0]-xs2)+(point[1]-ys1)*(point[1]-ys2);
//                    if (zi<=0) return s;
//                 }
//                 return TGeoShape::Big();
//              }
//              if (s<0) return TGeoShape::Big();
//           }
//           else {
//              // Surface is curved
//              Double_t d=b*b-4*a*c;
//              if (d<0) return TGeoShape::Big();
//              Double_t smin=0.5*(-b-signa*TMath::Sqrt(d))/a;
//              Double_t smax=0.5*(-b+signa*TMath::Sqrt(d))/a;
//              s = smin;
//              if (TMath::Abs(s)<1.E-6 && TMath::Abs(TMath::Abs(point[2])-fDz)>eps) {
//                 memcpy(dirp,dir,3*sizeof(Double_t));
//                 dirp[0] = -3;
//                 // Compute normal pointing outside
//                 ((TGeoArb8*)this)->ComputeNormal(point,dirp,norm);
//                 ndotd = dir[0]*norm[0]+dir[1]*norm[1]+dir[2]*norm[2];
//                 if (!in) ndotd*=-1.;
//                 if (ndotd>0) return TMath::Max(0.,s);
//                 s = 0.; // ignore
//              }
//              if (s>eps) {
//                 // Check smin
//                 zi=point[2]+s*dir[2];
//                 if (TMath::Abs(zi)<fDz) {
//                    x1=xs1+tx1*dir[2]*s;
//                    x2=xs2+tx2*dir[2]*s;
//                    xp=point[0]+s*dir[0];
//                    y1=ys1+ty1*dir[2]*s;
//                    y2=ys2+ty2*dir[2]*s;
//                    yp=point[1]+s*dir[1];
//                    zi = (xp-x1)*(xp-x2)+(yp-y1)*(yp-y2);
//                    if (zi<=0) return s;
//                 }
//              }
//              // Smin failed, try smax
//              s=smax;
//              if (TMath::Abs(s)<1.E-6 && TMath::Abs(TMath::Abs(point[2])-fDz)>eps) {
//                 memcpy(dirp,dir,3*sizeof(Double_t));
//                 dirp[0] = -3;
//                 // Compute normal pointing outside
//                 ((TGeoArb8*)this)->ComputeNormal(point,dirp,norm);
//                 ndotd = dir[0]*norm[0]+dir[1]*norm[1]+dir[2]*norm[2];
//                 if (!in) ndotd*=-1.;
//                 if (ndotd>0) s = TMath::Max(0.,s);
//                 else         s = TGeoShape::Big();
//                 return s;
//              }
//           }
//           if (s>eps) {
//              // Check smin
//              zi=point[2]+s*dir[2];
//              if (TMath::Abs(zi)<fDz) {
//                 x1=xs1+tx1*dir[2]*s;
//                 x2=xs2+tx2*dir[2]*s;
//                 xp=point[0]+s*dir[0];
//                 y1=ys1+ty1*dir[2]*s;
//                 y2=ys2+ty2*dir[2]*s;
//                 yp=point[1]+s*dir[1];
//                 zi = (xp-x1)*(xp-x2)+(yp-y1)*(yp-y2);
//                 if (zi<=0) return s;
//              }
//           }
//           return TGeoShape::Big();

}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void GenTrapImplementation<transCodeT, rotCodeT>::SafetyToInKernel(
    UnplacedGenTrap const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

  typedef typename Backend::precision_v Float_t;

  // to be done
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
template <class Backend>
VECGEOM_CUDA_HEADER_BOTH
void GenTrapImplementation<transCodeT, rotCodeT>::SafetyToOutKernel(
    UnplacedGenTrap const &unplaced,
    Vector3D<typename Backend::precision_v> const &point,
    typename Backend::precision_v &safety) {

   typedef typename Backend::precision_v Float_t;
   safety = 1.;
   return;
   // to be done
}


//*****************************
//**** Implementations end here
//*****************************

} } // End global namespace

#endif /* GENTRAPIMPLEMENTATION_H_ */
