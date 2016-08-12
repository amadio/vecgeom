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

#include "volumes/kernel/GenericKernels.h"
#include "volumes/kernel/BoxImplementation.h"
#include "volumes/UnplacedGenTrap.h"

#include <iostream>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct GenTrapImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, GenTrapImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedGenTrap;
class UnplacedGenTrap;

template <typename T>
struct GenTrapStruct;

struct GenTrapImplementation {

  using Vertex_t         = Vector3D<Precision>;
  using PlacedShape_t    = PlacedGenTrap;
  using UnplacedStruct_t = GenTrapStruct<double>;
  using UnplacedVolume_t = UnplacedGenTrap;

  VECGEOM_CUDA_HEADER_BOTH
  static void PrintType()
  {
    // printf("SpecializedGenTrap<%i, %i>", transCodeT, rotCodeT);
  }

  template <typename Stream>
  static void PrintType(Stream & /*s*/)
  {
    // s << "SpecializedGenTrap<" << transCodeT << "," << rotCodeT << ">";
  }

  template <typename Stream>
  static void PrintImplementationType(Stream & /*s*/)
  {
    // s << "GenTrapImplementation<" << transCodeT << "," << rotCodeT << ">";
  }

  template <typename Stream>
  static void PrintUnplacedType(Stream & /*s*/)
  {
    // s << "UnplacedGenTrap";
  }

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Contains(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Bool_v &inside);

  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Inside_t &inside);

  template <typename Real_v, bool ForInside>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void GenericKernelForContainsAndInside(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point,
                                                vecCore::Mask_v<Real_v> &completelyinside,
                                                vecCore::Mask_v<Real_v> &completelyoutside);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToIn(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point,
                           Vector3D<Real_v> const &direction, Real_v const &stepMax, Real_v &distance);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToOut(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point,
                            Vector3D<Real_v> const &direction, Real_v const &stepMax, Real_v &distance);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToIn(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Real_v &safety);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToOut(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Real_v &safety);

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static void NormalKernel(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Vector3D<Real_v> &normal,
                           Bool_v &valid);

  template <class Real_v>
  VECGEOM_CUDA_HEADER_BOTH
  static void GetClosestEdge(Vector3D<Real_v> const &point, Real_v vertexX[4], Real_v vertexY[4], Real_v &iseg,
                             Real_v &fraction);

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECGEOM_CUDA_HEADER_BOTH
  static vecCore::Mask_v<Real_v> IsInTopOrBottomPolygon(UnplacedStruct_t const &unplaced, Real_v const &pointx,
                                                        Real_v const &pointy, vecCore::Mask_v<Real_v> top);
}; // End struct GenTrapImplementation

//********************************
//**** implementations start here
//********************************/

//______________________________________________________________________________
template <typename Real_v, typename Bool_v>
VECGEOM_CUDA_HEADER_BOTH
void GenTrapImplementation::Contains(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Bool_v &inside)
{
  Bool_v unused;
  Bool_v outside;
  GenericKernelForContainsAndInside<Real_v, false>(unplaced, point, unused, outside);
  inside = !outside;
}

//______________________________________________________________________________
template <typename Real_v, bool ForInside>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void GenTrapImplementation::GenericKernelForContainsAndInside(UnplacedStruct_t const &unplaced,
                                                              Vector3D<Real_v> const &point,
                                                              vecCore::Mask_v<Real_v> &completelyinside,
                                                              vecCore::Mask_v<Real_v> &completelyoutside)
{

  using Bool_v                    = vecCore::Mask_v<Real_v>;
  constexpr Precision tolerancesq = 10000. * kTolerance * kTolerance;
  // Local point has to be translated in the bbox local frame.
  BoxImplementation::GenericKernelForContainsAndInside<Real_v, Bool_v, ForInside>(
      unplaced.fBBdimensions, point - unplaced.fBBorigin, completelyinside, completelyoutside);
  //  if (Backend::early_returns) {
  if (vecCore::MaskFull(completelyoutside)) {
    return;
  }
  //  }

  // analyse z
  Real_v cf = unplaced.fHalfInverseDz * (unplaced.fDz - point.z());
  // analyse if x-y coordinates of point are within polygon at z-height

  //  loop over edges connecting points i with i+4
  Real_v vertexX[4];
  Real_v vertexY[4];
  // vectorizes for scalar backend
  for (int i = 0; i < 4; i++) {
    // calculate x-y positions of vertex i at this z-height
    vertexX[i] = unplaced.fVerticesX[i + 4] + cf * unplaced.fConnectingComponentsX[i];
    vertexY[i] = unplaced.fVerticesY[i + 4] + cf * unplaced.fConnectingComponentsY[i];
  }

  for (int i = 0; i < 4; i++) {
    // this is based on the following idea:
    // we decided for each edge whether the point is above or below the
    // 2d line defined by that edge
    // In fact, this calculation is part of the calculation of the distance
    // of point to that line which is a cross product. In this case it is
    // an embedded cross product of 2D vectors in 3D. The resulting vector always points
    // in z-direction whose z-magnitude is directly related to the distance.
    // see, e.g.,  http://geomalgorithms.com/a02-_lines.html
    if (unplaced.fDegenerated[i]) continue;
    int j         = (i + 1) % 4;
    Real_v DeltaX = vertexX[j] - vertexX[i];
    Real_v DeltaY = vertexY[j] - vertexY[i];
    Real_v cross  = (point.x() - vertexX[i]) * DeltaY - (point.y() - vertexY[i]) * DeltaX;
    if (ForInside) {
      Bool_v onsurf     = (cross * cross < tolerancesq * (DeltaX * DeltaX + DeltaY * DeltaY));
      completelyoutside = completelyoutside || (((cross < MakeMinusTolerant<ForInside>(0.)) && (!onsurf)));
      completelyinside  = completelyinside && (cross > MakePlusTolerant<ForInside>(0.)) && (!onsurf);
    } else {
      completelyoutside = completelyoutside || (cross < MakeMinusTolerant<ForInside>(0.));
    }

    //    if (Backend::early_returns) {
    if (vecCore::MaskFull(completelyoutside)) {
      return;
    }
    //    }
  }
}

//______________________________________________________________________________
template <typename Real_v, typename Inside_t>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void GenTrapImplementation::Inside(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Inside_t &inside)
{

  using Bool_v       = vecCore::Mask_v<Real_v>;
  using InsideBool_v = vecCore::Mask_v<Inside_t>;
  Bool_v completelyinside;
  Bool_v completelyoutside;
  GenericKernelForContainsAndInside<Real_v, true>(unplaced, point, completelyinside, completelyoutside);

  inside = Inside_t(EInside::kSurface);
  vecCore::MaskedAssign(inside, (InsideBool_v)completelyoutside, Inside_t(EInside::kOutside));
  vecCore::MaskedAssign(inside, (InsideBool_v)completelyinside, Inside_t(EInside::kInside));
}

//______________________________________________________________________________
template <typename Real_v>
struct FillPlaneDataHelper {
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  static void FillPlaneData(GenTrapStruct<double> const &unplaced, Real_v &cornerx, Real_v &cornery, Real_v &deltax,
                            Real_v &deltay, vecCore::Mask_v<Real_v> const &top, int edgeindex)
  {

    // no vectorized data lookup for SIMD
    // need to fill the SIMD types individually

    for (size_t i = 0; i < vecCore::VectorSize<Real_v>(); ++i) {
      int index  = edgeindex + top[i] * 4;
      deltax[i]  = unplaced.fDeltaX[index];
      deltay[i]  = unplaced.fDeltaY[index];
      cornerx[i] = unplaced.fVerticesX[index];
      cornery[i] = unplaced.fVerticesY[index];
    }
  }
};

//______________________________________________________________________________
/** @brief A partial template specialization for nonSIMD cases (scalar, cuda, ... ) */
template <>
struct FillPlaneDataHelper<double> {
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  static void FillPlaneData(GenTrapStruct<double> const &unplaced, double &cornerx, double &cornery, double &deltax,
                            double &deltay, bool const &top, int edgeindex)
  {
    int index = edgeindex + top * 4;
    deltax    = unplaced.fDeltaX[index];
    deltay    = unplaced.fDeltaY[index];
    cornerx   = unplaced.fVerticesX[index];
    cornery   = unplaced.fVerticesY[index];
  }
};

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
vecCore::Mask_v<Real_v> GenTrapImplementation::IsInTopOrBottomPolygon(UnplacedStruct_t const &unplaced,
                                                                      Real_v const &pointx, Real_v const &pointy,
                                                                      vecCore::Mask_v<Real_v> top)
{
  // optimized "inside" check for top or bottom z-surfaces
  // this is a bit tricky if different tracks check different planes
  // ( for example in case of Backend = Vc when top is mixed )
  // ( this is because vector data lookup is tricky )

  // stripped down version of the Contains kernel ( not yet shared with that kernel )
  // std::cerr << "IsInTopOrBottom: pointx: " << pointx << "  pointy: " << pointy << "  top: " << top << "\n";

  using Bool_v             = vecCore::Mask_v<Real_v>;
  Bool_v completelyoutside = Bool_v(false);
  Bool_v degenerate        = Bool_v(true);
  for (int i = 0; i < 4; ++i) {
    Real_v deltaX;
    Real_v deltaY;
    Real_v cornerX;
    Real_v cornerY;

    // thats the only place where scalar and vector code diverge
    // IsSIMD misses...replaced with early_returns
    FillPlaneDataHelper<Real_v>::FillPlaneData(unplaced, cornerX, cornerY, deltaX, deltaY, top, i);

    // std::cerr << i << " CORNERS " << cornerX << " " << cornerY << " " << deltaX << " " << deltaY << "\n";

    Real_v cross = (pointx - cornerX) * deltaY;
    cross -= (pointy - cornerY) * deltaX;
    degenerate        = degenerate && (deltaX < MakePlusTolerant<true>(0.)) && (deltaY < MakePlusTolerant<true>(0.));
    completelyoutside = completelyoutside || (cross < MakeMinusTolerant<true>(0.));
    // if (Backend::early_returns) {
    if (vecCore::MaskFull(completelyoutside)) {
      return Bool_v(false);
    }
    // }
  }
  completelyoutside = completelyoutside || degenerate;
  return (!completelyoutside);
}

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void GenTrapImplementation::DistanceToIn(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point,
                                         Vector3D<Real_v> const &direction, Real_v const &stepMax, Real_v &distance)
{

  using Bool_v = vecCore::Mask_v<Real_v>;

//#define GENTRAPDEB = 1
#ifdef GENTRAPDEB
  std::cerr << "point: " << point << std::endl;
  std::cerr << "direction: " << direction << std::endl;
#endif
  // do a quick boundary box check (Arb8, USolids) is doing this
  // unplaced.GetBBox();
  // actually this could also give us some indication which face is likely to be hit

  Real_v bbdistance = Real_v(kInfinity);
  BoxImplementation::DistanceToIn(BoxStruct<Precision>(unplaced.fBBdimensions), point - unplaced.fBBorigin, direction,
                                  stepMax, bbdistance);

#ifdef GENTRAPDEB
  std::cerr << "BB gave " << bbdistance << "\n";
#endif
  distance = vecCore::NumericLimits<Real_v>::Infinity();

  // do a check on bbdistance
  // if none of the tracks can hit even the bounding box; just return
  Bool_v done = bbdistance >= vecCore::NumericLimits<Real_v>::Infinity();
  if (vecCore::MaskFull(done)) return;
#ifdef GENTRAPDEB
  Real_v x, y, z;
  x = point.x() + bbdistance * direction.x();
  y = point.y() + bbdistance * direction.y();
  z = point.z() + bbdistance * direction.z();
  std::cerr << "prolongated to box:  x " << x << " y " << y << " z " << z << "\n";
#endif

  // some particle could hit z
  Real_v zsafety = vecCore::math::Abs(point.z()) - unplaced.fDz;
  Bool_v canhitz = zsafety > MakeMinusTolerant<true>(0.);
  canhitz        = canhitz && (point.z() * direction.z() < 0); // coming towards the origin
  canhitz        = canhitz && (!done);
#ifdef GENTRAPDEB
  std::cerr << " canhitz " << canhitz << " \n";
#endif

  if (!vecCore::MaskEmpty(canhitz)) {
    //   std::cerr << "can potentially hit\n";
    // calculate distance to z-plane ( see Box algorithm )
    // check if hit point is inside top or bottom polygon
    Real_v next = zsafety / vecCore::math::Abs(direction.z() + kTiny);
#ifdef GENTRAPDEB
    std::cerr << " zdist " << next << "\n";
#endif
    // transport to z-height of planes
    Real_v coord1 = point.x() + next * direction.x();
    Real_v coord2 = point.y() + next * direction.y();
    Bool_v top    = direction.z() < 0;
    Bool_v hits   = IsInTopOrBottomPolygon<Real_v>(unplaced, coord1, coord2, top);
    hits          = hits && canhitz;
    vecCore::MaskedAssign(distance, hits, bbdistance);
    done = done || hits;
#ifdef GENTRAPDEB
    std::cerr << " hit result " << hits << " bbdistance " << distance << "\n";
#endif
    if (vecCore::MaskFull(done)) return;
  }

  // now treat lateral surfaces
  Real_v disttoplanes = unplaced.fSurfaceShell.DistanceToIn<Real_v>(point, direction, done);
#ifdef GENTRAPDEB
  std::cerr << "disttoplanes " << disttoplanes << "\n";
#endif

  vecCore::MaskedAssign(distance, !done, vecCore::math::Min(disttoplanes, distance));
#ifdef GENTRAPDEB
  std::cerr << distance << "\n";
#endif
}

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void GenTrapImplementation::DistanceToOut(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point,
                                          Vector3D<Real_v> const &direction, Real_v const & /* stepMax */,
                                          Real_v &distance)
{

  using Bool_v = vecCore::Mask_v<Real_v>;

  // we should check here the compilation condition
  // that treatNormal=true can only happen when Backend=kScalar
  // TODO: do this with some nice template features

  Bool_v negDirMask = direction.z() < 0;
  Real_v sign       = 1.;
  vecCore::MaskedAssign(sign, negDirMask, Real_v(-1.));
  //    Real_v invDirZ = 1./direction.z();
  // this construct costs one multiplication more
  Real_v distmin = (sign * unplaced.fDz - point.z()) / direction.z();

  Real_v distplane = unplaced.fSurfaceShell.DistanceToOut<Real_v>(point, direction);
  distance         = vecCore::math::Min(distmin, distplane);
}

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_FORCE_INLINE
VECGEOM_CUDA_HEADER_BOTH
void GenTrapImplementation::SafetyToIn(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Real_v &safety)
{

  using Bool_v = vecCore::Mask_v<Real_v>;

  Bool_v inside;
  // Check if all points are outside bounding box
  BoxImplementation::Contains(BoxStruct<double>(unplaced.fBBdimensions), point - unplaced.fBBorigin, inside);
  if (vecCore::MaskEmpty(inside)) {
    // All points outside, so compute safety using the bounding box
    // This is not optimal if top and bottom faces are not on top of each other
    BoxImplementation::SafetyToIn(BoxStruct<double>(unplaced.fBBdimensions), point - unplaced.fBBorigin, safety);
    return;
  }

  // Do Z
  safety = vecCore::math::Abs(point[2]) - unplaced.fDz;
  safety = unplaced.fSurfaceShell.SafetyToIn<Real_v>(point, safety);
}

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_CUDA_HEADER_BOTH
void GenTrapImplementation::SafetyToOut(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point, Real_v &safety)
{

  // Do Z
  safety = unplaced.fDz - vecCore::math::Abs(point[2]);
  safety = unplaced.fSurfaceShell.SafetyToOut<Real_v>(point, safety);
}

//______________________________________________________________________________
template <typename Real_v, typename Bool_v>
VECGEOM_CUDA_HEADER_BOTH
void GenTrapImplementation::NormalKernel(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point,
                                         Vector3D<Real_v> &normal, Bool_v &valid)
{

  // Computes the normal on a surface and returns it as a unit vector
  //   In case a point is further than tolerance_normal from a surface, set validNormal=false
  //   Must return a valid vector. (even if the point is not on the surface.)

  using Index_v = vecCore::Index_v<Real_v>;

  valid = Bool_v(true);
  normal.Set(0., 0., 0.);
  // Do bottom and top faces
  Real_v safz = vecCore::math::Abs(unplaced.fDz - vecCore::math::Abs(point.z()));
  Bool_v onZ  = (safz < 10. * kTolerance);
  vecCore::MaskedAssign(normal[2], onZ && (point.z() > 0), 1.);
  vecCore::MaskedAssign(normal[2], onZ && (point.z() < 0), -1.);

  //    if (Backend::early_returns) {
  if (vecCore::MaskFull(onZ)) {
    return;
  }
  //    }
  //  Real_v done = onZ;
  // Get the closest edge (point should be on this edge within tolerance)
  Real_v cf = unplaced.fHalfInverseDz * (unplaced.fDz - point.z());
  Real_v vertexX[4];
  Real_v vertexY[4];
  for (int i = 0; i < 4; i++) {
    // calculate x-y positions of vertex i at this z-height
    vertexX[i] = unplaced.fVerticesX[i + 4] + cf * unplaced.fConnectingComponentsX[i];
    vertexY[i] = unplaced.fVerticesY[i + 4] + cf * unplaced.fConnectingComponentsY[i];
  }
  Real_v seg;
  Real_v frac;
  GetClosestEdge<Real_v>(point, vertexX, vertexY, seg, frac);
  vecCore::MaskedAssign(frac, frac < 0., Real_v(0.));
  Index_v iseg = (Index_v)seg;
  if (unplaced.IsPlanar()) {
    // Normals for the planar case are pre-computed
    Vertex_t const *normals = unplaced.fSurfaceShell.GetNormals();
    normal                  = normals[iseg];
    return;
  }
  Index_v jseg = (iseg + 1) % 4;
  Real_v x0    = vertexX[iseg];
  Real_v y0    = vertexY[iseg];
  Real_v x2    = vertexX[jseg];
  Real_v y2    = vertexY[jseg];
  x0 += frac * (x2 - x0);
  y0 += frac * (y2 - y0);
  Real_v x1 = unplaced.fVerticesX[iseg + 4];
  Real_v y1 = unplaced.fVerticesY[iseg + 4];
  x1 += frac * (unplaced.fVerticesX[jseg + 4] - x1);
  y1 += frac * (unplaced.fVerticesY[jseg + 4] - y1);
  Real_v ax = x1 - x0;
  Real_v ay = y1 - y0;
  Real_v az = unplaced.fDz - point.z();
  Real_v bx = x2 - x0;
  Real_v by = y2 - y0;
  Real_v bz = 0.;
  // Cross product of the vector given by the section segment (that contains the
  // point) at z=point[2] and the vector connecting the point projection to its
  // correspondent on the top edge.
  normal.Set(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx);
  normal.Normalize();
}

//______________________________________________________________________________
template <typename Real_v>
VECGEOM_CUDA_HEADER_BOTH
void GenTrapImplementation::GetClosestEdge(Vector3D<Real_v> const &point, Real_v vertexX[4], Real_v vertexY[4],
                                           Real_v &iseg, Real_v &fraction)
{
  /// Get index of the edge of the quadrilater represented by vert closest to point.
  /// If [P1,P2] is the closest segment and P is the point, the function returns the fraction of the
  /// projection of (P1P) over (P1P2). If projection of P is not in range [P1,P2] return -1.

  using Bool_v = vecCore::Mask_v<Real_v>;

  iseg = Real_v(0.);
  //  Real_v p1X, p1Y, p2X, p2Y;
  Real_v lsq, dx, dy, dpx, dpy, u;
  fraction    = Real_v(-1.);
  Real_v safe = vecCore::NumericLimits<Real_v>::Infinity();
  Real_v ssq  = vecCore::NumericLimits<Real_v>::Infinity();
  for (int i = 0; i < 4; ++i) {
    int j = (i + 1) % 4;
    dx    = vertexX[j] - vertexX[i];
    dy    = vertexY[j] - vertexY[i];
    dpx   = point.x() - vertexX[i];
    dpy   = point.y() - vertexY[i];
    lsq   = dx * dx + dy * dy;
    // Current segment collapsed to a point
    Bool_v collapsed = lsq < kTolerance;
    if (!vecCore::MaskEmpty(collapsed)) {
      vecCore::MaskedAssign(ssq, lsq < kTolerance, dpx * dpx + dpy * dpy);
      // Missing a masked assign allowing to perform multiple assignments...
      vecCore::MaskedAssign(iseg, ssq < safe, (Precision)i);
      vecCore::MaskedAssign(fraction, ssq < safe, Real_v(-1.));
      vecCore::MaskedAssign(safe, ssq < safe, ssq);
      if (vecCore::MaskFull(collapsed)) continue;
    }
    // Projection fraction
    u = (dpx * dx + dpy * dy) / (lsq + kTiny);
    vecCore::MaskedAssign(dpx, u > 1 && !collapsed, point.x() - vertexX[j]);
    vecCore::MaskedAssign(dpy, u > 1 && !collapsed, point.y() - vertexY[j]);
    vecCore::MaskedAssign(dpx, u >= 0 && u <= 1 && !collapsed, dpx - u * dx);
    vecCore::MaskedAssign(dpy, u >= 0 && u <= 1 && !collapsed, dpy - u * dy);
    vecCore::MaskedAssign(u, (u > 1 || u < 0) && !collapsed, Real_v(-1.));
    ssq = dpx * dpx + dpy * dpy;
    vecCore::MaskedAssign(iseg, ssq < safe, (Precision)i);
    vecCore::MaskedAssign(fraction, ssq < safe, u);
    vecCore::MaskedAssign(safe, ssq < safe, ssq);
  }
}

//*****************************
//**** Implementations end here
//*****************************
}
} // End global namespace

#endif /* GENTRAPIMPLEMENTATION_H_ */
