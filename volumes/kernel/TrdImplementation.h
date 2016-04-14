//
/// @file TrdImplementation.h
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_TRDIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_TRDIMPLEMENTATION_H_

#include "base/Global.h"
#include "base/Transformation3D.h"
#include "volumes/kernel/GenericKernels.h"
#include "volumes/UnplacedTrd.h"
#include "volumes/kernel/shapetypes/TrdTypes.h"
#include <stdlib.h>
#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v_1t(TrdImplementation, TranslationCode, translation::kGeneric, RotationCode,
                                           rotation::kGeneric, typename)

    inline namespace VECGEOM_IMPL_NAMESPACE {

  namespace TrdUtilities {

  /*
   * Checks whether a point (x, y) falls on the left or right half-plane
   * of a line. The line is defined by a (vx, vy) vector, extended to infinity.
   *
   * Of course this can only be used for lines that pass through (0, 0), but
   * you can supply transformed coordinates for the point to check for any line.
   *
   * This simply calculates the magnitude of the cross product of vectors (px, py)
   * and (vx, vy), which is defined as |x| * |v| * sin theta.
   *
   * If the cross product is positive, the point is clockwise of V, or the "right"
   * half-plane. If it's negative, the point is CCW and on the "left" half-plane.
   */

  template <typename Backend>
  VECGEOM_INLINE VECGEOM_CUDA_HEADER_BOTH void
  PointLineOrientation(typename Backend::precision_v const &px, typename Backend::precision_v const &py,
                       Precision const &vx, Precision const &vy, typename Backend::precision_v &crossProduct) {
    crossProduct = vx * py - vy * px;
  }
  /*
   * Check intersection of the trajectory of a particle with a segment
   * that's bound from -Ylimit to +Ylimit.j
   *
   * All points of the along-vector of a plane lie on
   * s * (alongX, alongY)
   * All points of the trajectory of the particle lie on
   * (x, y) + t * (vx, vy)
   * Thefore, it must hold that s * (alongX, alongY) == (x, y) + t * (vx, vy)
   * Solving by t we get t = (alongY*x - alongX*y) / (vy*alongX - vx*alongY)
   *
   * t gives the distance, but how to make sure hitpoint is inside the
   * segment and not just the infinite line defined by the segment?
   *
   * Check that |hity| <= Ylimit
   */

  template <typename Backend>
  VECGEOM_INLINE VECGEOM_CUDA_HEADER_BOTH void
  PlaneTrajectoryIntersection(typename Backend::precision_v const &alongX, typename Backend::precision_v const &alongY,
                              typename Backend::precision_v const &ylimit, typename Backend::precision_v const &posx,
                              typename Backend::precision_v const &posy, typename Backend::precision_v const &dirx,
                              typename Backend::precision_v const &diry, typename Backend::precision_v &dist,
                              typename Backend::bool_v &ok) {
    typedef typename Backend::precision_v Float_t;

    dist = (alongY * posx - alongX * posy) / (diry * alongX - dirx * alongY);

    Float_t hity = posy + dist * diry;
    ok = Abs(hity) <= ylimit && dist > 0;
  }

  template <typename Backend, bool forY, bool mirroredPoint, bool toInside>
  VECGEOM_INLINE VECGEOM_CUDA_HEADER_BOTH void
  FaceTrajectoryIntersection(UnplacedTrd const &trd, Vector3D<typename Backend::precision_v> const &pos,
                             Vector3D<typename Backend::precision_v> const &dir, typename Backend::precision_v &dist,
                             typename Backend::bool_v &ok) {
    typedef typename Backend::precision_v Float_t;
    // typedef typename Backend::bool_v Bool_t;
    Float_t alongV, posV, dirV, posK, dirK, fV, fK, halfKplus, v1, ndotv;
    //    fNormals[0].Set(-fCalfX, 0., fFx*fCalfX);
    //    fNormals[1].Set(fCalfX, 0., fFx*fCalfX);
    //    fNormals[2].Set(0., -fCalfY, fFy*fCalfY);
    //    fNormals[3].Set(0., fCalfY, fFy*fCalfY);
    if (forY) {
      alongV = trd.y2minusy1();
      v1 = trd.dy1();
      posV = pos.y();
      posK = pos.x();
      dirV = dir.y();
      dirK = dir.x();
      fK = trd.fx();
      fV = trd.fy();
      halfKplus = trd.halfx1plusx2();
    } else {
      alongV = trd.x2minusx1();
      v1 = trd.dx1();
      posV = pos.x();
      posK = pos.y();
      dirV = dir.x();
      dirK = dir.y();
      fK = trd.fy();
      fV = trd.fx();
      halfKplus = trd.halfy1plusy2();
    }
    if (mirroredPoint) {
      posV *= -1.;
      dirV *= -1.;
    }

    ndotv = dirV + fV * dir.z();
    if (toInside)
      ok = ndotv < 0.;
    else
      ok = ndotv > 0.;
    if (IsEmpty(ok))
      return;
    Float_t alongZ = Float_t(2.0) * trd.dz();

    // distance from trajectory to face
    dist = (alongZ * (posV - v1) - alongV * (pos.z() + trd.dz())) / (dir.z() * alongV - dirV * alongZ + kTiny);
    ok &= dist > MakeMinusTolerant<true>(0.);
    if (Any(ok)) {
      // need to make sure z hit falls within bounds
      Float_t hitz = pos.z() + dist * dir.z();
      ok &= Abs(hitz) <= trd.dz();
      // need to make sure hit on varying dimension falls within bounds
      Float_t hitk = posK + dist * dirK;
      Float_t dK = halfKplus - fK * hitz; // calculate the width of the varying dimension at hitz
      ok &= Abs(hitk) <= dK;
      MaskedAssign(ok & (Abs(dist) < kHalfTolerance), 0., &dist);
    }
  }

  template <typename Backend, typename trdTypeT, bool inside>
  VECGEOM_INLINE VECGEOM_CUDA_HEADER_BOTH void Safety(UnplacedTrd const &trd,
                                                      Vector3D<typename Backend::precision_v> const &pos,
                                                      typename Backend::precision_v &dist) {
    using namespace TrdTypes;
    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    Float_t safz = trd.dz() - Abs(pos.z());
    // std::cout << "safz: " << safz << std::endl;
    dist = safz;

    Float_t distx = trd.halfx1plusx2() - trd.fx() * pos.z();
    Bool_t okx = distx >= 0;
    Float_t safx = (distx - Abs(pos.x())) * trd.calfx();
    MaskedAssign(okx && safx < dist, safx, &dist);
    // std::cout << "safx: " << safx << std::endl;

    if (checkVaryingY<trdTypeT>(trd)) {
      Float_t disty = trd.halfy1plusy2() - trd.fy() * pos.z();
      Bool_t oky = disty >= 0;
      Float_t safy = (disty - Abs(pos.y())) * trd.calfy();
      MaskedAssign(oky && safy < dist, safy, &dist);
    } else {
      Float_t safy = trd.dy1() - Abs(pos.y());
      MaskedAssign(safy < dist, safy, &dist);
    }
    if (!inside)
      dist = -dist;
  }

  template <typename Backend, typename trdTypeT, bool surfaceT>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE static void
  UnplacedInside(UnplacedTrd const &trd, Vector3D<typename Backend::precision_v> const &point,
                 typename Backend::bool_v &completelyinside, typename Backend::bool_v &completelyoutside) {

    using namespace TrdUtilities;
    using namespace TrdTypes;
    typedef typename Backend::precision_v Float_t;
    // typedef typename Backend::bool_v Bool_t;

    Float_t pzPlusDz = point.z() + trd.dz();

    // inside Z?
    completelyoutside = Abs(point.z()) > MakePlusTolerant<surfaceT>(trd.dz());
    if (surfaceT)
      completelyinside = Abs(point.z()) < MakeMinusTolerant<surfaceT>(trd.dz());

    // inside X?
    Float_t cross;
    // Note: we cannot compare directly the cross product with the surface tolerance, but with
    // the tolerance multiplied by the length of the lateral segment connecting dx1 and dx2
    PointLineOrientation<Backend>(Abs(point.x()) - trd.dx1(), pzPlusDz, trd.x2minusx1(), 2.0 * trd.dz(), cross);
    if (surfaceT) {
      completelyoutside |= cross < -trd.ToleranceX();
      completelyinside &= cross > trd.ToleranceX();
    } else {
      completelyoutside |= cross < 0;
    }

    // inside Y?
    if (HasVaryingY<trdTypeT>::value != TrdTypes::kNo) {
      // If Trd type is unknown don't bother with a runtime check, assume the general case
      PointLineOrientation<Backend>(Abs(point.y()) - trd.dy1(), pzPlusDz, trd.y2minusy1(), 2.0 * trd.dz(), cross);
      if (surfaceT) {
        completelyoutside |= cross < -trd.ToleranceY();
        completelyinside &= cross > trd.ToleranceY();
      } else {
        completelyoutside |= cross < 0;
      }
    } else {
      completelyoutside |= Abs(point.y()) > MakePlusTolerant<surfaceT>(trd.dy1());
      if (surfaceT)
        completelyinside &= Abs(point.y()) < MakeMinusTolerant<surfaceT>(trd.dy1());
    }
  }

  } // Trd utilities

  class PlacedTrd;

  template <TranslationCode transCodeT, RotationCode rotCodeT, typename trdTypeT> struct TrdImplementation {

    static const int transC = transCodeT;
    static const int rotC = rotCodeT;

    using PlacedShape_t = PlacedTrd;
    using UnplacedShape_t = UnplacedTrd;

    VECGEOM_CUDA_HEADER_BOTH
    static void PrintType() {
      printf("SpecializedTrd<%i, %i, %s>", transCodeT, rotCodeT, trdTypeT::toString());
    }

    template <typename Stream>
    static void PrintType( Stream & s ) {
      s << "SpecializedTrd<" << transCodeT << "," << rotCodeT << "," << trdTypeT::toString() << ">";
    }

    template <typename Stream>
    static void PrintImplementationType(Stream &s) {
      s << "TrdImplemenation<" << transCodeT << "," << rotCodeT << "," << trdTypeT::toString() << ">";
    }

    template <typename Stream>
    static void PrintUnplacedType(Stream &s) { s << "UnplacedTrd"; }

    template <typename Backend>
    VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE static void
    UnplacedContains(UnplacedTrd const &trd, Vector3D<typename Backend::precision_v> const &point,
                     typename Backend::bool_v &inside) {

      typename Backend::bool_v unused;
      TrdUtilities::UnplacedInside<Backend, trdTypeT, false>(trd, point, unused, inside);
      inside = !inside;
    }

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE static void
    Contains(UnplacedTrd const &trd, Transformation3D const &transformation,
             Vector3D<typename Backend::precision_v> const &point, Vector3D<typename Backend::precision_v> &localPoint,
             typename Backend::bool_v &inside) {

      typename Backend::bool_v unused;
      localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
      TrdUtilities::UnplacedInside<Backend, trdTypeT, false>(trd, localPoint, unused, inside);
      inside = !inside;
    }

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE static void
    Inside(UnplacedTrd const &trd, Transformation3D const &transformation,
           Vector3D<typename Backend::precision_v> const &point, typename Backend::inside_v &inside) {
      typedef typename Backend::bool_v Bool_t;
      Vector3D<typename Backend::precision_v> localpoint;
      localpoint = transformation.Transform<transCodeT, rotCodeT>(point);
      inside = EInside::kOutside;

      Bool_t completelyoutside, completelyinside;
      TrdUtilities::UnplacedInside<Backend, trdTypeT, true>(trd, localpoint, completelyinside, completelyoutside);
      inside = EInside::kSurface;
      MaskedAssign(completelyinside, EInside::kInside, &inside);
      MaskedAssign(completelyoutside, EInside::kOutside, &inside);
    }

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE static void
    DistanceToIn(UnplacedTrd const &trd, Transformation3D const &transformation,
                 Vector3D<typename Backend::precision_v> const &point,
                 Vector3D<typename Backend::precision_v> const &direction,
                 typename Backend::precision_v const & /*stepMax*/, typename Backend::precision_v &distance) {

      using namespace TrdUtilities;
      using namespace TrdTypes;
      typedef typename Backend::bool_v Bool_t;
      typedef typename Backend::precision_v Float_t;

      Float_t hitx, hity;
      // Float_t hitz;

      Vector3D<Float_t> pos_local;
      Vector3D<Float_t> dir_local;
      distance = kInfinity;

      transformation.Transform<transCodeT, rotCodeT>(point, pos_local);
      transformation.TransformDirection<rotCodeT>(direction, dir_local);

      // hit Z faces?
      Bool_t inz = Abs(pos_local.z()) < MakeMinusTolerant<true>(trd.dz());
      Float_t distx = trd.halfx1plusx2() - trd.fx() * pos_local.z();
      Bool_t inx = (distx - Abs(pos_local.x())) * trd.calfx() > MakePlusTolerant<true>(0.);
      Float_t disty;
      Bool_t iny;
      if (checkVaryingY<trdTypeT>(trd)) {
        disty = trd.halfy1plusy2() - trd.fy() * pos_local.z();
        iny = (disty - Abs(pos_local.y())) * trd.calfy() > MakePlusTolerant<true>(0.);
      } else {
        disty = Abs(pos_local.y()) - trd.dy1();
        iny = disty < MakeMinusTolerant<true>(0.);
      }
      Bool_t inside = inx & iny & inz;
      MaskedAssign(inside, -1., &distance);
      Bool_t done = inside;
      Bool_t okz = pos_local.z() * dir_local.z() < 0;
      okz &= !inz;
      if (!IsEmpty(okz)) {
        Float_t distz = (Abs(pos_local.z()) - trd.dz()) / Abs(dir_local.z());
        // exclude case in which particle is going away
        hitx = Abs(pos_local.x() + distz * dir_local.x());
        hity = Abs(pos_local.y() + distz * dir_local.y());

        // hitting top face?
        Bool_t okzt = pos_local.z() > (trd.dz() - kHalfTolerance) && hitx <= trd.dx2() && hity <= trd.dy2();
        // hitting bottom face?
        Bool_t okzb = pos_local.z() < (-trd.dz() + kHalfTolerance) && hitx <= trd.dx1() && hity <= trd.dy1();

        okz &= (okzt | okzb);
        MaskedAssign(okz, distz, &distance);
      }
      done |= okz;
      if (IsFull(done)) {
        MaskedAssign(Abs(distance) < kHalfTolerance, 0., &distance);
        return;
      }

      // hitting X faces?
      Bool_t okx = Backend::kFalse;
      if (!IsFull(inx)) {

        FaceTrajectoryIntersection<Backend, false, false, true>(trd, pos_local, dir_local, distx, okx);
        MaskedAssign(okx, distx, &distance);

        FaceTrajectoryIntersection<Backend, false, true, true>(trd, pos_local, dir_local, distx, okx);
        MaskedAssign(okx, distx, &distance);
      }
      done |= okx;
      if (IsFull(done)) {
        MaskedAssign(Abs(distance) < kHalfTolerance, 0., &distance);
        return;
      }

      // hitting Y faces?
      Bool_t oky;
      if (checkVaryingY<trdTypeT>(trd)) {
        if (!IsFull(iny)) {
          FaceTrajectoryIntersection<Backend, true, false, true>(trd, pos_local, dir_local, disty, oky);
          MaskedAssign(oky, disty, &distance);

          FaceTrajectoryIntersection<Backend, true, true, true>(trd, pos_local, dir_local, disty, oky);
          MaskedAssign(oky, disty, &distance);
        }
      } else {
        if (!IsFull(iny)) {
          disty /= Abs(dir_local.y());
          Float_t zhit = pos_local.z() + disty * dir_local.z();
          Float_t xhit = pos_local.x() + disty * dir_local.x();
          Float_t dx = trd.halfx1plusx2() - trd.fx() * zhit;
          oky = pos_local.y() * dir_local.y() < 0 && disty > -kHalfTolerance && Abs(xhit) < dx && Abs(zhit) < trd.dz();
          MaskedAssign(oky, disty, &distance);
        }
      }
      MaskedAssign(Abs(distance) < kHalfTolerance, 0., &distance);
    }

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE static void
    DistanceToOut(UnplacedTrd const &trd, Vector3D<typename Backend::precision_v> const &point,
                  Vector3D<typename Backend::precision_v> const &dir, typename Backend::precision_v const & /*stepMax*/,
                  typename Backend::precision_v &distance) {

      using namespace TrdUtilities;
      using namespace TrdTypes;
      typedef typename Backend::bool_v Bool_t;
      typedef typename Backend::precision_v Float_t;

      Float_t hitx, hity;
      // Float_t hitz;
      // Bool_t done = Backend::kFalse;
      distance = Float_t(0.0);

      // hit top Z face?
      Float_t invdir = 1. / Abs(dir.z() + kTiny);
      Float_t safz = trd.dz() - Abs(point.z());
      Bool_t out = safz < MakeMinusTolerant<true>(0.);
      Float_t distx = trd.halfx1plusx2() - trd.fx() * point.z();
      out |= (distx - Abs(point.x())) * trd.calfx() < MakeMinusTolerant<true>(0.);
      Float_t disty;
      if (checkVaryingY<trdTypeT>(trd)) {
        disty = trd.halfy1plusy2() - trd.fy() * point.z();
        out |= (disty - Abs(point.y())) * trd.calfy() < MakeMinusTolerant<true>(0.);
      } else {
        disty = trd.dy1() - Abs(point.y());
        out |= disty < MakeMinusTolerant<true>(0.);
      }
      if (/*Backend::early_returns && */ IsFull(out)) {
        distance = -1.;
        return;
      }
      Bool_t okzt = dir.z() > 0;
      if (!IsEmpty(okzt)) {
        Float_t distz = (trd.dz() - point.z()) * invdir;
        hitx = Abs(point.x() + distz * dir.x());
        hity = Abs(point.y() + distz * dir.y());
        okzt &= hitx <= trd.dx2() && hity <= trd.dy2();
        MaskedAssign(okzt, distz, &distance);
        if (Backend::early_returns && IsFull(okzt)) {
          MaskedAssign(Abs(distance) < kHalfTolerance, 0., &distance);
          return;
        }
      }

      // hit bottom Z face?
      Bool_t okzb = dir.z() < 0;
      if (!IsEmpty(okzb)) {
        Float_t distz = (point.z() + trd.dz()) * invdir;
        hitx = Abs(point.x() + distz * dir.x());
        hity = Abs(point.y() + distz * dir.y());
        okzb &= hitx <= trd.dx1() && hity <= trd.dy1();
        MaskedAssign(okzb, distz, &distance);
        if (Backend::early_returns && IsFull(okzb)) {
          MaskedAssign(Abs(distance) < kHalfTolerance, 0., &distance);
          return;
        }
      }

      // hitting X faces?
      Bool_t okx;

      FaceTrajectoryIntersection<Backend, false, false, false>(trd, point, dir, distx, okx);

      MaskedAssign(okx, distx, &distance);
      if (Backend::early_returns && IsFull(okx)) {
        MaskedAssign(Abs(distance) < kHalfTolerance, 0., &distance);
        return;
      }

      FaceTrajectoryIntersection<Backend, false, true, false>(trd, point, dir, distx, okx);
      MaskedAssign(okx, distx, &distance);
      if (Backend::early_returns && IsFull(okx)) {
        MaskedAssign(Abs(distance) < kHalfTolerance, 0., &distance);
        return;
      }

      // hitting Y faces?
      Bool_t oky;

      if (checkVaryingY<trdTypeT>(trd)) {
        FaceTrajectoryIntersection<Backend, true, false, false>(trd, point, dir, disty, oky);
        MaskedAssign(oky, disty, &distance);
        if (Backend::early_returns && IsFull(oky)) {
          MaskedAssign(Abs(distance) < kHalfTolerance, 0., &distance);
          return;
        }

        FaceTrajectoryIntersection<Backend, true, true, false>(trd, point, dir, disty, oky);
        MaskedAssign(oky, disty, &distance);
      } else {
        Float_t plane = trd.dy1();
        MaskedAssign(dir.y() < 0, -trd.dy1(), &plane);
        disty = (plane - point.y()) / dir.y();
        Float_t zhit = point.z() + disty * dir.z();
        Float_t xhit = point.x() + disty * dir.x();
        Float_t dx = trd.halfx1plusx2() - trd.fx() * zhit;
        oky = Abs(xhit) < dx && Abs(zhit) < trd.dz();
        MaskedAssign(oky, disty, &distance);
      }
      MaskedAssign(Abs(distance) < kHalfTolerance, 0., &distance);
      MaskedAssign(out, -1., &distance);
    }

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE static void
    SafetyToIn(UnplacedTrd const &trd, Transformation3D const &transformation,
               Vector3D<typename Backend::precision_v> const &point, typename Backend::precision_v &safety) {
      using namespace TrdUtilities;
      typedef typename Backend::precision_v Float_t;
      Vector3D<Float_t> pos_local;
      transformation.Transform<transCodeT, rotCodeT>(point, pos_local);
      Safety<Backend, trdTypeT, false>(trd, pos_local, safety);
    }

    template <class Backend>
    VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE static void
    SafetyToOut(UnplacedTrd const &trd, Vector3D<typename Backend::precision_v> const &point,
                typename Backend::precision_v &safety) {
      using namespace TrdUtilities;
      Safety<Backend, trdTypeT, true>(trd, point, safety);
    }
  };
}
} // End global namespace

#endif // VECGEOM_VOLUMES_KERNEL_TRDIMPLEMENTATION_H_
