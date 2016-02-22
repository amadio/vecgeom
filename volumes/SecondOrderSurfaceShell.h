/// \file SecondOrderSurfaceShell.h
/// \author: swenzel
///  Created on: Aug 1, 2014
///  Modified and completed: mihaela.gheata@cern.ch

#ifndef VECGEOM_SECONDORDERSURFACESHELL_H_
#define VECGEOM_SECONDORDERSURFACESHELL_H_

#include "base/Vector3D.h"
#include "backend/Backend.h"
#include "volumes/kernel/GenericKernels.h"
#include <iostream>

namespace vecgeom {
/**
 * Templated class providing a (SOA) encapsulation of
 * second order curved surfaces used in generic trap
 */
VECGEOM_DEVICE_FORWARD_DECLARE(class SecondOrderSurfaceShell;)

inline namespace VECGEOM_IMPL_NAMESPACE {

template <int N> class SecondOrderSurfaceShell {

  using Vertex_t = Vector3D<Precision>;

private:
  // caching some important values for each of the curved planes
  Precision fxa[N], fya[N], fxb[N], fyb[N], fxc[N], fyc[N], fxd[N], fyd[N]; /** Coordinates of vertices */
  Precision ftx1[N], fty1[N], ftx2[N], fty2[N];                             /** Connecting components */
  Precision ft1crosst2[N]; /** Cross term ftx1[i]*fty2[i] - ftx2[i]*fty1[i] */
  Precision fDeltatx[N];   /** Term ftx2[i] - ftx1[i] */
  Precision fDeltaty[N];   /** Term fty2[i] - fty1[i] */

  Precision fDz;          /** height of surface (coming from the height of the GenTrap) */
  Precision fDz2;         /** 0.5/fDz */
  Precision fiscurved[N]; /** Indicate which surfaces are planar */
  bool fisplanar;         /** Flag that all surfaces are planar */
  bool fdegenerated[N];   /** Flags for each top-bottom edge marking that this is degenerated */

  Vertex_t fNormals[N]; /** Pre-computed normals for the planar case */

  // pre-computed cross products for normal computation
  Vertex_t fViCrossHi0[N];  /** Pre-computed vi X hi0 */
  Vertex_t fViCrossVj[N];   /** Pre-computed vi X vj */
  Vertex_t fHi1CrossHi0[N]; /** Pre-computed hi1 X hi0 */

public:
  /** @brief SecondOrderSurfaceShell constructor
  * @param verticesx X positions of vertices in array form
  * @param verticesy Y positions of vertices in array form
  * @param dz The half-height of the GenTrap
  */
  VECGEOM_CUDA_HEADER_BOTH
  SecondOrderSurfaceShell(const Precision *verticesx, const Precision *verticesy, Precision dz)
      : fDz(dz), fDz2(0.5 / dz) {
    // Constructor
    // Store vertex coordinates
    Vertex_t va, vb, vc, vd;
    for (int i = 0; i < N; ++i) {
      int j = (i + 1) % N;
      va.Set(verticesx[i], verticesy[i], -dz);
      fxa[i] = verticesx[i];
      fya[i] = verticesy[i];
      vb.Set(verticesx[i + N], verticesy[i + N], dz);
      fxb[i] = verticesx[i + N];
      fyb[i] = verticesy[i + N];
      vc.Set(verticesx[j], verticesy[j], -dz);
      fxc[i] = verticesx[j];
      fyc[i] = verticesy[j];
      vd.Set(verticesx[j + N], verticesy[j + N], dz);
      fxd[i] = verticesx[N + j];
      fyd[i] = verticesy[N + j];
      fdegenerated[i] = (Abs(fxa[i] - fxc[i]) < kTolerance) && (Abs(fya[i] - fyc[i]) < kTolerance) &&
                        (Abs(fxb[i] - fxd[i]) < kTolerance) && (Abs(fyb[i] - fyd[i]) < kTolerance);
      ftx1[i] = fDz2 * (fxb[i] - fxa[i]);
      fty1[i] = fDz2 * (fyb[i] - fya[i]);
      ftx2[i] = fDz2 * (fxd[i] - fxc[i]);
      fty2[i] = fDz2 * (fyd[i] - fyc[i]);

      ft1crosst2[i] = ftx1[i] * fty2[i] - ftx2[i] * fty1[i];
      fDeltatx[i] = ftx2[i] - ftx1[i];
      fDeltaty[i] = fty2[i] - fty1[i];
      fNormals[i] = Vertex_t::Cross(vb - va, vc - va);
      // The computation of normals is done also for the curved surfaces, even if they will not be used
      if (fNormals[i].Mag2() < kTolerance) {
        // points i and i+1/N are overlapping - use i+N and j+N instead
        fNormals[i] = Vertex_t::Cross(vb - va, vd - vb);
        if (fNormals[i].Mag2() < kTolerance)
          fNormals[i].Set(0., 0., 1.); // No surface, just a line
      }
      fNormals[i].Normalize();
      // Cross products used for normal computation
      fViCrossHi0[i] = Vertex_t::Cross(vb - va, vc - va);
      fViCrossVj[i] = Vertex_t::Cross(vb - va, vd - vc);
      fHi1CrossHi0[i] = Vertex_t::Cross(vd - vb, vc - va);
    }

    // Analyze planarity and precompute normals
    fisplanar = true;
    for (int i = 0; i < N; ++i) {
      fiscurved[i] = (((Abs(fxc[i] - fxa[i]) < kTolerance) && (Abs(fyc[i] - fya[i]) < kTolerance)) ||
                      ((Abs(fxd[i] - fxb[i]) < kTolerance) && (Abs(fyd[i] - fyb[i]) < kTolerance)) ||
                      (Abs((fxc[i] - fxa[i]) * (fyd[i] - fyb[i]) - (fxd[i] - fxb[i]) * (fyc[i] - fya[i])) < kTolerance))
                         ? 0
                         : 1;
      if (fiscurved[i])
        fisplanar = false;
    }
  }

  //______________________________________________________________________________
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE
      /** @brief Compute distance to a set of curved/planar surfaces
       * @param point Starting point in the local frame
       * @param dir Direction in the local frame
       */
      typename Backend::precision_v
      DistanceToOut(Vector3D<typename Backend::precision_v> const &point,
                    Vector3D<typename Backend::precision_v> const &dir) const {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    // Planar case
    if (fisplanar)
      return DistanceToOutPlanar<Backend>(point, dir);

    // The algorithmic tolerance in distance
    const Float_t tolerance = 100. * kTolerance;

    Float_t dist(kInfinity);
    Float_t smin[N], smax[N];
    Vector3D<Float_t> unorm;
    Float_t r = -1.;
    Float_t rz;
    Bool_t inside = (Abs(point.z()) < fDz + kTolerance);
    Float_t cross;
    Float_t vertexX[N];
    Float_t vertexY[N];
    Float_t dzp = fDz + point[2];

    // Point may be on the wrong side - check this
    for (int i = 0; i < N; i++) {
      // calculate x-y positions of vertex i at this z-height
      vertexX[i] = fxa[i] + ftx1[i] * dzp;
      vertexY[i] = fya[i] + fty1[i] * dzp;
    }
    for (int i = 0; i < N; i++) {
      if (fdegenerated[i])
        continue;
      int j = (i + 1) % 4;
      Float_t DeltaX = vertexX[j] - vertexX[i];
      Float_t DeltaY = vertexY[j] - vertexY[i];
      cross = (point.x() - vertexX[i]) * DeltaY - (point.y() - vertexY[i]) * DeltaX;
      inside &= (cross > -tolerance);
    }
    // If on the wrong side, return -1.
    Float_t wrongsidedist = -1.;
    MaskedAssign(!inside, wrongsidedist, &dist);
    if (IsEmpty(inside))
      return dist;

    // Solve the second order equation and return distance solutions for each surface
    ComputeSminSmax<Backend>(point, dir, smin, smax);
    for (int i = 0; i < N; ++i) {
      // Check if point(s) is(are) on boundary, and in this case compute normal
      Bool_t crtbound = (Abs(smin[i]) < tolerance || Abs(smax[i]) < tolerance);
      if (!IsEmpty(crtbound)) {
        if (fiscurved[i])
          UNormal<Backend>(point, i, unorm, rz, r);
        else
          unorm = fNormals[i];
      }
      // Starting point may be propagated close to boundary
      // === MaskedMultipleAssign needed
      MaskedAssign(inside && Abs(smin[i]) < tolerance && dir.Dot(unorm) < 0, kInfinity, &smin[i]);
      MaskedAssign(inside && Abs(smax[i]) < tolerance && dir.Dot(unorm) < 0, kInfinity, &smax[i]);

      MaskedAssign(inside && (smin[i] > -tolerance) && (smin[i] < dist), Max(smin[i], 0.), &dist);
      MaskedAssign(inside && (smax[i] > -tolerance) && (smax[i] < dist), Max(smax[i], 0.), &dist);
    }
    MaskedAssign(dist<tolerance && dist>wrongsidedist, 0., &dist);
    return (dist);
  } // end of function

  //______________________________________________________________________________
  /** @brief Compute distance to exiting the set of surfaces in the planar case.
   * @param point Starting point in the local frame
   * @param dir Direction in the local frame
   */
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE typename Backend::precision_v
  DistanceToOutPlanar(Vector3D<typename Backend::precision_v> const &point,
                      Vector3D<typename Backend::precision_v> const &dir) const {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;
    const Float_t tolerance = 100. * kTolerance;

    Vertex_t va;          // vertex i of lower base
    Vector3D<Float_t> pa; // same vertex converted to backend type
    Float_t distance = kInfinity;

    // Check every surface
    Bool_t outside = (Abs(point.z()) > MakePlusTolerant<true>(fDz)); // If point is outside, we need to know
    for (int i = 0; i < N && (!IsFull(outside)); ++i) {
      // Point A is the current vertex on lower Z. P is the point we come from.
      pa.Set(fxa[i], fya[i], -fDz);
      Vector3D<Float_t> vecAP = point - pa;
      // Dot product between AP vector and normal to surface has to be negative
      Float_t dotAPNorm = vecAP.Dot(fNormals[i]);
      Bool_t otherside = (dotAPNorm > 10. * kTolerance);
      // Dot product between direction and normal to surface has to be positive
      Float_t dotDirNorm = dir.Dot(fNormals[i]);
      Bool_t outgoing = (dotDirNorm > 0.);
      dotDirNorm += kTiny; // Avoid division by 0 without changing result
      // Update globally outside flag
      outside |= otherside;
      Bool_t valid = outgoing & (!otherside);
      if (IsEmpty(valid))
        continue;
      Float_t snext = -dotAPNorm / dotDirNorm;
      MaskedAssign(valid && snext < distance, Max(snext, 0.), &distance);
    }
    // Return -1 for points actually outside
    MaskedAssign(distance < tolerance, 0., &distance);
    MaskedAssign(outside, -1., &distance);
    return distance;
  }

  //______________________________________________________________________________
  /** @brief Compute distance to entering the set of surfaces in the planar case.
   * @param point Starting point in the local frame
   * @param dir Direction in the local frame
   */
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE typename Backend::precision_v
  DistanceToInPlanar(Vector3D<typename Backend::precision_v> const &point,
                     Vector3D<typename Backend::precision_v> const &dir, typename Backend::bool_v &done) const {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;
    const Float_t tolerance = 100. * kTolerance;

    Vertex_t va;          // vertex i of lower base
    Vector3D<Float_t> pa; // same vertex converted to backend type
    Float_t distance = kInfinity;

    // Check every surface
    Bool_t inside = (Abs(point.z()) < MakeMinusTolerant<true>(fDz)); // If point is inside, we need to know
    for (int i = 0; i < N; ++i) {
      // Point A is the current vertex on lower Z. P is the point we come from.
      pa.Set(fxa[i], fya[i], -fDz);
      Vector3D<Float_t> vecAP = point - pa;
      // Dot product between AP vector and normal to surface has to be positive
      Float_t dotAPNorm = vecAP.Dot(fNormals[i]);
      Bool_t otherside = (dotAPNorm < -10. * kTolerance);
      // Dot product between direction and normal to surface has to be negative
      Float_t dotDirNorm = dir.Dot(fNormals[i]);
      Bool_t ingoing = (dotDirNorm < 0.);
      dotDirNorm += kTiny; // Avoid division by 0 without changing result
      // Update globally outside flag
      inside &= otherside;
      Bool_t valid = ingoing & (!otherside);
      if (IsEmpty(valid))
        continue;
      Float_t snext = -dotAPNorm / dotDirNorm;
      // Now propagate the point to surface and check if in range
      Vector3D<Float_t> psurf = point + snext * dir;
      valid &= InSurfLimits<Backend>(psurf, i);
      MaskedAssign((!done) && valid && snext < distance, Max(snext, 0.), &distance);
    }
    // Return -1 for points actually inside
    MaskedAssign((!done) && (distance < tolerance), 0., &distance);
    MaskedAssign((!done) && inside, -1., &distance);
    return distance;
  }

  //______________________________________________________________________________
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE
      /**
      * A generic function calculation the distance to a set of curved/planar surfaces
      *
      * things to improve: error handling for boundary cases
      *
      * another possibility relies on the following idea:
      * we always have an even number of planar/curved surfaces. We could organize them in separate substructures...
      */
      typename Backend::precision_v
      DistanceToIn(Vector3D<typename Backend::precision_v> const &point,
                   Vector3D<typename Backend::precision_v> const &dir, typename Backend::bool_v &done) const {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    // Planar case
    if (fisplanar)
      return DistanceToInPlanar<Backend>(point, dir, done);
    Float_t crtdist;
    Vector3D<Float_t> hit;
    Float_t resultdistance(kInfinity);
    Float_t tolerance = 100. * kTolerance;
    Vector3D<Float_t> unorm;
    Float_t r = -1.;
    Float_t rz;
    Bool_t inside = (Abs(point.z()) < fDz - kTolerance);
    Float_t cross;
    Float_t vertexX[N];
    Float_t vertexY[N];
    Float_t dzp = fDz + point[2];
    // Point may be on the wrong side - check this
    for (int i = 0; i < N; i++) {
      // calculate x-y positions of vertex i at this z-height
      vertexX[i] = fxa[i] + ftx1[i] * dzp;
      vertexY[i] = fya[i] + fty1[i] * dzp;
    }
    for (int i = 0; i < N; i++) {
      if (fdegenerated[i])
        continue;
      int j = (i + 1) % 4;
      Float_t DeltaX = vertexX[j] - vertexX[i];
      Float_t DeltaY = vertexY[j] - vertexY[i];
      cross = (point.x() - vertexX[i]) * DeltaY - (point.y() - vertexY[i]) * DeltaX;
      inside &= (cross > tolerance);
    }

    // If on the wrong side, return -1.
    Float_t wrongsidedist = -1.;
    MaskedAssign(inside & (!done), wrongsidedist, &resultdistance);
    Bool_t checked = inside | done;
    if (IsFull(checked))
      return (resultdistance);

    // Now solve the second degree equation to find crossings
    Float_t smin[N], smax[N];
    ComputeSminSmax<Backend>(point, dir, smin, smax);

    // now we need to analyse which of those distances is good
    // does not vectorize
    for (int i = 0; i < N; ++i) {
      crtdist = smin[i];
      // Extrapolate with hit distance candidate
      hit = point + crtdist * dir;
      Bool_t crossing = (crtdist > -tolerance) & (Abs(hit.z()) < fDz + kTolerance);
      // Early skip surface if not in Z range
      if (!IsEmpty(crossing & (!checked))) {
        ;
        // Compute local un-normalized outwards normal direction and hit ratio factors
        UNormal<Backend>(hit, i, unorm, rz, r);
        // Distance have to be positive within tolerance, and crossing must be inwards
        crossing &= (crtdist > -10 * tolerance) & (dir.Dot(unorm) < 0.);
        // Propagated hitpoint must be on surface (rz in [0,1] checked already)
        crossing &= (r >= 0.) & (r <= 1.);
        MaskedAssign(crossing && (!checked) && crtdist < resultdistance, Max(crtdist, 0.), &resultdistance);
      }
      // For the particle(s) not crossing at smin, try smax
      if (!IsFull(crossing | checked)) {
        // Treat only particles not crossing at smin
        crossing = !crossing;
        crtdist = smax[i];
        hit = point + crtdist * dir;
        crossing &= (Abs(hit.z()) < fDz + kTolerance);
        if (IsEmpty(crossing))
          continue;
        UNormal<Backend>(hit, i, unorm, rz, r);
        crossing &= (crtdist > -tolerance) & (dir.Dot(unorm) < 0.);
        crossing &= (r >= 0.) & (r <= 1.);
        MaskedAssign(crossing && (!checked) && crtdist < resultdistance, Max(crtdist, 0.), &resultdistance);
      }
    }
    MaskedAssign(resultdistance<tolerance && resultdistance>wrongsidedist, 0., &resultdistance);
    return (resultdistance);

  } // end distanceToIn function

  //______________________________________________________________________________
  /**
   * @brief A generic function calculation for the safety to a set of curved/planar surfaces.
           Should be smaller than safmax
   * @param point Starting point inside, in the local frame
   * @param safmax current safety value
   */
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE typename Backend::precision_v
  SafetyToOut(Vector3D<typename Backend::precision_v> const &point, typename Backend::precision_v const &safmax) const {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;
    constexpr Precision eps = 100. * kTolerance;

    Float_t safety = safmax;
    Bool_t done = (Abs(safety) < eps);
    if (IsFull(done))
      return (safety);
    Float_t safetyface = kInfinity;

    // loop lateral surfaces
    // We can use the surface normals to get safety for non-curved surfaces
    Vertex_t va;          // vertex i of lower base
    Vector3D<Float_t> pa; // same vertex converted to backend type
    if (fisplanar) {
      for (int i = 0; i < N; ++i) {
        va.Set(fxa[i], fya[i], -fDz);
        pa = va;
        safetyface = (pa - point).Dot(fNormals[i]);
        MaskedAssign((safetyface < safety) && (!done), safetyface, &safety);
      }
      return safety;
    }

    // Not fully planar - use mixed case
    safetyface = SafetyCurved<Backend>(point, Backend::kTrue);
    //  std::cout << "safetycurved = " << safetyface << std::endl;
    MaskedAssign((safetyface < safety) && (!done), safetyface, &safety);
    //  std::cout << "safety = " << safety << std::endl;
    MaskedAssign(safety>0. && safety<eps, 0., &safety);
    return safety;

  } // end SafetyToOut

  //______________________________________________________________________________
  /**
   * @brief A generic function calculation for the safety to a set of curved/planar surfaces.
           Should be smaller than safmax
   * @param point Starting point outside, in the local frame
   * @param safmax current safety value
   */
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE typename Backend::precision_v
  SafetyToIn(Vector3D<typename Backend::precision_v> const &point, typename Backend::precision_v const &safmax) const {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;
    constexpr Precision eps = 100. * kTolerance;

    Float_t safety = safmax;
    Bool_t done = (Abs(safety) < eps);
    if (IsFull(done))
      return (safety);
    Float_t safetyface = kInfinity;

    // loop lateral surfaces
    // We can use the surface normals to get safety for non-curved surfaces
    Vertex_t va;          // vertex i of lower base
    Vector3D<Float_t> pa; // same vertex converted to backend type
    if (fisplanar) {
      for (int i = 0; i < N; ++i) {
        va.Set(fxa[i], fya[i], -fDz);
        pa = va;
        safetyface = (point - pa).Dot(fNormals[i]);
        MaskedAssign((safetyface > safety) && (!done), safetyface, &safety);
      }
      return safety;
    }

    // Not fully planar - use mixed case
    safetyface = SafetyCurved<Backend>(point, Backend::kFalse);
    //  std::cout << "safetycurved = " << safetyface << std::endl;
    MaskedAssign((safetyface > safety) && (!done), safetyface, &safety);
    //  std::cout << "safety = " << safety << std::endl;
    MaskedAssign(safety>0. && safety<eps, 0., &safety);
    return (safety);

  } // end SafetyToIn

  //______________________________________________________________________________
  /**
   * @brief A generic function calculation for the safety to a set of curved/planar surfaces.
           Should be smaller than safmax
   * @param point Starting point
   * @param in Inside value for starting point
   */
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE typename Backend::precision_v
  SafetyCurved(Vector3D<typename Backend::precision_v> const &point, typename Backend::bool_v in) const {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    Float_t safety = kInfinity;
    Float_t safplanar = kInfinity;
    Float_t tolerance = 100 * kTolerance;
    MaskedAssign(!in, -tolerance, &tolerance);

    //  loop over edges connecting points i with i+4
    Float_t vertexX[N];
    Float_t vertexY[N];
    Float_t dx, dy, dpx, dpy, lsq, u;
    Float_t dx1 = 0.0;
    Float_t dx2 = 0.0;
    Float_t dy1 = 0.0;
    Float_t dy2 = 0.0;
    Float_t dzp = fDz + point[2];
    // vectorizes for scalar backend
    for (int i = 0; i < N; i++) {
      // calculate x-y positions of vertex i at this z-height
      vertexX[i] = fxa[i] + ftx1[i] * dzp;
      vertexY[i] = fya[i] + fty1[i] * dzp;
    }
    // Check if point is where it is supposed to be
    Bool_t inside = (Abs(point.z()) < fDz + tolerance);
    Float_t cross;
    for (int i = 0; i < N; i++) {
      if (fdegenerated[i])
        continue;
      int j = (i + 1) % 4;
      Float_t DeltaX = vertexX[j] - vertexX[i];
      Float_t DeltaY = vertexY[j] - vertexY[i];
      cross = (point.x() - vertexX[i]) * DeltaY - (point.y() - vertexY[i]) * DeltaX;
      inside &= (cross > -tolerance);
    }
    Bool_t wrong = in & (!inside);
    wrong |= (!in) & inside;
    if (IsFull(wrong)) {
      safety = -kTolerance;
      return safety;
    }
    Float_t umin = 0.0;
    for (int i = 0; i < N; i++) {
      if (fiscurved[i] == 0) {
        // We can use the surface normals to get safety for non-curved surfaces
        Vertex_t va;          // vertex i of lower base
        Vector3D<Float_t> pa; // same vertex converted to backend type
        va.Set(fxa[i], fya[i], -fDz);
        pa = va;
        Float_t sface = Abs((point - pa).Dot(fNormals[i]));
        MaskedAssign(sface < safplanar, sface, &safplanar);
        continue;
      }
      int j = (i + 1) % N;
      dx = vertexX[j] - vertexX[i];
      dy = vertexY[j] - vertexY[i];
      dpx = point[0] - vertexX[i];
      dpy = point[1] - vertexY[i];
      lsq = dx * dx + dy * dy;
      u = (dpx * dx + dpy * dy) / (lsq + kTiny);
      MaskedAssign(u > 1, point[0] - vertexX[j], &dpx);
      MaskedAssign(u > 1, point[1] - vertexY[j], &dpy);
      MaskedAssign(u >= 0 && u <= 1, dpx - u * dx, &dpx);
      MaskedAssign(u >= 0 && u <= 1, dpy - u * dy, &dpy);
      Float_t ssq = dpx * dpx + dpy * dpy; // safety squared
      MaskedAssign(ssq < safety, fxc[i] - fxa[i], &dx1);
      MaskedAssign(ssq < safety, fxd[i] - fxb[i], &dx2);
      MaskedAssign(ssq < safety, fyc[i] - fya[i], &dy1);
      MaskedAssign(ssq < safety, fyd[i] - fyb[i], &dy2);
      MaskedAssign(ssq < safety, u, &umin);
      MaskedAssign(ssq < safety, ssq, &safety);
    }
    MaskedAssign(umin < 0 || umin > 1, 0.0, &umin);
    dx = dx1 + umin * (dx2 - dx1);
    dy = dy1 + umin * (dy2 - dy1);
    safety *= 1. - 4. * fDz * fDz / (dx * dx + dy * dy + 4. * fDz * fDz);
    safety = Sqrt(safety);
    MaskedAssign(safplanar < safety, safplanar, &safety);
    MaskedAssign(wrong, -safety, &safety);
    return safety;
  } // end SafetyFace

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vertex_t const *GetNormals() const { return fNormals; }

  //______________________________________________________________________________
  /**
   * @brief Computes if point on surface isurf is within surface limits
   * @param point Starting point
   * @param isurf Surface index
   */
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE typename Backend::bool_v
  InSurfLimits(Vector3D<typename Backend::precision_v> const &point, int isurf) const {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;
    // Check first if Z is in range
    Float_t rz = fDz2 * (point.z() + fDz);
    Bool_t insurf = (rz > MakeMinusTolerant<true>(0.)) & (rz < MakePlusTolerant<true>(1.));
    if (IsEmpty(insurf))
      return insurf;

    Float_t r = kInfinity;
    Float_t num = (point.x() - fxa[isurf]) - rz * (fxb[isurf] - fxa[isurf]);
    Float_t denom = (fxc[isurf] - fxa[isurf]) + rz * (fxd[isurf] - fxc[isurf] - fxb[isurf] + fxa[isurf]);
    MaskedAssign((Abs(denom) > 1.e-6), num / denom, &r);
    num = (point.y() - fya[isurf]) - rz * (fyb[isurf] - fya[isurf]);
    denom = (fyc[isurf] - fya[isurf]) + rz * (fyd[isurf] - fyc[isurf] - fyb[isurf] + fya[isurf]);
    MaskedAssign((Abs(denom) > 1.e-6), num / denom, &r);
    insurf &= (r > MakeMinusTolerant<true>(0.)) & (r < MakePlusTolerant<true>(1.));
    return insurf;
  }

  //______________________________________________________________________________
  /** @brief Computes un-normalized normal to surface isurf, on the input point */
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE void UNormal(Vector3D<typename Backend::precision_v> const &point, int isurf,
                                                       Vector3D<typename Backend::precision_v> &unorm,
                                                       typename Backend::precision_v &rz,
                                                       typename Backend::precision_v &r) const {

    // unorm = (vi X hi0) + rz*(vi X vj) + r*(hi1 X hi0)
    //    where: vi, vj are the vectors (AB) and (CD) (see constructor)
    //           hi0 = (AC) and hi1 = (BD)
    //           rz = 0.5*(point.z()+dz)/dz is the vertical ratio
    //           r = ((AP)-rz*vi) / (hi0+rz(vj-vi)) is the horizontal ratio
    // Any point within the surface range should reurn r and rz in the range [0,1]
    // These can be used as surface crossing criteria
    typedef typename Backend::precision_v Float_t;
    rz = fDz2 * (point.z() + fDz);
    /*
      Vector3D<Float_t> a(fxa[isurf], fya[isurf], -fDz);
      Vector3D<Float_t> vi(fxb[isurf]-fxa[isurf], fyb[isurf]-fya[isurf], 2*fDz);
      Vector3D<Float_t> vj(fxd[isurf]-fxc[isurf], fyd[isurf]-fyc[isurf], 2*fDz);
      Vector3D<Float_t> hi0(fxc[isurf]-fxa[isurf], fyc[isurf]-fya[isurf], 0.);
    */
    Float_t num = (point.x() - fxa[isurf]) - rz * (fxb[isurf] - fxa[isurf]);
    Float_t denom = (fxc[isurf] - fxa[isurf]) + rz * (fxd[isurf] - fxc[isurf] - fxb[isurf] + fxa[isurf]);
    MaskedAssign(Abs(denom) > 1.e-6, num / denom, &r);
    num = (point.y() - fya[isurf]) - rz * (fyb[isurf] - fya[isurf]);
    denom = (fyc[isurf] - fya[isurf]) + rz * (fyd[isurf] - fyc[isurf] - fyb[isurf] + fya[isurf]);
    MaskedAssign(Abs(denom) > 1.e-6, num / denom, &r);

    unorm = (Vector3D<Float_t>)fViCrossHi0[isurf] + rz * (Vector3D<Float_t>)fViCrossVj[isurf] +
            r * (Vector3D<Float_t>)fHi1CrossHi0[isurf];
  } // end UNormal

  //______________________________________________________________________________
  /** @brief Solver for the second degree equation for curved surface crossing */
  template <typename Backend>
  VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE
      /**
       * Function to compute smin and smax crossings with the N lateral surfaces.
       */
      void
      ComputeSminSmax(Vector3D<typename Backend::precision_v> const &point,
                      Vector3D<typename Backend::precision_v> const &dir, typename Backend::precision_v smin[N],
                      typename Backend::precision_v smax[N]) const {

    typedef typename Backend::precision_v Float_t;
    const Float_t big = kInfinity;

    Float_t dzp = fDz + point[2];
    // calculate everything needed to solve the second order equation
    Float_t a[N], b[N], c[N], d[N];
    Float_t signa[N], inva[N];

    // vectorizes
    for (int i = 0; i < N; ++i) {
      Float_t xs1 = fxa[i] + ftx1[i] * dzp;
      Float_t ys1 = fya[i] + fty1[i] * dzp;
      Float_t xs2 = fxc[i] + ftx2[i] * dzp;
      Float_t ys2 = fyc[i] + fty2[i] * dzp;
      Float_t dxs = xs2 - xs1;
      Float_t dys = ys2 - ys1;
      a[i] = (fDeltatx[i] * dir[1] - fDeltaty[i] * dir[0] + ft1crosst2[i] * dir[2]) * dir[2];
      b[i] = dxs * dir[1] - dys * dir[0] +
             (fDeltatx[i] * point[1] - fDeltaty[i] * point[0] + fty2[i] * xs1 - fty1[i] * xs2 + ftx1[i] * ys2 -
              ftx2[i] * ys1) *
                 dir[2];
      c[i] = dxs * point[1] - dys * point[0] + xs1 * ys2 - xs2 * ys1;
      d[i] = b[i] * b[i] - 4 * a[i] * c[i];
    }

    // does not vectorize
    for (int i = 0; i < N; ++i) {
      // zero or one to start with
      signa[i] = 0.;
      MaskedAssign(a[i] < -kTolerance, (Float_t)(-Backend::kOne), &signa[i]);
      MaskedAssign(a[i] > kTolerance, (Float_t)Backend::kOne, &signa[i]);
      inva[i] = c[i] / (b[i] * b[i] + kTiny);
      MaskedAssign(Abs(a[i]) > kTolerance, 1. / (2. * a[i]), &inva[i]);
    }

    // vectorizes
    for (int i = 0; i < N; ++i) {
      // treatment for curved surfaces. Invalid solutions will be excluded.

      Float_t sqrtd = signa[i] * Sqrt(Abs(d[i]));
      MaskedAssign(d[i]<0., big, &sqrtd);
      // what is the meaning of this??
      smin[i] = (-b[i] - sqrtd) * inva[i];
      smax[i] = (-b[i] + sqrtd) * inva[i];
    }
    // For the planar surfaces, the above may be wrong, redo the work using
    // just the normal. This does not vectorize
    for (int i = 0; i < N; ++i) {
      if (fiscurved[i])
        continue;
      Vertex_t va;          // vertex i of lower base
      Vector3D<Float_t> pa; // same vertex converted to backend type
      // Point A is the current vertex on lower Z. P is the point we come from.
      pa.Set(fxa[i], fya[i], -fDz);
      Vector3D<Float_t> vecAP = point - pa;
      // Dot product between AP vector and normal to surface has to be negative
      Float_t dotAPNorm = vecAP.Dot(fNormals[i]);
      // Dot product between direction and normal to surface has to be positive
      Float_t dotDirNorm = dir.Dot(fNormals[i]);
      dotDirNorm += kTiny; // Avoid division by 0 without changing result
      smin[i] = -dotAPNorm / dotDirNorm;
      smax[i] = kInfinity; // not to be checked
    }
  } // end ComputeSminSmax

}; // end class definition

} // End inline namespace

} // End global namespace

#endif /* SECONDORDERSURFACESHELL_H_ */
