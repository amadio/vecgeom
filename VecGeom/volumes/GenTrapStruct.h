/*
 * GenTrapStruct.h
 *
 *  Created on: 17.07.2016
 *      Author: mgheata
 */

#ifndef VECGEOM_VOLUMES_GENTRAPSTRUCT_H_
#define VECGEOM_VOLUMES_GENTRAPSTRUCT_H_
#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/SecondOrderSurfaceShell.h"

#ifndef VECCORE_CUDA
#include "VecGeom/volumes/TessellatedSection.h"
#endif

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

// A plain struct without member functions to encapsulate just the parameters
// of a generic trapezoid
template <typename T = double>
struct GenTrapStruct {
  using Vertex_t = Vector3D<T>;

  Vertex_t fBBdimensions; /** Bounding box dimensions */
  Vertex_t fBBorigin;     /** Bounding box origin */
  Vertex_t fVertices[8];  /** The eight points that define the Arb8 */

  // we also store this in SOA form
  T fVerticesX[8]; /** Backed-up X positions of vertices */
  T fVerticesY[8]; /** Backed-up Y positions of vertices */
  T fTwist[4];     /** Twist angles */

  T fDz;            /** The half-height of the GenTrap */
  T fInverseDz;     /** Pre-computed 1/fDz */
  T fHalfInverseDz; /** Pre-computed 0.5/fDz */
  bool fIsTwisted;  /** Twisted flag */

  // we store the connecting vectors in SOA Form
  // these vectors are used to calculate the polygon at a certain z-height
  // moreover: they can be precomputed !!
  // Compute intersection between Z plane containing point and the shape
  //
  T fConnectingComponentsX[4]; /** X components of connecting bottom-top vectors vi */
  T fConnectingComponentsY[4]; /** Y components of connecting bottom-top vectors vi */

  T fDeltaX[8]; /** X components of connecting horizontal vectors hij */
  T fDeltaY[8]; /** Y components of connecting horizontal vectors hij */

  bool fDegenerated[4]; /** Flags for each top-bottom edge marking that this is degenerated */

  SecondOrderSurfaceShell<4> fSurfaceShell; /** Utility class for twisted surface algorithms */

#ifndef VECCORE_CUDA
  TessellatedSection<T> *fTslHelper = nullptr; /** SIMD helper using tessellated clusters */
#endif

  VECCORE_ATT_HOST_DEVICE
  GenTrapStruct()
      : fBBdimensions(), fBBorigin(), fVertices(), fVerticesX(), fVerticesY(), fDz(0.), fInverseDz(0.),
        fHalfInverseDz(0.), fIsTwisted(false), fConnectingComponentsX(), fConnectingComponentsY(), fDeltaX(), fDeltaY(),
        fSurfaceShell()
  {
    // Dummy constructor
  }

  VECCORE_ATT_HOST_DEVICE
  GenTrapStruct(const Precision verticesx[], const Precision verticesy[], Precision halfzheight)
      : fBBdimensions(), fBBorigin(), fVertices(), fVerticesX(), fVerticesY(), fDz(0.), fInverseDz(0.),
        fHalfInverseDz(0.), fIsTwisted(false), fConnectingComponentsX(), fConnectingComponentsY(), fDeltaX(), fDeltaY(),
        fSurfaceShell()
  {
    // Constructor
    Initialize(verticesx, verticesy, halfzheight);
  }

  VECCORE_ATT_HOST_DEVICE
  bool Initialize(const Precision verticesx[], const Precision verticesy[], Precision halfzheight)
  {
    // Initialization based on vertices and half length
    fDz            = halfzheight;
    fInverseDz     = 1. / halfzheight;
    fHalfInverseDz = 0.5 / halfzheight;
    fIsTwisted     = false;
    fSurfaceShell.Initialize(verticesx, verticesy, halfzheight);

    // Set vertices in Vector3D form
    for (int i = 0; i < 4; ++i) {
      fVertices[i].operator[](0) = verticesx[i];
      fVertices[i].operator[](1) = verticesy[i];
      fVertices[i].operator[](2) = -halfzheight;
    }
    for (int i = 4; i < 8; ++i) {
      fVertices[i].operator[](0) = verticesx[i];
      fVertices[i].operator[](1) = verticesy[i];
      fVertices[i].operator[](2) = halfzheight;
    }

    for (int i = 0; i < 4; ++i) {
      int j           = (i + 1) % 4;
      fDegenerated[i] = (Abs(verticesx[i] - verticesx[j]) < kTolerance) &&
                        (Abs(verticesy[i] - verticesy[j]) < kTolerance) &&
                        (Abs(verticesx[i + 4] - verticesx[j + 4]) < kTolerance) &&
                        (Abs(verticesy[i + 4] - verticesy[j + 4]) < kTolerance);
    }

    // Make sure vertices are defined clockwise
    Precision sum1 = 0.;
    Precision sum2 = 0.;
    for (int i = 0; i < 4; ++i) {
      int j = (i + 1) % 4;
      sum1 += fVertices[i].x() * fVertices[j].y() - fVertices[j].x() * fVertices[i].y();
      sum2 += fVertices[i + 4].x() * fVertices[j + 4].y() - fVertices[j + 4].x() * fVertices[i + 4].y();
    }

    // Me should generate an exception here
    if (sum1 * sum2 < -kTolerance) {
      printf("ERROR: Unplaced generic trap defined with opposite clockwise\n");
      Print();
      return false;
    }

    // Revert sequence of vertices to have them clockwise
    if (sum1 > kTolerance) {
      printf("INFO: Reverting to clockwise vertices of GenTrap shape:\n");
      Print();
      Vertex_t vtemp;
      vtemp        = fVertices[1];
      fVertices[1] = fVertices[3];
      fVertices[3] = vtemp;
      vtemp        = fVertices[5];
      fVertices[5] = fVertices[7];
      fVertices[7] = vtemp;
    }

    // Initialize the vertices components and connecting components
    for (int i = 0; i < 4; ++i) {
      fConnectingComponentsX[i] = (fVertices[i] - fVertices[i + 4]).x();
      fConnectingComponentsY[i] = (fVertices[i] - fVertices[i + 4]).y();
      fVerticesX[i]             = fVertices[i].x();
      fVerticesX[i + 4]         = fVertices[i + 4].x();
      fVerticesY[i]             = fVertices[i].y();
      fVerticesY[i + 4]         = fVertices[i + 4].y();
    }

    // Initialize components of horizontal connecting vectors
    for (int i = 0; i < 4; ++i) {
      int j          = (i + 1) % 4;
      fDeltaX[i]     = fVerticesX[j] - fVerticesX[i];
      fDeltaX[i + 4] = fVerticesX[j + 4] - fVerticesX[i + 4];
      fDeltaY[i]     = fVerticesY[j] - fVerticesY[i];
      fDeltaY[i + 4] = fVerticesY[j + 4] - fVerticesY[i + 4];
    }

    // Check that opposite segments are not crossing -> fatal exception
    if (SegmentsCrossing(fVertices[0], fVertices[1], fVertices[3], fVertices[2]) ||
        SegmentsCrossing(fVertices[1], fVertices[2], fVertices[0], fVertices[3]) ||
        SegmentsCrossing(fVertices[4], fVertices[5], fVertices[7], fVertices[6]) ||
        SegmentsCrossing(fVertices[5], fVertices[6], fVertices[4], fVertices[7])) {
      printf("ERROR: Unplaced generic trap defined with crossing opposite segments\n");
      Print();
      return false;
    }

    // Check that top and bottom quadrilaterals are convex
    if (!ComputeIsConvexQuadrilaterals()) {
      printf("ERROR: Unplaced generic trap defined with top/bottom quadrilaterals not convex\n");
      Print();
      return false;
    }

    fIsTwisted = ComputeIsTwisted();

#ifndef VECCORE_CUDA
    // Create the tessellated helper if  the faces are planar
    if (IsPlanar()) {
      fTslHelper = new TessellatedSection<T>(4, -fDz, fDz);
      fTslHelper->AddQuadrilateralFacet(fVertices[0], fVertices[4], fVertices[5], fVertices[1]);
      fTslHelper->AddQuadrilateralFacet(fVertices[1], fVertices[5], fVertices[6], fVertices[2]);
      fTslHelper->AddQuadrilateralFacet(fVertices[2], fVertices[6], fVertices[7], fVertices[3]);
      fTslHelper->AddQuadrilateralFacet(fVertices[3], fVertices[7], fVertices[4], fVertices[0]);
    }
#endif
    ComputeBoundingBox();
    return true;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsPlanar() const { return (!fIsTwisted); }

  VECCORE_ATT_HOST_DEVICE
  void ComputeBoundingBox()
  {
    // Computes bounding box parameters
    Vertex_t aMin, aMax;
    Extent(aMin, aMax);
    fBBorigin     = 0.5 * (aMin + aMax);
    fBBdimensions = 0.5 * (aMax - aMin);
  }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vertex_t &aMin, Vertex_t &aMax) const
  {
    // Returns the full 3D cartesian extent of the solid.
    aMin = aMax = fVertices[0];
    aMin[2]     = -fDz;
    aMax[2]     = fDz;
    for (int i = 0; i < 4; ++i) {
      // lower -fDz vertices
      if (aMin[0] > fVertices[i].x()) aMin[0] = fVertices[i].x();
      if (aMax[0] < fVertices[i].x()) aMax[0] = fVertices[i].x();
      if (aMin[1] > fVertices[i].y()) aMin[1] = fVertices[i].y();
      if (aMax[1] < fVertices[i].y()) aMax[1] = fVertices[i].y();
      // upper fDz vertices
      if (aMin[0] > fVertices[i + 4].x()) aMin[0] = fVertices[i + 4].x();
      if (aMax[0] < fVertices[i + 4].x()) aMax[0] = fVertices[i + 4].x();
      if (aMin[1] > fVertices[i + 4].y()) aMin[1] = fVertices[i + 4].y();
      if (aMax[1] < fVertices[i + 4].y()) aMax[1] = fVertices[i + 4].y();
    }
  }

  VECCORE_ATT_HOST_DEVICE
  bool ComputeIsConvexQuadrilaterals()
  {
    // Computes if this gentrap top and bottom quadrilaterals are convex. The vertices
    // have to be pre-ordered clockwise in the XY plane.

    // The cross product of all vector pairs corresponding to ordered consecutive
    // segments has to be positive.
    for (int i = 0; i < 4; ++i) {
      int j = (i + 1) % 4;
      // Bottom face
      Precision crossij = fDeltaX[i] * fDeltaY[j] - fDeltaY[i] * fDeltaX[j];
      if (crossij > kTolerance) return false;
      // Top face
      crossij = fDeltaX[i + 4] * fDeltaY[j + 4] - fDeltaY[i + 4] * fDeltaX[j + 4];
      if (crossij > kTolerance) return false;
    }
    return true;
  }

  VECCORE_ATT_HOST_DEVICE
  bool ComputeIsTwisted()
  {
    // Check if the trapezoid is twisted. A lateral face is twisted if the top and
    // bottom segments are not parallel (cross product not null)

    bool twisted = false;
    Precision dx1, dy1, dx2, dy2;
    const int nv = 4; // half the number of verices

    for (int i = 0; i < 4; ++i) {
      dx1 = fVertices[(i + 1) % nv].x() - fVertices[i].x();
      dy1 = fVertices[(i + 1) % nv].y() - fVertices[i].y();
      if ((dx1 == 0) && (dy1 == 0)) {
        continue;
      }

      dx2 = fVertices[nv + (i + 1) % nv].x() - fVertices[nv + i].x();
      dy2 = fVertices[nv + (i + 1) % nv].y() - fVertices[nv + i].y();

      if ((dx2 == 0 && dy2 == 0)) {
        continue;
      }
      fTwist[i] = std::fabs(dy1 * dx2 - dx1 * dy2);
      if (fTwist[i] < kTolerance) {
        fTwist[i] = 0.;
        continue;
      }
      twisted = true;
    }
    return twisted;
  }

  VECCORE_ATT_HOST_DEVICE
  void Print() const
  {
    printf("UnplacedGenTrap: { halfZ: %f mm,  planar: %s }\n",
       fDz, (IsPlanar() ? "true" : "false"));
    printf("        --------------------------------------\n");
    for (int i = 0; i < 8; ++i) {
      printf("        #%d", i);
      printf("   vx = %f mm", fVertices[i].x());
      printf("   vy = %f mm\n", fVertices[i].y());
    }
    printf("        --------------------------------------\n");
  }

  VECCORE_ATT_HOST_DEVICE
  bool SegmentsCrossing(Vertex_t p, Vertex_t p1, Vertex_t q, Vertex_t q1) const
  {
    // Check if 2 segments defined by (p,p1) and (q,q1) are crossing.
    // See: http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    using Vector     = Vertex_t;
    Vector r         = p1 - p; // p1 = p+r
    Vector s         = q1 - q; // q1 = q+s
    Vector r_cross_s = Vector::Cross(r, s);
    if (r_cross_s.Mag2() < kTolerance) // parallel, colinear or degenerated - ignore crossing
      return false;
    // The segments are crossing if the vector equations:
    //   t * (r x s) = (q-p) x s
    //   u * (r x s) = (q-p) x r
    // give t and u values in the range (0,1).
    // We allow crossing within kTolerance, so the interval becomes (kTolerance, 1-kTolerance)
    Precision t = Vector::Cross(q - p, s).z() / r_cross_s.z();
    if (t < kTolerance || t > 1. - kTolerance) return false;
    Precision u = Vector::Cross(q - p, r).z() / r_cross_s.z();
    if (u < kTolerance || u > 1. - kTolerance) return false;
    return true;
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
