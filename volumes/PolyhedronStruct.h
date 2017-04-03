/*
 * PolyhedronStruct.h
 *
 *  Created on: 09.12.2016
 *      Author: mgheata
 */
#ifndef VECGEOM_POLYHEDRONSTRUCT_H_
#define VECGEOM_POLYHEDRONSTRUCT_H_

#include <ostream>

#include <VecCore/VecCore>
#include "base/Vector3D.h"
#include "volumes/Quadrilaterals.h"
#include "volumes/Wedge_Evolution.h"
#include "base/Array.h"
#include "base/SOA3D.h"
#include "volumes/TubeStruct.h"

// These enums should be in the scope vecgeom::Polyhedron, but when used in the
// shape implementation helper instantiations, nvcc gets confused:

enum struct EInnerRadii { kFalse = -1, kGeneric = 0, kTrue = 1 };
enum struct EPhiCutout { kFalse = -1, kGeneric = 0, kTrue = 1, kLarge = 2 };

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct ZSegment;);
VECGEOM_DEVICE_DECLARE_CONV(struct, ZSegment);

// Declare types shared by cxx and cuda.
namespace Polyhedron {
using ::EInnerRadii;
using ::EPhiCutout;
}

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Represents one segment along the Z-axis, containing one or more sets of
/// quadrilaterals that represent the outer, inner and phi shells.
struct ZSegment {
  Quadrilaterals outer; ///< Should always be non-empty.
  Quadrilaterals phi;   ///< Is empty if fHasPhiCutout is false.
  Quadrilaterals inner; ///< Is empty hasInnerRadius is false.
  bool hasInnerRadius;  ///< Indicates whether any inner quadrilaterals are present in this segment.
};

// a plain and lightweight struct to encapsulate data members of a polyhedron
template <typename T = double>
struct PolyhedronStruct {
  int fSideCount;                 ///< Number of segments along phi.
  bool fHasInnerRadii;            ///< Has any Z-segments with an inner radius != 0.
  bool fHasPhiCutout;             ///< Has a cutout angle along phi.
  bool fHasLargePhiCutout;        ///< Phi cutout is larger than pi.
  T fPhiStart;                    ///< Phi start in radians (input to constructor)
  T fPhiDelta;                    ///< Phi delta in radians (input to constructor)
  evolution::Wedge fPhiWedge;     ///< Phi wedge
  Array<ZSegment> fZSegments;     ///< AOS'esque collections of quadrilaterals
  Array<T> fZPlanes;              ///< Z-coordinate of each plane separating segments
  Array<T> fRMin;                 ///< Inner radii as specified in constructor.
  Array<T> fRMax;                 ///< Outer radii as specified in constructor.
  Array<bool> fSameZ;             ///< Array of flags marking that the following plane is at same Z
  SOA3D<T> fPhiSections;          ///< Unit vectors marking the bounds between
                                  ///  phi segments, represented by planes
                                  ///  through the origin with the normal
                                  ///  point along the positive phi direction.
  TubeStruct<T> fBoundingTube;    ///< Tube enclosing the outer bounds of the
                                  ///  polyhedron. Used in Contains, Inside and
                                  ///  DistanceToIn.
  T fBoundingTubeOffset;          ///< Offset in Z of the center of the bounding
                                  ///  tube. Used as a quick substitution for
                                  ///  running a full transformation.
  mutable Precision fSurfaceArea; ///< Stored SurfaceArea
  mutable Precision fCapacity;    ///< Stored Capacity

  // These data member and member functions are added for convexity detection
  bool fContinuousInSlope;
  bool fConvexityPossible;
  bool fEqualRmax;

  VECCORE_ATT_HOST_DEVICE
  PolyhedronStruct()
      : fSideCount(0), fHasInnerRadii(false), fHasPhiCutout(false), fHasLargePhiCutout(false), fPhiStart(0),
        fPhiDelta(0), fPhiWedge(0., 0.), fBoundingTube(0, 0, 0, 0, 0), fBoundingTubeOffset(0)
  {
  }

  VECCORE_ATT_HOST_DEVICE
  PolyhedronStruct(Precision phiStart, Precision phiDelta, const int sideCount, const int zPlaneCount,
                   Precision const zPlanes[], Precision const rMin[], Precision const rMax[])
      : fSideCount(sideCount), fHasInnerRadii(false), fHasPhiCutout(phiDelta < kTwoPi),
        fHasLargePhiCutout(phiDelta < kPi), fPhiStart(NormalizeAngle<kScalar>(phiStart)),
        fPhiDelta((phiDelta > kTwoPi) ? kTwoPi : phiDelta), fPhiWedge(fPhiDelta, fPhiStart),
        fZSegments(zPlaneCount - 1), fZPlanes(zPlaneCount), fRMin(zPlaneCount), fRMax(zPlaneCount),
        fPhiSections(sideCount + 1), fBoundingTube(0, 1, 1, fPhiStart, fPhiDelta), fSurfaceArea(0.), fCapacity(0.),
        fContinuousInSlope(true), fConvexityPossible(true), fEqualRmax(true)
  {
    // initialize polyhedron internals
    Initialize(phiStart, phiDelta, sideCount, zPlaneCount, zPlanes, rMin, rMax);
  }

  PolyhedronStruct(Precision phiStart, Precision phiDelta, const int sideCount, const int zPlaneCount,
                   Precision const r[], Precision const z[])
      : fSideCount(sideCount), fHasInnerRadii(false), fHasPhiCutout(phiDelta < kTwoPi),
        fHasLargePhiCutout(phiDelta < kPi), fPhiStart(NormalizeAngle<kScalar>(phiStart)),
        fPhiDelta((phiDelta > kTwoPi) ? kTwoPi : phiDelta), fPhiWedge(fPhiDelta, fPhiStart),
        fZSegments(zPlaneCount - 1), fZPlanes(zPlaneCount), fRMin(zPlaneCount), fRMax(zPlaneCount),
        fPhiSections(sideCount + 1), fBoundingTube(0, 1, 1, fPhiStart, fPhiDelta), fSurfaceArea(0.), fCapacity(0.)
  {
    if (zPlaneCount == 0) throw std::runtime_error("A Polyhedron need at least one plane");

    // Geant4-like construction:
    // ind (   0       ...    n-1      n      ...    2*n -1   )
    // r = ( rmin[n-1], ... , rmin[0], rmax[0], ..., rmax[n-1] )
    // z = ( z[n-1], ...    , z[0]   , z[0]   , ..., z[n-1]    )
    // data integrity checks
    for (int i = 0; i <= zPlaneCount; ++i) {
      assert(z[i] == z[2 * zPlaneCount - 1 - i] && "UnplPolyhedron ERROR: z[] array is not symmetrical, please fix.\n");
    }

    // reuse input array as argument, in ascending order
    int Nz                 = zPlaneCount;
    bool ascendingZ        = true;
    const Precision *zarg  = &z[0];
    const Precision *r1arg = r;
    if (z[0] > z[1]) {
      ascendingZ = false;
      zarg       = z + Nz; // second half of input z[] is ascending due to symmetry already verified
      r1arg      = r + Nz;
    }

    // reorganize remainder of r[] data in ascending-z order
    Precision *r2arg = new Precision[Nz];
    for (int i = 0; i < Nz; ++i)
      r2arg[i] = (ascendingZ ? r[2 * Nz - 1 - i] : r[Nz - 1 - i]);

    // identify which rXarg is rmax and rmin and ensure that Rmax > Rmin for all points provided
    const Precision *rmin = r1arg, *rmax = r2arg;
    if (r1arg[0] > r2arg[0]) {
      rmax = r1arg;
      rmin = r2arg;
    }

    // final data integrity cross-check
    for (int i = 0; i < Nz; ++i) {
      assert(rmax[i] > rmin[i] &&
             "UnplPolycone ERROR: r[] provided has problems of the Rmax < Rmin type, please check!\n");
    }
    // Delegate to full constructor
    Initialize(phiStart, phiDelta, sideCount, zPlaneCount, zarg, rmin, rmax);
    delete[] r2arg;
  }

  VECCORE_ATT_HOST_DEVICE
  bool CheckContinuityInSlope(const double rOuter[], const double zPlane[], const unsigned int nz)
  {

    Precision prevSlope = kInfLength;
    for (unsigned int j = 0; j < nz - 1; ++j) {
      if (zPlane[j + 1] == zPlane[j]) {
        if (rOuter[j + 1] != rOuter[j]) return false;
      } else {
        Precision currentSlope = (rOuter[j + 1] - rOuter[j]) / (zPlane[j + 1] - zPlane[j]);
        if (currentSlope > prevSlope) return false;
        prevSlope = currentSlope;
      }
    }
    return true;
  }

  // This method does the proper construction of planes and segments.
  // Used by multiple constructors.
  VECCORE_ATT_HOST_DEVICE
  void Initialize(Precision phiStart, Precision phiDelta, const int sideCount, const int zPlaneCount,
                  Precision const zPlanes[], Precision const rMin[], Precision const rMax[])
  {
    typedef Vector3D<Precision> Vec_t;

    // Sanity check of input parameters
    assert(zPlaneCount > 1);
    assert(fSideCount > 0);

    copy(zPlanes, zPlanes + zPlaneCount, &fZPlanes[0]);
    copy(rMin, rMin + zPlaneCount, &fRMin[0]);
    copy(rMax, rMax + zPlaneCount, &fRMax[0]);
    fSameZ.Allocate(zPlaneCount);

    double startRmax = rMax[0];
    for (int i = 0; i < zPlaneCount; i++) {
      fConvexityPossible &= (rMin[i] == 0.);
      fEqualRmax &= (startRmax == rMax[i]);
      fSameZ[i]                                                                     = false;
      if (i > 0 && i < zPlaneCount - 1 && fZPlanes[i] == fZPlanes[i + 1]) fSameZ[i] = true;
    }
    fContinuousInSlope = CheckContinuityInSlope(rMax, zPlanes, zPlaneCount);

    // Initialize segments
    // sometimes there will be no quadrilaterals: for instance when
    // rmin jumps at some z and rmax remains continouus
    for (int i = 0; i < zPlaneCount - 1; ++i) {
      // Z-planes must be monotonically increasing
      assert(zPlanes[i] <= zPlanes[i + 1]);

      fZSegments[i].hasInnerRadius = rMin[i] > 0 || rMin[i + 1] > 0;

      int multiplier = (zPlanes[i] == zPlanes[i + 1] && rMax[i] == rMax[i + 1]) ? 0 : 1;

      // create quadrilaterals in a predefined place with placement new
      new (&fZSegments[i].outer) Quadrilaterals(sideCount * multiplier);

      // no phi segment here if degenerate z;
      if (fHasPhiCutout) {
        multiplier = (zPlanes[i] == zPlanes[i + 1]) ? 0 : 1;
        new (&fZSegments[i].phi) Quadrilaterals(2 * multiplier);
      }

      multiplier = (zPlanes[i] == zPlanes[i + 1] && rMin[i] == rMin[i + 1]) ? 0 : 1;

      if (fZSegments[i].hasInnerRadius) {
        new (&fZSegments[i].inner) Quadrilaterals(sideCount * multiplier);
        fHasInnerRadii = true;
      }
    }

    // Compute the cylindrical coordinate phi along which the corners are placed
    assert(phiDelta > 0);
    phiStart                        = NormalizeAngle<kScalar>(phiStart);
    if (phiDelta > kTwoPi) phiDelta = kTwoPi;
    Precision sidePhi               = phiDelta / sideCount;
    vecgeom::unique_ptr<Precision[]> vertixPhi(new Precision[sideCount + 1]);
    for (int i = 0, iMax = sideCount + 1; i < iMax; ++i) {
      vertixPhi[i]                     = NormalizeAngle<kScalar>(phiStart + i * sidePhi);
      Vector3D<Precision> cornerVector = Vec_t::FromCylindrical(1., vertixPhi[i], 0).Normalized().FixZeroes();
      fPhiSections.set(i, cornerVector.Normalized().Cross(Vector3D<Precision>(0, 0, -1)));
    }
    if (!fHasPhiCutout) {
      // If there is no phi cutout, last phi is equal to the first
      vertixPhi[sideCount] = vertixPhi[0];
    }

    // Specified radii are to the sides, not to the corners. Change these values,
    // as corners and not sides are used to build the structure
    Precision cosHalfDeltaPhi = cos(0.5 * sidePhi);
    Precision innerRadius = kInfLength, outerRadius = -kInfLength;
    for (int i = 0; i < zPlaneCount; ++i) {
      // Use distance to side for minimizing inner radius of bounding tube
      if (rMin[i] < innerRadius) innerRadius = rMin[i];
      // rMin[i] /= cosHalfDeltaPhi;
      // rMax[i] /= cosHalfDeltaPhi;
      assert(rMin[i] >= 0 && rMax[i] > 0);
      // Use distance to corner for minimizing outer radius of bounding tube
      if (rMax[i] > outerRadius) outerRadius = rMax[i];
    }
    // need to convert from distance to planes to real radius in case of outerradius
    // the inner radius of the bounding tube is given by min(rMin[])
    outerRadius /= cosHalfDeltaPhi;

    // Create bounding tube with biggest outer radius and smallest inner radius
    Precision boundingTubeZ = 0.5 * (zPlanes[zPlaneCount - 1] - zPlanes[0] + kTolerance);
    // Make bounding tube phi range a bit larger to contain all points on phi boundaries
    const Precision kPhiTolerance = 100 * kTolerance;
    // The increase in the angle has to be large enough to contain most of
    // kSurface points. There will be some points close to the Z axis which will
    // not be contained. The value is empirical to satisfy ShapeTester
    Precision boundsPhiStart = !fHasPhiCutout ? 0 : phiStart - kPhiTolerance;
    Precision boundsPhiDelta = !fHasPhiCutout ? kTwoPi : phiDelta + 2 * kPhiTolerance;
    // correct inner and outer Radius with conversion factor
    // innerRadius /= cosHalfDeltaPhi;
    // outerRadius /= cosHalfDeltaPhi;

    fBoundingTube = TubeStruct<double>(innerRadius - kHalfTolerance, outerRadius + kHalfTolerance, boundingTubeZ,
                                       boundsPhiStart, boundsPhiDelta);

    // The offset has to match the middle of the polyhedron
    fBoundingTubeOffset = 0.5 * (zPlanes[0] + zPlanes[zPlaneCount - 1]);

    // Ease indexing into twodimensional vertix array
    auto VertixIndex = [&sideCount](int plane, int corner) { return plane * (sideCount + 1) + corner; };

    // Precompute all vertices to ensure that there are no numerical cracks in the
    // surface.
    const int nVertices = zPlaneCount * (sideCount + 1);
    vecgeom::unique_ptr<Vec_t[]> outerVertices(new Vec_t[nVertices]);
    vecgeom::unique_ptr<Vec_t[]> innerVertices(new Vec_t[nVertices]);
    for (int i = 0; i < zPlaneCount; ++i) {
      for (int j = 0, jMax = sideCount + fHasPhiCutout; j < jMax; ++j) {
        int index            = VertixIndex(i, j);
        outerVertices[index] = Vec_t::FromCylindrical(rMax[i] / cosHalfDeltaPhi, vertixPhi[j], zPlanes[i]).FixZeroes();
        innerVertices[index] = Vec_t::FromCylindrical(rMin[i] / cosHalfDeltaPhi, vertixPhi[j], zPlanes[i]).FixZeroes();
      }
      // Non phi cutout case
      if (!fHasPhiCutout) {
        // Make last vertices identical to the first phi coordinate
        outerVertices[VertixIndex(i, sideCount)] = outerVertices[VertixIndex(i, 0)];
        innerVertices[VertixIndex(i, sideCount)] = innerVertices[VertixIndex(i, 0)];
      }
    }

    // Build segments by drawing quadrilaterals between vertices
    for (int iPlane = 0; iPlane < zPlaneCount - 1; ++iPlane) {

      auto WrongNormal = [](Vector3D<Precision> const &normal, Vector3D<Precision> const &corner) {
        return normal[0] * corner[0] + normal[1] * corner[1] < 0;
      };

      // Draw the regular quadrilaterals along phi
      for (int iSide = 0; iSide < fZSegments[iPlane].outer.size(); ++iSide) {
        fZSegments[iPlane].outer.Set(
            iSide, outerVertices[VertixIndex(iPlane, iSide)], outerVertices[VertixIndex(iPlane, iSide + 1)],
            outerVertices[VertixIndex(iPlane + 1, iSide + 1)], outerVertices[VertixIndex(iPlane + 1, iSide)]);
        // Normal has to point away from Z-axis
        if (WrongNormal(fZSegments[iPlane].outer.GetNormal(iSide), outerVertices[VertixIndex(iPlane, iSide)])) {
          fZSegments[iPlane].outer.FlipSign(iSide);
        }
      }
      if (fZSegments[iPlane].hasInnerRadius) {
        for (int iSide = 0; iSide < fZSegments[iPlane].inner.size(); ++iSide) {
          fZSegments[iPlane].inner.Set(
              iSide, innerVertices[VertixIndex(iPlane, iSide)], innerVertices[VertixIndex(iPlane, iSide + 1)],
              innerVertices[VertixIndex(iPlane + 1, iSide + 1)], innerVertices[VertixIndex(iPlane + 1, iSide)]);
          // Normal has to point away from Z-axis
          if (WrongNormal(fZSegments[iPlane].inner.GetNormal(iSide), innerVertices[VertixIndex(iPlane, iSide)])) {
            fZSegments[iPlane].inner.FlipSign(iSide);
          }
        }
      }

      if (fHasPhiCutout && fZSegments[iPlane].phi.size() == 2) {
        // If there's a phi cutout, draw two quadrilaterals connecting the four
        // corners (two inner, two outer) of the first and last phi coordinate,
        // respectively
        fZSegments[iPlane].phi.Set(0, innerVertices[VertixIndex(iPlane, 0)], innerVertices[VertixIndex(iPlane + 1, 0)],
                                   outerVertices[VertixIndex(iPlane + 1, 0)], outerVertices[VertixIndex(iPlane, 0)]);
        // Make sure normal points backwards along phi
        if (fZSegments[iPlane].phi.GetNormal(0).Dot(fPhiSections[0]) > 0) {
          fZSegments[iPlane].phi.FlipSign(0);
        }
        fZSegments[iPlane].phi.Set(
            1, outerVertices[VertixIndex(iPlane, sideCount)], outerVertices[VertixIndex(iPlane + 1, sideCount)],
            innerVertices[VertixIndex(iPlane + 1, sideCount)], innerVertices[VertixIndex(iPlane, sideCount)]);
        // Make sure normal points forwards along phi
        if (fZSegments[iPlane].phi.GetNormal(1).Dot(fPhiSections[fSideCount]) < 0) {
          fZSegments[iPlane].phi.FlipSign(1);
        }
      }

    } // End loop over segments
  }
};
}
} // end global namespace

#endif
