/// \file TessellatedStruct.h
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_VOLUMES_TRIANGLEFACET_H_
#define VECGEOM_VOLUMES_TRIANGLEFACET_H_

#define TEST_TCPERF = 1

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {
//______________________________________________________________________________
// Basic structure of indices to 3 vertices making a triangle.
// The vertices making the triangle have to be given in anti-clockwise
// order looking from the outsider of the solid where it belongs.
// This helper structure is used temporarily by the tesselated solid in the
// creation and clustering phase.
//______________________________________________________________________________
template <typename T = double>
struct TriangleFacet {
  Vector3D<T> fVertices[3]; ///< vertices of the triangle
  Vector3D<T> fCenter;      ///< Center of the triangle
  Vector3D<int> fIndices;   ///< indices for 3 distinct vertices
  Vector<int> fNeighbors;   ///< indices to triangle neighbors
  T fSurfaceArea = 0;       ///< surface area
  Vector3D<T> fNormal;      ///< normal vector pointing outside
  T fDistance;              ///< distance between the origin and the triangle plane
#ifdef TEST_TCPERF
  Vector3D<T> fSideVectors[3]; ///< side vectors perpendicular to edges
#endif

  VECCORE_ATT_HOST_DEVICE
  TriangleFacet() { fNeighbors.reserve(3); }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool SetVertices(Vector3D<T> const &vtx0, Vector3D<T> const &vtx1, Vector3D<T> const &vtx2, int ind0 = 0,
                   int ind1 = 0, int ind2 = 0)
  {
    fVertices[0] = vtx0;
    fVertices[1] = vtx1;
    fVertices[2] = vtx2;
    fIndices.Set(ind0, ind1, ind2);
    // Check validity
    Vector3D<T> e1 = vtx1 - vtx0;
    Vector3D<T> e2 = vtx2 - vtx0;
    double eMag1   = e1.Mag();
    double eMag2   = e2.Mag();
    double eMag3   = (e2 - e1).Mag();

    if (eMag1 <= kTolerance || eMag2 <= kTolerance || eMag3 <= kTolerance) {
      // TO DO: add more verbosity
      std::cout << "Length of sides of facet are too small." << std::endl;
      return false;
    }
    // Compute normal
    fNormal = e1.Cross(e2).Unit();
#ifdef TEST_TCPERF
    // Compute side vectors (testing)
    for (int i        = 0; i < 3; i++)
      fSideVectors[i] = fNormal.Cross(fVertices[(i + 1) % 3] - fVertices[i]).Normalized();
#endif
    // Distace to facet
    fDistance    = -fNormal.Dot(vtx0);
    fSurfaceArea = 0.5 * (e1.Cross(e2)).Mag();
    if (fSurfaceArea < kTolerance * kTolerance) {
      // TO DO: add more verbosity
      std::cout << "Flat triangle." << std::endl;
      return false;
    }
    // Center of the triangle
    fCenter = (vtx0 + vtx1 + vtx2) / 3.;
    return true;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsSurrounded() const { return fNeighbors.size() > 2; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int IsNeighbor(TriangleFacet const &other)
  {
    // Check if a segment is common
    int ncommon = 0;
    for (int ind1 = 0; ind1 < 3; ++ind1) {
      for (int ind2 = 0; ind2 < 3; ++ind2) {
        if (fIndices[ind1] == other.fIndices[ind2]) ncommon++;
      }
    }
    return ncommon;
  }

#ifdef TEST_TCPERF
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  bool Inside(Vector3D<Precision> const &point) const
  {
    // Check id point within the triangle plane is inside the triangle.
    bool inside = true;
    for (size_t i = 0; i < 3; ++i) {
      Precision saf = (point - fVertices[i]).Dot(fSideVectors[i]);
      inside &= saf > -kTolerance;
    }
    return inside;
  }

  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  T DistPlane(Vector3D<Precision> const &point) const
  {
    // Returns distance from point to plane. This is positive if the point is on
    // the outside halfspace, negative otherwise.
    return (point.Dot(fNormal) + fDistance);
  }

  VECCORE_ATT_HOST_DEVICE
  T DistanceToIn(Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
                 Precision const & /*stepMax*/) const
  {
    T ndd      = NonZero(direction.Dot(fNormal));
    T saf      = DistPlane(point);
    bool valid = ndd < 0. && saf > -kTolerance;
    if (!valid) return InfinityLength<T>();
    T distance = -saf / ndd;
    // Propagate the point with the distance to the plane.
    Vector3D<Precision> point_prop = point + distance * direction;
    // Check if propagated points hit the triangle
    if (!Inside(point_prop)) return InfinityLength<T>();
    return distance;
  }

  VECCORE_ATT_HOST_DEVICE
  Precision DistanceToOut(Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
                          Precision const & /*stepMax*/) const
  {
    T ndd      = NonZero(direction.Dot(fNormal));
    T saf      = DistPlane(point);
    bool valid = ndd > 0. && saf < kTolerance;
    if (!valid) return InfinityLength<T>();
    T distance = -saf / ndd;
    // Propagate the point with the distance to the plane.
    Vector3D<Precision> point_prop = point + distance * direction;
    // Check if propagated points hit the triangle
    if (!Inside(point_prop)) return InfinityLength<T>();
    return distance;
  }

  template <bool ToIn>
  VECCORE_ATT_HOST_DEVICE
  T SafetySq(Vector3D<Precision> const &point) const
  {
    T safety = DistPlane(point);
    // Find the projection of the point on each plane
    Vector3D<Precision> intersection = point - safety * fNormal;
    bool withinBound                 = Inside(intersection);
    if (ToIn)
      withinBound &= safety > -kTolerance;
    else
      withinBound &= safety < kTolerance;
    safety *= safety;
    if (withinBound) return safety;

    Vector3D<T> safety_outbound = InfinityLength<T>();
    for (int ivert = 0; ivert < 3; ++ivert) {
      safety_outbound[ivert] =
          DistanceToLineSegmentSquared<kScalar>(fVertices[ivert], fVertices[(ivert + 1) % 3], point);
    }
    return (safety_outbound.Min());
  }
#endif
};

std::ostream &operator<<(std::ostream &os, TriangleFacet<double> const &facet);

} // end VECGEOM_IMPL_NAMESPACE
} // end namespace vecgeom

#endif
