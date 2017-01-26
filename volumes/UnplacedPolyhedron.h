/// \file UnplacedPolyhedron.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDPOLYHEDRON_H_
#define VECGEOM_VOLUMES_UNPLACEDPOLYHEDRON_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/PolyhedronStruct.h"
#include "volumes/kernel/PolyhedronImplementation.h"
#include "volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedPolyhedron;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedPolyhedron);

template <typename Stream>
Stream &operator<<(Stream &st, EInnerRadii a)
{
  if (a == EInnerRadii::kFalse) st << "EInnerRadii::kFalse";
  if (a == EInnerRadii::kGeneric) st << "EInnerRadii::kGeneric";
  if (a == EInnerRadii::kTrue) st << "EInnerRadii::kTrue";
  return st;
}

template <typename Stream>
Stream &operator<<(Stream &st, EPhiCutout a)
{
  if (a == EPhiCutout::kFalse) st << "EPhiCutout::kFalse";
  if (a == EPhiCutout::kGeneric) st << "EPhiCutout::kGeneric";
  if (a == EPhiCutout::kTrue) st << "EPhiCutout::kTrue";
  if (a == EPhiCutout::kLarge) st << "EPhiCutout::kLarge";
  return st;
}

inline namespace VECGEOM_IMPL_NAMESPACE {

/// \class UnplacedPolyhedron
/// \brief A series of regular n-sided segments along the Z-axis with varying
///        radii and mutual distance in Z.
///
///
/// ---- Cross section of single Z segment ----
///
/// R/Phi--->    -o- Z
/// |        ________________
/// v       /        ^      .\,
///        /    rMax |     .  \,
///       /          |    . <------ fPhiSections[1]
///      /       ____|___.      \,
///     /       /    ^   \       \,
///    /       /     |rMin\       \,
///   /       /      |     \_______\ phiStart/fPhiSections[0]
///   \       \                ^
///    \       \               |
///     \       \________      |
///      \           ^   \<---fZSegments.phi
///      fZSegments.inner \,
///        \               \,
///         \_______________\,
///           ^              phiStart+phiDelta/fPhiSections[n-1]
/// zSegment.outer
///
///
/// ---- Segments along Z ----
///
///                          fZPlanes[size-1]
/// fRMax[1]_____fRMax[2] __       |
///       /|     |\     /|  \___   v
///      / |     | \___/ |  |   |\.
///     |  |     | |   | |  |   | \.
///     |  |     | |   | |  |   |  |
///     |  |     | |___| |  |   | /
///      \ |     | /   \ |  |___|/    ^ R/Phi
///     ^ \|_____|/     \|__/         |
///     |                             |     Z
///     fZPlanes[0]/fRMax[0]           ----->

class UnplacedPolyhedron
    : public LoopUnplacedVolumeImplHelper<
          PolyhedronImplementation<Polyhedron::EInnerRadii::kGeneric, Polyhedron::EPhiCutout::kGeneric>>,
      public AlignedBase {

private:
  PolyhedronStruct<double> fPoly; ///< Structure holding polyhedron data

public:
  UnplacedPolyhedron() : fPoly() {}
  /// \param sideCount Number of sides along phi in each Z-segment.
  /// \param zPlaneCount Number of Z-planes to draw segments between. The number
  ///                    of segments will always be this number minus one.
  /// \param zPlanes Z-coordinates of each Z-plane to draw segments between.
  /// \param rMin Radius to the sides (not to the corners!) of the inner shell
  ///             for the corresponding Z-plane.
  /// \param rMin Radius to the sides (not to the corners!) of the outer shell
  ///             for the corresponding Z-plane.
  UnplacedPolyhedron(const int sideCount, const int zPlaneCount, Precision const zPlanes[], Precision const rMin[],
                     Precision const rMax[]);

  /// \param phiStart Angle in phi of first corner. This will be one phi angle
  ///                 of the phi cutout, if any cutout is specified. Specified
  ///                 in radians.
  /// \param phiDelta Total angle in phi over which the sides of each segment
  ///                 will be drawn. When added to the starting angle, this will
  ///                 mark one of the angles of the phi cutout, if any cutout is
  ///                 specified.
  /// \param sideCount Number of sides along phi in each Z-segment.
  /// \param zPlaneCount Number of Z-planes to draw segments between. The number
  ///                    of segments will always be this number minus one.
  /// \param zPlanes Z-coordinates of each Z-plane to draw segments between.
  /// \param rMin Radius to the sides (not to the corners!) of the inner shell
  ///             for the corresponding Z-plane.
  /// \param rMin Radius to the sides (not to the corners!) of the outer shell
  ///             for the corresponding Z-plane.
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedPolyhedron(Precision phiStart, Precision phiDelta, const int sideCount, const int zPlaneCount,
                     Precision const zPlanes[], Precision const rMin[], Precision const rMax[]);

  /// Alternative constructor, required for integration with Geant4.
  /// This constructor mirrors one in UnplacedPolycone(), for which the r[],z[] idea makes more sense.
  /// Input must be such that r[i],z[i] arrays describe the outer,inner or inner,outer envelope of the
  /// polyhedron, after connecting all adjacent points, and closing the polygon by connecting last -> first points.
  /// Hence z[] array must be symmetrical: z[0..Nz] = z[2Nz, 2Nz-1, ..., Nz+1], where Nz = zPlaneCount.
  ///
  /// \param phiStart Angle in phi of first corner. This will be one phi angle of the phi cutout, if any
  ///                 cutout is specified. Specified in radians.
  /// \param phiDelta Total angle in phi over which the sides of each segment will be drawn. When added to the
  ///                 starting angle, this will mark one of the angles of the phi cutout, if a cutout is specified.
  /// \param sideCount Number of sides along phi in each Z-segment.
  /// \param zPlaneCount Number of Z-planes to draw segments between. The number
  ///                    of segments will always be this number minus one.
  /// \param zPlanes Z-coordinates of each Z-plane to draw segments between.
  /// \param rMin Radius to the sides (not to the corners!) of the inner shell for the corresponding Z-plane.
  /// \param rMax Radius to the sides (not to the corners!) of the outer shell for the corresponding Z-plane.
  UnplacedPolyhedron(Precision phiStart, Precision phiDelta, const int sideCount, const int zPlaneCount,
                     Precision const r[], Precision const z[]);

  VECGEOM_CUDA_HEADER_BOTH
  virtual ~UnplacedPolyhedron() {}

  VECGEOM_CUDA_HEADER_BOTH
  PolyhedronStruct<double> const &GetStruct() const { return fPoly; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  int GetSideCount() const { return fPoly.fSideCount; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  int GetZSegmentCount() const { return fPoly.fZSegments.size(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  bool HasInnerRadii() const { return fPoly.fHasInnerRadii; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  bool HasPhiCutout() const { return fPoly.fHasPhiCutout; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  bool HasLargePhiCutout() const { return fPoly.fHasLargePhiCutout; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  ZSegment const &GetZSegment(int i) const { return fPoly.fZSegments[i]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Array<ZSegment> const &GetZSegments() const { return fPoly.fZSegments; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision GetZPlane(int i) const { return fPoly.fZPlanes[i]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Array<Precision> const &GetZPlanes() const { return fPoly.fZPlanes; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Array<Precision> const &GetRMin() const { return fPoly.fRMin; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Array<Precision> const &GetRMax() const { return fPoly.fRMax; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Vector3D<Precision> GetPhiSection(int i) const { return fPoly.fPhiSections[i]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  SOA3D<Precision> const &GetPhiSections() const { return fPoly.fPhiSections; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  evolution::Wedge const &GetPhiWedge() const { return fPoly.fPhiWedge; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  TubeStruct<double> const &GetBoundingTube() const { return fPoly.fBoundingTube; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision GetBoundingTubeOffset() const { return fPoly.fBoundingTubeOffset; }

#ifndef VECGEOM_NVCC
  VECGEOM_CUDA_HEADER_BOTH
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const;
#endif

#ifndef VECGEOM_NVCC
  Precision DistanceSquarePointToSegment(Vector3D<Precision> &v1, Vector3D<Precision> &v2,
                                         const Vector3D<Precision> &p) const;
  bool InsideTriangle(Vector3D<Precision> &v1, Vector3D<Precision> &v2, Vector3D<Precision> &v3,
                      const Vector3D<Precision> &p) const;

  // calculate array of triangle spanned by points v1,v2,v3
  // TODO: this function has nothing to do with a Polyhedron. It should live somewhere else ( indeed: the Quadriteral
  // seems to have such a function, too )
  Precision GetTriangleArea(Vector3D<Precision> const &v1, Vector3D<Precision> const &v2,
                            Vector3D<Precision> const &v3) const;

  // returns a random point inside the triangle described by v1,v2,v3
  // TODO: this function has nothing to do with a Polyhedron. It should live somewhere else ( indeed: the Quadriteral
  // seems to have such a function, too )
  Vector3D<Precision> GetPointOnTriangle(Vector3D<Precision> const &v1, Vector3D<Precision> const &v2,
                                         Vector3D<Precision> const &v3) const;

  Precision Capacity() const;

  Precision SurfaceArea() const;

  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const;

  Vector3D<Precision> GetPointOnSurface() const;

  std::string GetEntityType() const { return "Polyhedron"; }
#endif // !VECGEOM_NVCC

  /// Not a stored value, and should not be called from performance critical code.
  /// \return The angle along phi where the first corner is placed, specified in radians.
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision GetPhiStart() const { return fPoly.fPhiStart; }

  /// Not a stored value, and should not be called from performance critical code.
  /// \return The angle along phi where the last corner is placed, specified in degrees.
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision GetPhiEnd() const { return fPoly.fPhiStart + fPoly.fPhiDelta; }

  /// Not a stored value, and should not be called from performance critical code.
  /// \return The difference in angle along phi between the last corner and the first corner.
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision GetPhiDelta() const { return fPoly.fPhiDelta; }

  // \return the number of quadrilaterals (including triangles) that this
  // polyhedra consists of; this should be all visible surfaces except the endcaps
  VECGEOM_CUDA_HEADER_BOTH
  int GetNQuadrilaterals() const;

  // reconstructs fZPlanes, fRmin, fRMax from Quadrilaterals
  template <typename PushableContainer>
  void ReconstructSectionArrays(PushableContainer &zplanes, PushableContainer &rmin, PushableContainer &rmax) const
  {
    // iterate over sections;
    // pick one inner quadrilateral and one outer quadrilateral
    // reconstruct rmin, rmax and z from these

    // TODO: this might not yet be correct when we have degenerate
    // z-plane values

    AOS3D<Precision> const *innercorners;
    AOS3D<Precision> const *outercorners;

    // lambda function to recalculate the radii
    auto getradius = [](Vector3D<Precision> const &a, Vector3D<Precision> const &b) {
      return std::sqrt(a.Perp2() - (a - b).Mag2() / 4.);
    };

    Array<ZSegment>::const_iterator s;
    Array<ZSegment>::const_iterator end = fPoly.fZSegments.cend();

    for (s = fPoly.fZSegments.cbegin(); s != end; ++s) {
      outercorners          = (*s).outer.GetCorners();
      Vector3D<Precision> a = outercorners[0][0];
      Vector3D<Precision> b = outercorners[1][0];
      rmax.push_back(getradius(a, b));
      zplanes.push_back(a.z());

      if (fPoly.fHasInnerRadii) {
        innercorners = (*s).inner.GetCorners();
        a            = innercorners[0][0];
        b            = innercorners[1][0];
        rmin.push_back(getradius(a, b));
      } else {
        rmin.push_back(0.);
      }
    }
    // for last segment need to add addidional plane

    Vector3D<Precision> a = outercorners[2][0];
    Vector3D<Precision> b = outercorners[3][0];
    rmax.push_back(getradius(a, b));
    zplanes.push_back(a.z());

    if (fPoly.fHasInnerRadii) {
      a = innercorners[2][0];
      b = innercorners[3][0];
      rmin.push_back(getradius(a, b));
    } else {
      rmin.push_back(0.);
    }
  }

  VECGEOM_CUDA_HEADER_BOTH
  void DetectConvexity();

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const final;

  VECGEOM_CUDA_HEADER_BOTH
  void PrintSegments() const;

  virtual void Print(std::ostream &os) const final;

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECGEOM_CUDA_HEADER_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

  VECGEOM_CUDA_HEADER_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
                                           const int id,
#endif
                                           VPlacedVolume *const placement = NULL) const final;
  VECGEOM_FORCE_INLINE
  virtual int memory_size() const final { return sizeof(*this); }

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedPolyhedron>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
#endif

private:
  // This method does the proper construction of planes and segments.
  // Used by multiple constructors.
  VECGEOM_CUDA_HEADER_BOTH
  void Initialize(Precision phiStart, Precision phiDelta, const int sideCount, const int zPlaneCount,
                  Precision const zPlanes[], Precision const rMin[], Precision const rMax[]);

}; // End class UnplacedPolyhedron

} // End inline namespace

} // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDPOLYHEDRON_H_
