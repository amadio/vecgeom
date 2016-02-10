/// \file UnplacedGenTrap.h
/// \author: swenzel
///  Modified and completed: mihaela.gheata@cern.ch

#ifndef VECGEOM_VOLUMES_UNPLACEDGENTRAP_H_
#define VECGEOM_VOLUMES_UNPLACEDGENTRAP_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/SecondOrderSurfaceShell.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedGenTrap;)
VECGEOM_DEVICE_DECLARE_CONV(UnplacedGenTrap)

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * A generic trap:
 * see TGeoArb8 or UGenericTrap
 */
class UnplacedGenTrap : public VUnplacedVolume, public AlignedBase {

public:
  using Vertex_t = Vector3D<Precision>;

  Vertex_t fBBdimensions; // Bounding box dimensions
  Vertex_t fBBorigin;     // Bounding box origin

  // the eight points that define the Arb8
  // actually we will neglect the z coordinates of those
  Vertex_t fVertices[8];

  // we also store this in SOA form
  Precision fVerticesX[8];
  Precision fVerticesY[8];

  // the half-height of the GenTrap
  Precision fDz;
  Precision fInverseDz;
  Precision fHalfInverseDz;
  bool fIsTwisted;

  // we store the connecting vectors in SOA Form
  // these vectors are used to calculate the polygon at a certain z-height
  // less elegant than UVector2 but enables vectorization
  // TODO: make this elegant
  // moreover: they can be precomputed !!
  // Compute intersection between Z plane containing point and the shape
  //
  Precision fConnectingComponentsX[4];
  Precision fConnectingComponentsY[4];

  Precision fDeltaX[8]; // int  j = (i + 1) % 4;
  Precision fDeltaY[8];

  // Utility class for twisted surface algorithms
  SecondOrderSurfaceShell<4> fSurfaceShell;

public:
  // constructor
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedGenTrap(const Precision verticesx[], const Precision verticesy[], Precision halfzheight)
      : fBBdimensions(0., 0., 0.), fBBorigin(0., 0., 0.), fVertices(), fVerticesX(), fVerticesY(), fDz(halfzheight),
        fInverseDz(1. / halfzheight), fHalfInverseDz(0.5 / halfzheight), fIsTwisted(false), fConnectingComponentsX(),
        fConnectingComponentsY(), fDeltaX(), fDeltaY(), fSurfaceShell(verticesx, verticesy, halfzheight) {
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

    // Make sure vertices are defined clockwise
    Precision sum1 = 0.;
    Precision sum2 = 0.;
    for (int i = 0; i < 4; ++i) {
      int j = (i + 1) % 4;
      sum1 += fVertices[i].x() * fVertices[j].y() - fVertices[j].x() * fVertices[i].y();
      sum2 += fVertices[i + 4].x() * fVertices[j + 4].y() - fVertices[j + 4].x() * fVertices[i + 4].y();
    }

    // we should generate an exception here
    if (sum1 * sum2 < -kTolerance) {
      printf("ERROR: Unplaced generic trap defined with opposite clockwise\n");
      Print();
      return;
    }

    // revert sequence of vertices to have them clockwise
    if (sum1 > kTolerance) {
      printf("INFO: Reverting to clockwise vertices of GenTrap shape:\n");
      Print();
      Vertex_t vtemp;
      vtemp = fVertices[1];
      fVertices[1] = fVertices[3];
      fVertices[3] = vtemp;
      vtemp = fVertices[5];
      fVertices[5] = fVertices[7];
      fVertices[7] = vtemp;
    }

    // Check that opposite segments are not crossing -> exception
    if (SegmentsCrossing(fVertices[0], fVertices[1], fVertices[3], fVertices[2]) ||
        SegmentsCrossing(fVertices[1], fVertices[2], fVertices[0], fVertices[3]) ||
        SegmentsCrossing(fVertices[4], fVertices[5], fVertices[7], fVertices[6]) ||
        SegmentsCrossing(fVertices[5], fVertices[6], fVertices[4], fVertices[7])) {
      printf("ERROR: Unplaced generic trap defined with crossing opposite segments\n");
      Print();
      return;
    }

    // Check that top and bottom quadrilaterals are convex
    if (!ComputeIsConvexQuadrilaterals()) {
      printf("ERROR: Unplaced generic trap defined with top/bottom quadrilaterals not convex\n");
      Print();
      return;
    }

    // initialize the connecting components
    for (int i = 0; i < 4; ++i) {
      fConnectingComponentsX[i] = (fVertices[i] - fVertices[i + 4]).x();
      fConnectingComponentsY[i] = (fVertices[i] - fVertices[i + 4]).y();
      fVerticesX[i] = fVertices[i].x();
      fVerticesX[i + 4] = fVertices[i + 4].x();
      fVerticesY[i] = fVertices[i].y();
      fVerticesY[i + 4] = fVertices[i + 4].y();
    }
    for (int i = 0; i < 4; ++i) {
      int j = (i + 1) % 4;
      fDeltaX[i] = fVerticesX[j] - fVerticesX[i];
      fDeltaX[i + 4] = fVerticesX[j + 4] - fVerticesX[i + 4];
      fDeltaY[i] = fVerticesY[j] - fVerticesY[i];
      fDeltaY[i + 4] = fVerticesY[j + 4] - fVerticesY[i + 4];
    }
    fIsTwisted = ComputeIsTwisted();
    // std::cout << "twisted= " << fIsTwisted ? "true":"false" << std::endl;
    ComputeBoundingBox();
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual ~UnplacedGenTrap() {}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  SecondOrderSurfaceShell<4> const &GetShell() const { return (fSurfaceShell); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetDZ() const { return (fDz); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vertex_t const &GetVertex(int i) const { return fVertices[i]; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  const Precision *GetVerticesX() const { return fVerticesX; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  const Precision *GetVerticesY() const { return fVerticesY; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  const Vertex_t *GetVertices() const { return fVertices; }

  // computes if this gentrap is twisted
  VECGEOM_CUDA_HEADER_BOTH
  bool ComputeIsTwisted();

  // computes if the top and bottom quadrilaterals are convex (mandatory)
  VECGEOM_CUDA_HEADER_BOTH
  bool ComputeIsConvexQuadrilaterals();

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool IsPlanar() const { return (!fIsTwisted); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool IsConvex() const final { return (!fIsTwisted); }

  // computes if opposite segments are crossing, making a malformed shape
  // This can become a general utility
  VECGEOM_CUDA_HEADER_BOTH
  bool SegmentsCrossing(Vertex_t pa, Vertex_t pb, Vertex_t pc, Vertex_t pd) const;

  // computes and sets the bounding box member of this class
  VECGEOM_CUDA_HEADER_BOTH
  void ComputeBoundingBox();

  virtual int memory_size() const final { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const final;

  virtual void Print(std::ostream &os) const final;

#ifdef VECGEOM_CUDA_INTERFACE
  size_t DeviceSizeOf() const final { return DevicePtr<cuda::UnplacedGenTrap>::SizeOf(); }
  DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const final;
  DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const final;
#endif

  Precision Capacity() { return volume(); }

  VECGEOM_INLINE
  Precision volume() const {
    int i, j;
    Precision capacity = 0;
    for (i = 0; i < 4; i++) {
      j = (i + 1) % 4;

      capacity += 0.25 * fDz * ((fVerticesX[i] + fVerticesX[i + 4]) * (fVerticesY[j] + fVerticesY[j + 4]) -
                                (fVerticesX[j] + fVerticesX[j + 4]) * (fVerticesY[i] + fVerticesY[i + 4]) +
                                (1. / 3) * ((fVerticesX[i + 4] - fVerticesX[i]) * (fVerticesY[j + 4] - fVerticesY[j]) -
                                            (fVerticesX[j] - fVerticesX[j + 4]) * (fVerticesY[i] - fVerticesY[i + 4])));
    }
    return Abs(capacity);
  }

  VECGEOM_INLINE
  Precision SurfaceArea() const {
    Vertex_t vi, vj, hi0, vres;
    Precision surfTop = 0.;
    Precision surfBottom = 0.;
    Precision surfLateral = 0;
    for (int i = 0; i < 4; ++i) {
      int j = (i + 1) % 4;
      surfBottom += 0.5 * (fVerticesX[i] * fVerticesY[j] - fVerticesX[j] * fVerticesY[i]);
      surfTop += 0.5 * (fVerticesX[i + 4] * fVerticesY[j + 4] - fVerticesX[j + 4] * fVerticesY[i + 4]);
      vi.Set(fVerticesX[i + 4] - fVerticesX[i], fVerticesY[i + 4] - fVerticesY[i], 2 * fDz);
      vj.Set(fVerticesX[j + 4] - fVerticesX[j], fVerticesY[j + 4] - fVerticesY[j], 2 * fDz);
      hi0.Set(fVerticesX[j] - fVerticesX[i], fVerticesY[j] - fVerticesY[i], 0.);
      vres = 0.5 * (Vertex_t::Cross(vi + vj, hi0) + Vertex_t::Cross(vi, vj));
      surfLateral += vres.Mag();
    }
    return (Abs(surfTop) + Abs(surfBottom) + surfLateral);
  }

  VECGEOM_CUDA_HEADER_BOTH
  void Extent(Vertex_t &, Vertex_t &) const;

  Vertex_t GetPointOnSurface() const;

  virtual std::string GetEntityType() const { return "GenTrap"; }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECGEOM_CUDA_HEADER_DEVICE static VPlacedVolume *Create(LogicalVolume const *const logical_volume,
                                                          Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
                                                          const int id,
#endif
                                                          VPlacedVolume *const placement = NULL);

/*
  VECGEOM_CUDA_HEADER_DEVICE
  static VPlacedVolume *CreateSpecializedVolume(LogicalVolume const *const volume,
                                                Transformation3D const *const transformation,
                                                const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
                                                const int id,
#endif
                                                VPlacedVolume *const placement = NULL);
*/
#if defined(VECGEOM_USOLIDS)
  std::ostream &StreamInfo(std::ostream &os) const;
#endif

private:
  VECGEOM_CUDA_HEADER_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
                                           const int id,
#endif
                                           VPlacedVolume *const placement = NULL) const final;

}; // end of class declaration
}
} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDGENTRAP_H_
