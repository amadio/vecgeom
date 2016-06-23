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
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedGenTrap)

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * A generic trap:
 * see TGeoArb8 or UGenericTrap
 */
class UnplacedGenTrap : public VUnplacedVolume, public AlignedBase {

public:
  using Vertex_t = Vector3D<Precision>;

  Vertex_t fBBdimensions; /** Bounding box dimensions */
  Vertex_t fBBorigin;     /** Bounding box origin */
  Vertex_t fVertices[8];  /** The eight points that define the Arb8 */

  // we also store this in SOA form
  Precision fVerticesX[8]; /** Backed-up X positions of vertices */
  Precision fVerticesY[8]; /** Backed-up Y positions of vertices */

  Precision fDz;            /** The half-height of the GenTrap */
  Precision fInverseDz;     /** Pre-computed 1/fDz */
  Precision fHalfInverseDz; /** Pre-computed 0.5/fDz */
  bool fIsTwisted;          /** Twisted flag */

  // we store the connecting vectors in SOA Form
  // these vectors are used to calculate the polygon at a certain z-height
  // moreover: they can be precomputed !!
  // Compute intersection between Z plane containing point and the shape
  //
  Precision fConnectingComponentsX[4]; /** X components of connecting bottom-top vectors vi */
  Precision fConnectingComponentsY[4]; /** Y components of connecting bottom-top vectors vi */

  Precision fDeltaX[8]; /** X components of connecting horizontal vectors hij */
  Precision fDeltaY[8]; /** Y components of connecting horizontal vectors hij */

  bool fDegenerated[4]; /** Flags for each top-bottom edge marking that this is degenerated */

  SecondOrderSurfaceShell<4> fSurfaceShell; /** Utility class for twisted surface algorithms */

public:
  /** @brief UnplacedGenTrap constructor
  * @param verticesx X positions of vertices in array form
  * @param verticesy Y positions of vertices in array form
  * @param halfzheight The half-height of the GenTrap
  */
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedGenTrap(const Precision verticesx[], const Precision verticesy[], Precision halfzheight);

  /** @brief UnplacedGenTrap destructor */
  VECGEOM_CUDA_HEADER_BOTH
  virtual ~UnplacedGenTrap() = default;

  /** @brief Getter for the surface shell */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  SecondOrderSurfaceShell<4> const &GetShell() const { return (fSurfaceShell); }

  /** @brief Getter for the half-height */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision GetDZ() const { return (fDz); }

  /** @brief Getter for one of the 8 vertices in Vector3D<Precision> form */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vertex_t const &GetVertex(int i) const { return fVertices[i]; }

  /** @brief Getter for the array of X coordinates of vertices */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  const Precision *GetVerticesX() const { return fVerticesX; }

  /** @brief Getter for the array of Y coordinates of vertices */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  const Precision *GetVerticesY() const { return fVerticesY; }

  /** @brief Getter for the list of vertices */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  const Vertex_t *GetVertices() const { return fVertices; }

  /** @brief Computes if this gentrap is twisted */
  VECGEOM_CUDA_HEADER_BOTH
  bool ComputeIsTwisted();

  /** @brief Computes if the top and bottom quadrilaterals are convex (mandatory) */
  VECGEOM_CUDA_HEADER_BOTH
  bool ComputeIsConvexQuadrilaterals();

  /** @brief Getter for the planarity of lateral surfaces */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool IsPlanar() const { return (!fIsTwisted); }

  /** @brief Getter for the global convexity of the trapezoid */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  bool IsDegenerated(int i) const { return (fDegenerated[i]); }

  /** @brief Computes if opposite segments are crossing, making a malformed shape */
  // This can become a general utility
  VECGEOM_CUDA_HEADER_BOTH
  bool SegmentsCrossing(Vertex_t pa, Vertex_t pb, Vertex_t pc, Vertex_t pd) const;

  /** @brief Computes and sets the bounding box dimensions/origin */
  VECGEOM_CUDA_HEADER_BOTH
  void ComputeBoundingBox();

  /** @brief Memory size in bytes */
  virtual int memory_size() const final { return sizeof(*this); }

  /** @brief Print parameters of the trapezoid */
  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const final;

  /** @brief Print parameters of the trapezoid to stream */
  virtual void Print(std::ostream &os) const final;

#ifdef VECGEOM_CUDA_INTERFACE
  /** @brief Size of object on the device */
  size_t DeviceSizeOf() const final { return DevicePtr<cuda::UnplacedGenTrap>::SizeOf(); }
  /** @brief Copy to GPU interface function */
  DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const final;
  /** @brief Copy to GPU implementation */
  DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const final;
#endif

  /** @brief Interface method for computing capacity */
  Precision Capacity() { return volume(); }

  /** @brief Implementation of capacity computation */
  Precision volume() const;

  /** @brief Implementation of surface area computation */
  Precision SurfaceArea() const;

  /** @brief Computes the extent on X/Y/Z of the trapezoid */
  VECGEOM_CUDA_HEADER_BOTH
  void Extent(Vertex_t &, Vertex_t &) const;

  /** @brief Generates randomly a point on the surface of the trapezoid */
  Vertex_t GetPointOnSurface() const;

  /** @brief Get type name */
  virtual std::string GetEntityType() const { return "GenTrap"; }

  /** @brief Templated factory for creating a placed volume */
  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECGEOM_CUDA_HEADER_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

/*
  // Is this still needed?
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
  /** @brief Stream trapezoid information in the USolids style */
  std::ostream &StreamInfo(std::ostream &os) const;
#endif

private:
  /** @brief Factory for specializing the volume */
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

#endif // VECGEOM_VOLUMES_UNPLACEDGENTRAP_H_
