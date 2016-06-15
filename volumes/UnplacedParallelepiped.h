/// \file UnplacedParallelepiped.h
/// \author: Johannes de Fine Licht (johannes.definelicht@cern.ch)
///  Modified and completed: mihaela.gheata@cern.ch
#ifndef VECGEOM_VOLUMES_UNPLACEDPARALLELEPIPED_H_
#define VECGEOM_VOLUMES_UNPLACEDPARALLELEPIPED_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedParallelepiped;)
VECGEOM_DEVICE_DECLARE_CONV(class,UnplacedParallelepiped)

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedParallelepiped : public VUnplacedVolume, public AlignedBase {

private:
  Vector3D<Precision> fDimensions; /** Dimensions dx, dy, dx */
  Precision fAlpha;                /** Angle dx versus dy (degrees)*/
  Precision fTheta;                /** Theta angle of parallelepiped axis*/
  Precision fPhi;                  /** Phi angle of parallelepiped axis*/
  Precision fCtx;                  /** Cosine of xz angle */
  Precision fCty;                  /** Cosine of yz angle */
  Vector3D<Precision> fNormals[3]; /** Precomputed normals */

  // Precomputed values computed from parameters
  Precision fTanAlpha, fTanThetaSinPhi, fTanThetaCosPhi;

  /** @brief Compute normals */
  VECGEOM_CUDA_HEADER_BOTH
  void ComputeNormals();

public:
  /** @brief Constructor */
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedParallelepiped(Vector3D<Precision> const &dimensions, const Precision alpha, const Precision theta,
                         const Precision phi);

  /** @brief Constructor */
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedParallelepiped(const Precision x, const Precision y, const Precision z, const Precision alpha,
                         const Precision theta, const Precision phi);

  /** @brief Getter for parallelepiped dimensions */
  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> const &GetDimensions() const { return fDimensions; }

  /** @brief Getter for parallelepiped normals */
  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> const &GetNormal(int i) const { return fNormals[i]; }

  /** @brief Getter for dX */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetX() const { return fDimensions[0]; }

  /** @brief Getter for dY */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetY() const { return fDimensions[1]; }

  /** @brief Getter for dZ */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetZ() const { return fDimensions[2]; }

  /** @brief Getter for alpha */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetAlpha() const { return fAlpha; }

  /** @brief Getter for theta */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTheta() const { return fTheta; }

  /** @brief Getter for phi */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetPhi() const { return fPhi; }

  /** @brief Getter for tan(alpha) */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanAlpha() const { return fTanAlpha; }

  /** @brief Getter for tan(th)*sin(phi) */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanThetaSinPhi() const { return fTanThetaSinPhi; }

  /** @brief Getter for tan(th)*cos(phi) */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanThetaCosPhi() const { return fTanThetaCosPhi; }

  /** @brief Getter for fCtx */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetCtx() const { return fCtx; }

  /** @brief Getter for fCty */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetCty() const { return fCty; }

  /** @brief Setter for dimensions on x,y,z */
  VECGEOM_CUDA_HEADER_BOTH
  void SetDimensions(Vector3D<Precision> const &dimensions) { fDimensions = dimensions; }

  /** @brief Setter for dimensions on x,y,z */
  VECGEOM_CUDA_HEADER_BOTH
  void SetDimensions(const Precision x, const Precision y, const Precision z) { fDimensions.Set(x, y, z); }

  /** @brief Setter for alpha */
  VECGEOM_CUDA_HEADER_BOTH
  void SetAlpha(const Precision alpha);

  /** @brief Setter for theta */
  VECGEOM_CUDA_HEADER_BOTH
  void SetTheta(const Precision theta);

  /** @brief Setter for phi */
  VECGEOM_CUDA_HEADER_BOTH
  void SetPhi(const Precision phi);

  /** @brief Setter for theta and phi */
  VECGEOM_CUDA_HEADER_BOTH
  void SetThetaAndPhi(const Precision theta, const Precision phi);

  virtual int memory_size() const final { return sizeof(*this); }

  /** @brief Print parameters of the parallelepiped */
  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const final;

  /** @brief Computes the extent on X/Y/Z of the parallelepiped */
  VECGEOM_CUDA_HEADER_BOTH
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const;

#ifndef VECGEOM_NVCC
  /** @brief Generates randomly a point on the surface of the parallelepiped */
  Vector3D<Precision> GetPointOnSurface() const;

  /** @brief Implementation of capacity computation */
  VECGEOM_INLINE
  Precision volume() const { return 8.0 * fDimensions[0] * fDimensions[1] * fDimensions[2]; }

  /** @brief Interface method for computing capacity */
  Precision Capacity() { return volume(); }

  /** @brief Implementation of surface area computation */
  VECGEOM_INLINE
  Precision SurfaceArea() const {
    // factor 8 because dimensions_ are half-lengths
    Precision ctinv = 1. / cos(kDegToRad * fTheta);
    return 8.0 * (fDimensions[0] * fDimensions[1] +
                  fDimensions[1] * fDimensions[2] * sqrt(ctinv * ctinv - fTanThetaSinPhi * fTanThetaSinPhi) +
                  fDimensions[2] * fDimensions[0] * sqrt(ctinv * ctinv - fTanThetaCosPhi * fTanThetaCosPhi));
  }
#endif

  /** @brief Compute normal vector to surface */
  VECGEOM_CUDA_HEADER_BOTH
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const;

  /** @brief Get type name */
  std::string GetEntityType() const { return "parallelepiped"; }

  /** @brief Templated factory for creating a placed volume */
  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECGEOM_CUDA_HEADER_DEVICE static VPlacedVolume *Create(LogicalVolume const *const logical_volume,
                                                          Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
                                                          const int id,
#endif
                                                          VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedParallelepiped>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
#endif

private:
  virtual void Print(std::ostream &os) const final;

  VECGEOM_CUDA_HEADER_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
                                           const int id,
#endif
                                           VPlacedVolume *const placement = NULL) const final;
};
}
} // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDPARALLELEPIPED_H_
