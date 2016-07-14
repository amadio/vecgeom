/// \file UnplacedParallelepiped.h
/// \author: Johannes de Fine Licht (johannes.definelicht@cern.ch)
///  Modified and completed: mihaela.gheata@cern.ch
#ifndef VECGEOM_VOLUMES_UNPLACEDPARALLELEPIPED_H_
#define VECGEOM_VOLUMES_UNPLACEDPARALLELEPIPED_H_

#include "base/AlignedBase.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedVolume.h"
#include "ParallelepipedStruct.h"
#include "volumes/kernel/ParallelepipedImplementation.h"
#include "volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedParallelepiped;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedParallelepiped);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedParallelepiped : public SIMDUnplacedVolumeImplHelper<ParallelepipedImplementation>, public AlignedBase {

private:
  ParallelepipedStruct<double> fPara; /** The parallelepiped structure */

public:
  /** @brief Constructor */
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedParallelepiped(Vector3D<Precision> const &dimensions, const Precision alpha, const Precision theta,
                         const Precision phi)
      : fPara(dimensions, alpha, theta, phi)
  {
    fGlobalConvexity = true;
  }

  /** @brief Constructor */
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedParallelepiped(const Precision x, const Precision y, const Precision z, const Precision alpha,
                         const Precision theta, const Precision phi)
      : fPara(x, y, z, alpha, theta, phi)
  {
    fGlobalConvexity = true;
  }

  /** @brief Interface getter for parallelepiped struct */
  VECGEOM_CUDA_HEADER_BOTH
  ParallelepipedStruct<double> const &GetStruct() const { return fPara; }

  /** @brief Getter for parallelepiped dimensions */
  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> const &GetDimensions() const { return fPara.fDimensions; }

  /** @brief Getter for parallelepiped normals */
  VECGEOM_CUDA_HEADER_BOTH
  Vector3D<Precision> const &GetNormal(int i) const { return fPara.fNormals[i]; }

  /** @brief Getter for dX */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetX() const { return fPara.fDimensions[0]; }

  /** @brief Getter for dY */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetY() const { return fPara.fDimensions[1]; }

  /** @brief Getter for dZ */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetZ() const { return fPara.fDimensions[2]; }

  /** @brief Getter for alpha */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetAlpha() const { return fPara.fAlpha; }

  /** @brief Getter for theta */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTheta() const { return fPara.fTheta; }

  /** @brief Getter for phi */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetPhi() const { return fPara.fPhi; }

  /** @brief Getter for tan(alpha) */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanAlpha() const { return fPara.fTanAlpha; }

  /** @brief Getter for tan(th)*sin(phi) */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanThetaSinPhi() const { return fPara.fTanThetaSinPhi; }

  /** @brief Getter for tan(th)*cos(phi) */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanThetaCosPhi() const { return fPara.fTanThetaCosPhi; }

  /** @brief Getter for fCtx */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetCtx() const { return fPara.fCtx; }

  /** @brief Getter for fCty */
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetCty() const { return fPara.fCty; }

  /** @brief Setter for dimensions on x,y,z */
  VECGEOM_CUDA_HEADER_BOTH
  void SetDimensions(Vector3D<Precision> const &dimensions) { fPara.fDimensions = dimensions; }

  /** @brief Setter for dimensions on x,y,z */
  VECGEOM_CUDA_HEADER_BOTH
  void SetDimensions(const Precision x, const Precision y, const Precision z) { fPara.fDimensions.Set(x, y, z); }

  /** @brief Setter for alpha */
  VECGEOM_CUDA_HEADER_BOTH
  void SetAlpha(const Precision alpha) { fPara.SetAlpha(alpha); }

  /** @brief Setter for theta */
  VECGEOM_CUDA_HEADER_BOTH
  void SetTheta(const Precision theta) { fPara.SetTheta(theta); }

  /** @brief Setter for phi */
  VECGEOM_CUDA_HEADER_BOTH
  void SetPhi(const Precision phi) { fPara.SetPhi(phi); }

  /** @brief Setter for theta and phi */
  VECGEOM_CUDA_HEADER_BOTH
  void SetThetaAndPhi(const Precision theta, const Precision phi) { fPara.SetThetaAndPhi(theta, phi); }

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
  VECGEOM_FORCE_INLINE
  Precision volume() const { return 8.0 * fPara.fDimensions[0] * fPara.fDimensions[1] * fPara.fDimensions[2]; }

  /** @brief Interface method for computing capacity */
  Precision Capacity() { return volume(); }

  /** @brief Implementation of surface area computation */
  VECGEOM_FORCE_INLINE
  Precision SurfaceArea() const
  {
    // factor 8 because dimensions_ are half-lengths
    Precision ctinv = 1. / cos(kDegToRad * fPara.fTheta);
    return 8.0 * (fPara.fDimensions[0] * fPara.fDimensions[1] +
                  fPara.fDimensions[1] * fPara.fDimensions[2] *
                      sqrt(ctinv * ctinv - fPara.fTanThetaSinPhi * fPara.fTanThetaSinPhi) +
                  fPara.fDimensions[2] * fPara.fDimensions[0] *
                      sqrt(ctinv * ctinv - fPara.fTanThetaCosPhi * fPara.fTanThetaCosPhi));
  }
#endif

  /** @brief Compute normal vector to surface */
  VECGEOM_CUDA_HEADER_BOTH
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const;

  /** @brief Get type name */
  std::string GetEntityType() const { return "parallelepiped"; }

  /** @brief Templated factory for creating a placed volume */
  template <TranslationCode transCodeT, RotationCode rotCodeT>
#ifdef VECGEOM_NVCC
  VECGEOM_CUDA_HEADER_DEVICE
#endif
      static VPlacedVolume *
      Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
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

#ifdef VECGEOM_NVCC
  VECGEOM_CUDA_HEADER_DEVICE
#endif
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
