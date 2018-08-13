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
  using Kernel = ParallelepipedImplementation;
  /** @brief Constructor */
  VECCORE_ATT_HOST_DEVICE
  UnplacedParallelepiped(Vector3D<Precision> const &dimensions, const Precision alpha, const Precision theta,
                         const Precision phi)
      : fPara(dimensions, alpha, theta, phi)
  {
    fGlobalConvexity = true;
  }

  /** @brief Constructor */
  VECCORE_ATT_HOST_DEVICE
  UnplacedParallelepiped(const Precision x, const Precision y, const Precision z, const Precision alpha,
                         const Precision theta, const Precision phi)
      : fPara(x, y, z, alpha, theta, phi)
  {
    fGlobalConvexity = true;
  }

  /** @brief Default constructor */
  VECCORE_ATT_HOST_DEVICE
  UnplacedParallelepiped() : fPara(0., 0., 0., 0., 0., 0.) { fGlobalConvexity = true; }

  /** @brief Interface getter for parallelepiped struct */
  VECCORE_ATT_HOST_DEVICE
  ParallelepipedStruct<double> const &GetStruct() const { return fPara; }

  /** @brief Getter for parallelepiped dimensions */
  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> const &GetDimensions() const { return fPara.fDimensions; }

  /** @brief Getter for parallelepiped normals */
  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> const &GetNormal(int i) const { return fPara.fNormals[i]; }

  /** @brief Getter for dX */
  VECCORE_ATT_HOST_DEVICE
  Precision GetX() const { return fPara.fDimensions[0]; }

  /** @brief Getter for dY */
  VECCORE_ATT_HOST_DEVICE
  Precision GetY() const { return fPara.fDimensions[1]; }

  /** @brief Getter for dZ */
  VECCORE_ATT_HOST_DEVICE
  Precision GetZ() const { return fPara.fDimensions[2]; }

  /** @brief Getter for alpha */
  VECCORE_ATT_HOST_DEVICE
  Precision GetAlpha() const { return fPara.fAlpha; }

  /** @brief Getter for theta */
  VECCORE_ATT_HOST_DEVICE
  Precision GetTheta() const { return fPara.fTheta; }

  /** @brief Getter for phi */
  VECCORE_ATT_HOST_DEVICE
  Precision GetPhi() const { return fPara.fPhi; }

  /** @brief Getter for tan(alpha) */
  VECCORE_ATT_HOST_DEVICE
  Precision GetTanAlpha() const { return fPara.fTanAlpha; }

  /** @brief Getter for tan(th)*sin(phi) */
  VECCORE_ATT_HOST_DEVICE
  Precision GetTanThetaSinPhi() const { return fPara.fTanThetaSinPhi; }

  /** @brief Getter for tan(th)*cos(phi) */
  VECCORE_ATT_HOST_DEVICE
  Precision GetTanThetaCosPhi() const { return fPara.fTanThetaCosPhi; }

  /** @brief Getter for fCtx */
  VECCORE_ATT_HOST_DEVICE
  Precision GetCtx() const { return fPara.fCtx; }

  /** @brief Getter for fCty */
  VECCORE_ATT_HOST_DEVICE
  Precision GetCty() const { return fPara.fCty; }

  /** @brief Setter for dimensions on x,y,z */
  VECCORE_ATT_HOST_DEVICE
  void SetDimensions(Vector3D<Precision> const &dimensions) { fPara.fDimensions = dimensions; }

  /** @brief Setter for dimensions on x,y,z */
  VECCORE_ATT_HOST_DEVICE
  void SetDimensions(const Precision x, const Precision y, const Precision z) { fPara.fDimensions.Set(x, y, z); }

  /** @brief Setter for alpha */
  VECCORE_ATT_HOST_DEVICE
  void SetAlpha(const Precision alpha) { fPara.SetAlpha(alpha); }

  /** @brief Setter for theta */
  VECCORE_ATT_HOST_DEVICE
  void SetTheta(const Precision theta) { fPara.SetTheta(theta); }

  /** @brief Setter for phi */
  VECCORE_ATT_HOST_DEVICE
  void SetPhi(const Precision phi) { fPara.SetPhi(phi); }

  /** @brief Setter for theta and phi */
  VECCORE_ATT_HOST_DEVICE
  void SetThetaAndPhi(const Precision theta, const Precision phi) { fPara.SetThetaAndPhi(theta, phi); }

  virtual int MemorySize() const final { return sizeof(*this); }

  /** @brief Print parameters of the parallelepiped */
  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const final;

  virtual void Print(std::ostream &os) const final;

  /** @brief Computes the extent on X/Y/Z of the parallelepiped */
  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  /** @brief Implementation of capacity computation */
  Precision volume() const { return 8.0 * fPara.fDimensions[0] * fPara.fDimensions[1] * fPara.fDimensions[2]; }

  /** @brief Interface method for computing capacity */
  Precision Capacity() const override { return volume(); }
#ifndef VECCORE_CUDA
  /** @brief Generates randomly a point on the surface of the parallelepiped */
  Vector3D<Precision> SamplePointOnSurface() const override;
#endif

  /** @brief Implementation of surface area computation */
  Precision SurfaceArea() const override
  {
    // factor 8 because dimensions_ are half-lengths
    Precision ctinv = 1. / cos(kDegToRad * fPara.fTheta);
    return 8.0 * (fPara.fDimensions[0] * fPara.fDimensions[1] +
                  fPara.fDimensions[1] * fPara.fDimensions[2] *
                      sqrt(ctinv * ctinv - fPara.fTanThetaSinPhi * fPara.fTanThetaSinPhi) +
                  fPara.fDimensions[2] * fPara.fDimensions[0] *
                      sqrt(ctinv * ctinv - fPara.fTanThetaCosPhi * fPara.fTanThetaCosPhi));
  }

  /** @brief Compute normal vector to surface */
  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override;

  /** @brief Get type name */
  std::string GetEntityType() const { return "parallelepiped"; }

  /** @brief Templated factory for creating a placed volume */
  template <TranslationCode transCodeT, RotationCode rotCodeT>
#ifdef VECCORE_CUDA
  VECCORE_ATT_DEVICE
#endif
      static VPlacedVolume *
      Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
             const int id,
#endif
             VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedParallelepiped>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

#ifdef VECCORE_CUDA
  VECCORE_ATT_DEVICE
#endif
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECCORE_CUDA
                                           const int id,
#endif
                                           VPlacedVolume *const placement = NULL) const final;
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_UNPLACEDPARALLELEPIPED_H_
