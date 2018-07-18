/// \file UnplacedOrb.h
/// \author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDORB_H_
#define VECGEOM_VOLUMES_UNPLACEDORB_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/OrbStruct.h" // the pure Orb struct
#include "volumes/kernel/OrbImplementation.h"
#include "volumes/UnplacedVolumeImplHelper.h"
#ifdef VECGEOM_ROOT
class TGeoShape;
#endif
#ifdef VECGEOM_GEANT4
class G4VSolid;
#endif

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedOrb;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedOrb);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedOrb : public SIMDUnplacedVolumeImplHelper<OrbImplementation>, public AlignedBase {

private:
  OrbStruct<double> fOrb;

  // Caching the Volume and SurfaceArea
  Precision fCubicVolume, fSurfaceArea;

  Precision fEpsilon, fRTolerance;

public:
  using Kernel = OrbImplementation;

  VECCORE_ATT_HOST_DEVICE
  UnplacedOrb();

  VECCORE_ATT_HOST_DEVICE
  UnplacedOrb(const Precision r);

  VECCORE_ATT_HOST_DEVICE
  void SetRadialTolerance();

  VECCORE_ATT_HOST_DEVICE
  Precision GetRadialTolerance() const { return fRTolerance; }

  VECCORE_ATT_HOST_DEVICE
  OrbStruct<double> const &GetStruct() const { return fOrb; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetRadius() const { return fOrb.fR; }

  VECCORE_ATT_HOST_DEVICE
  // VECGEOM_FORCE_INLINE
  void SetRadius(Precision r);

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  Precision Capacity() const override { return fCubicVolume; }

  // VECCORE_ATT_HOST_DEVICE
  Precision SurfaceArea() const override { return fSurfaceArea; }

  virtual Vector3D<Precision> SamplePointOnSurface() const override;

  VECCORE_ATT_HOST_DEVICE
  virtual bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const override
  {
    bool valid;
    normal = OrbImplementation::NormalKernel(fOrb, p, valid);
    return valid;
  }

  std::string GetEntityType() const;

  VECCORE_ATT_HOST_DEVICE
  void GetParametersList(int aNumber, double *aArray) const;

  VECCORE_ATT_HOST_DEVICE
  UnplacedOrb *Clone() const;

  std::ostream &StreamInfo(std::ostream &os) const;

public:
  virtual int MemorySize() const final { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override;

  virtual void Print(std::ostream &os) const override;

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedOrb>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

#ifndef VECCORE_CUDA
  // this is the function called from the VolumeFactory
  // this may be specific to the shape
  template <TranslationCode trans_code, RotationCode rot_code>
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               VPlacedVolume *const placement = NULL);

  VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume, Transformation3D const *const transformation,
                                   const TranslationCode trans_code, const RotationCode rot_code,
                                   VPlacedVolume *const placement) const override;
#else
  template <TranslationCode trans_code, RotationCode rot_code>
  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               const int id, VPlacedVolume *const placement = NULL);
  VECCORE_ATT_DEVICE VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                                      Transformation3D const *const transformation,
                                                      const TranslationCode trans_code, const RotationCode rot_code,
                                                      const int id, VPlacedVolume *const placement) const override;

#endif

// Comparison specific conversion functions
#ifndef VECCORE_CUDA
#ifdef VECGEOM_ROOT
  TGeoShape const *ConvertToRoot(char const *label = "") const;
#endif

#ifdef VECGEOM_GEANT4
  G4VSolid const *ConvertToGeant4(char const *label = "") const;
#endif
#endif // VECCORE_CUDA
};
}
} // End global namespace

#endif
