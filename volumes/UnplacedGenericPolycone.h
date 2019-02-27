/// @file UnplacedGenericPolycone.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDGENERICPOLYCONE_H_
#define VECGEOM_VOLUMES_UNPLACEDGENERICPOLYCONE_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/GenericPolyconeStruct.h" // the pure GenericPolycone struct
#include "volumes/kernel/GenericPolyconeImplementation.h"
#include "volumes/UnplacedVolumeImplHelper.h"
#include "volumes/ReducedPolycone.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedGenericPolycone;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedGenericPolycone);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedGenericPolycone : public LoopUnplacedVolumeImplHelper<GenericPolyconeImplementation>, public AlignedBase {

private:
  // Vector<Vector<ConeParam>> fSectionsParamVector;
  GenericPolyconeStruct<Precision> fGenericPolycone;

  // Original Polycone Parameters
  Precision fSPhi;
  Precision fDPhi;
  int fNumRZ;
  Vector<Precision> fR;
  Vector<Precision> fZ;

  // Used for Extent
  Vector3D<Precision> fAMin;
  Vector3D<Precision> fAMax;

  void CheckParameters();

public:
  /*
   * All the required Parametric GenericPolycone Constructor
   *
   */

  /*
    // This constructor will be used internally, and NOT to be exposed to the user
    VECCORE_ATT_HOST_DEVICE
    UnplacedGenericPolycone(Vector<Vector<ConeParam>> sectionsParamVector, Vector<Precision> zS)
        : fSectionsParamVector(sectionsParamVector), fGenericPolycone(sectionsParamVector, zS)
    {
    }
  */

  VECCORE_ATT_HOST_DEVICE
  UnplacedGenericPolycone() {}

  VECCORE_ATT_HOST_DEVICE
  UnplacedGenericPolycone(Precision phiStart,  // initial phi starting angle
                          Precision phiTotal,  // total phi angle
                          int numRZ,           // number corners in r,z space (must be an even number)
                          Precision const *r,  // r coordinate of these corners
                          Precision const *z); // z coordinate of these corners

  VECCORE_ATT_HOST_DEVICE
  GenericPolyconeStruct<Precision> const &GetStruct() const { return fGenericPolycone; }

  /*
   * Check Parameter function if required
   *
   */

  /*
   * Required Getters and Setters
   *
   */

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetSPhi() const { return fSPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDPhi() const { return fDPhi; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  int GetNumRz() const { return fNumRZ; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector<Precision> GetR() const { return fR; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector<Precision> GetZ() const { return fZ; }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;
  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &norm) const override;

  Precision Capacity() const override { return fGenericPolycone.fCubicVolume; }

  // Using the Generic implementation from UnplacedVolume
  Precision SurfaceArea() const override { return EstimateSurfaceArea(); }

  // Curently using the Generic Implementation from UnplacedVolume
  // Vector3D<Precision> SamplePointOnSurface() const override;

  std::string GetEntityType() const { return "GenericPolycone"; }

  std::ostream &StreamInfo(std::ostream &os) const;

public:
  virtual int MemorySize() const final { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override;

  virtual void Print(std::ostream &os) const override;

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedGenericPolycone>::SizeOf(); }
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
};

using GenericUnplacedGenericPolycone = UnplacedGenericPolycone;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
