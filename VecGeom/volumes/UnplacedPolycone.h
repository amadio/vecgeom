/*
 * UnplacedPolycone.h
 *
 *  Created on: Dec 8, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_UNPLACEDPOLYCONE_H_
#define VECGEOM_VOLUMES_UNPLACEDPOLYCONE_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/PolyconeStruct.h"
#include "VecGeom/volumes/kernel/PolyconeImplementation.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"
#include "VecGeom/volumes/kernel/shapetypes/ConeTypes.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedPolycone;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedPolycone);
VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(class, SUnplacedPolycone, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedPolycone : public VUnplacedVolume {

private:
  PolyconeStruct<Precision> fPolycone;

public:
  // Constructor needed by specialization when Polycone becomes Cone
  UnplacedPolycone(Precision rmin1, Precision rmax1, Precision rmin2, Precision rmax2, Precision dz, Precision phistart,
                   Precision deltaphi)
  {
    int Nz = 2;
    Precision rMin[2];
    Precision rMax[2];
    Precision z[2];
    rMin[0]                      = rmin1;
    rMin[1]                      = rmin2;
    rMax[0]                      = rmax1;
    rMax[1]                      = rmax2;
    z[0]                         = -dz;
    z[1]                         = dz;
    fPolycone.fContinuityOverAll = true;
    fPolycone.fConvexityPossible = true;
    fPolycone.fEqualRmax         = true;
    fPolycone.Init(phistart, deltaphi, Nz, z, rMin, rMax);
    DetectConvexity();
    ComputeBBox();
  }

  // Constructor needed by specialization when Polycone becomes Tube
  UnplacedPolycone(Precision rmin, Precision rmax, Precision dz, Precision phistart, Precision deltaphi)
  {
    int Nz = 2;
    Precision rMin[2];
    Precision rMax[2];
    Precision z[2];
    rMin[0]                      = rmin;
    rMin[1]                      = rmin;
    rMax[0]                      = rmax;
    rMax[1]                      = rmax;
    z[0]                         = -dz;
    z[1]                         = dz;
    fPolycone.fContinuityOverAll = true;
    fPolycone.fConvexityPossible = true;
    fPolycone.fEqualRmax         = true;
    fPolycone.Init(phistart, deltaphi, Nz, z, rMin, rMax);
    DetectConvexity();
    ComputeBBox();
  }

  // the constructor
  VECCORE_ATT_HOST_DEVICE
  UnplacedPolycone(Precision phistart, Precision deltaphi, int Nz, Precision const *z, Precision const *rmin,
                   Precision const *rmax)
  {
    // init internal members
    fPolycone.fContinuityOverAll = true;
    fPolycone.fConvexityPossible = true;
    fPolycone.fEqualRmax         = true;
    fPolycone.Init(phistart, deltaphi, Nz, z, rmin, rmax);
    DetectConvexity();
    ComputeBBox();
  }

  // alternative constructor, required for integration with Geant4
  VECCORE_ATT_HOST_DEVICE
  UnplacedPolycone(Precision phiStart,  // initial phi starting angle
                   Precision phiTotal,  // total phi angle
                   int numRZ,           // number corners in r,z space
                   Precision const *r,  // r coordinate of these corners
                   Precision const *z); // z coordinate of these corners

  VECCORE_ATT_HOST_DEVICE
  void DetectConvexity();

  VECCORE_ATT_HOST_DEVICE
  void Reset();

  VECCORE_ATT_HOST_DEVICE
  PolyconeHistorical *GetOriginalParameters() const { return fPolycone.GetOriginalParameters(); }

  VECCORE_ATT_HOST_DEVICE
  PolyconeStruct<Precision> const &GetStruct() const { return fPolycone; }
  VECCORE_ATT_HOST_DEVICE
  unsigned int GetNz() const { return fPolycone.fNz; }
  VECCORE_ATT_HOST_DEVICE
  int GetNSections() const { return fPolycone.fSections.size(); }
  VECCORE_ATT_HOST_DEVICE
  Precision GetStartPhi() const { return fPolycone.fStartPhi; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetDeltaPhi() const { return fPolycone.fDeltaPhi; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetEndPhi() const { return fPolycone.fStartPhi + fPolycone.fDeltaPhi; }
  VECCORE_ATT_HOST_DEVICE
  evolution::Wedge const &GetWedge() const { return fPolycone.fPhiWedge; }

  VECCORE_ATT_HOST_DEVICE
  int GetSectionIndex(Precision zposition) const { return fPolycone.GetSectionIndex(zposition); }

  VECCORE_ATT_HOST_DEVICE
  PolyconeSection const &GetSection(Precision zposition) const { return fPolycone.GetSection(zposition); }

  VECCORE_ATT_HOST_DEVICE
  // GetSection if index is known
  PolyconeSection const &GetSection(int index) const { return fPolycone.fSections[index]; }

  VECCORE_ATT_HOST_DEVICE
  Precision GetRminAtPlane(int index) const { return fPolycone.GetRminAtPlane(index); }

  VECCORE_ATT_HOST_DEVICE
  Precision GetRmaxAtPlane(int index) const { return fPolycone.GetRmaxAtPlane(index); }

  VECCORE_ATT_HOST_DEVICE
  Precision GetZAtPlane(int index) const { return fPolycone.GetZAtPlane(index); }

  Precision Capacity() const override
  {
    Precision cubicVolume = 0.;
    for (int i = 0; i < GetNSections(); i++) {
      PolyconeSection const &section = fPolycone.fSections[i];
      cubicVolume += section.fSolid->Capacity();
    }
    return cubicVolume;
  }

  Precision SurfaceArea() const override;

#if !defined(VECCORE_CUDA)
  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &norm) const override;

  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override;

  Vector3D<Precision> SamplePointOnSurface() const override;

  // Methods for random point generation
  Vector3D<Precision> GetPointOnCone(Precision fRmin1, Precision fRmax1, Precision fRmin2, Precision fRmax2,
                                     Precision zOne, Precision zTwo, Precision &totArea) const;

  Vector3D<Precision> GetPointOnTubs(Precision fRMin, Precision fRMax, Precision zOne, Precision zTwo,
                                     Precision &totArea) const;

  Vector3D<Precision> GetPointOnCut(Precision fRMin1, Precision fRMax1, Precision fRMin2, Precision fRMax2,
                                    Precision zOne, Precision zTwo, Precision &totArea) const;

  Vector3D<Precision> GetPointOnRing(Precision fRMin, Precision fRMax, Precision fRMin2, Precision fRMax2,
                                     Precision zOne) const;

#endif // !VECCORE_CUDA

  // a method to reconstruct "plane" section arrays for z, rmin and rmax
  template <typename PushableContainer>
  void ReconstructSectionArrays(PushableContainer &z, PushableContainer &rmin, PushableContainer &rmax) const;

  // these methods are required by VUnplacedVolume
  //
public:
  virtual int MemorySize() const override { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const final;
  virtual void Print(std::ostream &os) const final;

#ifndef VECCORE_CUDA
  virtual SolidMesh *CreateMesh3D(Transformation3D const &trans, size_t nSegments) const override;
#endif

  std::ostream &StreamInfo(std::ostream &os) const;
  std::string GetEntityType() const { return "Polycone"; }

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override
  {
    return DevicePtr<cuda::SUnplacedPolycone<cuda::ConeTypes::UniversalCone>>::SizeOf();
  }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
  static void CopyToGpu(std::vector<VUnplacedVolume const *> const & volumes, std::vector<DevicePtr<cuda::VUnplacedVolume>> const & devicePointers);
#endif

#ifndef VECCORE_CUDA
#ifdef VECGEOM_ROOT
  TGeoShape const *ConvertToRoot(char const *label) const;
#endif

#ifdef VECGEOM_GEANT4
  G4VSolid const *ConvertToGeant4(char const *label) const;
#endif
#endif

}; // end class UnplacedPolycone

template <>
struct Maker<UnplacedPolycone> {
  template <typename... ArgTypes>
  static UnplacedPolycone *MakeInstance(Precision phistart, Precision deltaphi, int Nz, Precision const *z,
                                        Precision const *rmin, Precision const *rmax);
  template <typename... ArgTypes>
  static UnplacedPolycone *MakeInstance(Precision phistart, Precision deltaphi, int Nz, Precision const *r,
                                        Precision const *z);
};

template <typename PushableContainer>
void UnplacedPolycone::ReconstructSectionArrays(PushableContainer &z, PushableContainer &rmin,
                                                PushableContainer &rmax) const
{

  Precision prevrmin, prevrmax;
  bool putlowersection = true;
  for (int i = 0; i < GetNSections(); ++i) {
    ConeStruct<Precision> const *cone = fPolycone.GetSection(i).fSolid;
    if (putlowersection) {
      rmin.push_back(cone->fRmin1); // GetRmin1());
      rmax.push_back(cone->fRmax1); // GetRmax1());
      z.push_back(-cone->fDz + fPolycone.GetSection(i).fShift);
    }
    rmin.push_back(cone->fRmin2); // GetRmin2());
    rmax.push_back(cone->fRmax2); // GetRmax2());
    z.push_back(cone->fDz + fPolycone.GetSection(i).fShift);

    prevrmin = cone->fRmin2; // GetRmin2();
    prevrmax = cone->fRmax2; // GetRmax2();

    // take care of a possible discontinuity
    if (i < GetNSections() - 1 && (prevrmin != fPolycone.GetSection(i + 1).fSolid->fRmin1 ||
                                   prevrmax != fPolycone.GetSection(i + 1).fSolid->fRmax1)) {
      putlowersection = true;
    } else {
      putlowersection = false;
    }
  }
}

template <typename PolyconeType = ConeTypes::UniversalCone>
class SUnplacedPolycone : public UnplacedVolumeImplHelper<PolyconeImplementation<PolyconeType>, UnplacedPolycone>,
                          public AlignedBase {
public:
  using BaseType_t = UnplacedVolumeImplHelper<PolyconeImplementation<PolyconeType>, UnplacedPolycone>;
  using BaseType_t::BaseType_t;

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                               const int id, const int copy_no, const int child_id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifndef VECCORE_CUDA
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           const TranslationCode trans_code, const RotationCode rot_code,
                                           VPlacedVolume *const placement = NULL) const override
  {
    return VolumeFactory::CreateByTransformation<SUnplacedPolycone<PolyconeType>>(volume, transformation, trans_code,
                                                                                  rot_code, placement);
  }

#else
  VECCORE_ATT_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           const TranslationCode trans_code, const RotationCode rot_code, const int id,
                                           const int copy_no, const int child_id,
                                           VPlacedVolume *const placement = NULL) const override
  {
    return VolumeFactory::CreateByTransformation<SUnplacedPolycone<PolyconeType>>(
        volume, transformation, trans_code, rot_code, id, copy_no, child_id, placement);
  }
#endif
};

using GenericUnplacedPolycone = SUnplacedPolycone<ConeTypes::UniversalCone>;

} // namespace VECGEOM_IMPL_NAMESPACE

} // namespace vecgeom

#include "VecGeom/volumes/SpecializedPolycone.h"

#endif /* VECGEOM_VOLUMES_UNPLACEDPOLYCONE_H_ */
