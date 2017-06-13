/*
 * UnplacedPolycone.h
 *
 *  Created on: Dec 8, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_UNPLACEDPOLYCONE_H_
#define VECGEOM_VOLUMES_UNPLACEDPOLYCONE_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/PolyconeStruct.h"
#include "volumes/kernel/PolyconeImplementation.h"
#include "volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedPolycone;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedPolycone);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedPolycone : public LoopUnplacedVolumeImplHelper<PolyconeImplementation>, public AlignedBase {

private:
  PolyconeStruct<double> fPolycone;

public:
  // the constructor
  VECCORE_ATT_HOST_DEVICE
  UnplacedPolycone(Precision phistart, Precision deltaphi, int Nz, Precision const *z, Precision const *rmin,
                   Precision const *rmax)
  {
    // init internal members
    fPolycone.fContinuityOverAll = true;
    fPolycone.fConvexityPossible = true;
    fPolycone.Init(phistart, deltaphi, Nz, z, rmin, rmax);
    DetectConvexity();
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
  PolyconeStruct<double> const &GetStruct() const { return fPolycone; }
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

#if !defined(VECCORE_CUDA)
  Precision Capacity() const
  {
    Precision cubicVolume = 0.;
    for (int i = 0; i < GetNSections(); i++) {
      PolyconeSection const &section = fPolycone.fSections[i];
      cubicVolume += section.fSolid->Capacity();
    }
    return cubicVolume;
  }

  Precision SurfaceArea() const;

  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &norm) const;

  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const;

  Vector3D<Precision> SamplePointOnSurface() const;

  // Methods for random point generation
  Vector3D<Precision> GetPointOnCone(Precision fRmin1, Precision fRmax1, Precision fRmin2, Precision fRmax2,
                                     Precision zOne, Precision zTwo, Precision &totArea) const;

  Vector3D<Precision> GetPointOnTubs(Precision fRMin, Precision fRMax, Precision zOne, Precision zTwo,
                                     Precision &totArea) const;

  Vector3D<Precision> GetPointOnCut(Precision fRMin1, Precision fRMax1, Precision fRMin2, Precision fRMax2,
                                    Precision zOne, Precision zTwo, Precision &totArea) const;

  Vector3D<Precision> GetPointOnRing(Precision fRMin, Precision fRMax, Precision fRMin2, Precision fRMax2,
                                     Precision zOne) const;

  std::string GetEntityType() const { return "Polycone"; }
#endif // !VECCORE_CUDA

  // a method to reconstruct "plane" section arrays for z, rmin and rmax
  template <typename PushableContainer>
  void ReconstructSectionArrays(PushableContainer &z, PushableContainer &rmin, PushableContainer &rmax) const;

  // these methods are required by VUnplacedVolume
  //
public:
  virtual int MemorySize() const final { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const final;
  virtual void Print(std::ostream &os) const final;

#if defined(VECGEOM_USOLIDS)
  std::ostream &StreamInfo(std::ostream &os) const;
#endif

  VECCORE_ATT_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECCORE_CUDA
                                           const int id,
#endif
                                           VPlacedVolume *const placement = NULL) const final;

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedPolycone>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
#endif

}; // end class UnplacedPolycone

template <typename PushableContainer>
void UnplacedPolycone::ReconstructSectionArrays(PushableContainer &z, PushableContainer &rmin,
                                                PushableContainer &rmax) const
{

  double prevrmin, prevrmax;
  bool putlowersection = true;
  for (int i = 0; i < GetNSections(); ++i) {
    ConeStruct<double> const *cone = fPolycone.GetSection(i).fSolid;
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

} // end inline namespace

} // end vecgeom namespace

#endif /* VECGEOM_VOLUMES_UNPLACEDPOLYCONE_H_ */
