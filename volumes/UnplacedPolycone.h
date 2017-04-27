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
#include "base/Vector3D.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/UnplacedCone.h"
#include "base/Vector.h"
#include <vector>
#include "volumes/Wedge.h"
#include "volumes/PolyconeHistorical.h"
namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedPolycone;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedPolycone);

VECGEOM_DEVICE_FORWARD_DECLARE(struct PolyconeSection;);
VECGEOM_DEVICE_DECLARE_CONV(struct, PolyconeSection);

inline namespace VECGEOM_IMPL_NAMESPACE {

// helper structure to encapsulate a section
struct PolyconeSection {
  VECCORE_ATT_HOST_DEVICE
  PolyconeSection() : fSolid(0), fShift(0.0), fTubular(0), fConvex(0) {}

  VECCORE_ATT_HOST_DEVICE
  ~PolyconeSection() = default;

  UnplacedCone *fSolid;
  double fShift;
  bool fTubular;
  bool fConvex; // TRUE if all points in section are concave in regards to whole polycone, will be determined
};

// typedef std::vector<PolyconeSection> vec;
// typedef Vector<PolyconeSection> vec;

class UnplacedPolycone : public VUnplacedVolume, public AlignedBase {

public:
  // the members
  // for the phi section --> will be replaced by a wedge
  Precision fStartPhi;
  Precision fDeltaPhi;

  unsigned int fNz; // number of planes the polycone was constructed with; It should not be modified
  // Precision * fRmin;
  // Precision * fRmax;
  // Precision * fZ;

  // actual internal storage
  Vector<PolyconeSection> fSections;
  Vector<double> fZs;
  PolyconeHistorical  *fOriginal_parameters;  // original input parameters


  // These private data member and member functions are added for convexity detection
private:
  bool fEqualRmax;
  bool fContinuityOverAll;
  bool fConvexityPossible;
  Wedge fPhiWedge;
  VECCORE_ATT_HOST_DEVICE
  bool CheckContinuity(const double rOuter[], const double rInner[], const double zPlane[],
                       Vector<Precision> &newROuter, Vector<Precision> &newRInner, Vector<Precision> &zewZPlane);
  VECCORE_ATT_HOST_DEVICE
  bool CheckContinuityInRmax(const Vector<Precision> &rOuter);
  VECCORE_ATT_HOST_DEVICE
  bool CheckContinuityInSlope(const Vector<Precision> &rOuter, const Vector<Precision> &zPlane);

public:
  VECCORE_ATT_HOST_DEVICE
  void Init(double phiStart,         // initial phi starting angle
            double phiTotal,         // total phi angle
            unsigned int numZPlanes, // number of z planes
            const double zPlane[],   // position of z planes
            const double rInner[],   // tangent distance to inner surface
            const double rOuter[]);

  // the constructor
  VECCORE_ATT_HOST_DEVICE
  UnplacedPolycone(Precision phistart, Precision deltaphi, int Nz, Precision const *z, Precision const *rmin,
                   Precision const *rmax)
      : fStartPhi(phistart), fDeltaPhi(deltaphi), fNz(Nz), fSections(), fZs(Nz), fEqualRmax(true),
        fContinuityOverAll(true), fConvexityPossible(true), fPhiWedge(deltaphi, phistart)
  {
    // init internal members
    Init(phistart, deltaphi, Nz, z, rmin, rmax);
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
  unsigned int GetNz() const { return fNz; }
  VECCORE_ATT_HOST_DEVICE
  int GetNSections() const { return fSections.size(); }
  VECCORE_ATT_HOST_DEVICE
  Precision GetStartPhi() const { return fStartPhi; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetDeltaPhi() const { return fDeltaPhi; }
  VECCORE_ATT_HOST_DEVICE
  Precision GetEndPhi() const { return fStartPhi + fDeltaPhi; }
  VECCORE_ATT_HOST_DEVICE
  Wedge const &GetWedge() const { return fPhiWedge; }

  VECCORE_ATT_HOST_DEVICE
  int GetSectionIndex(Precision zposition) const
  {
    // TODO: consider binary search
    // TODO: consider making these comparisons tolerant in case we need it
    if (zposition < fZs[0]) return -1;
    for (int i = 0; i < GetNSections(); ++i) {
      if (zposition >= fZs[i] && zposition <= fZs[i + 1]) return i;
    }
    return -2;
  }

  VECCORE_ATT_HOST_DEVICE
  PolyconeSection const &GetSection(Precision zposition) const
  {
    // TODO: consider binary search
    int i = GetSectionIndex(zposition);
    return fSections[i];
  }

  VECCORE_ATT_HOST_DEVICE
  // GetSection if index is known
  PolyconeSection const &GetSection(int index) const { return fSections[index]; }

  VECCORE_ATT_HOST_DEVICE
  Precision GetRminAtPlane(int index) const
  {
    int nsect = GetNSections();
    assert(index >= 0 && index <= nsect);
    if (index == nsect)
      return fSections[index - 1].fSolid->GetRmin2();
    else
      return fSections[index].fSolid->GetRmin1();
  }

  VECCORE_ATT_HOST_DEVICE
  Precision GetRmaxAtPlane(int index) const
  {
    int nsect = GetNSections();
    assert(index >= 0 || index <= nsect);
    if (index == nsect)
      return fSections[index - 1].fSolid->GetRmax2();
    else
      return fSections[index].fSolid->GetRmax1();
  }

  VECCORE_ATT_HOST_DEVICE
  Precision GetZAtPlane(int index) const
  {
    assert(index >= 0 || index <= GetNSections());
    return fZs[index];
  }

#if !defined(VECCORE_CUDA)
  Precision Capacity() const
  {
    Precision cubicVolume = 0.;
    for (int i = 0; i < GetNSections(); i++) {
      PolyconeSection const &section = fSections[i];
      cubicVolume += section.fSolid->Capacity();
    }
    return cubicVolume;
  }

  Precision SurfaceArea() const;

  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &norm) const;

  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const;

  Vector3D<Precision> GetPointOnSurface() const;

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

#if defined(VECGEOM_USOLIDS)
  std::ostream &StreamInfo(std::ostream &os) const;
#endif

private:
  virtual void Print(std::ostream &os) const final;

  VECCORE_ATT_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECCORE_CUDA
                                           const int id,
#endif
                                           VPlacedVolume *const placement = NULL) const final;

}; // end class UnplacedPolycone

template <typename PushableContainer>
void UnplacedPolycone::ReconstructSectionArrays(PushableContainer &z, PushableContainer &rmin,
                                                PushableContainer &rmax) const
{

  double prevrmin, prevrmax;
  bool putlowersection = true;
  for (int i = 0; i < GetNSections(); ++i) {
    UnplacedCone const *cone = GetSection(i).fSolid;
    if (putlowersection) {
      rmin.push_back(cone->GetRmin1());
      rmax.push_back(cone->GetRmax1());
      z.push_back(-cone->GetDz() + GetSection(i).fShift);
    }
    rmin.push_back(cone->GetRmin2());
    rmax.push_back(cone->GetRmax2());
    z.push_back(cone->GetDz() + GetSection(i).fShift);

    prevrmin = cone->GetRmin2();
    prevrmax = cone->GetRmax2();

    // take care of a possible discontinuity
    if (i < GetNSections() - 1 &&
        (prevrmin != GetSection(i + 1).fSolid->GetRmin1() || prevrmax != GetSection(i + 1).fSolid->GetRmax1())) {
      putlowersection = true;
    } else {
      putlowersection = false;
    }
  }
}

} // end inline namespace

} // end vecgeom namespace

#endif /* VECGEOM_VOLUMES_UNPLACEDPOLYCONE_H_ */
