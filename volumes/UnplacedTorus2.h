/// @file UnplacedTorus2.h

#ifndef VECGEOM_VOLUMES_UNPLACEDTORUS2_H_
#define VECGEOM_VOLUMES_UNPLACEDTORUS2_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/UnplacedTube.h"
#include "volumes/Wedge.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class UnplacedTorus2; )
VECGEOM_DEVICE_DECLARE_CONV( class, UnplacedTorus2 )

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedTorus2 : public VUnplacedVolume, public AlignedBase {

private:
  // torus defining parameters ( like G4torus )
    Precision fRmin; // outer radius of torus "tube"
    Precision fRmax; // inner radius of torus "tube"
    Precision fRtor; // bending radius of torus
    Precision fSphi; // start angle
    Precision fDphi; // delta angle of torus section
    Wedge     fPhiWedge; // the Phi bounding of the torus (not the cutout)

  // cached values
  Precision fRmin2, fRmax2, fRtor2, fAlongPhi1x, fAlongPhi1y, fAlongPhi2x, fAlongPhi2y;
  Precision fTolIrmin2, fTolOrmin2, fTolIrmax2, fTolOrmax2;
  // bounding tube
  UnplacedTube fBoundingTube;

  VECGEOM_CUDA_HEADER_BOTH
  static void GetAlongVectorToPhiSector(Precision phi, Precision &x, Precision &y) {
    x = std::cos(phi);
    y = std::sin(phi);
  }

  VECGEOM_CUDA_HEADER_BOTH
  void calculateCached() {
    fRmin2 = fRmin * fRmin;
    fRmax2 = fRmax * fRmax;
    fRtor2 = fRtor * fRtor;

    fTolOrmin2 = (fRmin - kTolerance)*(fRmin - kTolerance);
    fTolIrmin2 = (fRmin + kTolerance)*(fRmin + kTolerance);

    fTolOrmax2 = (fRmax + kTolerance)*(fRmax + kTolerance);
    fTolIrmax2 = (fRmax - kTolerance)*(fRmax - kTolerance);

    GetAlongVectorToPhiSector(fSphi, fAlongPhi1x, fAlongPhi1y);
    GetAlongVectorToPhiSector(fSphi + fDphi, fAlongPhi2x, fAlongPhi2y);
  }

public:

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTorus2(const Precision rminVal, const Precision rmaxVal, const Precision rtorVal,
               const Precision sphiVal, const Precision dphiVal) : fRmin(rminVal), fRmax(rmaxVal),
    fRtor(rtorVal), fSphi(sphiVal), fDphi(dphiVal), fPhiWedge(dphiVal,sphiVal), fBoundingTube(0, 1, 1, 0, dphiVal) {
    calculateCached();

    fBoundingTube = UnplacedTube(fRtor-fRmax - kTolerance,
    fRtor+fRmax + kTolerance, fRmax,
     sphiVal, dphiVal);
    DetectConvexity();
  }

  VECGEOM_CUDA_HEADER_BOTH
  void DetectConvexity();
//  VECGEOM_CUDA_HEADER_BOTH
//  UnplacedTorus2(UnplacedTorus2 const &other) :
//  fRmin(other.fRmin), fRmax(other.fRmax), fRtor(other.fRtor), fSphi(other.fSphi), fDphi(other.fDphi),fBoundingTube(other.fBoundingTube) {
//    calculateCached();
//
//  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rmin() const { return fRmin; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rmax() const { return fRmax; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rtor() const { return fRtor; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision sphi() const { return fSphi; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision dphi() const { return fDphi; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rmin2() const { return fRmin2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rmax2() const { return fRmax2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision rtor2() const { return fRtor2; }

  VECGEOM_CUDA_HEADER_BOTH
    VECGEOM_INLINE
    Wedge const & GetWedge() const { return fPhiWedge; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision alongPhi1x() const { return fAlongPhi1x; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision alongPhi1y() const { return fAlongPhi1y; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision alongPhi2x() const { return fAlongPhi2x; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision alongPhi2y() const { return fAlongPhi2y; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision tolOrmin2() const { return fTolOrmin2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision tolIrmin2() const { return fTolIrmin2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision tolOrmax2() const { return fTolOrmax2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision tolIrmax2() const { return fTolIrmax2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision volume() const {
    return fDphi*kPi*fRtor*(fRmax*fRmax-fRmin*fRmin);
  }

  VECGEOM_CUDA_HEADER_BOTH
  void SetRMin(Precision arg) { fRmin = arg; calculateCached(); }
  VECGEOM_CUDA_HEADER_BOTH
  void SetRMax(Precision arg) { fRmax = arg; calculateCached(); }
  VECGEOM_CUDA_HEADER_BOTH
  void SetRTor(Precision arg) { fRtor = arg; calculateCached(); }
  VECGEOM_CUDA_HEADER_BOTH
  void SetSPhi(Precision arg) { fSphi = arg; calculateCached(); }
  VECGEOM_CUDA_HEADER_BOTH
  void SetDPhi(Precision arg) { fDphi = arg; calculateCached(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision SurfaceArea() const {
     Precision surfaceArea = fDphi*kTwoPi*fRtor*(fRmax+fRmin);
     if(fDphi < kTwoPi) {
       surfaceArea = surfaceArea + kTwoPi*(fRmax*fRmax-fRmin*fRmin);
     } 
     return surfaceArea;
  }

   
  VECGEOM_CUDA_HEADER_BOTH
  bool Normal(Vector3D<Precision> const& point, Vector3D<Precision>& norm) const;

  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  UnplacedTube const &GetBoundingTube() const { return fBoundingTube; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Extent(Vector3D<Precision> & min, Vector3D<Precision> & max ) const {
      GetBoundingTube().Extent( min, max );
  }

  Vector3D<Precision> GetPointOnSurface() const;

  virtual int memory_size() const final { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const final;

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECGEOM_CUDA_HEADER_DEVICE
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedTorus2>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
#endif

#if defined(VECGEOM_USOLIDS)
  std::ostream& StreamInfo(std::ostream &os) const;
#endif

  std::string GetEntityType() const { return "Torus2";}

private:

  virtual void Print(std::ostream &os) const final;

  VECGEOM_CUDA_HEADER_DEVICE
  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
      const int id,
#endif
      VPlacedVolume *const placement = NULL) const final;

};

} } // end global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDTORUS2_H_
