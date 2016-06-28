#ifndef VECGEOM_TUBESTRUCT_H_
#define VECGEOM_TUBESTRUCT_H_

#include "base/Global.h"
#include "volumes/Wedge_Evolution.h"

namespace vecgeom {
// VECGEOM_DEVICE_FORWARD_DECLARE( struct TubeStruct; )
// VECGEOM_DEVICE_DECLARE_BoxStruct )

inline namespace VECGEOM_IMPL_NAMESPACE {

// a plain and lightweight struct to encapsulate data members of a tube
template <typename T = double>
struct TubeStruct {
  // tube defining parameters
  T fRmin; //< inner radius
  T fRmax; //< outer radius
  T fZ;    //< half-length in +z and -z direction
  T fSphi; //< starting phi value (in radians)
  T fDphi; //< delta phi value of tube segment (in radians)

  // cached complex values (to avoid recomputation during usage)
  T fRmin2;
  T fRmax2;
  T fAlongPhi1x;
  T fAlongPhi1y;
  T fAlongPhi2x;
  T fAlongPhi2y;
  T fTolIrmin2;
  T fTolOrmin2;
  T fTolIrmax2;
  T fTolOrmax2;
  T fTolIz;
  T fTolOz;
  T fTolIrmin;
  T fTolOrmin;
  T fTolIrmax;
  T fTolOrmax;
  evolution::Wedge fPhiWedge;

  VECGEOM_CUDA_HEADER_BOTH
  void UpdatedZ()
  {
    fTolIz = fZ - kHalfTolerance;
    fTolOz = fZ + kHalfTolerance;
  }

  VECGEOM_CUDA_HEADER_BOTH
  void UpdatedRMin()
  {
    fRmin2     = fRmin * fRmin;
    fTolOrmin  = (fRmin - kHalfTolerance);
    fTolIrmin  = (fRmin + kHalfTolerance);
    fTolOrmin2 = fTolOrmin * fTolOrmin;
    fTolIrmin2 = fTolIrmin * fTolIrmin;
  }

  VECGEOM_CUDA_HEADER_BOTH
  void UpdatedRMax()
  {
    fRmax2     = fRmax * fRmax;
    fTolOrmax  = (fRmax + kHalfTolerance);
    fTolIrmax  = (fRmax - kHalfTolerance);
    fTolOrmax2 = fTolOrmax * fTolOrmax;
    fTolIrmax2 = fTolIrmax * fTolIrmax;
  }

public:
  VECGEOM_CUDA_HEADER_BOTH
  T rmin() const { return fRmin; }
  VECGEOM_CUDA_HEADER_BOTH
  T z() const { return fZ; }

private:
  // can go into source
  VECGEOM_CUDA_HEADER_BOTH
  static void GetAlongVectorToPhiSector(Precision phi, Precision &x, Precision &y)
  {
    x = std::cos(phi);
    y = std::sin(phi);
  }

public:
  VECGEOM_CUDA_HEADER_BOTH
  void SetRMin(Precision const &_rmin)
  {
    fRmin = _rmin;
    // Update cached values;
    UpdatedRMin();
  }

  VECGEOM_CUDA_HEADER_BOTH
  void SetRMax(Precision const &_rmax)
  {
    fRmax = _rmax;
    // Update cached values;
    UpdatedRMax();
  }

  VECGEOM_CUDA_HEADER_BOTH
  void SetDz(Precision const &_z)
  {
    fZ = _z;
    // Update cached values;
    UpdatedZ();
  }

  // can go into source
  VECGEOM_CUDA_HEADER_BOTH
  void SetAndCheckSPhiAngle(T sPhi)
  {
    // Ensure fSphi in 0-2PI or -2PI-0 range if shape crosses 0
    if (sPhi < 0) {
      fSphi = kTwoPi - std::fmod(std::fabs(sPhi), kTwoPi);
    } else {
      fSphi = std::fmod(sPhi, kTwoPi);
    }
    if (fSphi + fDphi > kTwoPi) {
      fSphi -= kTwoPi;
    }
    // Update cached values.
    GetAlongVectorToPhiSector(fSphi, fAlongPhi1x, fAlongPhi1y);
    GetAlongVectorToPhiSector(fSphi + fDphi, fAlongPhi2x, fAlongPhi2y);
  }

  // can go into source
  VECGEOM_CUDA_HEADER_BOTH
  void SetAndCheckDPhiAngle(T dPhi)
  {
    if (dPhi >= kTwoPi - 0.5 * kAngTolerance) {
      fDphi = kTwoPi;
      fSphi = 0;
    } else {
      if (dPhi > 0) {
        fDphi = dPhi;
      } else {
        //        std::ostringstream message;
        //        message << "Invalid dphi.\n"
        //                << "Negative or zero delta-Phi (" << dPhi << ")\n";
        //        std::cout<<"UnplacedTube::CheckDPhiAngle(): Fatal error: "<< message.str().c_str() <<"\n";
      }
    }
    // Update cached values.
    GetAlongVectorToPhiSector(fSphi + fDphi, fAlongPhi2x, fAlongPhi2y);
  }

  // can go into source
  VECGEOM_CUDA_HEADER_BOTH
  void CalculateCached()
  {
    UpdatedZ();
    UpdatedRMin();
    UpdatedRMax();

    GetAlongVectorToPhiSector(fSphi, fAlongPhi1x, fAlongPhi1y);
    GetAlongVectorToPhiSector(fSphi + fDphi, fAlongPhi2x, fAlongPhi2y);
  }

public:
  // constructors
  VECGEOM_CUDA_HEADER_BOTH
  TubeStruct(T const &_rmin, T const &_rmax, T const &_z, T const &_sphi, T const &_dphi)
      : fRmin(_rmin < 0.0 ? 0.0 : _rmin), fRmax(_rmax), fZ(_z), fSphi(_sphi), fDphi(_dphi), fRmin2(0), fRmax2(0),
        fAlongPhi1x(0), fAlongPhi1y(0), fAlongPhi2x(0), fAlongPhi2y(0), fTolIrmin2(0), fTolOrmin2(0), fTolIrmax2(0),
        fTolOrmax2(0), fTolIz(0), fTolOz(0), fPhiWedge(_dphi, _sphi)
  {
    CalculateCached();
    // DetectConvexity();
  }
};
}
} // end global namespace

#endif
