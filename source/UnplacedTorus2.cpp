/// \file UnplacedTorus2.cpp

#include "volumes/UnplacedTorus2.h"
#include "volumes/SpecializedTorus2.h"

#include "volumes/utilities/VolumeUtilities.h"
#include "management/VolumeFactory.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

void UnplacedTorus2::Print() const
{
  printf("UnplacedTorus2 {%.2f, %.2f, %.2f, %.2f, %.2f}", rmin(), rmax(), rtor(), sphi(), dphi());
}

void UnplacedTorus2::Print(std::ostream &os) const
{
  os << "UnplacedTorus2 {" << rmin() << ", " << rmax() << ", " << rtor() << ", " << sphi() << ", " << dphi();
}

#ifndef VECCORE_CUDA
VPlacedVolume *UnplacedTorus2::SpecializedVolume(LogicalVolume const *const volume,
                                                 Transformation3D const *const transformation,
                                                 const TranslationCode trans_code, const RotationCode rot_code,
                                                 VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedTorus2>(volume, transformation, trans_code, rot_code, placement);
}
#else
__device__ VPlacedVolume *UnplacedTorus2::SpecializedVolume(LogicalVolume const *const volume,
                                                            Transformation3D const *const transformation,
                                                            const TranslationCode trans_code,
                                                            const RotationCode rot_code, const int id,
                                                            VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedTorus2>(volume, transformation, trans_code, rot_code, id,
                                                               placement);
}
#endif

Vector3D<Precision> UnplacedTorus2::SamplePointOnSurface() const
{
  // taken from Geant4
  Precision cosu, sinu, cosv, sinv, aOut, aIn, aSide, chose, phi, theta, rRand;

  phi   = RNG::Instance().uniform(fTorus.fSphi, fTorus.fSphi + fTorus.fDphi);
  theta = RNG::Instance().uniform(0., vecgeom::kTwoPi);

  cosu = std::cos(phi);
  sinu = std::sin(phi);
  cosv = std::cos(theta);
  sinv = std::sin(theta);

  // compute the areas

  aOut  = (fTorus.fDphi) * vecgeom::kTwoPi * fTorus.fRtor * fTorus.fRmax;
  aIn   = (fTorus.fDphi) * vecgeom::kTwoPi * fTorus.fRtor * fTorus.fRmin;
  aSide = vecgeom::kPi * (fTorus.fRmax * fTorus.fRmax - fTorus.fRmin * fTorus.fRmin);

  if ((fTorus.fSphi == 0.) && (fTorus.fDphi == vecgeom::kTwoPi)) {
    aSide = 0;
  }
  chose = RNG::Instance().uniform(0., aOut + aIn + 2. * aSide);

  if (chose < aOut) {
    return Vector3D<Precision>((fTorus.fRtor + fTorus.fRmax * cosv) * cosu, (fTorus.fRtor + fTorus.fRmax * cosv) * sinu,
                               fTorus.fRmax * sinv);
  } else if ((chose >= aOut) && (chose < aOut + aIn)) {
    return Vector3D<Precision>((fTorus.fRtor + fTorus.fRmin * cosv) * cosu, (fTorus.fRtor + fTorus.fRmin * cosv) * sinu,
                               fTorus.fRmin * sinv);
  } else if ((chose >= aOut + aIn) && (chose < aOut + aIn + aSide)) {
    rRand = volumeUtilities::GetRadiusInRing(fTorus.fRmin, fTorus.fRmax);
    return Vector3D<Precision>((fTorus.fRtor + rRand * cosv) * std::cos(fTorus.fSphi),
                               (fTorus.fRtor + rRand * cosv) * std::sin(fTorus.fSphi), rRand * sinv);
  } else {
    rRand = volumeUtilities::GetRadiusInRing(fTorus.fRmin, fTorus.fRmax);
    return Vector3D<Precision>((fTorus.fRtor + rRand * cosv) * std::cos(fTorus.fSphi + fTorus.fDphi),
                               (fTorus.fRtor + rRand * cosv) * std::sin(fTorus.fSphi + fTorus.fDphi), rRand * sinv);
  }
}

VECCORE_ATT_HOST_DEVICE
void UnplacedTorus2::DetectConvexity()
{
  // Default safe convexity value
  fGlobalConvexity = false;

  // Logic to calculate the convexity
  if (fTorus.fRtor == 0.) {   // This will turn Torus2 to Spherical Shell
    if (fTorus.fRmin == 0.) { // This will turn the Spherical shell to Orb
      if (fTorus.fDphi <= kPi || fTorus.fDphi == kTwoPi) fGlobalConvexity = true;
    }
  }
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedTorus2::Create(LogicalVolume const *const logical_volume,
                                      Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                      const int id,
#endif
                                      VPlacedVolume *const placement)
{
  (void)placement;
  return new SimpleTorus2(logical_volume, transformation
#ifdef VECCORE_CUDA
                          ,
                          id
#endif
                          );
}

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedTorus2::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<UnplacedTorus2>(in_gpu_ptr, fTorus.fRmin, fTorus.fRmax, fTorus.fRtor, fTorus.fSphi,
                                       fTorus.fDphi);
}

DevicePtr<cuda::VUnplacedVolume> UnplacedTorus2::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedTorus2>();
}

#endif // VECGEOM_CUDA_INTERFACE

// Return unit normal of surface closest to p
// - note if point on z axis, ignore phi divided sides
// - unsafe if point close to z axis a rmin=0 - no explicit checks
bool UnplacedTorus2::Normal(Vector3D<Precision> const &point, Vector3D<Precision> &norm) const
{

  int noSurfaces = 0;
  bool valid     = true;

  Precision rho2, rho, pt2, pt, pPhi;
  Precision distRMin = kInfLength;
  Precision distSPhi = kInfLength, distEPhi = kInfLength;

  // To cope with precision loss
  //
  const Precision delta  = Max(10.0 * kTolerance, 1.0e-8 * (fTorus.fRtor + fTorus.fRmax));
  const Precision dAngle = 10.0 * kTolerance;

  Vector3D<Precision> nR, nPs, nPe;
  Vector3D<Precision> sumnorm(0., 0., 0.);

  rho2 = point.x() * point.x() + point.y() * point.y();
  rho  = Sqrt(rho2);
  pt2  = rho2 + point.z() * point.z() + fTorus.fRtor * (fTorus.fRtor - 2 * rho);
  pt2  = Max(pt2, 0.0); // std::fabs(pt2);
  pt   = Sqrt(pt2);

  Precision distRMax         = Abs(pt - fTorus.fRmax);
  if (fTorus.fRmin) distRMin = Abs(pt - fTorus.fRmin);

  if (rho > delta && pt != 0.0) {
    Precision redFactor = (rho - fTorus.fRtor) / rho;
    nR                  = Vector3D<Precision>(point.x() * redFactor, // p.x()*(1.-fRtor/rho),
                             point.y() * redFactor,                  // p.y()*(1.-fRtor/rho),
                             point.z());
    nR *= 1.0 / pt;
  }

  if (fTorus.fDphi < kTwoPi) // && rho ) // old limitation against (0,0,z)
  {
    if (rho) {
      pPhi = std::atan2(point.y(), point.x());

      if (pPhi < fTorus.fSphi - delta) {
        pPhi += kTwoPi;
      } else if (pPhi > fTorus.fSphi + fTorus.fDphi + delta) {
        pPhi -= kTwoPi;
      }

      distSPhi = Abs(pPhi - fTorus.fSphi);
      distEPhi = Abs(pPhi - fTorus.fSphi - fTorus.fDphi);
    }
    nPs = Vector3D<Precision>(sin(fTorus.fSphi), -cos(fTorus.fSphi), 0);
    nPe = Vector3D<Precision>(-sin(fTorus.fSphi + fTorus.fDphi), cos(fTorus.fSphi + fTorus.fDphi), 0);
  }
  if (distRMax <= delta) {
    noSurfaces++;
    sumnorm += nR;
  } else if (fTorus.fRmin && (distRMin <= delta)) // Must not be on both Outer and Inner
  {
    noSurfaces++;
    sumnorm -= nR;
  }

  //  To be on one of the 'phi' surfaces,
  //  it must be within the 'tube' - with tolerance

  if ((fTorus.fDphi < kTwoPi) && (fTorus.fRmin - delta <= pt) && (pt <= (fTorus.fRmax + delta))) {
    if (distSPhi <= dAngle) {
      noSurfaces++;
      sumnorm += nPs;
    }
    if (distEPhi <= dAngle) {
      noSurfaces++;
      sumnorm += nPe;
    }
  }
  if (noSurfaces == 0) {

    valid = false;
  } else if (noSurfaces == 1) {
    norm = sumnorm;
  } else {
    norm = sumnorm.Unit();
  }

  return valid;
}

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedTorus2>::SizeOf();
template void DevicePtr<cuda::UnplacedTorus2>::Construct(const Precision rmin, const Precision rmax,
                                                         const Precision rtor, const Precision sphi,
                                                         const Precision dphi) const;

} // namespace cxx

#endif

} // namespace vecgeom
