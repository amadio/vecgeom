/// \file UnplacedTorus2.cpp

#include "VecGeom/volumes/UnplacedTorus2.h"
#include "VecGeom/volumes/SpecializedTorus2.h"

#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/management/VolumeFactory.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
void UnplacedTorus2::Print() const
{
  printf("UnplacedTorus2 {%.2f, %.2f, %.2f, %.2f, %.2f}", rmin(), rmax(), rtor(), sphi(), dphi());
}

void UnplacedTorus2::Print(std::ostream &os) const
{
  os << "UnplacedTorus2 {" << rmin() << ", " << rmax() << ", " << rtor() << ", " << sphi() << ", " << dphi();
}

#ifndef VECCORE_CUDA
SolidMesh *UnplacedTorus2::CreateMesh3D(Transformation3D const &trans, size_t nSegments) const
{
  typedef Vector3D<Precision> Vec_t;
  bool isFull   = dphi() == 2 * kPi;
  SolidMesh *sm = new SolidMesh();

  sm->ResetMesh(2 * (nSegments + 1) * (nSegments + 1), (nSegments * nSegments) + 2);
  Vec_t *vertices = new Vec_t[2 * (nSegments + 1) * (nSegments + 1)];

  Precision phi_step   = dphi() / nSegments;
  Precision theta_step = 2 * M_PI / nSegments;
  Precision phi        = sphi();
  Precision theta      = 0.;
  Precision intermediate1, intermediate2;

  for (size_t p = 0, i = 0; i <= nSegments; ++i, theta += theta_step, phi = sphi()) {
    intermediate1 = (rtor() + rmax() * std::cos(theta));
    intermediate2 = (rtor() + rmin() * std::cos(theta));
    for (size_t j = 0; j <= nSegments; ++j, p++, phi += phi_step) {
      vertices[p] = Vec_t(intermediate1 * std::cos(phi), intermediate1 * std::sin(phi), rmax() * std::sin(theta));
      vertices[p + (nSegments + 1) * (nSegments + 1)] =
          Vec_t(intermediate2 * std::cos(phi), intermediate2 * std::sin(phi), rmin() * std::sin(theta));
    }
  }

  sm->SetVertices(vertices, 2 * (nSegments + 1) * (nSegments + 1));
  delete[] vertices;
  sm->TransformVertices(trans);

  // outer surface
  for (size_t j = 0, k = 0; j < nSegments; j++, k++) {
    for (size_t i = 0, l = k + nSegments + 1; i < nSegments; i++, k++, l++) {
      sm->AddPolygon(4, {l + 1, l, k, k + 1}, true);
    }
  }
  // inner surface
  for (size_t j = 0, k = (nSegments + 1) * (nSegments + 1); j < nSegments; j++, k++) {
    for (size_t i = 0, l = k + nSegments + 1; i < nSegments; i++, k++, l++) {
      sm->AddPolygon(4, {k + 1, k, l, l + 1}, true);
    }
  }

  // surfaces due to torus not being full
  if (!isFull) {

    for (size_t i = 0, j = 0; i < nSegments; i++, j += nSegments + 1) {
      sm->AddPolygon(4,
                     {j, j + nSegments + 1, j + nSegments + 1 + (nSegments + 1) * (nSegments + 1),
                      j + (nSegments + 1) * (nSegments + 1)},
                     true);
    }

    for (size_t i = 0, j = nSegments; i < nSegments; i++, j += nSegments + 1) {
      sm->AddPolygon(4,
                     {j + (nSegments + 1) * (nSegments + 1), j + (nSegments + 1) * (nSegments + 1) + nSegments + 1,
                      j + nSegments + 1, j},
                     true);
    }
  }

  return sm;
}
#endif

#ifndef VECCORE_CUDA
VPlacedVolume *UnplacedTorus2::SpecializedVolume(LogicalVolume const *const volume,
                                                 Transformation3D const *const transformation,
                                                 VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedTorus2>(volume, transformation, placement);
}
#else
__device__ VPlacedVolume *UnplacedTorus2::SpecializedVolume(LogicalVolume const *const volume,
                                                            Transformation3D const *const transformation, const int id,
                                                            const int copy_no, const int child_id,
                                                            VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedTorus2>(volume, transformation, id, copy_no, child_id,
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

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedTorus2::Create(LogicalVolume const *const logical_volume,
                                      Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                      const int id, const int copy_no, const int child_id,
#endif
                                      VPlacedVolume *const placement)
{
  (void)placement;
  return new SimpleTorus2(logical_volume, transformation
#ifdef VECCORE_CUDA
                          ,
                          id, copy_no, child_id
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
VECCORE_ATT_HOST_DEVICE
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
  pt2  = Max(pt2, Precision(0.)); // std::fabs(pt2);
  pt   = Sqrt(pt2);

  Precision distRMax = Abs(pt - fTorus.fRmax);
  if (fTorus.fRmin) distRMin = Abs(pt - fTorus.fRmin);

  if (rho > delta && pt != Precision(0.)) {
    Precision redFactor = (rho - fTorus.fRtor) / rho;
    nR                  = Vector3D<Precision>(point.x() * redFactor, // p.x()*(1.-fRtor/rho),
                             point.y() * redFactor, // p.y()*(1.-fRtor/rho),
                             point.z());
    nR *= Precision(1.) / pt;
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
