/*
 * UnplacedCutTube.cpp
 *
 *  Created on: 03.11.2016
 *      Author: mgheata
 */
#include "volumes/UnplacedCutTube.h"
#include "volumes/SpecializedCutTube.h"

#ifndef VECGEOM_NVCC
#include "base/RNG.h"
//#include <cmath>
#include <iostream>
#endif

#include "volumes/utilities/GenerationUtilities.h"
#include "management/VolumeFactory.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

void UnplacedCutTube::Print() const
{
  printf("UnplacedCutTube {rmin=%.2f, rmax=%.2f, z=%.2f, sphi=%.2f, dphi=%.2f bottom=(%f, %f, %f)  top=(%f, %f, %f)}",
         rmin(), rmax(), z(), sphi(), dphi(), BottomNormal().x(), BottomNormal().y(), BottomNormal().z(),
         TopNormal().x(), TopNormal().y(), TopNormal().z());
}

void UnplacedCutTube::Print(std::ostream &os) const
{
  os << "UnplacedCutTube { rmin=" << rmin() << ", rmax=" << rmax() << ", z=" << z() << ", sphi=" << sphi()
     << ", dphi=" << dphi() << ", bottom=" << BottomNormal() << ", top=" << TopNormal() << "}\n";
}

Precision UnplacedCutTube::volume() const
{
  constexpr Precision onethird = 1. / 3.;
  Precision volNocut           = z() * dphi() * (rmax() * rmax() - rmin() * rmin());
  Precision vDelta             = onethird * (rmax() * rmax() * rmax() - rmin() * rmin() * rmin()) *
                     (((TopNormal().x() / TopNormal().z() - BottomNormal().x() / BottomNormal().z()) *
                       (fCutTube.fSinPhi2 - fCutTube.fSinPhi1)) -
                      ((TopNormal().y() / TopNormal().z() - BottomNormal().y() / BottomNormal().z()) *
                       (fCutTube.fCosPhi2 - fCutTube.fCosPhi1)));
  return (volNocut - vDelta);
}

Vector3D<Precision> UnplacedCutTube::GetPointOnSurface() const
{
  Precision xVal = 0., yVal = 0., zVal = 0.;
#ifndef VECGEOM_NVCC
  Precision area[6];
  Precision atotal = 0.;

  area[0] = GetBottomArea();
  area[1] = GetTopArea();
  area[2] = GetLateralArea(rmax());
  area[3] = GetLateralArea(rmin());
  area[4] = GetLateralPhi1Area();
  area[5] = GetLateralPhi2Area();

  for (int i = 0; i < 6; ++i)
    atotal += area[i];

  RNG &rng = RNG::Instance();

  /* random value to choose surface to place the point */
  Precision rand = rng.uniform() * atotal;

  int surface = 0;
  while (rand > area[surface])
    rand -= area[surface], surface++;
  // assert (surface < 6);

  Precision rVal, phiVal, zmin, zmax;
  switch (surface) {
  case 0: // bottom
    rVal   = rng.uniform(rmin(), rmax());
    phiVal = rng.uniform(sphi(), sphi() + dphi());
    zVal   = ZlimitBottom(rVal, phiVal);
    break;
  case 1: // top
    rVal   = rng.uniform(rmin(), rmax());
    phiVal = rng.uniform(sphi(), sphi() + dphi());
    zVal   = ZlimitTop(rVal, phiVal);
    break;
  case 2: // outer
    rVal   = rmax();
    phiVal = rng.uniform(sphi(), sphi() + dphi());
    zmin   = ZlimitBottom(rVal, phiVal);
    zmax   = ZlimitTop(rVal, phiVal);
    // assert(zmax-zmin > 0);
    zVal = zmin + (zmax - zmin) * rng.uniform();
    break;
  case 3: // inner
    rVal   = rmin();
    phiVal = rng.uniform(sphi(), sphi() + dphi());
    zmin   = ZlimitBottom(rVal, phiVal);
    zmax   = ZlimitTop(rVal, phiVal);
    // assert(zmax-zmin > 0);
    zVal = rng.uniform(zmin, zmax);
    break;
  case 4: // phi
    rVal   = rng.uniform(rmin(), rmax());
    phiVal = sphi();
    zmin   = ZlimitBottom(rVal, phiVal);
    zmax   = ZlimitTop(rVal, phiVal);
    // assert(zmax-zmin > 0);
    zVal = rng.uniform(zmin, zmax);
    break;
  case 5: // phi + dphi
    rVal   = rng.uniform(rmin(), rmax());
    phiVal = sphi() + dphi();
    zmin   = ZlimitBottom(rVal, phiVal);
    zmax   = ZlimitTop(rVal, phiVal);
    // assert(zmax-zmin > 0);
    zVal = rng.uniform(zmin, zmax);
    break;
  }
  xVal = rVal * vecCore::math::Cos(phiVal);
  yVal = rVal * vecCore::math::Sin(phiVal);
#endif

  return Vector3D<Precision>(xVal, yVal, zVal);
}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedCutTube::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  Precision dztop    = rmax() * Sqrt(1. - TopNormal().z() * TopNormal().z()) / TopNormal().z();
  Precision dzbottom = -rmax() * vecCore::math::Sqrt(1. - BottomNormal().z() * BottomNormal().z()) / BottomNormal().z();
  aMin               = Vector3D<Precision>(-rmax(), -rmax(), -z() - dzbottom);
  aMax               = Vector3D<Precision>(rmax(), rmax(), z() + dztop);

  if (dphi() == kTwoPi) return;

  // The phi cut can reduce the extent in Z
  bool topReduce =
      !fCutTube.fTubeStruct.fPhiWedge.Contains(Vector3D<Precision>(-TopNormal().x(), -TopNormal().y(), 0.));
  if (topReduce) {
    aMax.z() = vecCore::math::Max(ZlimitTop(rmax(), sphi()), ZlimitTop(rmax(), sphi() + dphi()));
    aMax.z() = vecCore::math::Max(aMax.z(), ZlimitTop(rmin(), sphi()), ZlimitTop(rmin(), sphi() + dphi()));
  }
  bool bottomReduce =
      !fCutTube.fTubeStruct.fPhiWedge.Contains(Vector3D<Precision>(-BottomNormal().x(), -BottomNormal().y(), 0.));
  if (bottomReduce) {
    aMin.z() = vecCore::math::Min(ZlimitBottom(rmax(), sphi()), ZlimitBottom(rmax(), sphi() + dphi()));
    aMin.z() = vecCore::math::Min(aMin.z(), ZlimitBottom(rmin(), sphi()), ZlimitBottom(rmin(), sphi() + dphi()));
  }
  // The phi cut can also reduce the extent in x,y
  bool xPlusReduce = !fCutTube.fTubeStruct.fPhiWedge.Contains(Vector3D<Precision>(1., 0., 0.));
  if (xPlusReduce) {
    aMax.x() = vecCore::math::Max(rmax() * fCutTube.fCosPhi1, rmax() * fCutTube.fCosPhi2);
    aMax.x() = vecCore::math::Max(aMax.x(), rmin() * fCutTube.fCosPhi1, rmin() * fCutTube.fCosPhi2);
  }
  bool xMinusReduce = !fCutTube.fTubeStruct.fPhiWedge.Contains(Vector3D<Precision>(-1., 0., 0.));
  if (xMinusReduce) {
    aMin.x() = vecCore::math::Min(rmax() * fCutTube.fCosPhi1, rmax() * fCutTube.fCosPhi2);
    aMin.x() = vecCore::math::Min(aMin.x(), rmin() * fCutTube.fCosPhi1, rmin() * fCutTube.fCosPhi2);
  }

  bool yPlusReduce = !fCutTube.fTubeStruct.fPhiWedge.Contains(Vector3D<Precision>(0., 1., 0.));
  if (yPlusReduce) {
    aMax.y() = vecCore::math::Max(rmax() * fCutTube.fSinPhi1, rmax() * fCutTube.fSinPhi2);
    aMax.y() = vecCore::math::Max(aMax.y(), rmin() * fCutTube.fSinPhi1, rmin() * fCutTube.fSinPhi2);
  }
  bool yMinusReduce = !fCutTube.fTubeStruct.fPhiWedge.Contains(Vector3D<Precision>(0., -1., 0.));
  if (yMinusReduce) {
    aMin.y() = vecCore::math::Min(rmax() * fCutTube.fSinPhi1, rmax() * fCutTube.fSinPhi2);
    aMin.y() = vecCore::math::Min(aMin.y(), rmin() * fCutTube.fSinPhi1, rmin() * fCutTube.fSinPhi2);
  }
}

VECGEOM_CUDA_HEADER_BOTH
void UnplacedCutTube::DetectConvexity()
{
  // Default safe convexity value
  fGlobalConvexity = false;

  // Logic to calculate the convexity
  if (rmin() == 0.) {
    if (dphi() <= kPi || dphi() == kTwoPi) fGlobalConvexity = true;
  }
}

VECGEOM_CUDA_HEADER_BOTH
bool UnplacedCutTube::Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const
{
  bool valid;
  CutTubeImplementation::NormalKernel<double>(fCutTube, point, normal, valid);
  return valid;
}

template <TranslationCode trans_code, RotationCode rot_code>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume *UnplacedCutTube::Create(LogicalVolume const *const logical_volume,
                                       Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
                                       const int id,
#endif
                                       VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedCutTube<trans_code, rot_code>(logical_volume, transformation
#ifdef VECGEOM_NVCC
                                                             ,
                                                             id
#endif
                                                             );
    return placement;
  }
  return new SpecializedCutTube<trans_code, rot_code>(logical_volume, transformation
#ifdef VECGEOM_NVCC
                                                      ,
                                                      id
#endif
                                                      );
}

VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume *UnplacedCutTube::SpecializedVolume(LogicalVolume const *const volume,
                                                  Transformation3D const *const transformation,
                                                  const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
                                                  const int id,
#endif
                                                  VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedCutTube>(volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
                                                                id,
#endif
                                                                placement);
}

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedCutTube::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<UnplacedCutTube>(in_gpu_ptr, rmin(), rmax(), z(), sphi(), dphi(), BottomNormal().x(),
                                        BottomNormal().y(), BottomNormal().z(), TopNormal().x(), TopNormal().y(),
                                        TopNormal().z());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedCutTube::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedCutTube>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedCutTube>::SizeOf();
template void DevicePtr<cuda::UnplacedCutTube>::Construct(const Precision rmin, const Precision rmax, const Precision z,
                                                          const Precision sphi, const Precision dphi,
                                                          const Precision bx, const Precision by, const Precision bz,
                                                          const Precision tx, const Precision ty,
                                                          const Precision tz) const;

} // End cxx namespace

#endif

} // End global namespace
