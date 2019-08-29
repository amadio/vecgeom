// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// @file source/UnplacedEllipticalTube.cpp
/// @author Raman Sehgal, Evgueni Tcherniaev

#include "volumes/EllipticUtilities.h"
#include "volumes/UnplacedEllipticalTube.h"
#include "management/VolumeFactory.h"
#include "volumes/SpecializedEllipticalTube.h"
#include "base/RNG.h"
#include <stdio.h>
#include <cmath>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
UnplacedEllipticalTube::UnplacedEllipticalTube(Precision dx, Precision dy, Precision dz)
{
  SetParameters(dx, dy, dz);
  fGlobalConvexity = true;
}

VECCORE_ATT_HOST_DEVICE
void UnplacedEllipticalTube::CheckParameters()
{
  Precision tol = 2. * kTolerance;
  if (fEllipticalTube.fDx < tol || fEllipticalTube.fDy < tol || fEllipticalTube.fDy < tol) {
#ifndef VECCORE_CUDA
    std::cerr << "Wrong parameteres EllipticalTube { " << fEllipticalTube.fDx << ", " << fEllipticalTube.fDy << ", "
              << fEllipticalTube.fDz << " }" << std::endl;
#endif
    fEllipticalTube.fDx = fEllipticalTube.fDy = fEllipticalTube.fDz = tol;
  }
  Precision X = fEllipticalTube.fDx;
  Precision Y = fEllipticalTube.fDy;
  Precision Z = fEllipticalTube.fDz;
  Precision R = vecCore::math::Min(X, Y);

  fEllipticalTube.fSurfaceArea = 2. * (kPi * X * Y + vecgeom::EllipticUtilities::EllipsePerimeter(X, Y) * Z);
  fEllipticalTube.fCubicVolume = 2. * kPi * X * Y * Z;

  // Precalculated values
  fEllipticalTube.fRsph = vecCore::math::Sqrt(X * X + Y * Y + Z * Z);
  fEllipticalTube.fDDx  = X * X; // Dx squared
  fEllipticalTube.fDDy  = Y * Y; // Dy squared
  fEllipticalTube.fSx   = R / X; // X scale factor
  fEllipticalTube.fSy   = R / Y; // Y scale factor
  fEllipticalTube.fR    = R;     // resulting Radius, after scaling elipse to circle

  // Coefficient for approximation of distance : Q1 * (x^2 + y^2) - Q2
  fEllipticalTube.fQ1 = 0.5 / R;
  fEllipticalTube.fQ2 = 0.5 * (R + kHalfTolerance * kHalfTolerance / R);

  // Half length of scratching segment squared
  fEllipticalTube.fScratch = 2. * R * R * kEpsilon; // scratch within calculation error thickness
  // fEllipticalTube.fScratch = (B * B / A) * (2. + kHalfTolerance / A) * kHalfTolerance;
};

VECCORE_ATT_HOST_DEVICE
void UnplacedEllipticalTube::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  Precision dx = GetDx();
  Precision dy = GetDy();
  Precision dz = GetDz();
  aMin.Set(-dx, -dy, -dz);
  aMax.Set(dx, dy, dz);
}

Vector3D<Precision> UnplacedEllipticalTube::SamplePointOnSurface() const
{
  Precision dx = GetDx();
  Precision dy = GetDy();
  Precision dz = GetDz();

  // Select surface (0 - base at -Z, 1 - base at +Z, 2 - lateral surface)
  //
  Precision sbase  = kPi * dx * dy;
  Precision select = SurfaceArea() * RNG::Instance().uniform();

  int k = 0;
  if (select > sbase) k = 1;
  if (select > 2. * sbase) k = 2;

  // Pick random point on selected surface (rejection sampling)
  //
  Vector3D<Precision> p;
  switch (k) {
  case 0: // base at -Z
  {
    Vector2D<Precision> rho = EllipticUtilities::RandomPointInEllipse(dx, dy);
    p.Set(rho.x(), rho.y(), -dz);
    break;
  }
  case 1: // base at +Z
  {
    Vector2D<Precision> rho = EllipticUtilities::RandomPointInEllipse(dx, dy);
    p.Set(rho.x(), rho.y(), dz);
    break;
  }
  case 2: // lateral surface
  {
    Vector2D<Precision> rho = EllipticUtilities::RandomPointOnEllipse(dx, dy);
    p.Set(rho.x(), rho.y(), (2. * RNG::Instance().uniform() - 1.) * dz);
    break;
  }
  }
  return p;
}

// VECCORE_ATT_HOST_DEVICE
std::ostream &UnplacedEllipticalTube::StreamInfo(std::ostream &os) const
// Definition taken from UEllipticalTube
{
  int oldprc = os.precision(16);
  os << "-----------------------------------------------------------\n"
     //  << "     *** Dump for solid - " << GetName() << " ***\n"
     //  << "     ===================================================\n"

     << " Solid type: EllipticalTube\n"
     << " Parameters: \n"

     << "-----------------------------------------------------------\n";
  os.precision(oldprc);

  return os;
}

void UnplacedEllipticalTube::Print() const
{
  printf("EllipticalTube {%.2f, %.2f, %.2f}", fEllipticalTube.fDx, fEllipticalTube.fDy, fEllipticalTube.fDz);
}

void UnplacedEllipticalTube::Print(std::ostream &os) const
{
  os << "EllipticalTube {" << fEllipticalTube.fDx << ", " << fEllipticalTube.fDy << ", " << fEllipticalTube.fDz << "}";
}

#ifndef VECCORE_CUDA
SolidMesh *UnplacedEllipticalTube::CreateMesh3D(Transformation3D const &trans, size_t nSegments) const
{

  double a = GetDx();
  double b = GetDy();
  double c = GetDz();

  SolidMesh *sm = new SolidMesh();
  sm->ResetMesh(2*(nSegments + 1), nSegments + 2);

  typedef Vector3D<double> Vec_t;
  Vec_t *const vertices = new Vec_t[2*(nSegments + 1)];
  double acos, bsin;
  for (size_t i = 0; i <= nSegments; i++) {
	  acos = a*std::cos(i * 2 * M_PI / nSegments);
	  bsin = b*std::sin(i * 2 * M_PI / nSegments);
    vertices[i]     = Vec_t(acos, bsin, -c); // lower vertices
    vertices[i + nSegments + 1] = Vec_t(acos, bsin, c); // upper vertices
  }
  sm->SetVertices(vertices, 2*(nSegments + 1));
  delete[] vertices;

  sm->TransformVertices(trans);

  Utils3D::vector_t<size_t> indices;
  indices.reserve(nSegments);
  // upper surface
   for (size_t i = 0; i < nSegments; i++) {
     indices.push_back(i+ nSegments + 1);
   }

   sm->AddPolygon(nSegments, indices, true);
   indices.clear();

   // lower surface
   for (size_t i = nSegments; i > 0; i--) {
     indices.push_back(i - 1);
   }
   sm->AddPolygon(nSegments, indices, true);

   // lateral surfaces
   for (size_t i = 0; i < nSegments; i++) {
     sm->AddPolygon(4, {i, i + 1, i + 1 + nSegments + 1, i + nSegments + 1}, true);
   }

  return sm;
}
#endif

#ifndef VECCORE_CUDA
template <TranslationCode trans_code, RotationCode rot_code>
VPlacedVolume *UnplacedEllipticalTube::Create(LogicalVolume const *const logical_volume,
                                              Transformation3D const *const transformation,
                                              VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedEllipticalTube<trans_code, rot_code>(logical_volume, transformation);
    return placement;
  }
  return new SpecializedEllipticalTube<trans_code, rot_code>(logical_volume, transformation);
}

VPlacedVolume *UnplacedEllipticalTube::SpecializedVolume(LogicalVolume const *const volume,
                                                         Transformation3D const *const transformation,
                                                         const TranslationCode trans_code, const RotationCode rot_code,
                                                         VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedEllipticalTube>(volume, transformation, trans_code, rot_code,
                                                                       placement);
}
#else

template <TranslationCode trans_code, RotationCode rot_code>
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedEllipticalTube::Create(LogicalVolume const *const logical_volume,
                                              Transformation3D const *const transformation, const int id,
                                              VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedEllipticalTube<trans_code, rot_code>(logical_volume, transformation, id);
    return placement;
  }
  return new SpecializedEllipticalTube<trans_code, rot_code>(logical_volume, transformation, id);
}

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedEllipticalTube::SpecializedVolume(LogicalVolume const *const volume,
                                                         Transformation3D const *const transformation,
                                                         const TranslationCode trans_code, const RotationCode rot_code,
                                                         const int id, VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedEllipticalTube>(volume, transformation, trans_code, rot_code, id,
                                                                       placement);
}

#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedEllipticalTube::CopyToGpu(
    DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<UnplacedEllipticalTube>(in_gpu_ptr, GetDx(), GetDy(), GetDz());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedEllipticalTube::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedEllipticalTube>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedEllipticalTube>::SizeOf();
template void DevicePtr<cuda::UnplacedEllipticalTube>::Construct(const Precision dx, const Precision dy,
                                                                 const Precision dz) const;

} // namespace cxx

#endif
} // namespace vecgeom
