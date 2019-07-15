// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// @file source/UnplacedParallelepiped.cpp
/// @author Created by Johannes de Fine Licht
/// @author Modified and completed by Mihaela Gheata, Evgueni Tcherniaev

#include "volumes/UnplacedParallelepiped.h"

#include <stdio.h>
#include "management/VolumeFactory.h"
#include "volumes/SpecializedParallelepiped.h"
#include "volumes/utilities/GenerationUtilities.h"
#include "base/RNG.h"


namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA
SolidMesh *UnplacedParallelepiped::CreateMesh3D(Transformation3D const &trans, const size_t nFaces) const
{
  SolidMesh *sm = new SolidMesh();
  sm->ResetMesh(8, 6);

  double dx = fPara.fDimensions[0] * 2;
  double dy = fPara.fDimensions[1] * 2;
  double dz = fPara.fDimensions[2] * 2;

  double gamma = fPara.fPhi;
  double alpha = fPara.fTheta;
  double beta  = fPara.fAlpha;

  double intermediate = (std::cos(alpha) - std::cos(beta) * std::cos(gamma)) / std::sin(gamma);

  Vector3D<double> a = Vector3D<double>(dx, 0, 0);
  Vector3D<double> b = Vector3D<double>(dy * std::cos(gamma), dy * std::sin(gamma), 0);
  Vector3D<double> c = Vector3D<double>(dz * std::cos(beta), dz * intermediate,
                                        dz * std::sqrt(1 - std::cos(beta) * std::cos(beta) - intermediate * intermediate ));



  Utils3D::Vec_t vertices[] = {a, a + b, a + b + c, a + c, Vector3D<double>(),
                                     b, b + c, c};

  // subtract to move the origin to center
  double origin = (a + b + c) / 2;
  for(auto & vertex: vertices)
	  vertex -= origin;


  sm->SetVertices(vertices, 8);
  sm->TransformVertices(trans);
  sm->InitConvexHexahedron();
  return sm;
}
#endif

//______________________________________________________________________________
void UnplacedParallelepiped::Print() const
{
  printf("UnplacedParallelepiped {%.2f, %.2f, %.2f, %.2f, %.2f, %.2f}", GetX(), GetY(), GetZ(), GetTanAlpha(),
         GetTanThetaCosPhi(), GetTanThetaSinPhi());
}

//______________________________________________________________________________
void UnplacedParallelepiped::Print(std::ostream &os) const
{
  os << "UnplacedParallelepiped {" << GetX() << ", " << GetY() << ", " << GetZ() << ", " << GetTanAlpha() << ", "
     << GetTanThetaCosPhi() << ", " << GetTanThetaSinPhi();
}

//______________________________________________________________________________
VECCORE_ATT_HOST_DEVICE
void UnplacedParallelepiped::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  // Returns the full 3D cartesian extent of the solid.
  Precision dx = fPara.fDimensions[0] + fPara.fDimensions[1] * Abs(fPara.fTanAlpha) +
                 fPara.fDimensions[2] * Abs(fPara.fTanThetaCosPhi);
  Precision dy = fPara.fDimensions[1] + fPara.fDimensions[2] * Abs(fPara.fTanThetaSinPhi);
  Precision dz = fPara.fDimensions[2];
  aMin.Set(-dx, -dy, -dz);
  aMax.Set(dx, dy, dz);
}

//______________________________________________________________________________
Vector3D<Precision> UnplacedParallelepiped::SamplePointOnSurface() const
{
  // Set array for selection of facet
  Precision surf[6];
  surf[0] = fPara.fAreas[0];
  surf[1] = surf[0] + fPara.fAreas[0];
  surf[2] = surf[1] + fPara.fAreas[1];
  surf[3] = surf[2] + fPara.fAreas[1];
  surf[4] = surf[3] + fPara.fAreas[2];
  surf[5] = surf[4] + fPara.fAreas[2];

  // Select facet
  Precision select = surf[5] * RNG::Instance().uniform();
  int isurf        = 5;
  if (select <= surf[4]) isurf = 4;
  if (select <= surf[3]) isurf = 3;
  if (select <= surf[2]) isurf = 2;
  if (select <= surf[1]) isurf = 1;
  if (select <= surf[0]) isurf = 0;

  // Set working variables
  Precision Dx               = fPara.fDimensions[0];
  Precision Dy               = fPara.fDimensions[1];
  Precision Dz               = fPara.fDimensions[2];
  Precision DyTanAlpha       = Dy * fPara.fTanAlpha;
  Precision DzTanThetaCosPhi = Dz * fPara.fTanThetaCosPhi;
  Precision DzTanThetaSinPhi = Dz * fPara.fTanThetaSinPhi;

  // Sample random point
  Vector3D<Precision> p0, p1, p2;
  switch (isurf) {
  case 0: // -x
    p0.Set(-DzTanThetaCosPhi - DyTanAlpha - Dx, -DzTanThetaSinPhi - Dy, -Dz);
    p1.Set(-DzTanThetaCosPhi + DyTanAlpha - Dx, -DzTanThetaSinPhi + Dy, -Dz);
    p2.Set(DzTanThetaCosPhi - DyTanAlpha - Dx, DzTanThetaSinPhi - Dy, Dz);
    break;
  case 1: // +x
    p0.Set(-DzTanThetaCosPhi - DyTanAlpha + Dx, -DzTanThetaSinPhi - Dy, -Dz);
    p1.Set(DzTanThetaCosPhi - DyTanAlpha + Dx, DzTanThetaSinPhi - Dy, Dz);
    p2.Set(-DzTanThetaCosPhi + DyTanAlpha + Dx, -DzTanThetaSinPhi + Dy, -Dz);
    break;
  case 2: // -y
    p0.Set(-DzTanThetaCosPhi - DyTanAlpha - Dx, -DzTanThetaSinPhi - Dy, -Dz);
    p1.Set(DzTanThetaCosPhi - DyTanAlpha - Dx, DzTanThetaSinPhi - Dy, Dz);
    p2.Set(-DzTanThetaCosPhi - DyTanAlpha + Dx, -DzTanThetaSinPhi - Dy, -Dz);
    break;
  case 3: // +y
    p0.Set(-DzTanThetaCosPhi + DyTanAlpha - Dx, -DzTanThetaSinPhi + Dy, -Dz);
    p1.Set(-DzTanThetaCosPhi + DyTanAlpha + Dx, -DzTanThetaSinPhi + Dy, -Dz);
    p2.Set(DzTanThetaCosPhi + DyTanAlpha - Dx, DzTanThetaSinPhi + Dy, Dz);
    break;
  case 4: // -z
    p0.Set(-DzTanThetaCosPhi - DyTanAlpha - Dx, -DzTanThetaSinPhi - Dy, -Dz);
    p1.Set(-DzTanThetaCosPhi - DyTanAlpha + Dx, -DzTanThetaSinPhi - Dy, -Dz);
    p2.Set(-DzTanThetaCosPhi + DyTanAlpha - Dx, -DzTanThetaSinPhi + Dy, -Dz);
    break;
  case 5: // +z
    p0.Set(DzTanThetaCosPhi - DyTanAlpha - Dx, DzTanThetaSinPhi - Dy, Dz);
    p1.Set(DzTanThetaCosPhi + DyTanAlpha - Dx, DzTanThetaSinPhi + Dy, Dz);
    p2.Set(DzTanThetaCosPhi - DyTanAlpha + Dx, DzTanThetaSinPhi - Dy, Dz);
    break;
  }
  Precision u = RNG::Instance().uniform();
  Precision v = RNG::Instance().uniform();
  return p0 + u * (p1 - p0) + v * (p2 - p0);
}

//______________________________________________________________________________
template <TranslationCode transCodeT, RotationCode rotCodeT>
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedParallelepiped::Create(LogicalVolume const *const logical_volume,
                                              Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                              const int id,
#endif
                                              VPlacedVolume *const placement)
{

  return CreateSpecializedWithPlacement<SpecializedParallelepiped<transCodeT, rotCodeT>>(
#ifdef VECCORE_CUDA
      logical_volume, transformation, id, placement); // TODO: add bounding box?
#else
      logical_volume, transformation, placement);
#endif
}

//______________________________________________________________________________
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedParallelepiped::SpecializedVolume(LogicalVolume const *const volume,
                                                         Transformation3D const *const transformation,
                                                         const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECCORE_CUDA
                                                         const int id,
#endif
                                                         VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedParallelepiped>(volume, transformation, trans_code, rot_code,
#ifdef VECCORE_CUDA
                                                                       id,
#endif
                                                                       placement);
}

#ifdef VECGEOM_CUDA_INTERFACE

//______________________________________________________________________________
DevicePtr<cuda::VUnplacedVolume> UnplacedParallelepiped::CopyToGpu(
    DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<UnplacedParallelepiped>(in_gpu_ptr, GetX(), GetY(), GetZ(), GetAlpha(), GetTheta(), GetPhi());
}

//______________________________________________________________________________
DevicePtr<cuda::VUnplacedVolume> UnplacedParallelepiped::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedParallelepiped>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedParallelepiped>::SizeOf();
template void DevicePtr<cuda::UnplacedParallelepiped>::Construct(const Precision x, const Precision y,
                                                                 const Precision z, const Precision alpha,
                                                                 const Precision theta, const Precision phi) const;

} // namespace cxx

#endif

} // namespace vecgeom
