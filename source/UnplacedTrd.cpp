/// @file UnplacedTrd.cpp
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#include "volumes/UnplacedTrd.h"
#include "volumes/SpecializedTrd.h"
#include "volumes/utilities/GenerationUtilities.h"
#include "base/RNG.h"

#include "management/VolumeFactory.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

void UnplacedTrd::Print() const
{
  printf("UnplacedTrd {%.2f, %.2f, %.2f, %.2f, %.2f}", dx1(), dx2(), dy1(), dy2(), dz());
}

void UnplacedTrd::Print(std::ostream &os) const
{
  os << "UnplacedTrd {" << dx1() << ", " << dx2() << ", " << dy1() << ", " << dy2() << ", " << dz();
}

#ifndef VECCORE_CUDA
Precision UnplacedTrd::Capacity() const
{
  return 2 * (fTrd.fDX1 + fTrd.fDX2) * (fTrd.fDY1 + fTrd.fDY2) * fTrd.fDZ +
         (2. / 3.) * (fTrd.fDX1 - fTrd.fDX2) * (fTrd.fDY1 - fTrd.fDY2) * fTrd.fDZ;
}

Precision UnplacedTrd::SurfaceArea() const
{
  Precision dz = 2 * fTrd.fDZ;
  bool xvert   = (fTrd.fDX1 == fTrd.fDX2) ? true : false;
  Precision SA = 0.0;

  // Sum of area for planes Perp. to +X and -X
  Precision ht = (xvert) ? dz : Sqrt((fTrd.fDX1 - fTrd.fDX2) * (fTrd.fDX1 - fTrd.fDX2) + dz * dz);
  SA += 2.0 * 2.0 * 0.5 * (fTrd.fDY1 + fTrd.fDY2) * ht;

  // Sum of area for planes Perp. to +Y and -Y
  SA += 2.0 * 2.0 * 0.5 * (fTrd.fDX1 + fTrd.fDX2) * ht; // if xvert then topology forces to become yvert for closing

  // Sum of area for top and bottom planes +Z and -Z
  SA += 4. * (fTrd.fDX1 * fTrd.fDY1) + 4. * (fTrd.fDX2 * fTrd.fDY2);

  return SA;
}

int UnplacedTrd::ChooseSurface() const
{
  int choice = 0; // 0-1 = zm, 2-3 = zp, 4-5 = ym, 6-7 = yp, 8-9 = xm, 10-11 = xp
  Precision S[12], Stotal = 0.0;

  S[0] = S[1] = GetMinusZArea() / 2.0;
  S[2] = S[3] = GetPlusZArea() / 2.0;

  Precision yarea = GetPlusYArea() / (fTrd.fDX1 + fTrd.fDX2);
  S[4] = S[6] = yarea * fTrd.fDX1;
  S[5] = S[7] = yarea * fTrd.fDX2;

  Precision xarea = GetPlusXArea() / (fTrd.fDY1 + fTrd.fDY2);
  S[8] = S[10] = xarea * fTrd.fDY1;
  S[9] = S[11] = xarea * fTrd.fDY2;

  for (int i = 0; i < 12; ++i)
    Stotal += S[i];

  // random value to choose triangle to place the point
  Precision rand = RNG::Instance().uniform() * Stotal;

  while (rand > S[choice])
    rand -= S[choice], choice++;

  assert(choice < 12);
  return choice;
}

Vector3D<Precision> UnplacedTrd::SamplePointOnSurface() const
//                                             /
//                 p6-------p7                 /
//                / |       | \                /
//               / p4-------p5 \               /
//              /  /         \  \              /
//            p2--/-----------\--p3            /
//             | /             \ |             /
//             |/               \|             /
//            p0-----------------p1            /
//                                             /
{
  Vector3D<Precision> A, B, C;

  int surface = ChooseSurface();
  switch (surface) {
  case 0: // -Z face, 1st triangle (p0-p1-p2)
    A.Set(-fTrd.fDX1, -fTrd.fDY1, -fTrd.fDZ);
    B.Set(fTrd.fDX1, -fTrd.fDY1, -fTrd.fDZ);
    C.Set(-fTrd.fDX1, fTrd.fDY1, -fTrd.fDZ);
    break;
  case 1: // -Z face, 2nd triangle (p3-p1-p2)
    A.Set(fTrd.fDX1, fTrd.fDY1, -fTrd.fDZ);
    B.Set(fTrd.fDX1, -fTrd.fDY1, -fTrd.fDZ);
    C.Set(-fTrd.fDX1, fTrd.fDY1, -fTrd.fDZ);
    break;
  case 2: // +Z face, 1st triangle (p4-p5-p6)
    A.Set(-fTrd.fDX2, -fTrd.fDY2, fTrd.fDZ);
    B.Set(fTrd.fDX2, -fTrd.fDY2, fTrd.fDZ);
    C.Set(-fTrd.fDX2, fTrd.fDY2, fTrd.fDZ);
    break;
  case 3: // +Z face, 2nd triangle (p7-p5-p6)
    A.Set(fTrd.fDX2, fTrd.fDY2, fTrd.fDZ);
    B.Set(fTrd.fDX2, -fTrd.fDY2, fTrd.fDZ);
    C.Set(-fTrd.fDX2, fTrd.fDY2, fTrd.fDZ);
    break;
  case 4: // -Y face, 1st triangle (p0-p1-p4)
    A.Set(-fTrd.fDX1, -fTrd.fDY1, -fTrd.fDZ);
    B.Set(fTrd.fDX1, -fTrd.fDY1, -fTrd.fDZ);
    C.Set(-fTrd.fDX2, -fTrd.fDY2, fTrd.fDZ);
    break;
  case 5: // -Y face, 2nd triangle (p5-p1-p4)
    A.Set(fTrd.fDX2, -fTrd.fDY2, fTrd.fDZ);
    B.Set(fTrd.fDX1, -fTrd.fDY1, -fTrd.fDZ);
    C.Set(-fTrd.fDX2, -fTrd.fDY2, fTrd.fDZ);
    break;
  case 6: // +Y face, 1st triangle (p3-p2-p7)
    A.Set(fTrd.fDX1, fTrd.fDY1, -fTrd.fDZ);
    B.Set(-fTrd.fDX1, fTrd.fDY1, -fTrd.fDZ);
    C.Set(fTrd.fDX2, fTrd.fDY2, fTrd.fDZ);
    break;
  case 7: // +Y face, 2nd triangle (p6-p2-p7)
    A.Set(-fTrd.fDX2, fTrd.fDY2, fTrd.fDZ);
    B.Set(-fTrd.fDX1, fTrd.fDY1, -fTrd.fDZ);
    C.Set(fTrd.fDX2, fTrd.fDY2, fTrd.fDZ);
    break;
  case 8: // -X face, 1st triangle (p0-p4-p2)
    A.Set(-fTrd.fDX1, -fTrd.fDY1, -fTrd.fDZ);
    B.Set(-fTrd.fDX2, -fTrd.fDY2, fTrd.fDZ);
    C.Set(-fTrd.fDX1, fTrd.fDY1, -fTrd.fDZ);
    break;
  case 9: // -X face, 2nd triangle (p6-p4-p2)
    A.Set(-fTrd.fDX2, fTrd.fDY2, fTrd.fDZ);
    B.Set(-fTrd.fDX2, -fTrd.fDY2, fTrd.fDZ);
    C.Set(-fTrd.fDX1, fTrd.fDY1, -fTrd.fDZ);
    break;
  case 10: // +X face, 1st triangle (p3-p7-p1)
    A.Set(fTrd.fDX1, fTrd.fDY1, -fTrd.fDZ);
    B.Set(fTrd.fDX2, fTrd.fDY2, fTrd.fDZ);
    C.Set(fTrd.fDX1, -fTrd.fDY1, -fTrd.fDZ);
    break;
  case 11: // +X face, 2nd triangle (p3-p7-p1)
    A.Set(fTrd.fDX2, -fTrd.fDY2, fTrd.fDZ);
    B.Set(fTrd.fDX2, fTrd.fDY2, fTrd.fDZ);
    C.Set(fTrd.fDX1, -fTrd.fDY1, -fTrd.fDZ);
    break;
  }

  Precision r1 = RNG::Instance().uniform();
  Precision r2 = RNG::Instance().uniform();
  if (r1 + r2 > 1.0) {
    r1 = 1.0 - r1;
    r2 = 1.0 - r2;
  }

  return A + r1 * (B - A) + r2 * (C - A);
}

bool UnplacedTrd::Normal(Vector3D<Precision> const &point, Vector3D<Precision> &norm) const
{

  int noSurfaces = 0;
  Vector3D<Precision> sumnorm(0., 0., 0.), vecnorm(0., 0., 0.);
  Precision distz;

  distz = std::fabs(std::fabs(point[2]) - fTrd.fDZ);

  Precision xnorm = 1.0 / sqrt(4 * fTrd.fDZ * fTrd.fDZ + (fTrd.fDX2 - fTrd.fDX1) * (fTrd.fDX2 - fTrd.fDX1));
  Precision ynorm = 1.0 / sqrt(4 * fTrd.fDZ * fTrd.fDZ + (fTrd.fDY2 - fTrd.fDY1) * (fTrd.fDY2 - fTrd.fDY1));

  Precision distmx =
      -2.0 * fTrd.fDZ * point[0] - (fTrd.fDX2 - fTrd.fDX1) * point[2] - fTrd.fDZ * (fTrd.fDX1 + fTrd.fDX2);
  distmx *= xnorm;

  Precision distpx =
      2.0 * fTrd.fDZ * point[0] - (fTrd.fDX2 - fTrd.fDX1) * point[2] - fTrd.fDZ * (fTrd.fDX1 + fTrd.fDX2);
  distpx *= xnorm;

  Precision distmy =
      -2.0 * fTrd.fDZ * point[1] - (fTrd.fDY2 - fTrd.fDY1) * point[2] - fTrd.fDZ * (fTrd.fDY1 + fTrd.fDY2);
  distmy *= ynorm;

  Precision distpy =
      2.0 * fTrd.fDZ * point[1] - (fTrd.fDY2 - fTrd.fDY1) * point[2] - fTrd.fDZ * (fTrd.fDY1 + fTrd.fDY2);
  distpy *= ynorm;

  if (fabs(distmx) <= kHalfTolerance) {
    noSurfaces++;
    sumnorm += Vector3D<Precision>(-2.0 * fTrd.fDZ, 0.0, -(fTrd.fDX2 - fTrd.fDX1)) * xnorm;
  }
  if (fabs(distpx) <= kHalfTolerance) {
    noSurfaces++;
    sumnorm += Vector3D<Precision>(2.0 * fTrd.fDZ, 0.0, -(fTrd.fDX2 - fTrd.fDX1)) * xnorm;
  }
  if (fabs(distpy) <= kHalfTolerance) {
    noSurfaces++;
    sumnorm += Vector3D<Precision>(0.0, 2.0 * fTrd.fDZ, -(fTrd.fDY2 - fTrd.fDY1)) * ynorm;
  }
  if (fabs(distmy) <= kHalfTolerance) {
    noSurfaces++;
    sumnorm += Vector3D<Precision>(0.0, -2.0 * fTrd.fDZ, -(fTrd.fDY2 - fTrd.fDY1)) * ynorm;
  }

  if (std::fabs(distz) <= kHalfTolerance) {
    noSurfaces++;
    if (point[2] >= 0.)
      sumnorm += Vector3D<Precision>(0., 0., 1.);
    else
      sumnorm += Vector3D<Precision>(0., 0., -1.);
  }
  if (noSurfaces == 0) {
#ifdef UDEBUG
    UUtils::Exception("UnplacedTrapezoid::SurfaceNormal(point)", "GeomSolids1002", Warning, 1,
                      "Point is not on surface.");
#endif
    // vecnorm = ApproxSurfaceNormal( Vector3D<Precision>(point[0],point[1],point[2]) );
    vecnorm =
        Vector3D<Precision>(0., 0., 1.); // any plane will do it, since false is returned, so save the CPU cycles...
  } else if (noSurfaces == 1)
    vecnorm = sumnorm;
  else
    vecnorm = sumnorm.Unit();

  norm[0] = vecnorm[0];
  norm[1] = vecnorm[1];
  norm[2] = vecnorm[2];

  return noSurfaces != 0;
}

#endif

template <TranslationCode transCodeT, RotationCode rotCodeT>
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedTrd::Create(LogicalVolume const *const logical_volume,
                                   Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                   const int id,
#endif
                                   VPlacedVolume *const placement)
{

  using namespace TrdTypes;

#ifndef VECGEOM_NO_SPECIALIZATION

  __attribute__((unused)) const UnplacedTrd &trd =
      static_cast<const UnplacedTrd &>(*(logical_volume->GetUnplacedVolume()));

#define GENERATE_TRD_SPECIALIZATIONS
#ifdef GENERATE_TRD_SPECIALIZATIONS
  if (trd.dy1() == trd.dy2()) {
    //          std::cout << "trd1" << std::endl;
    return CreateSpecializedWithPlacement<SpecializedTrd<transCodeT, rotCodeT, TrdTypes::Trd1>>(logical_volume,
                                                                                                transformation
#ifdef VECCORE_CUDA
                                                                                                ,
                                                                                                id
#endif
                                                                                                ,
                                                                                                placement);
  } else {
    //          std::cout << "trd2" << std::endl;
    return CreateSpecializedWithPlacement<SpecializedTrd<transCodeT, rotCodeT, TrdTypes::Trd2>>(logical_volume,
                                                                                                transformation
#ifdef VECCORE_CUDA
                                                                                                ,
                                                                                                id
#endif
                                                                                                ,
                                                                                                placement);
  }
#endif

#endif // VECGEOM_NO_SPECIALIZATION

  //    std::cout << "universal trd" << std::endl;
  return CreateSpecializedWithPlacement<SpecializedTrd<transCodeT, rotCodeT, TrdTypes::UniversalTrd>>(logical_volume,
                                                                                                      transformation
#ifdef VECCORE_CUDA
                                                                                                      ,
                                                                                                      id
#endif
                                                                                                      ,
                                                                                                      placement);
}

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedTrd::SpecializedVolume(LogicalVolume const *const volume,
                                              Transformation3D const *const transformation,
                                              const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECCORE_CUDA
                                              const int id,
#endif
                                              VPlacedVolume *const placement) const
{

  return VolumeFactory::CreateByTransformation<UnplacedTrd>(volume, transformation, trans_code, rot_code,
#ifdef VECCORE_CUDA
                                                            id,
#endif
                                                            placement);
}

#if defined(VECGEOM_USOLIDS)
VECCORE_ATT_HOST_DEVICE
std::ostream &UnplacedTrd::StreamInfo(std::ostream &os) const
{
  int oldprc = os.precision(16);
  os << "-----------------------------------------------------------\n"
     << "     *** Dump for solid - " << GetEntityType() << " ***\n"
     << "     ===================================================\n"
     << " Solid type: Trd\n"
     << " Parameters: \n"
     << "     half lengths X1,X2: " << fTrd.fDX1 << "mm, " << fTrd.fDX2 << "mm \n"
     << "     half lengths Y1,Y2: " << fTrd.fDY1 << "mm, " << fTrd.fDY2 << "mm \n"
     << "     half length Z: " << fTrd.fDZ << "mm \n"
     << "-----------------------------------------------------------\n";
  os.precision(oldprc);
  return os;
}
#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedTrd::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<UnplacedTrd>(in_gpu_ptr, dx1(), dx2(), dy1(), dy2(), dz());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedTrd::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedTrd>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedTrd>::SizeOf();
template void DevicePtr<cuda::UnplacedTrd>::Construct(const Precision dx1, const Precision dx2, const Precision dy1,
                                                      const Precision dy2, const Precision z) const;

} // End cxx namespace

#endif

} // End global namespace
