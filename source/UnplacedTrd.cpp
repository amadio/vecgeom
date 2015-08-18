/// @file UnplacedTrd.cpp
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#include "volumes/UnplacedTrd.h"
#include "volumes/SpecializedTrd.h"
#include "volumes/utilities/GenerationUtilities.h"
#include "base/RNG.h"

#include "management/VolumeFactory.h"


namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

void UnplacedTrd::Print() const {
  printf("UnplacedTrd {%.2f, %.2f, %.2f, %.2f, %.2f}",
         dx1(), dx2(), dy1(), dy2(), dz() );
}

void UnplacedTrd::Print(std::ostream &os) const {
  os << "UnplacedTrd {" << dx1() << ", " << dx2() << ", " << dy1()
     << ", " << dy2() << ", " << dz();
}

#ifndef VECGEOM_NVCC
Precision UnplacedTrd::Capacity() const {
   return  2*(fDX1+fDX2)*(fDY1+fDY2)*fDZ
     + (2./3.)*(fDX1-fDX2)*(fDY1-fDY2)*fDZ;
}

Precision UnplacedTrd::SurfaceArea() const {
  Precision dz = 2*fDZ;
  bool xvert = (fDX1 == fDX2) ? true : false;
  Precision SA = 0.0;

  // Sum of area for planes Perp. to +X and -X
  Precision ht = (xvert) ? dz : Sqrt((fDX1-fDX2)*(fDX1-fDX2) + dz*dz);
  SA += 2.0 * 2.0 * 0.5 * (fDY1 + fDY2) * ht;

  // Sum of area for planes Perp. to +Y and -Y
  SA += 2.0 * 2.0 * 0.5 * (fDX1 + fDX2) * ht;    // if xvert then topology forces to become yvert for closing

  // Sum of area for top and bottom planes +Z and -Z
  SA += 4.*(fDX1 * fDY1) + 4.*(fDX2 * fDY2);

  return SA;
}

/*

void UnplacedTrd::Extent(Vector3D<Precision>& aMin, Vector3D<Precision>& aMax) const {
    aMin.x() = -1.0 * Min(dx1(), dx2());
    aMax.x() = Max(dx1(), dx2());
    aMin.y() = -1.0 * Min(dy1(), dy2());
    aMax.y() = Max(dy1(), dy2());
    aMin.z() = -dz();
    aMax.z() = dz();
}
*/

int UnplacedTrd::ChooseSurface() const {
    int choice = 0; // 0 = zm, 1 = zp, 2 = ym, 3 = yp, 4 = xm, 5 = xp
    Precision S[6], Stotal = 0.0;

    S[0] = S[1] = GetPlusXArea();
    S[2] = S[3] = GetPlusYArea();
    S[4] = GetMinusZArea();
    S[5] = GetPlusZArea();

    for (int i = 0; i < 6; ++i)
        Stotal += S[i];

    // random value to choose surface to place the point
    Precision rand = RNG::Instance().uniform() * Stotal;

    while (rand > S[choice])
        rand -= S[choice], choice++;

    assert(choice < 6);

    return choice;
}

Vector3D<Precision> UnplacedTrd::GetPointOnSurface() const {
    int surface = ChooseSurface();

    Precision invHeight = 0.5 / fDZ;
    Precision slopeX = (fDX2 - fDX1) * invHeight;
    Precision slopeY = (fDY2 - fDY1) * invHeight;
    Precision midX = 0.5*(fDX1+fDX2);
    Precision midY = 0.5*(fDY1+fDY2);

    // Note that randoms are drawn in range [-1,1], then we just need scales, e.g. xval *= xmax for x = [-xmax, xmax] and so on
    Precision xval = RNG::Instance().uniform(-1.0, 1.0);
    Precision yval = RNG::Instance().uniform(-1.0, 1.0);
    Precision zval = RNG::Instance().uniform(-1.0, 1.0);
    Precision vmax;
    switch (surface) {
    case 0:  // -Z face
      xval *= fDX1;  // = ran[-fDX1, fDX1]
      yval *= fDY1;
      zval = -fDZ;
      break;
    case 1:  // +Z face
      xval *= fDX2;  // = ran[-fDX2, fDX2]
      yval *= fDY2;
      zval =  fDZ;
      break;
    case 2:  // +X face
      xval = midX + slopeX*zval;
      vmax = midY + slopeY*zval;
      yval *= vmax;  // = rand[-vmax,vmax]
      break;
    case 3:  // -X face
      xval = -(midX + slopeX*zval);
      vmax = midY + slopeY*zval;
      yval *= vmax;  // = rand[-vmax,vmax]
      break;
    case 4:  // +Y face
      vmax = midX + slopeX*zval;
      xval *= vmax;  // = rand[-vmax,vmax]
      yval = midY + slopeY*zval;
      break;
    case 5:  // -Y face
      vmax = midX + slopeX*zval;
      xval *= vmax;  // = rand[-vmax,vmax]
      yval = -(midY + slopeY*zval);
      break;
    }

    return Vector3D<Precision>(xval, yval, zval);
}


bool UnplacedTrd::Normal(Vector3D<Precision> const& point, Vector3D<Precision>& norm) const {
    int nosurface = 0;
    bool onSurf(false);

    Precision xp = point[0];
    Precision yp = point[1];
    Precision zp = point[2];
    Precision xx1 = dx1();
    Precision xx2 = dx2();
    Precision yy1 = dy1();
    Precision yy2 = dy2();
    Precision zz = dz();
    Precision XplusX   = 2.0 * yy1 * 2.0 * zz;
    Precision XminusX  = XplusX;
    Precision XplusY   = 0.0;
    Precision XminusY  = 0.0;
    Precision XplusZ   = 2.0 * yy1 * (xx1 - xx2);
    Precision XminusZ  = 2.0 * yy1 * (-xx1 + xx2);

    Precision YplusX   = 0.0;
    Precision YminusX  = 0.0;
    Precision YplusY   = -2.0 * xx1 * 2.0 * zz;
    Precision YminusY  = YplusY;
    Precision YplusZ   = 2.0 * xx1 * (-yy1 + yy2);
    Precision YminusZ  = 2.0 * xx1 * (yy1 - yy2);
    Precision ZplusX   = 0.0;
    Precision ZminusX  = 0.0;
    Precision ZplusY   = 0.0;
    Precision ZminusY  = 0.0;
    Precision ZplusZ   = -2.0 * xx2 * 2.0 * yy2;
    Precision ZminusZ  = 2.0 * xx2 * 2.0 * yy1;

    // Checking for each plane whether the point is on Surface, if yes transfer normal
    bool FacPlusX  = XplusX * (xp - xx2) + XplusY * (yp - yy2) + XplusZ * (zp - zz);
    bool FacMinusX = XminusX * (xp + xx2) + XminusY * (yp - yy2) + XminusZ * (zp - zz);
    bool FacPlusY  = YplusX * (xp - xx2)  + YplusY * (yp - yy2) + YplusZ * (zp - zz);
    bool FacMinusY = YplusX * (xp - xx2) + YminusY * (yp + yy2) + YminusZ * (zp - zz);
    bool FacPlusZ  = ZplusX * (xp - xx2) + ZplusY * (yp - yy2) + ZplusZ * (zp - zz);
    bool FacMinusZ = ZminusX * (xp - xx2) + ZminusY * (yp + yy2) + ZminusZ * (zp - zz);
    onSurf = FacPlusX || FacMinusX || FacPlusY || FacMinusY || FacPlusZ || FacMinusZ;
    if (onSurf && FacPlusX)  norm[0] = XplusX;   norm[1] = XplusY;  norm[2] = XplusZ;  nosurface++;
    if (onSurf && FacMinusX) norm[0] = XminusX;  norm[1] = XminusY; norm[2] = XminusZ; nosurface++;
    if (onSurf && FacPlusY)  norm[0] = YplusX;   norm[1] = YplusY;  norm[2] = YplusZ;  nosurface++;
    if (onSurf && FacMinusY) norm[0] = YminusX;  norm[1] = YminusY; norm[2] = YminusZ; nosurface++;
    if (onSurf && FacPlusZ)  norm[0] = ZplusX;   norm[1] = ZplusY;  norm[2] = ZplusZ;  nosurface++;
    if (onSurf && FacMinusZ) norm[0] = ZminusX;  norm[1] = ZminusY; norm[2] = ZminusZ; nosurface++;
    return nosurface != 0;
}


#endif

template <TranslationCode transCodeT, RotationCode rotCodeT>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedTrd::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) {

    using namespace TrdTypes;

#ifndef VECGEOM_NO_SPECIALIZATION

     __attribute__((unused)) const UnplacedTrd &trd = static_cast<const UnplacedTrd&>( *(logical_volume->unplaced_volume()) );

    #define GENERATE_TRD_SPECIALIZATIONS
    #ifdef GENERATE_TRD_SPECIALIZATIONS
      if(trd.dy1() == trd.dy2()) {
    //          std::cout << "trd1" << std::endl;
          return CreateSpecializedWithPlacement<SpecializedTrd<transCodeT, rotCodeT, TrdTypes::Trd1> >(logical_volume, transformation
#ifdef VECGEOM_NVCC
                 ,id
#endif
                 , placement);
      } else {
    //          std::cout << "trd2" << std::endl;
          return CreateSpecializedWithPlacement<SpecializedTrd<transCodeT, rotCodeT, TrdTypes::Trd2> >(logical_volume, transformation
#ifdef VECGEOM_NVCC
                 ,id
#endif
                 , placement);
    }
    #endif

#endif // VECGEOM_NO_SPECIALIZATION

      //    std::cout << "universal trd" << std::endl;
    return CreateSpecializedWithPlacement<SpecializedTrd<transCodeT, rotCodeT, TrdTypes::UniversalTrd> >(logical_volume, transformation
#ifdef VECGEOM_NVCC
                ,id
#endif
                , placement);
}

VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedTrd::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {

  return VolumeFactory::CreateByTransformation<
      UnplacedTrd>(volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
                              id,
#endif
                              placement);
}

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedTrd::CopyToGpu(
   DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
   return CopyToGpuImpl<UnplacedTrd>(in_gpu_ptr, dx1(), dx2(), dy1(), dy2(), dz());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedTrd::CopyToGpu() const
{
   return CopyToGpuImpl<UnplacedTrd>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedTrd>::SizeOf();
template void DevicePtr<cuda::UnplacedTrd>::Construct(
   const Precision dx1, const Precision dx2, const Precision dy1,
   const Precision dy2, const Precision d) const;

} // End cxx namespace

#endif

} // End global namespace
