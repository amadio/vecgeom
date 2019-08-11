/// \file UnplacedHype.cpp
/// \author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)

#include "volumes/UnplacedHype.h"

#include "management/VolumeFactory.h"
#include "volumes/SpecializedHype.h"
#include "volumes/utilities/GenerationUtilities.h"

#ifdef VECGEOM_ROOT
#include "TGeoHype.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Hype.hh"
#endif

#ifndef VECCORE_CUDA
#include "volumes/UnplacedImplAs.h"
#endif

#include <stdio.h>
#include "base/RNG.h"
#include "base/Global.h"

#ifndef VECGEOM_NO_SPECIALIZATION
#include "volumes/UnplacedTube.h"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <>
UnplacedHype *Maker<UnplacedHype>::MakeInstance(const Precision rMin, const Precision rMax, const Precision stIn,
                                                const Precision stOut, const Precision dz)
{
#ifndef VECGEOM_NO_SPECIALIZATION
#ifndef VECCORE_CUDA
  if (rMin <= 0) {
    // Solid Hype
    if ((stIn <= 0.) && (stOut > 0.)) {
      return new SUnplacedHype<HypeTypes::NonHollowHype>(0., rMax, 0., stOut, dz);
    }

    // Hype becomes solid Tube with outer radius equals rMax
    if ((stIn <= 0.) && (stOut <= 0.)) {
      return new SUnplacedImplAs<SUnplacedHype<HypeTypes::NonHollowHype>, SUnplacedTube<TubeTypes::NonHollowTube>>(
          0., rMax, dz, 0., 2 * kPi);
    }
  } else if (rMin > 0) {

    // Hype becomes hollow Tube with inner radius equals rMin and outer radius equals rMax
    if ((stIn <= 0.) && (stOut <= 0.)) {
      return new SUnplacedImplAs<SUnplacedHype<HypeTypes::HollowHype>, SUnplacedTube<TubeTypes::HollowTube>>(
          rMin, rMax, dz, 0., 2 * kPi);
    }
    if ((stIn > 0.) || (stOut > 0.)) {
      return new SUnplacedHype<HypeTypes::HollowHype>(rMin, rMax, stIn, stOut, dz);
    }
  }
  return new SUnplacedHype<HypeTypes::UniversalHype>(rMin, rMax, stIn, stOut, dz);
#endif
#endif
  return new SUnplacedHype<HypeTypes::UniversalHype>(rMin, rMax, stIn, stOut, dz);
}

#ifndef VECCORE_CUDA
#ifdef VECGEOM_ROOT
TGeoShape const *UnplacedHype::ConvertToRoot(char const *label) const
{
  return new TGeoHype(label, GetRmin(), GetStIn() * kRadToDeg, GetRmax(), GetStOut() * kRadToDeg, GetDz());
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *UnplacedHype::ConvertToGeant4(char const *label) const
{
  return new G4Hype(label, GetRmin(), GetRmax(), GetStIn(), GetStOut(), GetDz());
}
#endif
#endif

VECCORE_ATT_HOST_DEVICE
void UnplacedHype::DetectConvexity()
{
  // Default Convexity set to false
  fGlobalConvexity = false;
  // Logic to calculate the convexity
  if ((fHype.fRmin == 0.) && (fHype.fStIn == 0.) && (fHype.fStOut == 0.)) // Hype becomes Solid Tube.
    fGlobalConvexity = true;
}

VECCORE_ATT_HOST_DEVICE
void UnplacedHype::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  // Returns the full 3D cartesian extent of the solid.
  Precision rMax = GetEndOuterRadius();
  aMin.Set(-rMax, -rMax, -fHype.fDz);
  aMax.Set(rMax, rMax, fHype.fDz);
}

std::string UnplacedHype::GetEntityType() const
{
  return "Hyperboloid\n";
}

Vector3D<Precision> UnplacedHype::SamplePointOnSurface() const
{

  Precision xRand, yRand, zRand, r2, aOne, aTwo, aThree, chose, sinhu;
  Precision phi, cosphi, sinphi, rBar2Out, rBar2In, alpha, t, rOut, rIn2, rOut2;

  // we use the formula of the area of a surface of revolution to compute
  // the areas, using the equation of the hyperbola:
  // x^2 + y^2 = (z*tanphi)^2 + r^2

  rBar2Out = fHype.fRmax2;
  alpha    = 2. * kPi * rBar2Out * std::cos(fHype.fStOut) / fHype.fTOut;
  t        = fHype.fDz * fHype.fTOut / (fHype.fRmax * std::cos(fHype.fStOut));
  t        = std::log(t + std::sqrt(t * t + 1)); // sqr(t*t)
  aOne     = std::fabs(2. * alpha * (std::sinh(2. * t) / 4. + t / 2.));

  rBar2In = fHype.fRmin2;
  alpha   = 2. * kPi * rBar2In * std::cos(fHype.fStIn) / fHype.fTIn;
  t       = fHype.fDz * fHype.fTIn / (fHype.fRmin * std::cos(fHype.fStIn));
  t       = std::log(t + std::sqrt(t * t + 1)); // sqr(t*t)
  aTwo    = std::fabs(2. * alpha * (std::sinh(2. * t) / 4. + t / 2.));

  aThree = kPi * ((fHype.fRmax2 + (fHype.fDz * fHype.fTOut) * (fHype.fDz * fHype.fTOut) -
                   (fHype.fRmin2 + (fHype.fDz * fHype.fTIn) * (fHype.fDz * fHype.fTIn))));

  if (fHype.fStOut == 0.) {
    aOne = std::fabs(2. * kPi * fHype.fRmax * 2. * fHype.fDz);
  }
  if (fHype.fStIn == 0.) {
    aTwo = std::fabs(2. * kPi * fHype.fRmin * 2. * fHype.fDz);
  }

  phi    = RNG::Instance().uniform(0., 2. * kPi);
  cosphi = std::cos(phi);
  sinphi = std::sin(phi);
  sinhu  = RNG::Instance().uniform(-1. * fHype.fDz * fHype.fTOut / fHype.fRmax, fHype.fDz * fHype.fTOut / fHype.fRmax);

  chose = RNG::Instance().uniform(0., aOne + aTwo + 2. * aThree);
  if (chose >= 0. && chose < aOne) {
    if (fHype.fStOut != 0.) {
      zRand = fHype.fRmax * sinhu / fHype.fTOut;
      xRand = std::sqrt((sinhu * sinhu) + 1) * fHype.fRmax * cosphi;
      yRand = std::sqrt((sinhu * sinhu) + 1) * fHype.fRmax * sinphi;
      return Vector3D<Precision>(xRand, yRand, zRand);
    } else {
      return Vector3D<Precision>(fHype.fRmax * cosphi, fHype.fRmax * sinphi,
                                 RNG::Instance().uniform(-fHype.fDz, fHype.fDz)); // RandFlat::shoot
    }
  } else if (chose >= aOne && chose < aOne + aTwo) {
    if (fHype.fStIn != 0.) {
      sinhu = RNG::Instance().uniform(-1. * fHype.fDz * fHype.fTIn / fHype.fRmin, fHype.fDz * fHype.fTIn / fHype.fRmin);
      zRand = fHype.fRmin * sinhu / fHype.fTIn;
      xRand = std::sqrt((sinhu * sinhu) + 1) * fHype.fRmin * cosphi;
      yRand = std::sqrt((sinhu * sinhu) + 1) * fHype.fRmin * sinphi;
      return Vector3D<Precision>(xRand, yRand, zRand);
    } else {
      return Vector3D<Precision>(fHype.fRmin * cosphi, fHype.fRmin * sinphi,
                                 RNG::Instance().uniform(-1. * fHype.fDz, fHype.fDz));
    }
  } else if (chose >= aOne + aTwo && chose < aOne + aTwo + aThree) {
    rIn2  = fHype.fRmin2 + fHype.fTIn2 * fHype.fDz * fHype.fDz;
    rOut2 = fHype.fRmax2 + fHype.fTOut2 * fHype.fDz * fHype.fDz;
    rOut  = std::sqrt(rOut2);

    do {
      xRand = RNG::Instance().uniform(-rOut, rOut);
      yRand = RNG::Instance().uniform(-rOut, rOut);
      r2    = xRand * xRand + yRand * yRand;
    } while (!(r2 >= rIn2 && r2 <= rOut2));

    zRand = fHype.fDz;
    return Vector3D<Precision>(xRand, yRand, zRand);
  } else {
    rIn2  = fHype.fRmin2 + fHype.fTIn2 * fHype.fDz * fHype.fDz;
    rOut2 = fHype.fRmax2 + fHype.fTOut2 * fHype.fDz * fHype.fDz;
    rOut  = std::sqrt(rOut2);

    do {
      xRand = RNG::Instance().uniform(-rOut, rOut);
      yRand = RNG::Instance().uniform(-rOut, rOut);
      r2    = xRand * xRand + yRand * yRand;
    } while (!(r2 >= rIn2 && r2 <= rOut2));

    zRand = -1. * fHype.fDz;
    return Vector3D<Precision>(xRand, yRand, zRand);
  }
}

VECCORE_ATT_HOST_DEVICE
void UnplacedHype::GetParametersList(int, double *aArray) const
{
  aArray[0] = fHype.fRmin;  // GetRmin();
  aArray[1] = fHype.fStIn;  // GetStIn();
  aArray[2] = fHype.fRmax;  // GetRmax();
  aArray[3] = fHype.fStOut; // GetStOut();
  aArray[4] = fHype.fDz;    // GetDz();
}

VECCORE_ATT_HOST_DEVICE
UnplacedHype *UnplacedHype::Clone() const
{
  return new SUnplacedHype<HypeTypes::UniversalHype>(fHype.fRmin, fHype.fStIn, fHype.fRmax, fHype.fStOut, fHype.fDz);
}

std::ostream &UnplacedHype::StreamInfo(std::ostream &os) const
// Definition taken from
{

  int oldprc = os.precision(16);
  os << "-----------------------------------------------------------\n"
     << "     *** Dump for solid - " << GetEntityType() << " ***\n"
     << "     ===================================================\n"

     << " Solid type: VecGeomHype\n"
     << " Parameters: \n"

     << "               Inner radius: " << fHype.fRmin << " mm \n"
     << "               Inner Stereo Angle " << fHype.fStIn << " rad \n"
     << "               Outer radius: " << fHype.fRmax << "mm\n"
     << "               Outer Stereo Angle " << fHype.fStOut << " rad \n"
     << "               Half Height: " << fHype.fDz << " mm \n"
     << "-----------------------------------------------------------\n";
  os.precision(oldprc);
  return os;
}

void UnplacedHype::Print() const
{
  printf("UnplacedHype {%.2f, %.2f, %.2f, %.2f, %.2f}", fHype.fRmin, fHype.fRmax, fHype.fStIn, fHype.fStOut, fHype.fDz);
}

void UnplacedHype::Print(std::ostream &os) const
{
  os << "UnplacedHype {" << fHype.fRmin << ", " << fHype.fRmax << ", " << fHype.fStIn << ", " << fHype.fStOut << ", "
     << fHype.fDz << "}";
}

#ifndef VECCORE_CUDA
SolidMesh *UnplacedHype::CreateMesh3D(Transformation3D const &trans, size_t nSegments) const
{

  typedef Vector3D<double> Vec_t;
  SolidMesh *sm = new SolidMesh();



     //sm->ResetMesh(4 * nMeshVertices, nMeshPolygons);

     double x, y;

     double z_step           = 2*GetDz() / nSegments;
     double z                = -GetDz();
     double phi = 0;
     double phi_step  = 2 * M_PI / nSegments;

     double cos_angle, sin_angle, outerEq, innerEq;

     Vec_t *vertices = new Vec_t[2 * (nSegments + 1) * (nSegments + 1)];

     size_t idx0 = 0;
     size_t idx1 = (nSegments + 1) * (nSegments + 1);

     for (size_t i = 0; i <= nSegments; ++i, z += z_step, phi = 0.) {
         outerEq          = std::sqrt(GetRmax2() + (z * z * GetTOut2()));
         innerEq          = std::sqrt(GetRmin2() + (z * z * GetTIn2()));
       for (size_t j = 0; j <= nSegments; ++j, phi += phi_step) {
         cos_angle        = std::cos(phi);
         sin_angle        = std::sin(phi);

         x                                 = outerEq * cos_angle;
         y                                 = outerEq * sin_angle;
         vertices[idx0++]                     = Vec_t(x, y, z); // outer

         x                                 = innerEq * cos_angle;
         y                                 = innerEq * sin_angle;
         vertices[idx1++]     = Vec_t(x, y, z); //inner
       }
     }
     sm->SetVertices(vertices, 2 * (nSegments + 1) * (nSegments + 1));
     delete[] vertices;
     sm->TransformVertices(trans);


     // lower face
      for (size_t j = 0, k = (nSegments + 1) * (nSegments + 1); j < nSegments; j++, k++) {
        sm->AddPolygon(4, {j + 1, j, k, k + 1}, true);
      }


      // upper face
      for (size_t i = 0, m = (nSegments + 1) * (nSegments), n = m + (nSegments + 1) * (nSegments + 1); i < nSegments; i++, m++, n++) {
        sm->AddPolygon(4, {n + 1, n, m, m + 1}, true);
      }


      // outer surface

      for (size_t j = 0, k = 0; j < nSegments; j++, k++) {
        for (size_t i = 0, l = k + (nSegments + 1); i < nSegments; i++, k++, l++) {
          sm->AddPolygon(4, {l + 1, l, k, k + 1}, true);
        }
      }


      // inner surface

      for (size_t j = 0, k = (nSegments + 1) * (nSegments + 1); j < nSegments; j++, k++) {
        for (size_t i = 0, l = k + (nSegments + 1); i < nSegments; i++, k++, l++) {
          sm->AddPolygon(4, {l + 1, k + 1, k, l}, true);
        }
      }

      /*
      // outer surface (bottom to 0)

      for (size_t j = 0, k = 0; j < nVertical; j++, k++) {
        for (size_t i = 0, l = k + (nHorizontal + 1); i < nHorizontal; i++, k++, l++) {
          sm->AddPolygon(4, {l + 1, l, k, k + 1}, true);
        }
      }

      // outer surface (0 to top)
      for (size_t j = 0, k = 2 * nMeshVertices; j < nVertical; j++, k++) {
        for (size_t i = 0, l = k + (nHorizontal + 1); i < nHorizontal; i++, k++, l++){
          sm->AddPolygon(4, {k + 1, k, l, l + 1}, true);
        }
      }


      // inner surface (bottom to 0)
      for (size_t j = 0, k = nMeshVertices; j < nVertical; j++, k++) {
        for (size_t i = 0, l = k + nHorizontal + 1; i < nHorizontal; i++, k++, l++) {
          sm->AddPolygon(4, {k + 1, k, l, l + 1}, true);
        }
      }


      // inner surface (0 to top)
      for (size_t j = 0, k = 3 * nMeshVertices; j < nVertical; j++, k++) {
        for (size_t i = 0, l = k + nHorizontal + 1; i < nHorizontal; i++, k++, l++) {
          sm->AddPolygon(4, {l + 1, l, k, k + 1}, true);
        }
      }




     // lower face
     for (size_t j = 0, k = (nSegments + 1); j < nSegments; j++, k++) {
       sm->AddPolygon(4, {j + 1, j, k, k + 1}, true);
     }

     // upper face
     for (size_t i = 0, m = 2 * (nSegments + 1), n = 3 * (nSegments + 1); i < nSegments; i++, m++, n++) {
       sm->AddPolygon(4, {n + 1, n, m, m + 1}, true);
     }

     // outer surface
     for (size_t j = 0, k = 0; j < nSegments; j++, k++) {
       for (size_t i = 0, l = k + nSegments + 1; i < nSegments; i++, k++, l++) {
         sm->AddPolygon(4, {l + 1, l, k, k + 1}, true);
       }
     }

     // outer surface
     for (size_t j = 0, k = 0; j < nSegments; j++, k++) {
       for (size_t i = 0, l = k + nSegments + 1; i < nSegments; i++, k++, l++) {
         sm->AddPolygon(4, {l + 1, l, k, k + 1}, true);
       }
     }

*/

  /*
  if (hasNoInner) {
	  nVertical = nHorizontal   = std::ceil(std::sqrt(nFaces - 2));
    nMeshVertices = (nHorizontal + 1) * (nVertical + 1);
    nMeshPolygons = (nHorizontal * nVertical + 2);
    sm->ResetMesh(nMeshVertices, nMeshPolygons);

    double z_step, z, horizontal_angle, horizontal_step, intermediate;

    z_step           = 2 * GetDz() / nVertical;
    z                = -GetDz();
    horizontal_angle = 0;
    horizontal_step  = 2 * M_PI / nHorizontal;

    Vec_t *vertices = new Vec_t[nMeshVertices];

    size_t idx = 0;
    for (size_t i = 0; i <= nVertical; ++i, z += z_step, horizontal_angle = 0.) {
      intermediate = std::sqrt(GetRmax2() + (z * z * GetTOut2()));
      for (size_t j = 0; j <= nHorizontal; ++j, horizontal_angle += horizontal_step) {
        vertices[idx++] = Vec_t(intermediate * std::cos(horizontal_angle), intermediate  * std::sin(horizontal_angle), z);
      }
    }
    sm->SetVertices(vertices, nMeshVertices);
    delete[] vertices;
    sm->TransformVertices(trans);

    Utils3D::vector_t<size_t> indices;
    indices.reserve(nHorizontal);

    // lower surface
    for (size_t i = nHorizontal; i > 0; i--) {
      indices.push_back(i - 1);
    }
    sm->AddPolygon(nHorizontal, indices, true);

    // upper surface

    indices.clear();
    for (size_t i = 0, k = (nHorizontal + 1) * (nVertical); i < nHorizontal; i++, k++) {
      indices.push_back(k);
    }
    sm->AddPolygon(nHorizontal, indices, true);

    // lateral surface
    for (size_t j = 0, k = 0; j < nVertical; j++, k++) {
      for (size_t i = 0, l = k + nHorizontal + 1; i < nHorizontal; i++, k++, l++) {
        sm->AddPolygon(4, {l + 1, l, k, k + 1}, true);
      }
    }

  } else {
    nVertical = nHorizontal = std::ceil((-1 + std::sqrt(4 * nFaces + 1)) / 4);
    nMeshVertices           = (nHorizontal + 1) *
                    (nVertical + 1); // the total number of vertices is 4 * nMeshVertices
    nMeshPolygons = 2 * (nHorizontal + 2 * nHorizontal * nVertical);
    sm->ResetMesh(4 * nMeshVertices, nMeshPolygons);

    double x, y;

    double z_step           = GetDz() / nVertical;
    double z                = -GetDz();
    double horizontal_angle = 0;
    double horizontal_step  = 2 * M_PI / nHorizontal;

    double cos_angle, sin_angle, outerEq, innerEq;

    Vec_t *vertices = new Vec_t[4 * nMeshVertices];

    size_t idx0 = 0;
    size_t idx1 = nMeshVertices;
    size_t idx2 = 2*nMeshVertices;
    size_t idx3 = 3*nMeshVertices;

    for (size_t i = 0; i <= nVertical; ++i, z += z_step, horizontal_angle = 0.) {
        outerEq          = std::sqrt(GetRmax2() + (z * z * GetTOut2()));
        innerEq          = std::sqrt(GetRmin2() + (z * z * GetTIn2()));
      for (size_t j = 0; j <= nHorizontal; ++j, horizontal_angle += horizontal_step) {
        cos_angle        = std::cos(horizontal_angle);
        sin_angle        = std::sin(horizontal_angle);


        x                                 = outerEq * cos_angle;
        y                                 = outerEq * sin_angle;
        vertices[idx0++]                     = Vec_t(x, y, z); // bottom outer
        vertices[idx2++] = Vec_t(x, y, -z); //top outer

        x                                 = innerEq * cos_angle;
        y                                 = innerEq * sin_angle;
        vertices[idx1++]     = Vec_t(x, y, z); //bottom inner
        vertices[idx3++] = Vec_t(x, y, -z); // top inner
      }
    }
    sm->SetVertices(vertices, 4 * nMeshVertices);
    delete[] vertices;
    sm->TransformVertices(trans);

    // lower face
    for (size_t j = 0, k = nMeshVertices; j < nHorizontal; j++, k++) {
      sm->AddPolygon(4, {j + 1, j, k, k + 1}, true);
    }

    // upper face
    for (size_t i = 0, m = 2 * nMeshVertices, n = 3 * nMeshVertices; i < nHorizontal; i++, m++, n++) {
      sm->AddPolygon(4, {n + 1, n, m, m + 1}, true);
    }

    // outer surface (bottom to 0)

    for (size_t j = 0, k = 0; j < nVertical; j++, k++) {
      for (size_t i = 0, l = k + (nHorizontal + 1); i < nHorizontal; i++, k++, l++) {
        sm->AddPolygon(4, {l + 1, l, k, k + 1}, true);
      }
    }

    // outer surface (0 to top)
    for (size_t j = 0, k = 2 * nMeshVertices; j < nVertical; j++, k++) {
      for (size_t i = 0, l = k + (nHorizontal + 1); i < nHorizontal; i++, k++, l++){
        sm->AddPolygon(4, {k + 1, k, l, l + 1}, true);
      }
    }


    // inner surface (bottom to 0)
    for (size_t j = 0, k = nMeshVertices; j < nVertical; j++, k++) {
      for (size_t i = 0, l = k + nHorizontal + 1; i < nHorizontal; i++, k++, l++) {
        sm->AddPolygon(4, {k + 1, k, l, l + 1}, true);
      }
    }


    // inner surface (0 to top)
    for (size_t j = 0, k = 3 * nMeshVertices; j < nVertical; j++, k++) {
      for (size_t i = 0, l = k + nHorizontal + 1; i < nHorizontal; i++, k++, l++) {
        sm->AddPolygon(4, {l + 1, l, k, k + 1}, true);
      }
    }

  }

*/
  //sm->InitPolygons();
  return sm;
}
#endif

#ifdef VECGEOM_CUDA_INTERFACE
DevicePtr<cuda::VUnplacedVolume> UnplacedHype::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<SUnplacedHype<HypeTypes::UniversalHype>>(in_gpu_ptr, fHype.fRmin, fHype.fRmax, fHype.fStIn,
                                                                fHype.fStOut, fHype.fDz);
}

DevicePtr<cuda::VUnplacedVolume> UnplacedHype::CopyToGpu() const
{
  return CopyToGpuImpl<SUnplacedHype<HypeTypes::UniversalHype>>();
}
#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::SUnplacedHype<cuda::HypeTypes::UniversalHype>>::SizeOf();
template void DevicePtr<cuda::SUnplacedHype<cuda::HypeTypes::UniversalHype>>::Construct(
    const Precision rmin, const Precision rmax, const Precision stIn, const Precision stOut, const Precision z) const;

} // namespace cxx

#endif
} // namespace vecgeom
