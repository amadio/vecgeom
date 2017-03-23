/// @file UnplacedTessellated.cpp
/// @author Mihaela Gheata (mihaela.gheata@cern.ch)

#include "volumes/UnplacedTessellated.h"
#include "volumes/SpecializedTessellated.h"
#include "volumes/utilities/GenerationUtilities.h"
#include "base/RNG.h"

#include "management/VolumeFactory.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

void UnplacedTessellated::Print() const
{
  printf("UnplacedTessellated {%d facets}", fTessellation.fFacets.size());
}

void UnplacedTessellated::Print(std::ostream &os) const
{
  os << "UnplacedTessellated {" << fTessellation.fFacets.size() << " facets " << std::endl;
}

#ifndef VECGEOM_NVCC
Precision UnplacedTessellated::Capacity() const
{
  if (fTessellation.fCubicVolume != 0.) return fTessellation.fCubicVolume;

  // For explanation of the following algorithm see:
  // https://en.wikipedia.org/wiki/Polyhedron#Volume
  // http://wwwf.imperial.ac.uk/~rn/centroid.pdf

  int size = fTessellation.fFacets.size();
  for (int i = 0; i < size; ++i) {
    TriangularFacet &facet = fTessellation.fFacets[i];
    double area            = facet.fSurfaceArea;
    UVector3 unit_normal   = facet.GetSurfaceNormal();
    fTesselation.fCubicVolume += area * (facet.fVertices[0].Dot(facet.fNormal));
  }
  fTesselation.fCubicVolume /= 3.;
  return fTesselation.fCubicVolume;
}

Precision UnplacedTessellated::SurfaceArea() const
{
  if (fTessellation.fSurfaceArea != 0.) return fTessellation.fSurfaceArea;

  int size = fTessellation.fFacets.size();
  for (int i = 0; i < size; ++i) {
    TriangularFacet &facet = fTessellation.fFacets[i];
    fTessellation.fSurfaceArea += facet.fSurfaceArea;
  }
  return fTessellation.fSurfaceArea;
}

int UnplacedTessellated::ChooseSurface() const
{
  int choice       = 0; // 0 = zm, 1 = zp, 2 = ym, 3 = yp, 4 = xm, 5 = xp
  Precision Stotal = SurfaceArea;

  // random value to choose surface to place the point
  Precision rand = RNG::Instance().uniform() * Stotal;

  while (rand > fTessellation.fFacets[choice].fSurfaceArea)
    rand -= fTessellation.fFacets[choice].fSurfaceArea, choice++;

  return choice;
}

Vector3D<Precision> UnplacedTessellated::GetPointOnSurface() const
{
  int surface    = ChooseSurface();
  double alpha   = RNG::Instance().uniform(0., 1.);
  double beta    = RNG::Instance().uniform(0., 1.);
  double lambda1 = alpha * beta;
  double lambda0 = alpha - lambda1;

  return GetVertex(0) + lambda0 * fTessellation.fFacets[surface].fE1 + lambda1 * fTessellation.fFacets[surface].fE2;
}

bool UnplacedTessellated::Normal(Vector3D<Precision> const &point, Vector3D<Precision> &norm) const
{
  //
  norm[0] = vecnorm[0];
  norm[1] = vecnorm[1];
  norm[2] = vecnorm[2];
}

#endif

template <TranslationCode transCodeT, RotationCode rotCodeT>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume *UnplacedTessellated::Create(LogicalVolume const *const logical_volume,
                                           Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
                                           const int id,
#endif
                                           VPlacedVolume *const placement)
{

  using namespace TrdTypes;

#ifndef VECGEOM_NO_SPECIALIZATION

  __attribute__((unused)) const UnplacedTessellated &trd =
      static_cast<const UnplacedTessellated &>(*(logical_volume->GetUnplacedVolume()));

#define GENERATE_TRD_SPECIALIZATIONS
#ifdef GENERATE_TRD_SPECIALIZATIONS
  if (trd.dy1() == trd.dy2()) {
    //          std::cout << "trd1" << std::endl;
    return CreateSpecializedWithPlacement<SpecializedTessellated<transCodeT, rotCodeT, TrdTypes::Trd1>>(logical_volume,
                                                                                                        transformation
#ifdef VECGEOM_NVCC
                                                                                                        ,
                                                                                                        id
#endif
                                                                                                        ,
                                                                                                        placement);
  } else {
    //          std::cout << "trd2" << std::endl;
    return CreateSpecializedWithPlacement<SpecializedTessellated<transCodeT, rotCodeT, TrdTypes::Trd2>>(logical_volume,
                                                                                                        transformation
#ifdef VECGEOM_NVCC
                                                                                                        ,
                                                                                                        id
#endif
                                                                                                        ,
                                                                                                        placement);
  }
#endif

#endif // VECGEOM_NO_SPECIALIZATION

  //    std::cout << "universal trd" << std::endl;
  return CreateSpecializedWithPlacement<SpecializedTessellated<transCodeT, rotCodeT, TrdTypes::UniversalTrd>>(
      logical_volume, transformation
#ifdef VECGEOM_NVCC
      ,
      id
#endif
      ,
      placement);
}

VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume *UnplacedTessellated::SpecializedVolume(LogicalVolume const *const volume,
                                                      Transformation3D const *const transformation,
                                                      const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
                                                      const int id,
#endif
                                                      VPlacedVolume *const placement) const
{

  return VolumeFactory::CreateByTransformation<UnplacedTessellated>(volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
                                                                    id,
#endif
                                                                    placement);
}

#if defined(VECGEOM_USOLIDS)
VECGEOM_CUDA_HEADER_BOTH
std::ostream &UnplacedTessellated::StreamInfo(std::ostream &os) const
{
  int oldprc = os.precision(16);
  os << "-----------------------------------------------------------\n"
     << "     *** Dump for solid - " << GetEntityType() << " ***\n"
     << "     ===================================================\n"
     << " Solid type: Trd\n"
     << " Parameters: \n"
     << "-----------------------------------------------------------\n";
  os.precision(oldprc);
  return os;
}
#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedTessellated::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<UnplacedTessellated>(in_gpu_ptr);
}

DevicePtr<cuda::VUnplacedVolume> UnplacedTessellated::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedTessellated>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedTessellated>::SizeOf();
template void DevicePtr<cuda::UnplacedTessellated>::Construct() const;

} // End cxx namespace

#endif

} // End global namespace
