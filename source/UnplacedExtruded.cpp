/// @file UnplacedExtruded.cpp
/// @author Mihaela Gheata (mihaela.gheata@cern.ch)

#include "volumes/Tessellated.h"
#include "volumes/UnplacedExtruded.h"
#include "volumes/SpecializedExtruded.h"
#include "volumes/utilities/GenerationUtilities.h"
#include "base/RNG.h"

#include "management/VolumeFactory.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

void UnplacedExtruded::Print() const
{
  std::cout << "UnplacedExtruded: vertices {";
  int nvert = GetNVertices();
  double x, y;
  for (int i = 0; i < nvert - 1; ++i) {
    GetVertex(i, x, y);
    std::cout << "(" << x << ", " << y << "), ";
  }
  GetVertex(nvert - 1, x, y);
  std::cout << "(" << x << ", " << y << ")}\n";
  std::cout << "sections:\n";
  int nsect = GetNSections();
  for (int i = 0; i < nsect; ++i) {
    XtruSection sect = GetSection(i);
    std::cout << "orig: (" << sect.fOrigin.x() << ", " << sect.fOrigin.y() << ", " << sect.fOrigin.z()
              << ") scl = " << sect.fScale << std::endl;
  }
}

void UnplacedExtruded::Print(std::ostream &os) const
{
  os << "UnplacedExtruded: vertices {";
  int nvert = GetNVertices();
  double x, y;
  for (int i = 0; i < nvert - 1; ++i) {
    GetVertex(i, x, y);
    os << "(" << x << ", " << y << "), ";
  }
  GetVertex(nvert - 1, x, y);
  os << "(" << x << ", " << y << ")}\n";
  os << "sections:\n";
  int nsect = GetNSections();
  for (int i = 0; i < nsect; ++i) {
    XtruSection sect = GetSection(i);
    os << "orig: (" << sect.fOrigin.x() << ", " << sect.fOrigin.y() << ", " << sect.fOrigin.z()
       << ") scl = " << sect.fScale << std::endl;
  }
}

void UnplacedExtruded::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  if (fXtru.fIsSxtru) {
    fXtru.fSxtruHelper.Extent(aMin, aMax);
  } else {
    fXtru.fTslHelper.Extent(aMin, aMax);
  }
}

Precision UnplacedExtruded::Capacity() const
{
  if (fXtru.fCubicVolume != 0.) return fXtru.fCubicVolume;

  if (fXtru.fIsSxtru) {
    fXtru.fCubicVolume =
        fXtru.fSxtruHelper.GetPolygon().Area() * (fXtru.fSxtruHelper.GetUpperZ() - fXtru.fSxtruHelper.GetLowerZ());
  } else {
    int size = fXtru.fTslHelper.fFacets.size();
    for (int i = 0; i < size; ++i) {
      TriangleFacet<double> &facet = *fXtru.fTslHelper.fFacets[i];
      double area                  = facet.fSurfaceArea;
      fXtru.fCubicVolume += area * (facet.fVertices[0].Dot(facet.fNormal));
    }
    fXtru.fCubicVolume /= 3.;
  }
  return fXtru.fCubicVolume;
}

Precision UnplacedExtruded::SurfaceArea() const
{
  if (fXtru.fSurfaceArea != 0.) return fXtru.fSurfaceArea;

  if (fXtru.fIsSxtru) {
    fXtru.fSurfaceArea = fXtru.fSxtruHelper.SurfaceArea() + 2. * fXtru.fSxtruHelper.GetPolygon().Area();
  } else {
    int size = fXtru.fTslHelper.fFacets.size();
    for (int i = 0; i < size; ++i) {
      TriangleFacet<double> *facet = fXtru.fTslHelper.fFacets[i];
      fXtru.fSurfaceArea += facet->fSurfaceArea;
    }
  }
  return fXtru.fSurfaceArea;
}

int UnplacedExtruded::ChooseSurface() const
{
  int choice       = 0; // 0 = zm, 1 = zp, 2 = ym, 3 = yp, 4 = xm, 5 = xp
  Precision Stotal = SurfaceArea();

  // random value to choose surface to place the point
  Precision rand = RNG::Instance().uniform() * Stotal;

  while (rand > fXtru.fTslHelper.fFacets[choice]->fSurfaceArea)
    rand -= fXtru.fTslHelper.fFacets[choice]->fSurfaceArea, choice++;

  return choice;
}

Vector3D<Precision> UnplacedExtruded::SamplePointOnSurface() const
{
  int surface    = ChooseSurface();
  double alpha   = RNG::Instance().uniform(0., 1.);
  double beta    = RNG::Instance().uniform(0., 1.);
  double lambda1 = alpha * beta;
  double lambda0 = alpha - lambda1;
  auto facet     = fXtru.fTslHelper.fFacets[surface];
  return (facet->fVertices[0] + lambda0 * (facet->fVertices[1] - facet->fVertices[0]) +
          lambda1 * (facet->fVertices[2] - facet->fVertices[1]));
}

bool UnplacedExtruded::Normal(Vector3D<Precision> const &point, Vector3D<Precision> &norm) const
{
  // Redirect to normal implementation
  bool valid = false;
  if (fXtru.fIsSxtru) {
    norm = SExtruImplementation::NormalKernel(fXtru.fSxtruHelper, point, valid);
  } else {
    norm = TessellatedImplementation::NormalKernel<Precision>(fXtru.fTslHelper, point, valid);
  }
  return valid;
}

#ifdef VECCORE_CUDA
template <TranslationCode transCodeT, RotationCode rotCodeT>
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedExtruded::Create(LogicalVolume const *const logical_volume,
                                        Transformation3D const *const transformation, const int id,
                                        VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedExtruded<transCodeT, rotCodeT>(logical_volume, transformation, id);
    return placement;
  }
  return new SpecializedExtruded<transCodeT, rotCodeT>(logical_volume, transformation, id);
}
#else
template <TranslationCode transCodeT, RotationCode rotCodeT>
VPlacedVolume *UnplacedExtruded::Create(LogicalVolume const *const logical_volume,
                                        Transformation3D const *const transformation, VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedExtruded<transCodeT, rotCodeT>(logical_volume, transformation);
    return placement;
  }
  return new SpecializedExtruded<transCodeT, rotCodeT>(logical_volume, transformation);
}
#endif

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedExtruded::SpecializedVolume(LogicalVolume const *const volume,
                                                   Transformation3D const *const transformation,
                                                   const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECCORE_CUDA
                                                   const int id,
#endif
                                                   VPlacedVolume *const placement) const
{

  return VolumeFactory::CreateByTransformation<UnplacedExtruded>(volume, transformation, trans_code, rot_code,
#ifdef VECCORE_CUDA
                                                                 id,
#endif
                                                                 placement);
}

#if defined(VECGEOM_USOLIDS)
std::ostream &UnplacedExtruded::StreamInfo(std::ostream &os) const
{
  int oldprc = os.precision(16);
  os << "-----------------------------------------------------------\n"
     << "     *** Dump for solid - " << GetEntityType() << " ***\n"
     << "     ===================================================\n"
     << " Solid type: Extruded\n"
     << " Parameters: \n"
     << "-----------------------------------------------------------\n";
  os.precision(oldprc);
  return os;
}
#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedExtruded::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<UnplacedExtruded>(in_gpu_ptr);
}

DevicePtr<cuda::VUnplacedVolume> UnplacedExtruded::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedExtruded>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedExtruded>::SizeOf();
template void DevicePtr<cuda::UnplacedExtruded>::Construct() const;

} // End cxx namespace

#endif

} // End global namespace
