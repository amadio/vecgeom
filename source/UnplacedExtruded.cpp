/// @file UnplacedExtruded.cpp
/// @author Mihaela Gheata (mihaela.gheata@cern.ch)

#include "volumes/UnplacedExtruded.h"
#include "volumes/SpecializedExtruded.h"
#include "volumes/utilities/GenerationUtilities.h"
#include "base/RNG.h"

#include "management/VolumeFactory.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

UnplacedExtruded(int nvertices, Vertex2 const *vertices, int nsections, Section const *sections)
     : UnplacedTessellated()
{
  // Create the polygon
  double *x = new double[nvertices];
  double *y = new double[nvertices];
  for (int i=0; i<nvertices; ++i) {
    x[i] = vertices[i].x;
    y[i] = vertices[i].y;
  }
  fPolygon = new PlanarPolygon(nvertices, x, y);
  
  // Triangulate polygon

  // Fill a vector of vertex indices
  vector_t<size_t> verticesToBeDone;
  for (size_t i=0; i<nvertices; ++i)
    vtx.push_back(i);
  
  int i1 = 0;
  int i2 = 1;
  int i3 = 2;

  while (vtx.size() > 2)
  {
    // skip concave vertices
    int counter = 0;
    while (!IsConvexSide(vtx[i1], vtx[i2], vtx[i3])) {
      i1++;
      i2++;
      i3 = (i3+1)%vtx.size();
      counter++;
      assert(counter < nvertices && "Triangulation failed");
    }
    bool good = true;
    for (auto i : vtx) {
      if (i == vtx[i1] || i == vtx[i2] || i == vtx[i3]) continue;
      if (IsPointInside(i, vtx[i1], vtx[i2], vtx[i3]) {
        good = false;
        i1++;
        i2++;
        i3 = (i3+1)%vtx.size();
        break;
      }      
    }
    if (good) {
      // Make triangle
      ...
      vtx.erase(vtx.begin() + i2);
      i1 = 0;
      i2 = 1;
      i3 = 2;
    }
  }
  ...
}
  
void UnplacedExtruded::Print() const
{
  std::cout << "UnplacedExtruded: vertices {";
  int nvert = fVertices.size();
  for (int i = 0; i < nvert-1; ++i)
    std::cout << "(" << fVertices[i].x << ", " << fVertices[i].y << "), ";
  std::cout << "(" << fVertices[nvert-1].x << ", " << fVertices[nvert-1].y << ")}\n";
  std::cout << "sections:\n";
  int nsect = fSections.size();
  for (int i = 0; i < nsect; ++i)
    std::cout << "orig: (" << fSections[i].fOrigin.x() << ", "
              << fSections[i].fOrigin.y() << ", " << fSections[i].fOrigin.z()
              <<  ") scl = " << fSections[i].fScale << std::endl;
  std::cout << "nuber of facets: " << fExtruded.fFacets.size() << std::endl;
}

void UnplacedExtruded::Print(std::ostream &os) const
{
  os << "UnplacedExtruded: vertices {";
  int nvert = fVertices.size();
  for (int i = 0; i < nvert-1; ++i)
    os << "(" << fVertices[i].x << ", " << fVertices[i].y << "), ";
  os << "(" << fVertices[nvert-1].x << ", " << fVertices[nvert-1].y << ")}\n";
  os << "sections:\n";
  int nsect = fSections.size();
  for (int i = 0; i < nsect; ++i)
    os << "orig: (" << fSections[i].fOrigin.x() << ", "
              << fSections[i].fOrigin.y() << ", " << fSections[i].fOrigin.z()
              <<  ") scl = " << fSections[i].fScale << std::endl;
  os << "nuber of facets: " << fExtruded.fFacets.size() << std::endl;
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
