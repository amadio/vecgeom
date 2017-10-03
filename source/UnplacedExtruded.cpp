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




   // Decompose polygonal sides in triangular facets

  typedef std::pair < UVector2, int > Vertex;

  // Fill one more vector
  //
  std::vector< Vertex > verticesToBeDone;
  for (int i = 0; i < fNv; ++i)
  {
    verticesToBeDone.push_back(Vertex(fPolygon[i], i));
  }
  std::vector< Vertex > ears;

  std::vector< Vertex >::iterator c1 = verticesToBeDone.begin();
  std::vector< Vertex >::iterator c2 = c1 + 1;
  std::vector< Vertex >::iterator c3 = c1 + 2;
  while (verticesToBeDone.size() > 2)
  {
    // skip concave vertices
    //
    double angle = GetAngle(c2->first, c3->first, c1->first);

    int counter = 0;
    while (angle >= UUtils::kPi)
    {
      // try next three consecutive vertices
      //
      c1 = c2;
      c2 = c3;
      ++c3;
      if (c3 == verticesToBeDone.end())
      {
        c3 = verticesToBeDone.begin();
      }

      angle = GetAngle(c2->first, c3->first, c1->first);

      counter++;

      if (counter > fNv)
      {
        UUtils::Exception("UExtrudedSolid::AddGeneralPolygonFacets",
                          "GeomSolids0003", UFatalError, 1,
                          "Triangularisation has failed.");
        break;
      }
    }

    bool good = true;
    std::vector< Vertex >::iterator it;
    for (it = verticesToBeDone.begin(); it != verticesToBeDone.end(); ++it)
    {
      // skip vertices of tested triangle
      //
      if (it == c1 || it == c2 || it == c3)
      {
        continue;
      }

      if (IsPointInside(c1->first, c2->first, c3->first, it->first))
      {
        good = false;

        // try next three consecutive vertices
        //
        c1 = c2;
        c2 = c3;
        ++c3;
        if (c3 == verticesToBeDone.end())
        {
          c3 = verticesToBeDone.begin();
        }
        break;
      }
    }
    if (good)
    {
      // all points are outside triangle, we can make a facet

      bool result;
      result = AddFacet(MakeDownFacet(c1->second, c2->second, c3->second));
      if (! result)
      {
        return false;
      }

      result = AddFacet(MakeUpFacet(c1->second, c2->second, c3->second));
      if (! result)
      {
        return false;
      }

      std::vector<int> triangle(3);
      triangle[0] = c1->second;
      triangle[1] = c2->second;
      triangle[2] = c3->second;
      fTriangles.push_back(triangle);

      // remove the ear point from verticesToBeDone
      //
      verticesToBeDone.erase(c2);
      c1 = verticesToBeDone.begin();
      c2 = c1 + 1;
      c3 = c1 + 2;
    }
  }
  return true;
 
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
