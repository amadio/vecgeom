/// @file UnplacedExtruded.cpp
/// @author Mihaela Gheata (mihaela.gheata@cern.ch)

#include "volumes/Tessellated.h"
#include "volumes/UnplacedExtruded.h"
#include "volumes/UnplacedSExtruVolume.h"
#include "volumes/SpecializedExtruded.h"
#include "volumes/utilities/GenerationUtilities.h"
#include "base/RNG.h"

#include "management/VolumeFactory.h"

#ifndef VECCORE_CUDA
#include "volumes/UnplacedImplAs.h"
#endif

#ifndef VECCORE_CUDA
#ifdef VECGEOM_ROOT
#include "TGeoXtru.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4ExtrudedSolid.hh"
#include "G4TessellatedSolid.hh"
#include "G4TriangularFacet.hh"
#endif
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA
#ifdef VECGEOM_ROOT
TGeoShape const *UnplacedExtruded::ConvertToRoot(char const *label) const
{
  size_t nvert = GetNVertices();
  size_t nsect = GetNSections();

  // if(nsect > 1){
  double *x = new double[nvert];
  double *y = new double[nvert];
  for (size_t i = 0; i < nvert; ++i) {
    GetVertex(i, x[i], y[i]);
  }
  TGeoXtru *xtru = new TGeoXtru(nsect);
  xtru->DefinePolygon(nvert, x, y);
  for (size_t i = 0; i < nsect; ++i) {
    XtruSection sect = GetSection(i);
    xtru->DefineSection(i, sect.fOrigin.z(), sect.fOrigin.x(), sect.fOrigin.y(), sect.fScale);
  }
  return xtru;
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *UnplacedExtruded::ConvertToGeant4(char const *label) const
{
  std::vector<G4TwoVector> polygon;
  double x, y;
  size_t nvert = GetNVertices();
  for (size_t i = 0; i < nvert; ++i) {
    GetVertex(i, x, y);
    polygon.push_back(G4TwoVector(x, y));
  }
  std::vector<G4ExtrudedSolid::ZSection> sections;
  size_t nsect = GetNSections();
  for (size_t i = 0; i < nsect; ++i) {
    XtruSection sect = GetSection(i);
    sections.push_back(
        G4ExtrudedSolid::ZSection(sect.fOrigin.z(), G4TwoVector(sect.fOrigin.x(), sect.fOrigin.y()), sect.fScale));
  }
  G4ExtrudedSolid *g4xtru = new G4ExtrudedSolid(label, polygon, sections);
  return g4xtru;
}
#endif
#endif

template <>
UnplacedExtruded *Maker<UnplacedExtruded>::MakeInstance(const size_t nvertices, XtruVertex2 const *vertices,
                                                        const int nsections, XtruSection const *sections)
{

#ifndef VECGEOM_NO_SPECIALIZATION
  bool isSExtru = false;
  for (int i = 0; i < (nsections - 1); i++) {
    if (i == 0) {
      isSExtru = ((sections[i].fOrigin - sections[i + 1].fOrigin).Perp2() < kTolerance &&
                  vecCore::math::Abs(sections[i].fScale - sections[i + 1].fScale) < kTolerance);
    } else {
      isSExtru &= ((sections[i].fOrigin - sections[i + 1].fOrigin).Perp2() < kTolerance &&
                   vecCore::math::Abs(sections[i].fScale - sections[i + 1].fScale) < kTolerance);
    }
    if (!isSExtru) break;
  }
  if (isSExtru) {
    double *x = new double[nvertices];
    double *y = new double[nvertices];
    for (size_t i = 0; i < nvertices; ++i) {
      x[i] = vertices[i].x;
      y[i] = vertices[i].y;
    }
    Precision zmin = sections[0].fOrigin.z();
    Precision zmax = sections[nsections - 1].fOrigin.z();
    return new SUnplacedImplAs<UnplacedExtruded, UnplacedSExtruVolume>(nvertices, x, y, zmin, zmax);
  } else {
    return new UnplacedExtruded(nvertices, vertices, nsections, sections);
  }
#else
  return new UnplacedExtruded(nvertices, vertices, nsections, sections);
#endif
}

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
  int surface  = ChooseSurface();
  Precision r1 = RNG::Instance().uniform(0.0, 1.0);
  Precision r2 = RNG::Instance().uniform(0.0, 1.0);
  if (r1 + r2 > 1.) {
    r1 = 1. - r1;
    r2 = 1. - r2;
  }
  auto facet = fXtru.fTslHelper.fFacets[surface];
  return (facet->fVertices[0] + r1 * (facet->fVertices[1] - facet->fVertices[0]) +
          r2 * (facet->fVertices[2] - facet->fVertices[0]));
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


#ifndef VECCORE_CUDA
SolidMesh *UnplacedExtruded::CreateMesh3D(Transformation3D const &trans, size_t nSegments) const
{

  typedef Vector3D<double> Vec_t;

  SolidMesh *sm = new SolidMesh();

  size_t n      = GetNVertices();
  size_t nSections = GetNSections();


  Vec_t *vertices = new Vec_t[nSections * (n + 1)];
  size_t idx = 0;
  for(size_t i = 0; i < nSections; i++){
	  for(size_t j = n; j > 0; j--){
		  vertices[idx++] = GetStruct().VertexToSection(j - 1, i);
	  }
	  vertices[idx++] = GetStruct().VertexToSection(n-1, i);
  }

  sm->ResetMesh(nSections * (n + 1), (nSections - 1) * (n) + 2);
  sm->SetVertices(vertices, nSections * (n + 1));
  delete[] vertices;
  sm->TransformVertices(trans);



  std::vector<size_t> indices;
  for(size_t i = n; i > 0; i--){
	  indices.push_back(i - 1);
  }
  sm->AddPolygon(n, indices, GetStruct().IsConvexPolygon());

  indices.clear();

  for(size_t i = 0, k = (nSections - 1)*(n+1); i < n; i++,k++ ){
	  indices.push_back(k);
  }

  sm->AddPolygon(n, indices, GetStruct().IsConvexPolygon());



  size_t k = 0;
  for(size_t i = 0; i < nSections - 1; i++, k++){
	  for (size_t j = 0; j < n; j++, k++) {
		  sm->AddPolygon(4, {k,k + 1,k + 1 + n + 1, k + n + 1}, true);
	  }
  }



  /*
  double k      = std::cos(0.5 * GetPhiDelta() / nSides); // divide R_side by k to get the R_corner

  Vec_t *vertices = new Vec_t[2 * (n + 1) * (nSides + 1)];
  size_t idx      = 0;
  size_t idx2     = ((n + 1) * (nSides + 1));
  for (size_t i = 0; i <= n; i++) {
    for (size_t j = 0; j <= nSides; j++) {
      vertices[idx++]  = GetPhiSection(j) * GetRMax()[i] / k + Vec_t(0, 0, GetZPlane(i));
      vertices[idx2++] = GetPhiSection(j) * GetRMin()[i] / k + Vec_t(0, 0, GetZPlane(i));
    }
  }

  sm->ResetMesh(2 * (n + 1) * (nSides + 1), 2 * n * nSides + 2 * nSides + 2 * n);

  sm->SetVertices(vertices, 2 * (n + 1) * (nSides + 1));
  delete[] vertices;
  sm->TransformVertices(trans);

  for (size_t i = 0, p = 0, k = nSides + 1, offset = (n + 1) * (nSides + 1); i < n; i++, p++, k++) {
    for (size_t j = 0; j < nSides; j++, p++, k++) {
      sm->AddPolygon(4, {p, p + 1, k + 1, k}, true);                                     // outer
      sm->AddPolygon(4, {k + offset, k + 1 + offset, p + offset + 1, p + offset}, true); // inner
    }
  }

  for (size_t i = 0, l = n * (nSides + 1), k = l + nSides + 1, p = k + l; i < nSides; i++, k++, l++, p++) {
    sm->AddPolygon(4, {k, k + 1, i + 1, i}, true); // lower
    sm->AddPolygon(4, {l, l + 1, p + 1, p}, true); // upper
  }

  if (GetPhiDelta() != kTwoPi) {
    for (size_t i = 0, j = 0, k = nSides + 1, offset = (n + 1) * (nSides + 1); i < n;
         i++, j += nSides + 1, k += nSides + 1) {
      sm->AddPolygon(4, {j, k, k + offset, j + offset}, true); // surface at sPhi
      sm->AddPolygon(4, {j + offset + nSides, k + offset + nSides, k + nSides, j + nSides},
                     true); // surface at sPhi + dPhi
    }
  }
  sm->InitPolygons();
*/
  return sm;
}
#endif

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

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedExtruded>::SizeOf();
template void DevicePtr<cuda::UnplacedExtruded>::Construct() const;

} // namespace cxx

#endif

} // namespace vecgeom
