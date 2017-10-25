/// @file PlacedExtruded.cpp
/// @author Mihaela Gheata (mihaela.gheata@cern.ch)

#include "volumes/Extruded.h"

#ifndef VECCORE_CUDA
#ifdef VECGEOM_USOLIDS
#include "UExtrudedSolid.hh"
#endif

#ifdef VECGEOM_ROOT
#include "TGeoXtru.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4ExtrudedSolid.hh"
#include "G4TessellatedSolid.hh"
#include "G4TriangularFacet.hh"
#endif

#endif // VECCORE_CUDA

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA

VPlacedVolume const *PlacedExtruded::ConvertToUnspecialized() const
{
  return new SimpleExtruded(GetLabel().c_str(), GetLogicalVolume(), GetTransformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const *PlacedExtruded::ConvertToRoot() const
{
  size_t nvert = GetUnplacedVolume()->GetNVertices();
  size_t nsect = GetUnplacedVolume()->GetNSections();
  double *x    = new double[nvert];
  double *y    = new double[nvert];
  for (size_t i = 0; i < nvert; ++i) {
    GetUnplacedVolume()->GetVertex(i, x[i], y[i]);
  }
  TGeoXtru *xtru = new TGeoXtru(nsect);
  xtru->DefinePolygon(nvert, x, y);
  for (size_t i = 0; i < nsect; ++i) {
    XtruSection sect = GetUnplacedVolume()->GetSection(i);
    xtru->DefineSection(i, sect.fOrigin.z(), sect.fOrigin.x(), sect.fOrigin.y(), sect.fScale);
  }
  return xtru;
}
#endif

#if defined(VECGEOM_USOLIDS) && !defined(VECGEOM_REPLACE_USOLIDS)
::VUSolid const *PlacedExtruded::ConvertToUSolids() const
{
  std::vector<UVector2> polygon;
  double x, y;
  size_t nvert = GetUnplacedVolume()->GetNVertices();
  for (size_t i = 0; i < nvert; ++i) {
    GetUnplacedVolume()->GetVertex(i, x, y);
    polygon.push_back(UVector2(x, y));
  }
  std::vector<UExtrudedSolid::ZSection> sections;
  size_t nsect = GetUnplacedVolume()->GetNSections();
  for (size_t i = 0; i < nsect; ++i) {
    XtruSection sect = GetUnplacedVolume()->GetSection(i);
    sections.push_back(
        UExtrudedSolid::ZSection(sect.fOrigin.z(), UVector2(sect.fOrigin.x(), sect.fOrigin.y()), sect.fScale));
  }
  UExtrudedSolid *uxtru = new UExtrudedSolid("", polygon, sections);
  return uxtru;
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *PlacedExtruded::ConvertToGeant4() const
{
  std::vector<G4TwoVector> polygon;
  double x, y;
  size_t nvert = GetUnplacedVolume()->GetNVertices();
  for (size_t i = 0; i < nvert; ++i) {
    GetUnplacedVolume()->GetVertex(i, x, y);
    polygon.push_back(G4TwoVector(x, y));
  }
  std::vector<G4ExtrudedSolid::ZSection> sections;
  size_t nsect = GetUnplacedVolume()->GetNSections();
  for (size_t i = 0; i < nsect; ++i) {
    XtruSection sect = GetUnplacedVolume()->GetSection(i);
    sections.push_back(
        G4ExtrudedSolid::ZSection(sect.fOrigin.z(), G4TwoVector(sect.fOrigin.x(), sect.fOrigin.y()), sect.fScale));
  }
  G4ExtrudedSolid *g4xtru = new G4ExtrudedSolid("", polygon, sections);
  return g4xtru;
}
#endif

#endif // VECCORE_CUDA

} // End impl namespace

#ifdef VECCORE_CUDA

VECGEOM_DEVICE_INST_PLACED_VOLUME_ALLSPEC(SpecializedExtruded)

#endif

} // End global namespace
