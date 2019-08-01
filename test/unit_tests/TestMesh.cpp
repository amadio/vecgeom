///
/// file:    TestMesh.cpp
/// purpose: Unit tests for the meshes of 3D models.
///

//-- ensure asserts are compiled in
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "base/Utils3D.h"
#include "ApproxEqual.h"
#include "volumes/Box.h"
#include "test/benchmark/ArgParser.h"

#include "volumes/SolidMesh.h"
#include "volumes/UnplacedParallelepiped.h"
#include "volumes/UnplacedTrapezoid.h"
#include "volumes/UnplacedTet.h"
#include "volumes/UnplacedTrd.h"
#include "management/GeoManager.h"
#include "volumes/UnplacedSExtruVolume.h"
#include "volumes/UnplacedEllipticalTube.h"
#include "volumes/UnplacedEllipticalCone.h"
#include "volumes/UnplacedOrb.h"
#include "volumes/UnplacedParaboloid.h"
#include "volumes/UnplacedCone.h"
#include "volumes/UnplacedParaboloid.h"
#include "volumes/UnplacedPolycone.h"

#ifdef VECGEOM_ROOT
#include "utilities/Visualizer.h"
#include "TPolyMarker3D.h"
#include "TPolyLine3D.h"
#endif

#ifdef VECGEOM_ROOT
void DrawPolygon(vecgeom::Utils3D::Polygon const &poly, vecgeom::Visualizer &visualizer, size_t color)
{
  using namespace vecgeom;
  using Vec_t = Vector3D<double>;
  TPolyLine3D pl(poly.fN + 1);
  pl.SetLineColor(color);
  for (size_t i = 0; i < poly.fN; ++i)
    pl.SetNextPoint(poly.GetVertex(i).x(), poly.GetVertex(i).y(), poly.GetVertex(i).z());
  pl.SetNextPoint(poly.GetVertex(0).x(), poly.GetVertex(0).y(), poly.GetVertex(0).z());
  visualizer.AddLine(pl);

  // Compute center of polygon
  Vec_t center;
  for (size_t i = 0; i < poly.fN; ++i)
    center += poly.GetVertex(i);
  center *= 1. / poly.fN;
  TPolyLine3D plnorm(2);
  plnorm.SetLineColor(color);
  plnorm.SetNextPoint(center[0], center[1], center[2]);
  plnorm.SetNextPoint(center[0] + poly.fNorm[0], center[1] + poly.fNorm[1], center[2] + poly.fNorm[2]);
  visualizer.AddLine(plnorm);
}

void DrawPolyhedron(const vecgeom::Utils3D::Polyhedron &polyh, vecgeom::Visualizer &visualizer, size_t color)
{
  using namespace vecgeom;

  for (size_t i = 0; i < polyh.GetNpolygons(); ++i)
    DrawPolygon(polyh.GetPolygon(i), visualizer, color);
}
#endif

int main(int argc, char *argv[])
{
  using namespace vecgeom;
  using namespace vecCore::math;
  using Vec_t = Vector3D<double>;

  using vecgeom::Utils3D::Line;
  using vecgeom::Utils3D::Plane;
  using vecgeom::Utils3D::Polygon;
  using vecgeom::Utils3D::Polyhedron;

#ifdef VECGEOM_ROOT
  OPTION_STRING(v, "noVolume");
  // OPTION_BOOL(t, false);
#endif

  VUnplacedVolume *unplacedvolume = nullptr;

#ifdef VECGEOM_ROOT
#define WORLDSIZE 10

  if (!v.compare("noVolume")) {
    std::cout << "\nUsage:\n"
                 "./TestMesh -v [str] -t [bool]\n"
                 "\n"
                 "Available volumes: \"box\", \"parallelepiped\", \"sextruvolume\", \"tet\", \"trapezoid\", "
                 "\"trd\", \"ellipticaltube\", \"ellipticalcone\", \"orb\".\nUse -t to apply a random transformation.\n"
                 "\n"
                 "\n"
                 "";
    return 0;
  } else if (!v.compare("box")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedBox>(3., 3., 3.);
  } else if (!v.compare("parallelepiped")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedParallelepiped>(2., 2., 3., 90., 90., 90.);
  } else if (!v.compare("sextruvolume")) {
#define N 10
    double dx = 5;
    double dy = 5;
    double dz = 3;

    double x[N], y[N];
    for (size_t i = 0; i < (size_t)N; ++i) {
      x[i] = dx * std::sin(i * (2. * M_PI) / N);
      y[i] = dy * std::cos(i * (2. * M_PI) / N);
    }
    unplacedvolume = GeoManager::MakeInstance<UnplacedSExtruVolume>(N, x, y, -dz, dz);
  } else if (!v.compare("trapezoid")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedTrapezoid>(5., 0., 0., 3., 4., 5., 0., 3., 4., 5., 0.);
  } else if (!v.compare("trd")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedTrd>(2., 3., 2., 5.);
  } else if (!v.compare("tet")) {
    Vec_t p0(0., 0., 5.), p1(-5., -5., -5.), p2(5., -5., -5.), p3(-5., 5., -5.);
    unplacedvolume = GeoManager::MakeInstance<UnplacedTet>(p0, p1, p2, p3);
  } else if (!v.compare("ellipticaltube")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedEllipticalTube>(2, 4, 5);
  } else if (!v.compare("ellipticalcone")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedEllipticalCone>(1., 1., 5., 3.);
  } else if (!v.compare("orb")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedOrb>(8.);
  } else if (!v.compare("cone")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedCone>(3., 5, 2., 4., 5, 0, kPi);
  } else if (!v.compare("paraboloid")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedParaboloid>(0., 7., 5.);
  } else if (!v.compare("polycone")) {
    int Nz         = 4;
    double rmin[]  = {0., 3., 7., 8};
    double rmax[]  = {5., 4., 8., 9};
    double z[]     = {-3, 0, 3, 8};
    unplacedvolume = GeoManager::MakeInstance<UnplacedPolycone>(0, kPi, Nz, z, rmin, rmax);
  }

  Visualizer visualizer;
  SimpleBox boxshape("box", WORLDSIZE, WORLDSIZE, WORLDSIZE);
  visualizer.AddVolume(boxshape);
  DrawPolyhedron(unplacedvolume->CreateMesh3D(Transformation3D(), 100)->GetMesh(), visualizer, kBlue);
  visualizer.Show();

#endif

  return 0;
}
