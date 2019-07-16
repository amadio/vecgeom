///
/// file:    TestUtils3D.cpp
/// purpose: Unit tests for the 3D geometry utilities
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
  //OPTION_BOOL(t, false);
#endif

  VUnplacedVolume *unplacedvolume;

#ifdef VECGEOM_ROOT
#define WORLDSIZE 6

  if (!v.compare("noVolume")) {
    std::cout << "\nUsage:\n"
                 "./TestMesh -v [str] -t [bool]\n"
                 "\n"
                 "Available volumes: \"box\", \"parallelepiped\", \"sextruvolume\", \"tet\", \"trapezoid\", "
                 "\"trd\".\nUse -t to apply a random transformation.\n"
                 "\n"
                 "\n"
                 "";
    return 0;
  } else if (!v.compare("box")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedBox>(3., 3., 3.);
  } else if (!v.compare("parallelepiped")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedParallelepiped>(2., 2., 3., 90., 90., 90.);
  } else if (!v.compare("sextruvolume")) {
#define N 12
    double dx = 5;
    double dy = 5;
    double dz = 5;

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
  }

  Visualizer visualizer;
  SimpleBox boxshape("box", WORLDSIZE, WORLDSIZE, WORLDSIZE);
  visualizer.AddVolume(boxshape);
  DrawPolyhedron(unplacedvolume->CreateMesh3D(Transformation3D(), -1)->GetMesh(), visualizer, kBlue);
  visualizer.Show();

#endif

  ///* Test box crossings */

  /*
  Vec_t box1(1., 2., 3.);
  Vec_t box2(1., 1., 1.);
  Polyhedron polyh1, polyh2;

  UnplacedBox a1 = UnplacedBox(box1);
  UnplacedBox a2 = UnplacedBox(box2);

  SolidMesh *sm1       = a1.CreateMesh3D();
  const Polyhedron &p = sm1->GetMesh();

  UnplacedParallelepiped up = UnplacedParallelepiped(1, 2, 3, 3.14 / 2, 3.14 / 2, 3.14 / 4);
  SolidMesh *smup           = up.CreateMesh3D();
  const Polyhedron &pup     = smup->GetMesh();

  UnplacedTrapezoid ut  = UnplacedTrapezoid(1., 2., 3., 4.);
  SolidMesh *stup       = ut.CreateMesh3D();
  const Polyhedron &put = stup->GetMesh();

  Vec_t p0(0., 0., 5.), p1(0., 0., 0.), p2(5., 0., 0.), p3(0., 5., 0.);
  UnplacedTet tetrahedron = UnplacedTet(p0, p1, p2, p3);
  SolidMesh * sm_tetrahedron = tetrahedron.CreateMesh3D();
  const Polyhedron &ph_tetra = sm_tetrahedron->GetMesh();


    auto unplacedtrd = GeoManager::MakeInstance<UnplacedTrd>(1.,3.,2.,5.);
    SolidMesh* unplaced_trd_sm= unplacedtrd->CreateMesh3D();
    const Polyhedron &unplaced_trd_mesh = unplaced_trd_sm->GetMesh();









*/

  /*

    SolidMesh *s1 = a1.CreateSolidMesh3D();
    SolidMesh *s2 = a2.CreateSolidMesh3D();

    const Polyhedron  &p1 = s1->getMesh();
    const Polyhedron  &p2 = s2->getMesh();
    */
  // Utils3D::FillBoxPolyhedron(box1, polyh1);
  // Utils3D::FillBoxPolyhedron(box2, polyh2);

  // UnplacedParallelepiped up = UnplacedParallelepiped(5,5,5, 3.14 / 2, 3.14/2, 3.14/4);
  // SolidMesh *s3 = new SolidMesh();
  // s3->createMeshFromParallelepiped(up.GetStruct());

  // Polyhedron &p3 = s3->getMesh();

  // s.addPolyGonalMesh

  /*
#ifdef VECGEOM_ROOT

  Visualizer visualizer;
  SimpleBox boxshape("box", 17, 17, 17);
  visualizer.AddVolume(boxshape);
  Utils3D::vector_t<Utils3D::Line> lines;
  //DrawPolyhedron(polyhedron, visualizer, kBlue);
  // DrawPolyhedron(polyh2, visualizer, kGreen);

  if (PolyhedronXing(polyh1, polyh2, lines) == Utils3D::kOverlapping) {
    TPolyLine3D pl(2);
    pl.SetLineColor(kRed);
    for (auto line : lines) {
      pl.SetNextPoint(line.fPts[0].x(), line.fPts[0].y(), line.fPts[0].z());
      pl.SetNextPoint(line.fPts[1].x(), line.fPts[1].y(), line.fPts[1].z());
      visualizer.AddLine(pl);
    }
  }

  visualizer.Show();
#endif
  */

  return 0;
}
