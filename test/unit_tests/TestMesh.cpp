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

#ifdef VECGEOM_ROOT
#include "utilities/Visualizer.h"
#include "TPolyMarker3D.h"
#include "TPolyLine3D.h"
#endif

bool ValidXing(vecgeom::Vector3D<double> const &point, vecgeom::Vector3D<double> const &dir,
               vecgeom::Vector3D<double> const &n1, double p1, vecgeom::Vector3D<double> const &n2, double p2)
{
  bool valid_dir   = ApproxEqual(dir.Dot(n1), 0.) && ApproxEqual(dir.Dot(n2), 0.);
  bool valid_point = ApproxEqual(point.Dot(n1 - n2) + (p1 - p2), 0.);
  return valid_point & valid_dir;
}

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

  const Vec_t dirx(1., 0., 0.);
  const Vec_t diry(0., 1., 0.);
  const Vec_t dirz(0., 0., 1.);

#ifdef VECGEOM_ROOT
  OPTION_INT(vis, 0);
#endif

  ///* Test box crossings */
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

  /*

    SolidMesh *s1 = a1.CreateSolidMesh3D();
    SolidMesh *s2 = a2.CreateSolidMesh3D();

    const Polyhedron  &p1 = s1->getMesh();
    const Polyhedron  &p2 = s2->getMesh();
    */
  Utils3D::FillBoxPolyhedron(box1, polyh1);
  Utils3D::FillBoxPolyhedron(box2, polyh2);

  // UnplacedParallelepiped up = UnplacedParallelepiped(5,5,5, 3.14 / 2, 3.14/2, 3.14/4);
  // SolidMesh *s3 = new SolidMesh();
  // s3->createMeshFromParallelepiped(up.GetStruct());

  // Polyhedron &p3 = s3->getMesh();

  // s.addPolyGonalMesh

#ifdef VECGEOM_ROOT
  if (vis == 0) return 0;
  Visualizer visualizer;
  SimpleBox boxshape("box", 7, 7, 7);
  visualizer.AddVolume(boxshape);
  Utils3D::vector_t<Utils3D::Line> lines;
  DrawPolyhedron(ph_tetra, visualizer, kBlue);
  // DrawPolyhedron(polyh2, visualizer, kGreen);
  /*
  if (PolyhedronXing(polyh1, polyh2, lines) == Utils3D::kOverlapping) {
    TPolyLine3D pl(2);
    pl.SetLineColor(kRed);
    for (auto line : lines) {
      pl.SetNextPoint(line.fPts[0].x(), line.fPts[0].y(), line.fPts[0].z());
      pl.SetNextPoint(line.fPts[1].x(), line.fPts[1].y(), line.fPts[1].z());
      visualizer.AddLine(pl);
    }
  }
  */
  visualizer.Show();
#endif

  return 0;
}
