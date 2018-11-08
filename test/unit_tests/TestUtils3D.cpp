///
/// file:    TestUtils3D.cpp
/// purpose: Unit tests for the 3D geometry utilities
///

//-- ensure asserts are compiled in
#undef NDEBUG

#include "base/Utils3D.h"
#include "ApproxEqual.h"
#include "volumes/Box.h"

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
    pl.SetNextPoint(poly.fVert[i].x(), poly.fVert[i].y(), poly.fVert[i].z());
  pl.SetNextPoint(poly.fVert[0].x(), poly.fVert[0].y(), poly.fVert[0].z());
  visualizer.AddLine(pl);

  // Compute center of polygon
  Vec_t center;
  for (size_t i = 0; i < poly.fN; ++i)
    center += poly.fVert[i];
  center *= 1. / poly.fN;
  TPolyLine3D plnorm(2);
  plnorm.SetLineColor(color);
  plnorm.SetNextPoint(center[0], center[1], center[2]);
  plnorm.SetNextPoint(center[0] + poly.fNorm[0], center[1] + poly.fNorm[1], center[2] + poly.fNorm[2]);
  visualizer.AddLine(plnorm);
}

void DrawBox(vecgeom::Vector3D<double> const &box, vecgeom::Transformation3D const &tr, vecgeom::Visualizer &visualizer,
             size_t color)
{
  using namespace vecgeom;
  using Vec_t = Vector3D<double>;

  // compute transformed vertices
  const Vec_t mesh1[8] = {{-box[0], -box[1], -box[2]}, {-box[0], box[1], -box[2]}, {box[0], box[1], -box[2]},
                          {box[0], -box[1], -box[2]},  {-box[0], -box[1], box[2]}, {-box[0], box[1], box[2]},
                          {box[0], box[1], box[2]},    {box[0], -box[1], box[2]}};

  Vec_t mesh[8];
  for (auto i = 0; i < 8; ++i)
    tr.InverseTransform(mesh1[i], mesh[i]);

  // Generate lines
  TPolyLine3D pl(2);
  pl.SetLineColor(color);

  pl.SetPoint(0, mesh[0].x(), mesh[0].y(), mesh[0].z());
  pl.SetPoint(1, mesh[1].x(), mesh[1].y(), mesh[1].z());
  visualizer.AddLine(pl);
  pl.SetPoint(0, mesh[1].x(), mesh[1].y(), mesh[1].z());
  pl.SetPoint(1, mesh[2].x(), mesh[2].y(), mesh[2].z());
  visualizer.AddLine(pl);
  pl.SetPoint(0, mesh[2].x(), mesh[2].y(), mesh[2].z());
  pl.SetPoint(1, mesh[3].x(), mesh[3].y(), mesh[3].z());
  visualizer.AddLine(pl);
  pl.SetPoint(0, mesh[3].x(), mesh[3].y(), mesh[3].z());
  pl.SetPoint(1, mesh[0].x(), mesh[0].y(), mesh[0].z());
  visualizer.AddLine(pl);

  pl.SetPoint(0, mesh[4].x(), mesh[4].y(), mesh[4].z());
  pl.SetPoint(1, mesh[5].x(), mesh[5].y(), mesh[5].z());
  visualizer.AddLine(pl);
  pl.SetPoint(0, mesh[5].x(), mesh[5].y(), mesh[5].z());
  pl.SetPoint(1, mesh[6].x(), mesh[6].y(), mesh[6].z());
  visualizer.AddLine(pl);
  pl.SetPoint(0, mesh[6].x(), mesh[6].y(), mesh[6].z());
  pl.SetPoint(1, mesh[7].x(), mesh[7].y(), mesh[7].z());
  visualizer.AddLine(pl);
  pl.SetPoint(0, mesh[7].x(), mesh[7].y(), mesh[7].z());
  pl.SetPoint(1, mesh[4].x(), mesh[4].y(), mesh[4].z());
  visualizer.AddLine(pl);

  pl.SetPoint(0, mesh[0].x(), mesh[0].y(), mesh[0].z());
  pl.SetPoint(1, mesh[4].x(), mesh[4].y(), mesh[4].z());
  visualizer.AddLine(pl);
  pl.SetPoint(0, mesh[1].x(), mesh[1].y(), mesh[1].z());
  pl.SetPoint(1, mesh[5].x(), mesh[5].y(), mesh[5].z());
  visualizer.AddLine(pl);
  pl.SetPoint(0, mesh[2].x(), mesh[2].y(), mesh[2].z());
  pl.SetPoint(1, mesh[6].x(), mesh[6].y(), mesh[6].z());
  visualizer.AddLine(pl);
  pl.SetPoint(0, mesh[3].x(), mesh[3].y(), mesh[3].z());
  pl.SetPoint(1, mesh[7].x(), mesh[7].y(), mesh[7].z());
  visualizer.AddLine(pl);
}

#endif

int main(int argc, char *argv[])
{
  using namespace vecgeom;
  using namespace vecCore::math;
  using Vec_t = Vector3D<double>;

  const Vector3D<double> dirx(1., 0., 0.);
  const Vector3D<double> diry(0., 1., 0.);
  const Vector3D<double> dirz(0., 0., 1.);

#ifdef VECGEOM_ROOT
  Visualizer visualizer;
  SimpleBox boxshape("box", 10, 10, 10);
  visualizer.AddVolume(boxshape);
#endif

  ///* Plane transformations */
  Utils3D::HPlane pl1(Vec_t(1., 0., 0.), -10.);
  Utils3D::HPlane plrot;
  Vector3D<double> pmaster;
  Transformation3D transf1(10., 0., 0., 0., 0., 180.);
  Utils3D::TransformPlane(transf1, pl1, plrot);
  assert(ApproxEqual(plrot.fNorm[0], -1.) && ApproxEqual(plrot.fDist, 0.));

  // Polygon intersection
  Utils3D::Polygon poly1(4, true);
  poly1.AddVertex(0, 10., 4., 6.);
  poly1.AddVertex(1, 10., -4., 6.);
  poly1.AddVertex(2, 10., -4., -6.);
  poly1.AddVertex(3, 10., 4., -6.);
  poly1.Init();
  Utils3D::Polygon poly2(4, true);
  poly2.AddVertex(0, 7.75736, -3.17423, 10.5854);
  poly2.AddVertex(1, 7.75736, -7.17423, 3.65722);
  poly2.AddVertex(2, 16.2426, 0.174235, -0.585422);
  poly2.AddVertex(3, 16.2426, 4.17423, 6.34278);
  poly2.Init();

  Utils3D::Line line1;
  assert(Utils3D::PolygonXing(poly1, poly2, &line1) == Utils3D::kOverlapping);

#ifdef VECGEOM_ROOT_1
  DrawPolygon(poly1, visualizer, kBlue);
  DrawPolygon(poly2, visualizer, kGreen);
  TPolyLine3D pl(2);
  pl.SetNextPoint(line1.fPts[0].x(), line1.fPts[0].y(), line1.fPts[0].z());
  pl.SetNextPoint(line1.fPts[1].x(), line1.fPts[1].y(), line1.fPts[1].z());
  pl.SetLineColor(kRed);
  visualizer.AddLine(pl);
  visualizer.Show();
#endif

  ///* Test plane crossings */
  Vector3D<double> point, direction;
  Vector3D<double> n1, n2;
  double p1, p2;

  // identical planes
  n1.Set(0., 0., 1.);
  n2.Set(0., 0., 1.);
  p1 = -3.;
  p2 = -3.;
  assert(Utils3D::PlaneXing(n1, p1, n2, p2, point, direction) == Utils3D::kIdentical);

  // identical planes with opposite normals
  n1.Set(0., 0., 1.);
  n2.Set(0., 0., -1.);
  p1 = -3.;
  p2 = 3.;
  assert(Utils3D::PlaneXing(n1, p1, n2, p2, point, direction) == Utils3D::kIdentical);

  // opposite planes with opposite normals
  n1.Set(0., 0., 1.);
  n2.Set(0., 0., -1.);
  p1 = -3;
  p2 = -3;
  assert(Utils3D::PlaneXing(n1, p1, n2, p2, point, direction) == Utils3D::kParallel);

  // opposite planes with identical normal
  n1.Set(0., 0., 1.);
  n2.Set(0., 0., 1.);
  p1 = -3;
  p2 = 3;
  assert(Utils3D::PlaneXing(n1, p1, n2, p2, point, direction) == Utils3D::kParallel);

  // arbitrary parallel planes
  n1.Set(1., 2., 3.);
  n1.Normalize();
  n2 = -n1;
  p1 = 1;
  p2 = -2;
  assert(Utils3D::PlaneXing(n1, p1, n2, p2, point, direction) == Utils3D::kParallel);

  // +z face of a box with +x face of the same box
  n1.Set(0., 0., 1.);
  n2.Set(1., 0., 0.);
  p1 = -3;
  p2 = -2;
  assert(Utils3D::PlaneXing(n1, p1, n2, p2, point, direction) == Utils3D::kIntersecting);
  assert(ValidXing(point, direction, n1, p1, n2, p2));

  // same as above but 1 face has opposite normal
  n2 = -n2;
  p2 = -p2;
  assert(Utils3D::PlaneXing(n1, p1, n2, p2, point, direction) == Utils3D::kIntersecting);
  assert(ValidXing(point, direction, n1, p1, n2, p2));

  ///* Test box crossings */
  Vector3D<double> box1(1., 2., 3.);
  Vector3D<double> box2(2., 3., 4.);
  Transformation3D tr1, tr2, tr3;

  // Touching boxes
  tr1 = Transformation3D(0., 0., 0.);
  tr2 = Transformation3D(3., 5., 0.);
  assert(Utils3D::BoxCollision(box1, tr1, box2, tr2) == Utils3D::kTouching);

  // Disjoint boxes
  tr1 = Transformation3D(0., 0., 0.);
  tr2 = Transformation3D(2.5, 4.5, 10.2);
  assert(Utils3D::BoxCollision(box1, tr1, box2, tr2) == Utils3D::kDisjoint);

  // Overlapping boxes
  tr1 = Transformation3D(0., 0., 0.);
  tr2 = Transformation3D(2.5, 4.5, 6.5);
  assert(Utils3D::BoxCollision(box1, tr1, box2, tr2) == Utils3D::kOverlapping);

  tr1 = Transformation3D(-2.5, -0.5, 0.5);
  tr2 = Transformation3D(-3., -0.5, 0.5);
  tr3 = Transformation3D(1., 2., 3., 0., 45., 45.);
  assert(Utils3D::BoxCollision(box1, tr1, box2, tr3) == Utils3D::kOverlapping &&
         Utils3D::BoxCollision(box1, tr2, box2, tr3) == Utils3D::kDisjoint);

#ifdef VECGEOM_ROOT
  DrawBox(box1, tr1, visualizer, kBlue);
  DrawBox(box2, tr3, visualizer, kGreen);
  visualizer.Show();
#endif

  std::cout << "TestUtils3D passed\n";
  return 0;
}
