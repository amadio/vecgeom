///
/// file:    TestUtils3D.cpp
/// purpose: Unit tests for the 3D geometry utilities
///

//-- ensure asserts are compiled in
#undef NDEBUG

#include "base/Utils3D.h"
#include "ApproxEqual.h"

bool ValidXing(vecgeom::Vector3D<double> const &point, vecgeom::Vector3D<double> const &dir,
               vecgeom::Vector3D<double> const &n1, double p1, vecgeom::Vector3D<double> const &n2, double p2)
{
  bool valid_dir   = ApproxEqual(dir.Dot(n1), 0.) && ApproxEqual(dir.Dot(n2), 0.);
  bool valid_point = ApproxEqual(point.Dot(n1 - n2) + (p1 - p2), 0.);
  return valid_point & valid_dir;
}

int main(int argc, char *argv[])
{
  using namespace vecgeom;
  using namespace vecCore::math;
  using Vec_t = Vector3D<double>;

  const Vector3D<double> dirx(1., 0., 0.);
  const Vector3D<double> diry(0., 1., 0.);
  const Vector3D<double> dirz(0., 0., 1.);

  ///* Plane transformations */
  Utils3D::HPlane pl1(Vec_t(1., 0., 0.), -10.);
  Utils3D::HPlane plrot;
  Vector3D<double> pmaster;
  Transformation3D transf1(10., 0., 0., 0., 0., 180.);
  Utils3D::TransformPlane(transf1, pl1, plrot);
  assert(ApproxEqual(plrot.fNorm[0], -1.) && ApproxEqual(plrot.fDist, 0.));

  // Rectangle transformations
  Utils3D::HRectangle rect1(2., 3., -10., Vec_t(1., 0., 0.), Vec_t(10., 0., 0.), Vec_t(0., 0., 1.));
  Utils3D::HRectangle rect2;
  Transformation3D transf2(10., 0., 0., 0., 90., 0.);
  Utils3D::TransformRectangle(transf2, rect1, rect2);
  assert(ApproxEqual(rect2.fCenter[0], 20.) && ApproxEqual(rect2.fNorm[0], 1.) && ApproxEqual(rect2.fUpVect[1], -1.) &&
         ApproxEqual(rect2.fDist, -20.));
  // std::cout << rect2 << std::endl;

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
  Transformation3D tr1, tr2;

  // Touching boxes
  tr1 = Transformation3D(0., 0., 0.);
  tr2 = Transformation3D(3., 5., 0.);
  assert(Utils3D::BoxXing(box1, tr1, box2, tr2) == Utils3D::kTouching);

  // Disjoint boxes
  tr1 = Transformation3D(0., 0., 0.);
  tr2 = Transformation3D(2.5, 4.5, 10.2);
  assert(Utils3D::BoxXing(box1, tr1, box2, tr2) == Utils3D::kDisjoint);

  // Overlapping boxes
  tr1 = Transformation3D(0., 0., 0.);
  tr2 = Transformation3D(2.5, 4.5, 6.5);
  assert(Utils3D::BoxXing(box1, tr1, box2, tr2) == Utils3D::kOverlapping);

  std::cout << "TestUtils3D passed\n";
  return 0;
}
