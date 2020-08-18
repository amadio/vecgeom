#include "VecGeom/volumes/PlanarPolygon.h"
#include "VecGeom/volumes/PolygonalShell.h"
#include "VecGeom/volumes/SExtru.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/volumes/UnplacedSExtruVolume.h"

#ifdef VECGEOM_ROOT
#include "TGeoPolygon.h"
#include "TGeoXtru.h"
#include "TGeoBBox.h"
#endif

#include <iostream>
#include <cmath>
#include <chrono>

using namespace vecgeom;
using namespace std::chrono;

__attribute__((noinline)) void TimeVecGeom(PlanarPolygon const &poly, SOA3D<Precision> const &container, int &count)
{
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  const auto S                         = container.size();
  for (size_t i = 0; i < S; ++i) {
    if (poly.Contains(container[i])) count++;
  }
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span           = duration_cast<duration<double>>(t2 - t1);
  std::cout << "It took me " << time_span.count() << " seconds.\n";
}

__attribute__((noinline)) void TimeVecGeomSafety(PlanarPolygon const &poly, SOA3D<Precision> const &container,
                                                 double &totalsafety)
{
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  const auto S                         = container.size();
  for (size_t i = 0; i < S; ++i) {
    int segmentid;
    totalsafety += std::sqrt(poly.SafetySqr(container[i], segmentid));
  }
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span           = duration_cast<duration<double>>(t2 - t1);
  std::cout << "It took me " << time_span.count() << " seconds.\n";
}

__attribute__((noinline)) void TimeVecGeomDistanceToIn(UnplacedSExtruVolume const &poly,
                                                       SOA3D<Precision> const &pointcontainer,
                                                       SOA3D<Precision> const &dircontainer, int &hitcount)
{
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  const auto S                         = pointcontainer.size();
  for (size_t i = 0; i < S; ++i) {
    auto d = poly.DistanceToIn(pointcontainer[i], dircontainer[i]);
    //  std::cerr << "VECGEOM D " << pointcontainer[i] << " " << dircontainer[i] << "\n";
    // std::cerr << "VECGEOM D " << i << " " << d << "\n";
    if (d < 1E30) hitcount++;
  }
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span           = duration_cast<duration<double>>(t2 - t1);
  std::cout << "It took me " << time_span.count() << " seconds.\n";
}

__attribute__((noinline)) void TimeVecGeomDistanceToOut(UnplacedSExtruVolume const &poly,
                                                        SOA3D<Precision> const &pointcontainer,
                                                        SOA3D<Precision> const &dircontainer, double &totaldistout)
{
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  const auto S                         = pointcontainer.size();
  for (size_t i = 0; i < S; ++i) {
    auto d = poly.DistanceToOut(pointcontainer[i], dircontainer[i]);
    // std::cerr << "VG DO " << i << " " << d << "\n";
    totaldistout += d;
  }
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span           = duration_cast<duration<double>>(t2 - t1);
  std::cout << "It took me " << time_span.count() << " seconds.\n";
}

#ifdef VECGEOM_ROOT
__attribute__((noinline)) void TimeTGeo(TGeoPolygon const &poly, SOA3D<Precision> const &container, int &count)
{
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  const auto S                         = container.size();
  for (size_t i = 0; i < S; ++i) {
    auto p      = container[i];
    double x[3] = {p.x(), p.y(), p.z()};
    if (poly.Contains(x)) count++;
  }
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span           = duration_cast<duration<double>>(t2 - t1);
  std::cout << "It took me " << time_span.count() << " seconds.\n";
}

__attribute__((noinline)) void TimeTGeoDistanceToIn(TGeoXtru const &xtru, SOA3D<Precision> const &pointcontainer,
                                                    SOA3D<Precision> const &dircontainer, int &count)
{
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  const auto S                         = pointcontainer.size();
  for (size_t i = 0; i < S; ++i) {
    auto p      = pointcontainer[i];
    auto dir    = dircontainer[i];
    double x[3] = {p.x(), p.y(), p.z()};
    double d[3] = {dir.x(), dir.y(), dir.z()};
    auto dist   = xtru.TGeoXtru::DistFromOutside(x, d);
    // std::cerr << "ROOT D " << i << " " << dist << "\n";
    if (dist < 1E30) count++;
  }
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span           = duration_cast<duration<double>>(t2 - t1);
  std::cout << "It took me " << time_span.count() << " seconds.\n";
}

__attribute__((noinline)) void TimeTGeoDistanceToOut(TGeoXtru const &xtru, SOA3D<Precision> const &pointcontainer,
                                                     SOA3D<Precision> const &dircontainer, double &totaldistout)
{
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  const auto S                         = pointcontainer.size();
  for (size_t i = 0; i < S; ++i) {
    auto p      = pointcontainer[i];
    auto dir    = dircontainer[i];
    double x[3] = {p.x(), p.y(), p.z()};
    double d[3] = {dir.x(), dir.y(), dir.z()};
    auto dist   = xtru.TGeoXtru::DistFromInside(x, d);
    // std::cerr << "ROOT DO " << i << " " << dist << "\n";
    totaldistout += dist;
  }
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span           = duration_cast<duration<double>>(t2 - t1);
  std::cout << "It took me " << time_span.count() << " seconds.\n";
}

__attribute__((noinline)) void TimeTGeoDistanceToIn(TGeoBBox const &box, SOA3D<Precision> const &pointcontainer,
                                                    SOA3D<Precision> const &dircontainer, int &count)
{
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  const auto S                         = pointcontainer.size();
  for (size_t i = 0; i < S; ++i) {
    auto p      = pointcontainer[i];
    auto dir    = dircontainer[i];
    double x[3] = {p.x(), p.y(), p.z()};
    double d[3] = {dir.x(), dir.y(), dir.z()};
    auto dist   = box.TGeoBBox::DistFromOutside(x, d);
    // std::cerr << "ROOTB D " << i << " " << dist << "\n";
    if (dist < 1E30) count++;
  }
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span           = duration_cast<duration<double>>(t2 - t1);
  std::cout << "It took me " << time_span.count() << " seconds.\n";
}

__attribute__((noinline)) void TimeTGeoSafety(TGeoPolygon const &poly, SOA3D<Precision> const &container,
                                              double &totalsafety)
{
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  const auto S                         = container.size();
  for (size_t i = 0; i < S; ++i) {
    int segmentid;
    const auto p = container[i];
    // const Precision x[3] = {p.x(),p.y(),p.z()};
    totalsafety += poly.Safety(&Vector3D<double>(p)[0], segmentid);
  }
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span           = duration_cast<duration<double>>(t2 - t1);
  std::cout << "It took me " << time_span.count() << " seconds.\n";
}
#endif

int main()
{
  //  const int N = 100;
  //  Precision x[N], y[N];

  ////  for (size_t i = 0; i < N; ++i) {
  ////    x[i] = std::sin(i * (2. * M_PI) / N);
  ////    y[i] = std::cos(i * (2. * M_PI) / N);
  ////  }

  ////  box-like (has to be clockwise order)
  constexpr int N = 4;
  Precision x[N]  = {-4, -4, 4, 4};
  Precision y[N]  = {-4, 4, 4, -4};

#ifdef VECGEOM_ROOT
  double xd[N] = {-4, -4, 4, 4};
  double yd[N] = {-4, 4, 4, -4};
  TGeoBBox box(4, 4, 10);
#endif

  // Precision x[6]={0,1,1.5,1.5,1.0,-0.5};
  // Precision y[6]={0,0,1,2,2,1};
  vecgeom::PlanarPolygon poly(N, x, y);

  std::cerr << "convexity" << poly.IsConvex() << "\n";

  vecgeom::PolygonalShell shell(N, x, y, -10., 10.);
  auto d = shell.DistanceToIn(Vector3D<Precision>(-100, 0., 0.), Vector3D<Precision>(1., 0., 0.));
  std::cerr << "distance " << d << "\n";

  vecgeom::UnplacedSExtruVolume vol(N, x, y, -10., 10.);
  {
    auto d = vol.DistanceToIn(Vector3D<Precision>(-100, 0., 0.), Vector3D<Precision>(1., 0., 0.));
    std::cerr << "distance " << d << "\n";
  }
  {
    auto d = vol.DistanceToIn(Vector3D<Precision>(0, 0., 20), Vector3D<Precision>(0., 0., -1.));
    std::cerr << "distance " << d << "\n";
  }
  {
    auto d = vol.DistanceToIn(Vector3D<Precision>(0, 0., -20), Vector3D<Precision>(0., 0., 1.));
    std::cerr << "distance " << d << "\n";
  }

  std::cerr << poly.OnSegment<Precision, Precision, bool>(0, 1.0, 2.0) << "\n";
  std::cerr << poly.OnSegment<Precision, Precision, bool>(0, x[0], y[0]) << "\n";

  //  for (size_t i = 0; i < N; ++i) {
  //   // midpoints should be on Segment
  //    std::cerr << poly.OnSegment<Precision,Precision,bool>(i, 0.5*(x[i]+x[(i-1)%N]), 0.5*(y[i]+y[(i-1)%N])) << "\n";
  //  }

  Vector3D<Precision> lower(poly.GetMinX() - 1, poly.GetMinY() - 1, -15);
  Vector3D<Precision> upper(poly.GetMaxX() + 1, poly.GetMaxY() + 1, 15);

  SOA3D<Precision> container;
  volumeUtilities::FillRandomPoints(lower, upper, container);

  SOA3D<Precision> pointsout(100000);
  SOA3D<Precision> pointsin(100000);
  SOA3D<Precision> dircontainer(100000);
  volumeUtilities::FillRandomPoints<SOA3D<Precision>, UnplacedSExtruVolume, true>(lower, upper, vol, pointsout);
  volumeUtilities::FillRandomPoints<SOA3D<Precision>, UnplacedSExtruVolume, false>(lower, upper, vol, pointsin);
  volumeUtilities::FillRandomDirections(dircontainer);

  UnplacedSExtruVolume dummy(N, x, y, -10, 10);
  Vector3D<Precision> n;
  bool v = dummy.Normal(Vector3D<Precision>(-4, 0., 0.), n);
  std::cerr << n << " valid " << v << "\n";
  v = dummy.Normal(Vector3D<Precision>(0, 4., 0.), n);
  std::cerr << n << " valid " << v << "\n";
  v = dummy.Normal(Vector3D<Precision>(4, 0., 0.), n);
  std::cerr << n << " valid " << v << "\n";
  v = dummy.Normal(Vector3D<Precision>(0, -4., 0.), n);
  std::cerr << n << " valid " << v << "\n";
  v = dummy.Normal(Vector3D<Precision>(0, 0., 10.), n);
  std::cerr << n << " valid " << v << "\n";
  v = dummy.Normal(Vector3D<Precision>(0, 0., -10.), n);
  std::cerr << n << " valid " << v << "\n";

#ifdef VECGEOM_ROOT
  TGeoPolygon geopoly(N);
  geopoly.SetXY(xd, yd);
  geopoly.FinishPolygon();

  TGeoXtru geoxtru(2);
  geoxtru.DefinePolygon(N, xd, yd);
  geoxtru.DefineSection(0, -10, 0., 0., 1.);
  geoxtru.DefineSection(1, 10, 0., 0., 1.);
  {
    double p[3] = {-100, 0., 0.};
    double d[3] = {1, 0., 0.};

    std::cerr << "ROOT distance " << geoxtru.DistFromOutside(p, d) << "\n";
  }
  {
    double p[3] = {0, 0., 20.};
    double d[3] = {0, 0., -1.};

    std::cerr << "ROOT distance " << geoxtru.DistFromOutside(p, d) << "\n";
  }
  {
    double p[3] = {0, 0., -20.};
    double d[3] = {0, 0., 1.};

    std::cerr << "ROOT distance " << geoxtru.DistFromOutside(p, d) << "\n";
  }
// return 1;
#endif

  int counter = 0;
  TimeVecGeom(poly, container, counter);
  std::cerr << counter << "\n";

  {
    int counter = 0;
    TimeVecGeomDistanceToIn(vol, pointsout, dircontainer, counter);
    std::cerr << "HITTING :" << counter << "\n";
  }

  {
    double out = 0;
    TimeVecGeomDistanceToOut(vol, pointsin, dircontainer, out);
    std::cerr << "accum DO :" << out << "\n";
  }

#ifdef VECGEOM_ROOT
  counter = 0;
  TimeTGeo(geopoly, container, counter);
  std::cerr << counter << "\n";
  {
    counter = 0;
    TimeTGeoDistanceToIn(geoxtru, pointsout, dircontainer, counter);
    std::cerr << "ROOT HITTING :" << counter << "\n";
  }

  {
    double out = 0.;
    TimeTGeoDistanceToOut(geoxtru, pointsin, dircontainer, out);
    std::cerr << "ROOT accum DO :" << out << "\n";
  }

  {
    counter = 0;
    TimeTGeoDistanceToIn(box, pointsout, dircontainer, counter);
    std::cerr << "ROOT HITTING :" << counter << "\n";
  }

#endif

  {
    double totals = 0;
    TimeVecGeomSafety(poly, container, totals);
    std::cerr << totals << "\n";
  }

#ifdef VECGEOM_ROOT
  {
    double totals = 0;
    TimeTGeoSafety(geopoly, container, totals);
    std::cerr << totals << "\n";
  }
#endif

  return 0;
}
