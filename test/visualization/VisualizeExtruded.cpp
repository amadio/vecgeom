#include "utilities/Visualizer.h"
#include "volumes/Polycone.h"
#include "volumes/utilities/VolumeUtilities.h"
#include "base/AOS3D.h"
#include "volumes/Extruded.h"
#include "volumes/SExtru.h"
#include "management/GeoManager.h"

using namespace vecgeom;

int main()
{
  constexpr int nSamples = 512;

  int nvert = 8;
  int nsect = 4;

  double rmin = 10.;
  double rmax = 20.;

  bool convex = false;

  vecgeom::XtruVertex2 *vertices = new vecgeom::XtruVertex2[nvert];
  vecgeom::XtruSection *sections = new vecgeom::XtruSection[nsect];
  double *x                      = new double[nvert];
  double *y                      = new double[nvert];

  double phi = 2. * kPi / nvert;
  double r;
  for (int i = 0; i < nvert; ++i) {
    r = rmax;
    if (i % 2 > 0 && !convex) r = rmin;
    vertices[i].x = r * vecCore::math::Cos(i * phi);
    vertices[i].y = r * vecCore::math::Sin(i * phi);
    x[i]          = vertices[i].x;
    y[i]          = vertices[i].y;
  }
  for (int i = 0; i < nsect; ++i) {
    sections[i].fOrigin.Set(0, 0, -20. + i * 40. / (nsect - 1));
    sections[i].fScale = 1;
  }

  auto xtru = GeoManager::MakeInstance<UnplacedExtruded>(nvert, vertices, nsect, sections);
  LogicalVolume xtruLogical("xtru", xtru);
  VPlacedVolume *xtruPlaced = xtruLogical.Place();

  AOS3D<Precision> points(nSamples);
  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> sample;
    do {
      sample = volumeUtilities::SamplePoint(Vector3D<Precision>(20, 20, 20));
    } while (!xtruPlaced->Contains(sample));
    points.set(i, sample);
  }

  points.resize(nSamples);
  Visualizer visualizer;
  visualizer.AddVolume(*xtruPlaced);
  visualizer.AddPoints(points);
  visualizer.Show();
  return 0;
}
