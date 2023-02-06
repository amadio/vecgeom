#include "VecGeomTest/Visualizer.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/volumes/Cone.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/base/AOS3D.h"

using namespace vecgeom;

int main()
{
  constexpr int nSamples = 512;
  SimpleCone cone("Visualizer Cone", 0., 3., 0., 8., 10., 0., 4 * kPi / 3);
  AOS3D<Precision> points(nSamples);
  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> sample;
    do {
      sample = volumeUtilities::SamplePoint(Vector3D<Precision>(20, 20, 20));
    } while (!cone.Contains(sample));
    points.set(i, sample);
  }
  points.resize(nSamples);
  Visualizer visualizer;
  visualizer.AddVolume(cone);
  // visualizer.AddPoints(points);
  visualizer.Show();
  return 0;
}
