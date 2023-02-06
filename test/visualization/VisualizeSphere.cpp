#include "VecGeomTest/Visualizer.h"
#include "VecGeom/volumes/Sphere.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/base/AOS3D.h"

#define PI 3.14159265358979323846

using namespace vecgeom;

typedef Vector3D<Precision> Vec_t;

int main()
{
  constexpr int nSamples = 512;
  SimpleSphere sphere("Visualizer Sphere", 15., 20., 0., 2 * PI / 3., PI / 4., PI / 6.);
  AOS3D<Precision> points(nSamples);

  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> sample;
    do {
      sample = volumeUtilities::SamplePoint(Vector3D<Precision>(20, 20, 20));
    } while (!sphere.Contains(sample));
    points.set(i, sample);
  }
  points.resize(nSamples);
  Visualizer visualizer;
  visualizer.AddVolume(sphere);
  visualizer.AddPoints(points);
  visualizer.Show();
  return 0;
}
