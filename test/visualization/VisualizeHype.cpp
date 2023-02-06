#include "VecGeomTest/Visualizer.h"
#include "VecGeom/volumes/Hype.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/base/AOS3D.h"

using namespace vecgeom;

int main()
{
  constexpr int nSamples = 512;
  SimpleHype hype("Visualizer Hype", 10, 15, kPi / 4, kPi / 4, 50);
  AOS3D<Precision> points(nSamples);
  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> sample;
    do {
      sample = volumeUtilities::SamplePoint(Vector3D<Precision>(20, 20, 20));
    } while (!hype.Contains(sample));
    points.set(i, sample);
  }
  points.resize(nSamples);
  Visualizer visualizer;
  visualizer.AddVolume(hype);
  visualizer.AddPoints(points);
  visualizer.Show();
  return 0;
}
