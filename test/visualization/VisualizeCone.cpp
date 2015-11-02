#include "utilities/Visualizer.h"
#include "volumes/Cone.h"
#include "volumes/utilities/VolumeUtilities.h"

using namespace vecgeom;

int main() {
  constexpr int nSamples = 512;
  SimpleCone cone("Visualizer Cone", 0., 3., 0., 8., 10.,0.,4*kPi/3);
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
  //visualizer.AddPoints(points);
  visualizer.Show();
  return 0;
}
