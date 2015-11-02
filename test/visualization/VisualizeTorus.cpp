#include "utilities/Visualizer.h"
#include "volumes/Torus2.h"
#include "volumes/utilities/VolumeUtilities.h"

using namespace vecgeom;

int main() {
  constexpr int nSamples = 512;
  SimpleTorus2 torus2("Visualizer Torus2", 2., 3., 15., 0., 2*kPi);
  AOS3D<Precision> points(nSamples);
  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> sample;
    do {
      sample = volumeUtilities::SamplePoint(Vector3D<Precision>(20, 20, 20));
    } while (!torus2.Contains(sample));
    points.set(i, sample);
  }
  points.resize(nSamples);
  Visualizer visualizer;
  visualizer.AddVolume(torus2);
  //visualizer.AddPoints(points);
  visualizer.Show();
  return 0;
}
