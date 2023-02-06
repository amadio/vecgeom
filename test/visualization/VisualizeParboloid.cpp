#include "VecGeomTest/Visualizer.h"
#include "VecGeom/volumes/Paraboloid.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/base/AOS3D.h"

using namespace vecgeom;

int main()
{
  constexpr int nSamples = 512;
  SimpleParaboloid parabolid("Visualizer Paraboloid", 2., 5., 8.);
  AOS3D<Precision> points(nSamples);
  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> sample;
    do {
      sample = volumeUtilities::SamplePoint(Vector3D<Precision>(20, 20, 20));
    } while (!parabolid.Contains(sample));
    points.set(i, sample);
  }
  points.resize(nSamples);
  Visualizer visualizer;
  visualizer.AddVolume(parabolid);
  // visualizer.AddPoints(points);
  visualizer.Show();
  return 0;
}
