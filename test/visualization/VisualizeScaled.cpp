#include "utilities/Visualizer.h"
#include "volumes/Orb.h"
#include "volumes/ScaledShape.h"
#include "volumes/utilities/VolumeUtilities.h"
#include "TPolyMarker3D.h"

using namespace vecgeom;

int main() {
  constexpr int nSamples = 10000;
  SimpleOrb orb("Visualizer Orb", 3);
  SimpleScaledShape scaled("Scaled Orb", orb.GetUnplacedVolume(), 0.5, 1.2, 1.);
  TPolyMarker3D pm(nSamples);
  pm.SetMarkerColor(kRed);
  pm.SetMarkerStyle(6);
  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> sample;
    do {
      sample = volumeUtilities::SamplePoint(Vector3D<Precision>(4, 4, 4));
    } while (!scaled.Contains(sample));
    pm.SetNextPoint(sample[0], sample[1], sample[2]);
  }
  Visualizer visualizer;
  visualizer.AddVolume(scaled);
  visualizer.AddVolume(orb);
  visualizer.AddPoints(pm);
  visualizer.Show();
  return 0;
}
