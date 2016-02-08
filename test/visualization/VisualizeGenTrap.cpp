#include "utilities/Visualizer.h"
#include "volumes/utilities/VolumeUtilities.h"
#include "TPolyMarker3D.h"
#include "base/Vector3D.h"
#include "volumes/GenTrap.h"

using namespace vecgeom;

int main() {
  constexpr int nSamples = 10000;
  std::vector<Vector3D<Precision>> vertexlist;
  // no twist
  vertexlist.push_back(Vector3D<Precision>(-3, -2.5, 0));
  vertexlist.push_back(Vector3D<Precision>(-2.5, 3, 0));
  vertexlist.push_back(Vector3D<Precision>(3, 2.5, 0));
  vertexlist.push_back(Vector3D<Precision>(2.5, -3, 0));
  vertexlist.push_back(Vector3D<Precision>(-2, -2, 0));
  vertexlist.push_back(Vector3D<Precision>(-2, 2, 0));
  vertexlist.push_back(Vector3D<Precision>(2, 2, 0));
  vertexlist.push_back(Vector3D<Precision>(2, -2, 0));

  SimpleGenTrap trap("gentrap", &vertexlist[0], 10);
  TPolyMarker3D pm(nSamples);
  pm.SetMarkerColor(kRed);
  pm.SetMarkerStyle(6);
  Inside_t inside;
  int nerrors = 0;
  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> sample;
    sample = trap.GetPointOnSurface();
    inside = trap.Inside(sample);
    if (inside != kSurface)
      nerrors++;
    /*
       bool contains;
       do {
          sample = volumeUtilities::SamplePoint(Vector3D<Precision>(4, 4, 10));
          sample = trap.GetPointOnSurface();
          inside = trap.Inside(sample);
          contains = trap.Contains(sample);
          if (inside==kInside && !contains) {
            inside = trap.Inside(sample);
          }
        } while (inside != kInside);
    */
    pm.SetNextPoint(sample[0], sample[1], sample[2]);
  }
  Visualizer visualizer;
  visualizer.AddVolume(trap);
  visualizer.AddPoints(pm);
  visualizer.Show();
  printf("=== nerrors = %d\n", nerrors);
  return 0;
}
