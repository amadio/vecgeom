#include "utilities/Visualizer.h"
#include "volumes/utilities/VolumeUtilities.h"
#include "TPolyMarker3D.h"
#include "base/Vector3D.h"
#include "volumes/GenTrap.h"

using namespace vecgeom;

int main() {
  constexpr int nSamples = 50000;
  std::vector<Vector3D<Precision> > vertexlist;
  // no twist
  vertexlist.push_back( Vector3D<Precision>(-3,-2.5, 0 ) );
  vertexlist.push_back( Vector3D<Precision>(-3, 3, 0 ) );
  vertexlist.push_back( Vector3D<Precision>( 3, 3, 0 ) );
  vertexlist.push_back( Vector3D<Precision>( 3,-3, 0 ) );
  vertexlist.push_back( Vector3D<Precision>(-3,-3, 0 ) );
  vertexlist.push_back( Vector3D<Precision>(-3, 3, 0 ) );
  vertexlist.push_back( Vector3D<Precision>( 3, 2, 0 ) );
  vertexlist.push_back( Vector3D<Precision>( 3,-3, 0 ) );

  SimpleGenTrap trap("gentrap", &vertexlist[0], 10 );
  TPolyMarker3D pm(nSamples);
  pm.SetMarkerColor(kRed);
  pm.SetMarkerStyle(6);
  Inside_t inside;
  bool contains;
  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> sample;
    do {
      sample = volumeUtilities::SamplePoint(Vector3D<Precision>(4, 4, 10));
      inside = trap.Inside(sample);
      contains = trap.Contains(sample);
      if (inside==kInside && !contains) {
        inside = trap.Inside(sample);
      }
    } while (inside != kInside);
    pm.SetNextPoint(sample[0], sample[1], sample[2]);
  }
  Visualizer visualizer;
  visualizer.AddVolume(trap);
  visualizer.AddPoints(pm);
  visualizer.Show();
  return 0;
}
