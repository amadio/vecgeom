#include "VecGeomTest/Visualizer.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "TPolyMarker3D.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/GenTrap.h"

using namespace vecgeom;

int main()
{
  constexpr int nSamples = 10000;
  Precision verticesx[8] = {-3, -2.5, 3, 2.5, -2, -2, 2, 2};
  Precision verticesy[8] = {-2.5, 3, 2.5, -3, -2, 2, 2, -2};

  SimpleGenTrap trap("gentrap", verticesx, verticesy, 10);
  TPolyMarker3D pm(nSamples);
  pm.SetMarkerColor(kRed);
  pm.SetMarkerStyle(6);
  Inside_t inside;
  int nerrors = 0;
  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> sample;
    sample = trap.GetUnplacedVolume()->SamplePointOnSurface();
    inside = trap.Inside(sample);
    if (inside != EnumInside::kSurface) nerrors++;
    /*
       bool contains;
       do {
          sample = volumeUtilities::SamplePoint(Vector3D<Precision>(4, 4, 10));
          sample = trap.GetUnplacedVolume()->SamplePointOnSurface();
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
