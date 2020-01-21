/// \file VisualizeParallelepiped.h
/// \author: Mihaela Gheata (mihaela.gheata@cern.ch)
#include "utilities/Visualizer.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "TPolyMarker3D.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/Parallelepiped.h"

using namespace vecgeom;

int main()
{
  constexpr int nSamples = 10000;

  SimpleParallelepiped para("parallelepiped", 10, 7, 20, 30, 30, 45);
  TPolyMarker3D pm(nSamples);
  pm.SetMarkerColor(kRed);
  pm.SetMarkerStyle(6);
  Inside_t inside;
  int nerrors = 0;
  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> sample;
    sample = para.GetUnplacedVolume()->SamplePointOnSurface();
    inside = para.Inside(sample);
    if (inside != EnumInside::kSurface) nerrors++;
    /*
       bool contains;
       do {
          sample = volumeUtilities::SamplePoint(Vector3D<Precision>(4, 4, 10));
          sample = para.GetUnplacedVolume()->SamplePointOnSurface();
          inside = para.Inside(sample);
          contains = para.Contains(sample);
          if (inside==kInside && !contains) {
            inside = para.Inside(sample);
          }
        } while (inside != kInside);
    */
    pm.SetNextPoint(sample[0], sample[1], sample[2]);
  }
  Visualizer visualizer;
  visualizer.AddVolume(para);
  visualizer.AddPoints(pm);
  visualizer.Show();
  printf("=== nerrors = %d\n", nerrors);
  return 0;
}
