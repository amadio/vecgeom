#include "utilities/Visualizer.h"
#include "VecGeom/volumes/Polyhedron.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "TPolyMarker3D.h"

using namespace vecgeom;

int main()
{
  constexpr int nSamples = 10000;
  //  Precision rmin[4]={15.,8.,15.,15.};
  // Precision rmin[5]={10.,10.,0.,0.,0.};
  // Precision rmax[4]={20.,15.,15.,20.};
  // Precision rmax[5]={15.,15.,25.,15.,15.};
  // Precision z[5]={0.,20.,10.,40.,70.};

  // Precision rmin[2]={0.,0.};//,0.,0.};
  // Precision rmax[4]={20.,15.};//,15.,10.};
  // Precision z[4]={0.,20.};//,20.,50.};

  Precision phiStart = 0., deltaPhi = 2. * kPi / 3.;
  int sides = 4;
  // int nZ=10;
  // Precision rmin[4]={0.,0.,0.,0.};
  // Precision rmax[4]={15.,15.,15.,10.};
  // Precision z[4]={0.,20.,30.,40.};

  // Precision rmin[6]={0.,0.,0.,0.,0.,0.};
  // Precision rmax[6]={10.,20.,20.,10.,10.,5.};
  // Precision z[6]={0.,20.,20.,40.,40.,50.};

  // Precision rmin[8]={0.,0.,0.,0.,0.,0.,0.,0.};
  // Precision rmax[8]={10.,20.,20.,10.,10.,5.,5.,20.};
  // Precision z[8]={0.,20.,20.,40.,45.,50.,50.,60.};

  // constexpr int nPlanes = 4;
  // Precision zPlanes[nPlanes] = {-2, -1, 1, 2};
  // Precision rInner[nPlanes] = {0, 0, 0, 0};
  // Precision rOuter[nPlanes] = {2, 4, 5, 3};

  constexpr int nPlanes      = 4;
  Precision zPlanes[nPlanes] = {-2, -1, 1, 2};
  Precision rInner[nPlanes]  = {0, 1, 0.5, 0};
  Precision rOuter[nPlanes]  = {1, 2, 2, 1};

  SimplePolyhedron polyhedron("Visualizer Polyhedron", phiStart, deltaPhi, sides, nPlanes, zPlanes, rInner, rOuter);
  TPolyMarker3D pm(nSamples);
  pm.SetMarkerColor(kRed);
  pm.SetMarkerStyle(6);
  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> sample;
    do {
      sample = volumeUtilities::SamplePoint(Vector3D<Precision>(20, 20, 20));
    } while (!polyhedron.Contains(sample));
    pm.SetNextPoint(sample[0], sample[1], sample[2]);
  }
  Visualizer visualizer;
  visualizer.AddVolume(polyhedron);
  visualizer.AddPoints(pm);
  visualizer.Show();
  return 0;
}
