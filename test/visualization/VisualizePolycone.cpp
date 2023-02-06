#include "VecGeomTest/Visualizer.h"
#include "VecGeom/volumes/Polycone.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/base/AOS3D.h"

using namespace vecgeom;

int main()
{
  constexpr int nSamples = 512;
  //  Precision rmin[4]={15.,8.,15.,15.};
  // Precision rmin[5]={10.,10.,0.,0.,0.};
  // Precision rmax[4]={20.,15.,15.,20.};
  // Precision rmax[5]={15.,15.,25.,15.,15.};
  // Precision z[5]={0.,20.,10.,40.,70.};

  // Precision rmin[2]={0.,0.};//,0.,0.};
  // Precision rmax[4]={20.,15.};//,15.,10.};
  // Precision z[4]={0.,20.};//,20.,50.};

  Precision phiStart = 0., deltaPhi = kTwoPi;
  // int nZ=10;
  // Precision rmin[4]={0.,0.,0.,0.};
  // Precision rmax[4]={15.,15.,15.,10.};
  // Precision z[4]={0.,20.,30.,40.};

  // constexpr int nZ=3;
  // Precision rmin[nZ]={0.,0.,0.};
  // Precision rmax[nZ]={10.,20.,20};
  // Precision z[nZ]=    {0.,10.,10.};

  // int nZ = 5;
  // Precision rmin[5] = {0., 0., 0., 0., 0.};
  // Precision rmax[5] = {10., 20., 20., 15.,10.};
  // Precision z[5] = {0., 10., 20., 20., 30.};

  deltaPhi           = kTwoPi;
  constexpr int nZ   = 3;
  Precision rmin[nZ] = {5., 0., 0.};
  Precision rmax[nZ] = {10., 20., 30.};
  Precision z[nZ]    = {0., 10., 30.};

  // constexpr int nZ = 3;
  // Precision rmin[nZ] = {0., 0., 0.};
  /// Precision rmax[nZ] = {10., 20., 30.};
  // Precision z[nZ] = {0., 10., 30.};

  // constexpr int nZ = 3;
  // Precision rmin[nZ] = {0., 0., 0.};
  // Precision rmax[nZ] = {10., 20., 30.};
  // Precision z[nZ] = {0., 10., 30.};

  // int nZ=8;
  // Precision rmin[8]={0.,0.,0.,0.,0.,0.,0.,0.};
  // Precision rmax[8]={10.,20.,20.,10.,10.,5.,5.,20.};
  // Precision z[8]={0.,20.,20.,40.,45.,50.,50.,60.};

  // constexpr int nZ = 3;
  // Precision rmin[10]={0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  // Precision rmax[10]={10.,10.,10.,20.,20.,10.,10.,5.,5.,20.};
  // Precision z[10]={-20.,0.,0.,20.,20.,40.,45.,50.,50.,60.};

  // Precision rmin[nZ]={0.,0.,0.,0.,0.,0.};
  // Precision rmax[nZ]={10., 20., 20., 30., 20., 10.};
  // Precision z[nZ]={0., 10., 20., 20., 20.,30.};

  // Precision rmin[nZ]={0.,0.,0.};
  // Precision rmax[nZ]={10.,20.,30.};
  // Precision z[nZ]={0.,10.,10.};

  SimplePolycone polycone("Visualizer Polycone", phiStart, deltaPhi, nZ, z, rmin, rmax);
  std::cout << "Convexity : " << polycone.GetUnplacedVolume()->IsConvex() << std::endl;
  AOS3D<Precision> points(nSamples);
  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> sample;
    do {
      sample = volumeUtilities::SamplePoint(Vector3D<Precision>(20, 20, 20));
    } while (!polycone.Contains(sample));
    points.set(i, sample);
  }
  points.resize(nSamples);
  Visualizer visualizer;
  visualizer.AddVolume(polycone);
  // visualizer.AddPoints(points);
  visualizer.Show();
  return 0;
}
