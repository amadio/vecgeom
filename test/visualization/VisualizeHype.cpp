#include "utilities/Visualizer.h"
#include "volumes/Hype.h"
#include "volumes/utilities/VolumeUtilities.h"

using namespace vecgeom;

int main() {
  constexpr int nSamples = 512;
  SimpleHype hype("Visualizer Hype", 10, 15, kPi/4, kPi/4, 50);
  AOS3D<Precision> points(nSamples);
  int ex=1;
  for (int i = 0; i < nSamples; ++i) {
	ex=1;
    Vector3D<Precision> sample;
    do {
      sample = volumeUtilities::SamplePoint(Vector3D<Precision>(20, 20, 20));
	  //std::cout<<sample<<std::endl;
	  ex--;
     // std::cout<<ex<<std::endl;
    } while (!hype.Contains(sample));
    points.set(i, sample);
  }
  points.resize(nSamples);
  Visualizer visualizer;
  visualizer.AddVolume(hype);
  visualizer.AddPoints(points);
  visualizer.Show();
  return 0;
}
