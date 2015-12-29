#include "utilities/Visualizer.h"
#include "volumes/Sphere.h"
#include "volumes/utilities/VolumeUtilities.h"

#define PI 3.14159265358979323846

using namespace vecgeom;

typedef Vector3D<Precision> Vec_t;

int main() {
  constexpr int nSamples = 512;
  SimpleSphere sphere("Visualizer Sphere", 15. , 20. ,0.,  2*PI, 0.3 ,0.235869);

  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> sample;
    do {
      sample = volumeUtilities::SamplePoint(Vector3D<Precision>(20, 20, 20));
    } while (!sphere.Contains(sample));
    //points.set(i, sample);
  }
  //points.resize(nSamples);
  Visualizer visualizer;
  visualizer.AddVolume(sphere);
  
  Vec_t pt(-3.5981316788113346661, 6.2008742178794342692, 18.09297989965089215);
  Vec_t dir(-0.19942178669177773598, -0.9731075086342455549, -0.11529409192283343599);
  Vec_t pt2 = pt + 15.*dir;
  visualizer.AddPoint(pt);
  visualizer.AddPoint(pt2);
  visualizer.AddLine(pt,pt2);

  Precision Dist = sphere.DistanceToIn(pt,dir);
  std::cout<<"Should be zero : "<<Dist<<std::endl;


  std::cout<<" -------------- Taken from ShapeTester ------------------------"<<std::endl;
  bool convex2=false;
  Vec_t norm(0.,0.,0.);
  double DistanceOut2 = sphere.DistanceToOut(pt, dir, norm, convex2);

    UVector3 pointIn = pt + dir * DistanceOut2 * (1. - 10 * kTolerance);
    double DistanceOut = sphere.DistanceToOut(pointIn, dir, norm, convex2);
    UVector3 pointOut = pt + dir * DistanceOut2 * (1. + 10 * kTolerance);
    double DistanceIn = sphere.DistanceToIn(pointOut, -dir);
    // Calculate distances for convex or notconvex case
    double DistanceToInSurf = sphere.DistanceToIn(pt + dir * DistanceOut2, dir);
    double dmove=0.;
    if (DistanceToInSurf >= UUtils::kInfinity) {
      //dmove = maxXYZ;
      // if (!convex2) {
      //   bool testConvexity = false;
      //   UVector3 pointSurf = point + dir * DistanceOut2;
      //   for (int k = 0; k < 100; k++) {
      //     UVector3 rndDir = GetRandomDirection();
      //     double distTest = sphere->DistanceToIn(pointSurf, rndDir);
      //     if ((distTest <= UUtils::kInfinity) && (distTest > 0.)) {
      //       testConvexity = true;
      //       break;
      //     }
      //   }
      //   if (!testConvexity)
      //     ReportError(&nError, point, dir, DistanceToInSurf, "SD: Error in convexity, must be convex");
      // }

    } else { // reentering solid, it is not convex
      dmove = DistanceToInSurf * 0.5;
      //if (convex2)
        //ReportError(&nError, point, dir, DistanceToInSurf, "SD: Error in convexity, must be NOT convex");
        std::cout<<"SD: Error in convexity, must be NOT convex"<<std::endl;
    }
    double DistanceToIn2 = sphere.DistanceToIn(pt + dir * (DistanceOut2+dmove), -dir);

    Vec_t pt3 = pt + dir * dmove;
    visualizer.AddPoint(pt3);
    std::cout<<"Location of Pt3 moved by distance DMOVE : "<<sphere.Inside(pt3)<<std::endl;

    Vec_t pt4 = pt + dir * DistanceOut2;
    visualizer.AddPoint(pt4);
    std::cout<<"Location of Pt4 should be surface  : "<<sphere.Inside(pt4)<<std::endl;

    
    std::cout<<"DMOVE : "<<dmove<<std::endl;
    std::cout<<"DistanceOut2 : "<<DistanceOut2<<std::endl;

    double difDelta = dmove - DistanceOut2 - DistanceToIn2;
    if (difDelta > 1000. * kTolerance)
      std::cout<<"SD: Distances calculation is not precise"<<std::endl;
    // if (difDelta > delta)
    //   delta = std::fabs(difDelta);
  //visualizer.AddPoints(points);
  visualizer.Show();
  return 0;
}