#include "utilities/Visualizer.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "TPolyMarker3D.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/CutTube.h"
#include "test/benchmark/ArgParser.h"

#ifdef VECGEOM_ROOT
#include "TGeoTube.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4CutTubs.hh"
#endif

using namespace vecgeom;

void RandomDirection(Vector3D<double> &direction)
{
  double phi    = RNG::Instance().uniform(0., 2. * kPi);
  double theta  = std::acos(1. - 2. * RNG::Instance().uniform(0, 1));
  direction.x() = std::sin(theta) * std::cos(phi);
  direction.y() = std::sin(theta) * std::sin(phi);
  direction.z() = std::cos(theta);
}

int main(int argc, char *argv[])
{
  // -test 0 - points on surface, checking Inside
  //       1 - points inside
  OPTION_INT(npoints, 10000);
  OPTION_DOUBLE(rmin, 3);
  OPTION_DOUBLE(rmax, 5);
  OPTION_DOUBLE(dz, 10);
  OPTION_DOUBLE(sphi, 0);
  OPTION_DOUBLE(dphi, 2 * kPi / 3);
  OPTION_DOUBLE(thb, 3 * kPi / 4);
  OPTION_DOUBLE(phib, kPi / 3);
  OPTION_DOUBLE(tht, kPi / 4);
  OPTION_DOUBLE(phit, 2 * kPi / 3);
  OPTION_INT(test, 0);
  OPTION_INT(nsamples, 10000000);

  const char *stest[] = {
      "=== Testing Inside and Safety for points on surface ===", "=== Testing Contains and SafetyToOut ===",
      "=== Testing DistanceToIn and SafetyToIn ===", "=== Testing DistanceToOut ==="};
  Vector3D<double> nbottom(std::sin(thb) * std::cos(phib), std::sin(thb) * std::sin(phib), std::cos(thb));
  Vector3D<double> ntop(std::sin(tht) * std::cos(phit), std::sin(tht) * std::sin(phit), std::cos(tht));
  Vector3D<Precision> sample;
  Inside_t inside;
  bool contains;

  SimpleCutTube cuttube("cuttube", rmin, rmax, dz, sphi, dphi, nbottom, ntop);
#ifdef VECGEOM_ROOT
  TGeoCtub *rootctub = (TGeoCtub *)cuttube.ConvertToRoot();
  printf("ROOT shape parameters:\n");
  rootctub->InspectShape();
#endif

#ifdef VECGEOM_GEANT4
  G4CutTubs *g4ctub = (G4CutTubs *)cuttube.ConvertToGeant4();
#endif
  // Get the extent
  Vector3D<double> amin, amax;
  cuttube.Extent(amin, amax);
  std::cout << "VecGeom extent is: "
            << "min: " << amin << " max: " << amax << std::endl;
#ifdef VECGEOM_ROOT
  Vector3D<double> amin_root, amax_root;
  amin_root.Set(rootctub->GetOrigin()[0] - rootctub->GetDX(), rootctub->GetOrigin()[1] - rootctub->GetDY(),
                rootctub->GetOrigin()[2] - rootctub->GetDZ());
  amax_root.Set(rootctub->GetOrigin()[0] + rootctub->GetDX(), rootctub->GetOrigin()[1] + rootctub->GetDY(),
                rootctub->GetOrigin()[2] + rootctub->GetDZ());
  std::cout << "ROOT extent is: "
            << "min: " << amin_root << " max: " << amax_root << std::endl;
#endif
  std::cout << "VecGeom surface: " << cuttube.SurfaceArea() << std::endl;
  std::cout << "VecGeom capacity: " << cuttube.Capacity() << std::endl;

#ifdef VECGEOM_ROOT
  std::cout << "ROOT capacity: " << rootctub->Capacity() << std::endl;
#endif

#ifdef VECGEOM_GEANT4
  std::cout << "Geant4 surface: " << g4ctub->GetSurfaceArea() << std::endl;
  std::cout << "Geant4 capacity: " << g4ctub->GetCubicVolume() << std::endl;
#endif
  if (test > 3) {
    printf("=== Unknown test ===\n");
    return 1;
  }
  // Sample volume of the object
  int ninside = 0;
  for (int i = 0; i < nsamples; ++i) {
    sample.Set(RNG::Instance().uniform(amin.x(), amax.x()), RNG::Instance().uniform(amin.y(), amax.y()),
               RNG::Instance().uniform(amin.z(), amax.z()));
    contains = cuttube.Contains(sample);
    if (contains) ninside++;
  }
  double capacity_sampled =
      double(ninside) * (amax.x() - amin.x()) * (amax.y() - amin.y()) * (amax.z() - amin.z()) / nsamples;
  double err = capacity_sampled / std::sqrt(ninside);
  std::cout << "Sampled capacity: " << capacity_sampled << " +/- " << err << std::endl;

  printf("%s\n", stest[test]);
  TPolyMarker3D pm(npoints);
  pm.SetMarkerColor(kRed);
  pm.SetMarkerStyle(6);
  Vector3D<double> direction, start;
  double distance, safety;
  int nerrors = 0;
  for (int i = 0; i < npoints; ++i) {
    switch (test) {
    case 0: // Points on surface test
      sample = cuttube.GetUnplacedVolume()->SamplePointOnSurface();
      safety = cuttube.SafetyToIn(sample);
      inside = cuttube.Inside(sample);
      if (inside != EnumInside::kSurface || safety > kTolerance) nerrors++;
      break;
    case 1: // Contains test
      do {
        sample.Set(RNG::Instance().uniform(amin.x(), amax.x()), RNG::Instance().uniform(amin.y(), amax.y()),
                   RNG::Instance().uniform(amin.z(), amax.z()));
        contains = cuttube.Contains(sample);
      } while (!contains);
      safety = cuttube.SafetyToOut(sample);
      if (safety < 0 || safety > 100) nerrors++;
      break;
    case 2: // DistanceToIn test
      do {
        sample.Set(RNG::Instance().uniform(2 * amin.x(), 2 * amax.x()),
                   RNG::Instance().uniform(2 * amin.y(), 2 * amax.y()),
                   RNG::Instance().uniform(2 * amin.z(), 2 * amax.z()));
        contains = cuttube.Contains(sample);
        if (!contains) {
          do {
            RandomDirection(direction);
            distance = cuttube.DistanceToIn(sample, direction);
          } while (distance >= 1e10);
          safety = cuttube.SafetyToIn(sample);
          sample += distance * direction;
        }
      } while (contains);
      inside = cuttube.Inside(sample);
      if (inside != EnumInside::kSurface || safety < 0 || safety > 200) {
        distance = cuttube.DistanceToIn(sample - distance * direction, direction);
        nerrors++;
      }
      break;
    case 3: // DistanceToOut test
      do {
        sample.Set(RNG::Instance().uniform(amin.x(), amax.x()), RNG::Instance().uniform(amin.y(), amax.y()),
                   RNG::Instance().uniform(amin.z(), amax.z()));
        contains = cuttube.Contains(sample);
      } while (!contains);
      // Sample inside cut tube
      RandomDirection(direction);
      distance = cuttube.DistanceToOut(sample, direction);
      sample += distance * direction;
      inside = cuttube.Inside(sample);
      if (inside != EnumInside::kSurface) nerrors++;
      break;
    }
    pm.SetNextPoint(sample[0], sample[1], sample[2]);
  }
  printf("=== nerrors = %d\n", nerrors);
  Visualizer visualizer;
  visualizer.AddVolume(cuttube);
  visualizer.AddPoints(pm);
  visualizer.Show();
  return 0;
}
