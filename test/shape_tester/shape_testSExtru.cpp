#include "../benchmark/ArgParser.h"
#include "ShapeTester.h"
#include "volumes/PlacedVolume.h"

#include "volumes/SExtru.h"

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool usolids, bool debug, bool stat);

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_BOOL(usolids, false);

  int errCode = 0;

  int N = 20;
  double x[N], y[N];
  for (size_t i = 0; i < (size_t)N; ++i) {
    x[i] = 4 * std::sin(i * (2. * M_PI) / N);
    y[i] = 4 * std::cos(i * (2. * M_PI) / N);
  }

  auto volume = new vecgeom::SimpleSExtru("test_VecGeomSExtru", N, x, y, -5., 5.);
  volume->Print();

  ShapeTester<vecgeom::VPlacedVolume> tester;
  tester.setConventionsMode(usolids);
  tester.setDebug(debug);
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  errCode = tester.Run(volume);

  std::cout << "Final Error count for Shape *** " << volume->GetName() << "*** = " << errCode << " ("
            << (tester.getConventionsMode() ? "USolids" : "VecGeom") << " conventions)\n";
  std::cout << "=========================================================\n";

  if (volume) delete volume;
  return 0;
}
