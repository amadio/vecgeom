#include "ShapeTester.h"

#include "base/Vector3D.h"
#include "volumes/SExtru.h"

int main(int argc, char *argv[])
{
  int errCode = 0;
  VUSolid *volume;

  int N = 20;
  double x[N], y[N];
  for (size_t i = 0; i < (size_t)N; ++i) {
    x[i] = 4 * std::sin(i * (2. * M_PI) / N);
    y[i] = 4 * std::cos(i * (2. * M_PI) / N);
  }

  volume = new vecgeom::SimpleSExtru("test_VecGeomSExtru", N, x, y, -5., 5.);
  ShapeTester tester;
  errCode = tester.Run(volume);

  std::cout << "Final Error count for Shape *** " << volume->GetName() << "*** = " << errCode << std::endl;
  std::cout << "=========================================================\n";
  return 0;
}
