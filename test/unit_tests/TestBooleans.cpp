// Purpose: Simple Unit tests for the Boolean volumes

//-- ensure asserts are compiled in
#undef NDEBUG
#include "base/FpeEnable.h"

#include "base/Vector3D.h"
#include "volumes/BooleanVolume.h"
#include "volumes/Tube.h"
#include "volumes/Box.h"
#include "ApproxEqual.h"
#include <cmath>

using namespace vecgeom;

using Vec3D_t = vecgeom::Vector3D<vecgeom::Precision>;

int TestBooleans()
{
  double L = 10.; // global scale

  // make a combination of union and subtraction
  // BOX - ( TUBE + TUBE )   aka a box with 2 holes

  UnplacedBox box(L, L, L);
  auto bv = box.Capacity();
  auto ba = box.SurfaceArea();

  GenericUnplacedTube tube(0., 0.9 * L / 4., L, 0., vecgeom::kTwoPi);
  LogicalVolume lbox("box", &box);
  auto placedbox = lbox.Place();

  // test capacity + surface area
  auto tv  = tube.Capacity();
  auto etv = tube.EstimateCapacity(10000000);
  assert(std::abs(tv - etv) / etv < 1E-3);
  auto ta  = tube.SurfaceArea();
  auto eta = tube.EstimateSurfaceArea(10000000);
  assert(std::abs(ta - eta) / ta < 1E-2);

  LogicalVolume ltube("tube", &tube);

  double HalfDiagL    = std::sqrt(2.) * L; // Half diagonal length of box
  double QuarterDiagL = HalfDiagL / 2.;

  auto upperhole = ltube.Place(new Transformation3D(-L / 4, -L / 4, 0.));
  auto lowerhole = ltube.Place(new Transformation3D(L / 4, L / 4, 0.));

  UnplacedBooleanVolume<kUnion> holes(kUnion, upperhole, lowerhole); // 2 holes as a union
  LogicalVolume lholes("CombinedTubes", &holes);
  auto placedholes1 = lholes.Place();
  auto placedholes2 = lholes.Place(new Transformation3D(-L / 2, -L / 2, 0.));

  UnplacedBooleanVolume<kSubtraction> boxminusholes(kSubtraction, placedbox, placedholes2);
  LogicalVolume lboxminusholes("CombinedBoolean", &boxminusholes);
  auto placedcombinedboolean = lboxminusholes.Place();

  auto capacity = boxminusholes.Capacity();
  assert(capacity > 0);
  assert(std::fabs(capacity - (bv - 2. * tv)) / capacity < 1E-03);

  auto sarea = boxminusholes.SurfaceArea();
  assert(sarea > 0);
  // ?? assert(sarea < (ba + 2. * ta));
  assert(sarea > ba);

  // Tests on the union first

  // Contains
  {
    bool c = placedholes1->Contains(Vec3D_t(0., 0., 0.));
    assert(!c);
  }
  {
    bool c = placedholes1->Contains(Vec3D_t(-L / 4, -L / 4, 0.));
    assert(c);
  }
  {
    bool c = placedholes1->Contains(Vec3D_t(L / 4, L / 4, 0.));
    assert(c);
  }

  {
    double d = placedholes1->DistanceToIn(Vec3D_t(0., 0., 0.), Vec3D_t(0., 0., 1.));
    assert(d == kInfLength);
  }

  {
    double d = placedholes1->DistanceToIn(Vec3D_t(0., 0., 0.), Vec3D_t(1., 0., 0.));
    assert(d == kInfLength);
  }

  {
    double d = placedholes1->DistanceToIn(Vec3D_t(0., 0., 0.), Vec3D_t(0., 1., 0.));
    assert(d == kInfLength);
  }

  {
    double d   = placedholes1->DistanceToIn(Vec3D_t(0., 0., 0.), Vec3D_t(1., 1., 0.).Normalized());
    double d2  = placedholes1->DistanceToIn(Vec3D_t(0., 0., 0.), Vec3D_t(-1., -1., 0.).Normalized());
    double dt1 = lowerhole->DistanceToIn(Vec3D_t(0., 0., 0.), Vec3D_t(1., 1., 0.).Normalized());
    assert(ApproxEqual(dt1, d));
    assert(ApproxEqual(d, d2));
    assert(ApproxEqual(d, QuarterDiagL / 2. - tube.rmax()));
  }

  {
    double d = placedholes1->PlacedDistanceToOut(Vec3D_t(L / 4, L / 4, 0.), Vec3D_t(1., 1., 0.).Normalized());
    assert(ApproxEqual(d, tube.rmax()));
  }
  {
    double d = placedholes1->PlacedDistanceToOut(Vec3D_t(-L / 4, -L / 4, 0.), Vec3D_t(1., 1., 0.).Normalized());
    assert(ApproxEqual(d, tube.rmax()));
  }

  // Test the combined boolean
  {
    bool c = placedcombinedboolean->Contains(Vec3D_t(0., 0., 0.));
    assert(c);
  }
  {
    bool c = placedcombinedboolean->Contains(Vec3D_t(-L / 2, -L / 2, 0.));
    assert(c);
  }
  {
    bool c = placedcombinedboolean->Contains(Vec3D_t(-3. * L / 4., -3. * L / 4., 0.));
    assert(!c);
  }
  {
    bool c = placedcombinedboolean->Contains(Vec3D_t(-L / 4., -L / 4., 0.));
    assert(!c);
  }
  {
    bool c = placedcombinedboolean->Contains(Vec3D_t(+L / 4., +L / 4., 0.));
    assert(c);
  }
  {
    bool c = placedcombinedboolean->Contains(Vec3D_t(+3 * L / 4., +3. * L / 4., 0.));
    assert(c);
  }

  // Test the combined boolean DistanceToOut
  {
    double d = placedcombinedboolean->DistanceToOut(Vec3D_t(0., 0., 0.), Vec3D_t(1., 0., 0.));
    assert(ApproxEqual(d, 10.));
  }
  {
    double d = placedcombinedboolean->DistanceToOut(Vec3D_t(0., 0., 0.), Vec3D_t(-1., -0., -0.));
    assert(ApproxEqual(d, 10.));
  }
  {
    double d = placedcombinedboolean->DistanceToOut(Vec3D_t(0., 0., 0.), Vec3D_t(-0., -1., -0.));
    assert(ApproxEqual(d, 10.));
  }
  {
    double d = placedcombinedboolean->DistanceToOut(Vec3D_t(0., -0., 0.), Vec3D_t(-0., 0., -1.));
    assert(ApproxEqual(d, 10.));
  }
  {
    double d = placedcombinedboolean->DistanceToOut(Vec3D_t(-L / 4, 0., 0.), Vec3D_t(0., -1., 0));
    assert(ApproxEqual(d, (L / 2. - 2. * tube.rmax()) / 2.));
  }
  {
    double d = placedcombinedboolean->DistanceToOut(Vec3D_t(-L / 4, 0., 0.), Vec3D_t(0., 1., 0));
    assert(ApproxEqual(d, 10.));
  }
  {
    double d = placedcombinedboolean->DistanceToOut(Vec3D_t(-3. * L / 4, 0., 0.), Vec3D_t(0., -1., 0));
    assert(ApproxEqual(d, L / 2. + (L / 2. - 2. * tube.rmax()) / 2.));
  }
  {
    double d = placedcombinedboolean->DistanceToOut(Vec3D_t(-L / 2, -L / 2., 0.), Vec3D_t(1., 1., 0).Normalized());
    assert(ApproxEqual(d, QuarterDiagL / 2. - tube.rmax()));
  }
  {
    double d = placedcombinedboolean->DistanceToOut(Vec3D_t(L, L, 0.), Vec3D_t(-1., -1., 0).Normalized());
    assert(d > std::sqrt(2) * L && d < std::sqrt(2) * L * 3 / 2.);
  }

  // Test the combined boolean DistanceToIn
  {
      // needs fix
      // double d = placedcombinedboolean->DistanceToIn(Vec3D_t(0., 0., 0.), Vec3D_t(1., 0., 0.));
      // assert(d <= 0.);
  } {
    double d = placedcombinedboolean->DistanceToIn(Vec3D_t(-20., 0., 0.), Vec3D_t(1., -0., -0.));
    assert(ApproxEqual(d, 10.));
  }
  {
    double d = placedcombinedboolean->DistanceToIn(Vec3D_t(0., -20., 0.), Vec3D_t(0., 1., -0.));
    assert(ApproxEqual(d, 10.));
  }
  {
    double d = placedcombinedboolean->DistanceToIn(Vec3D_t(20., 20., 20.), Vec3D_t(-1., -1., -1.).Normalized());
    assert(ApproxEqual(d, 10. * std::sqrt(3.)));
  }
  {
    double d = placedcombinedboolean->DistanceToIn(Vec3D_t(-L / 4., -L / 4., 0.), Vec3D_t(-1., -0., -0.));
    assert(ApproxEqual(d, tube.rmax()));
  }
  {
    double d = placedcombinedboolean->DistanceToIn(Vec3D_t(-L / 4. - tube.rmax(), -L / 4., 0.), Vec3D_t(1., -0., -0.));
    assert(ApproxEqual(d, 2. * tube.rmax()));
  }
  {
    double d = placedcombinedboolean->DistanceToIn(Vec3D_t(-L / 4., -L / 4., 0.), Vec3D_t(0, -1., -0.));
    assert(ApproxEqual(d, tube.rmax()));
  }
  {
    double d = placedcombinedboolean->DistanceToIn(Vec3D_t(-L / 4., -L / 4., 0.), Vec3D_t(0, -0., -1.));
    assert(d == kInfLength);
  }
  {
    double d = placedcombinedboolean->DistanceToIn(Vec3D_t(-L / 4., -L / 4., 0.), Vec3D_t(-0, -0, -1.));
    assert(d == kInfLength);
  }
  {
      // needs reworking verification
      // double d = placedcombinedboolean->DistanceToIn(Vec3D_t(-L / 4 - tube.rmax(), -L / 4., 0.), Vec3D_t(-0, -0,
      // -1.));
      // assert(d == kInfLength);
  } {
    // needs reworking + verification
    // double d = placedcombinedboolean->DistanceToIn(Vec3D_t(-L / 4 - tube.rmax() - 0.01, -L / 4., 0.), Vec3D_t(-0, -0,
    // -1.));
    // assert(d <= 0.);
  }

  // Make combined union of the tubes with a slab
  double Lslab = L / 4. * std::sqrt(2.);
  UnplacedBox slab(Lslab, L / 10., L / 10.);
  LogicalVolume lslab("slab", &slab);

  // two disconnetected tubes are now connected via the rotated slab
  UnplacedBooleanVolume<kUnion> tubesandslabunion(kUnion, placedholes1,
                                                  lslab.Place(new Transformation3D(0., 0., 0., 0., 0., 45)));
  auto placedtubesslabunion = (new LogicalVolume("tubesandslabunion", &tubesandslabunion))->Place();

  {
    double d = placedtubesslabunion->PlacedDistanceToOut(Vec3D_t(0., 0., 0.), Vec3D_t(1., 1., 0.).Normalized());
    assert(ApproxEqual(d, L / 4 * std::sqrt(2) + tube.rmax()));
  }
  {
    double d = placedtubesslabunion->PlacedDistanceToOut(Vec3D_t(-L / 4, -L / 4, 0.), Vec3D_t(1., 1., 0.).Normalized());
    assert(ApproxEqual(d, L / 2 * std::sqrt(2) + tube.rmax()));
  }
  {
    double d = placedtubesslabunion->PlacedDistanceToOut(Vec3D_t(L / 4, L / 4, 0.), Vec3D_t(1., 1., 0.).Normalized());
    assert(ApproxEqual(d, tube.rmax()));
  }
  {
    double d =
        placedtubesslabunion->PlacedDistanceToOut(Vec3D_t(-L / 4, -L / 4, 0.), Vec3D_t(-1., -1., 0.).Normalized());
    assert(ApproxEqual(d, tube.rmax()));
  }
  {
    double d = placedtubesslabunion->PlacedDistanceToOut(Vec3D_t(L / 4, L / 4, 0.), Vec3D_t(-1., -1., 0.).Normalized());
    assert(ApproxEqual(d, L / 2 * std::sqrt(2) + tube.rmax()));
  }

  return 0;
}

int main(int argc, char *argv[])
{
  return TestBooleans();
}
