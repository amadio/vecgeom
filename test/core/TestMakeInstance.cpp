#include "volumes/UnplacedBox.h"
#include "volumes/UnplacedTube.h"
#include "volumes/UnplacedOrb.h"
#include "volumes/LogicalVolume.h"
#include "management/GeoManager.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>

using namespace vecgeom;

// a unit test checking the factory mechanism to produce specialized unplaced
// volumes
int main()
{
  // BOX: IS TRIVIAL
  auto ubox = GeoManager::MakeInstance<UnplacedBox>(1., 1., 2.);
  assert(ubox != nullptr);
  assert(dynamic_cast<UnplacedBox *>(ubox));
  // let me try to make a specialized placed box
  Transformation3D placement(0, 0, 0);
  LogicalVolume lv("mybox", ubox);
  auto pv = lv.Place(&placement);
  assert(pv->Contains(Vector3D<double>(0, 0, 0)));

  // ORB: IS TRIVIAL
  auto uorb = GeoManager::MakeInstance<UnplacedOrb>(1.);
  assert(uorb != nullptr);
  assert(dynamic_cast<UnplacedOrb *>(uorb));

  // CHECK THE TUBE CASES
  {
    // an ordinary tube without inner radius
    auto utube = GeoManager::MakeInstance<UnplacedTube>(0., 1., 1., 0., 2. * M_PI);
    assert(utube != nullptr);
    assert(dynamic_cast<UnplacedTube *>(utube));
#ifndef VECGEOM_NO_SPECIALIZATION
    assert(dynamic_cast<SUnplacedTube<TubeTypes::NonHollowTube> *>(utube));
    assert(dynamic_cast<SUnplacedTube<TubeTypes::HollowTube> *>(utube) == nullptr);
#else
    assert(dynamic_cast<SUnplacedTube<TubeTypes::UniversalTube> *>(utube));
#endif

    // let me try to make a specialized placed hollow tube
    Transformation3D placement(0, 0, 0);
    LogicalVolume lv("mytube", utube);
    auto pv = lv.Place(&placement);
    auto c  = pv->Contains(Vector3D<double>(0, 0, 0));
    assert(c);
  }

  {
    // an ordinary hollow tube
    auto utube = GeoManager::MakeInstance<UnplacedTube>(0.5, 1., 1., 0., 2. * M_PI);
    assert(utube != nullptr);
    assert(dynamic_cast<UnplacedTube *>(utube));
#ifndef VECGEOM_NO_SPECIALIZATION
    assert(dynamic_cast<SUnplacedTube<TubeTypes::HollowTube> *>(utube));
    assert(dynamic_cast<SUnplacedTube<TubeTypes::NonHollowTube> *>(utube) == nullptr);
#else
    assert(dynamic_cast<SUnplacedTube<TubeTypes::UniversalTube> *>(utube));
#endif
  }

  std::cout << "test passed \n";
  return 0;
}
