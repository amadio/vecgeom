#include "VecGeom/volumes/Box.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/navigation/VLevelLocator.h"

namespace vecgeom {

bool SetupBoxGeometry()
{
  UnplacedBox *worldUnplaced   = new UnplacedBox(10, 10, 10);
  UnplacedBox *boxUnplaced     = new UnplacedBox(4, 4, 4); // 4,4,4
  Transformation3D *placement1 = new Transformation3D(5, 5, 5, 0, 0, 0);
  Transformation3D *placement2 = new Transformation3D(-5, 5, 5, 0, 0, 0);   // 45, 0, 0);
  Transformation3D *placement3 = new Transformation3D(5, -5, 5, 0, 0, 0);   // 0, 45, 0);
  Transformation3D *placement4 = new Transformation3D(5, 5, -5, 0, 0, 0);   // 0, 0, 45);
  Transformation3D *placement5 = new Transformation3D(-5, -5, 5, 0, 0, 0);  // 45, 45, 0);
  Transformation3D *placement6 = new Transformation3D(-5, 5, -5, 0, 0, 0);  // 45, 0, 45);
  Transformation3D *placement7 = new Transformation3D(5, -5, -5, 0, 0, 0);  // 0, 45, 45);
  Transformation3D *placement8 = new Transformation3D(-5, -5, -5, 0, 0, 0); // 45, 45, 45);

  LogicalVolume *world = new LogicalVolume("world", worldUnplaced);
  LogicalVolume *box   = new LogicalVolume("box", boxUnplaced);
  world->PlaceDaughter("box1", box, placement1);
  world->PlaceDaughter("box2", box, placement2);
  world->PlaceDaughter("box3", box, placement3);
  world->PlaceDaughter("box4", box, placement4);
  world->PlaceDaughter("box5", box, placement5);
  world->PlaceDaughter("box6", box, placement6);
  world->PlaceDaughter("box7", box, placement7);
  world->PlaceDaughter("box8", box, placement8);

  VPlacedVolume *w = world->Place();
  GeoManager::Instance().SetWorld(w);
  GeoManager::Instance().CloseGeometry();

  return true;
}

} // namespace vecgeom
