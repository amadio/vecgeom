#include "base/FlatVoxelHashMap.h"
#include "base/Vector3D.h"
#undef NDEBUG
#include <cassert>

using namespace vecgeom;

void testGeneralVersion()
{
  int Nx = 10;
  int Ny = 10;
  int Nz = 10;
  Vector3D<float> lower(-5, -5, -5);
  Vector3D<float> dim(10, 10, 10);

  // a voxel structure mapping to vector of ints per voxel
  FlatVoxelHashMap<int> voxels(lower, dim, Nx, Ny, Nz);

  Vector3D<float> p1(-4.9, -4.9, -4.9);
  assert(voxels.getVoxelKey(p1) == 0);
  Vector3D<float> p3(4.99, 4.99, 4.99);
  assert(voxels.getVoxelKey(p3) == Nx * Ny * Nz - 1);

  int length{0};
  assert(voxels.isOccupied(p1) == false);
  assert(voxels.getProperties(p1, length) == nullptr);
  assert(length == 0);

  voxels.addProperty(p1, 111);
  voxels.addProperty(p1, 112);
  assert(voxels.isOccupied(p1) == true);
  assert(voxels.getProperties(p1, length) != nullptr);
  assert(length == 2);
  auto props = voxels.getProperties(p1, length);
  assert(props[0] == 111);
  assert(props[1] == 112);

  // nearby point in same voxel
  Vector3D<float> p2(-4.85, -4.85, -4.85);
  assert(voxels.isOccupied(p2) == true);
  assert(voxels.getProperties(p2, length) != nullptr);

#ifdef VECGEOM_ROOT
  voxels.dumpToTFile("foo.root");
  auto newvoxels = FlatVoxelHashMap<int, false>::readFromTFile("foo.root");
  assert(newvoxels != nullptr);
  assert(newvoxels && newvoxels->isOccupied(p2) == true);
  assert(newvoxels && newvoxels->getProperties(p2, length) != nullptr && length == 2);
  assert(newvoxels && newvoxels->getProperties(p1, length) != nullptr && length == 2);
  assert(newvoxels && newvoxels->getVoxelKey(p3) == Nx * Ny * Nz - 1);
#endif
}

void testScalarVersion()
{
  int Nx = 10;
  int Ny = 10;
  int Nz = 10;
  Vector3D<float> lower(-5, -5, -5);
  Vector3D<float> dim(10, 10, 10);

  // a voxel structure mapping to a single int per voxel
  FlatVoxelHashMap<int, true> voxels(lower, dim, Nx, Ny, Nz);

  Vector3D<float> p1(-4.9, -4.9, -4.9);
  assert(voxels.getVoxelKey(p1) == 0);
  Vector3D<float> p3(4.99, 4.99, 4.99);
  assert(voxels.getVoxelKey(p3) == Nx * Ny * Nz - 1);

  int length{0};
  assert(voxels.isOccupied(p1) == false);
  assert(voxels.getProperties(p1, length) == nullptr);
  assert(length == 0);

  voxels.addProperty(p1, 111);
  assert(voxels.isOccupied(p1) == true);
  assert(voxels.getProperties(p1, length) != nullptr);
  assert(length == 1);
  auto props = voxels.getProperties(p1, length);
  assert(props[0] == 111);

  // nearby point in same voxel
  Vector3D<float> p2(-4.85, -4.85, -4.85);
  assert(voxels.isOccupied(p2) == true);
  assert(voxels.getProperties(p2, length) != nullptr);

#ifdef VECGEOM_ROOT
  voxels.dumpToTFile("foo.root");
  auto newvoxels = FlatVoxelHashMap<int, true>::readFromTFile("foo.root");
  assert(newvoxels != nullptr);
  assert(newvoxels && newvoxels->isOccupied(p2) == true);
  assert(newvoxels && newvoxels->getProperties(p2, length) != nullptr);
  assert(newvoxels && newvoxels->getVoxelKey(p3) == Nx * Ny * Nz - 1);
#endif
}

int main()
{
  testGeneralVersion();
  testScalarVersion();
  std::cout << "test passed\n";
  return 0;
}
