/// \file CudaManager.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "VecGeom/management/CudaManager.h"

#include "VecGeom/backend/cuda/Interface.h"
#include "VecGeom/base/Array.h"
#include "VecGeom/base/Stopwatch.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/management/VolumeFactory.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/PlacedBooleanVolume.h"
#include "VecGeom/volumes/PlacedScaledShape.h"

#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <set>

namespace vecgeom {

namespace cuda {
// forward declare a global function
extern __global__ void InitDeviceCompactPlacedVolBufferPtr(void *gpu_ptr);
}

inline namespace cxx {

CudaManager::CudaManager() : world_gpu_(), fGPUtoCPUmapForPlacedVolumes()
{
  synchronized  = true;
  world_        = NULL;
  verbose_      = 0;
  total_volumes = 0;

  auto res = cudaDeviceSetLimit(cudaLimitStackSize, 4096);
  CudaAssertError(res);
}

VPlacedVolume const *CudaManager::world() const
{
  assert(world_ != nullptr);
  return world_;
}

vecgeom::cuda::VPlacedVolume const *CudaManager::world_gpu() const
{
  assert(world_gpu_ != nullptr);
  return world_gpu_;
}

vecgeom::DevicePtr<const vecgeom::cuda::VPlacedVolume> CudaManager::Synchronize()
{
  Stopwatch timer, overalltimer;
  overalltimer.Start();
  if (verbose_ > 0) std::cerr << "Starting synchronization to GPU.\n";

  // Will return null if no geometry is loaded
  if (synchronized) return vecgeom::DevicePtr<const vecgeom::cuda::VPlacedVolume>(world_gpu_);

  CleanGpu();

  // Populate the memory map with GPU addresses

  AllocateGeometry();

  // Create new objects with pointers adjusted to point to GPU memory, then
  // copy them to the allocated memory locations on the GPU.

  if (verbose_ > 1) std::cerr << "Copying geometry to GPU..." << std::endl;

  if (verbose_ > 2) std::cerr << "\nCopying logical volumes...";
  timer.Start();
  for (std::set<LogicalVolume const *>::const_iterator i = logical_volumes_.begin(); i != logical_volumes_.end(); ++i) {

    (*i)->CopyToGpu(LookupUnplaced((*i)->GetUnplacedVolume()), LookupDaughters((*i)->fDaughters), LookupLogical(*i));
  }
  timer.Stop();
  if (verbose_ > 2) std::cerr << " OK; TIME NEEDED " << timer.Elapsed() << "s \n";

  if (verbose_ > 2) std::cerr << "Copying unplaced volumes...";
  timer.Start();
  for (std::set<VUnplacedVolume const *>::const_iterator i = unplaced_volumes_.begin(); i != unplaced_volumes_.end();
       ++i) {

    (*i)->CopyToGpu(LookupUnplaced(*i));
  }
  timer.Stop();
  if (verbose_ > 2) std::cout << " OK; TIME NEEDED " << timer.Elapsed() << "s \n";

  if (verbose_ > 2) std::cout << "Copying transformations_...";
  timer.Start();
  for (std::set<Transformation3D const *>::const_iterator i = transformations_.begin(); i != transformations_.end();
       ++i) {

    (*i)->CopyToGpu(LookupTransformation(*i));
  }
  timer.Stop();
  if (verbose_ > 2) std::cout << " OK; TIME NEEDED " << timer.Elapsed() << "s \n";

  if (verbose_ > 2) std::cout << "Copying placed volumes...";
  // TODO: eventually we want to copy the placed volumes in one go (since they live now in contiguous buffers on both
  // sides
  // (the catch is that we will need to fix the virtual table pointers on the device side manually )

  timer.Start();
  for (std::set<VPlacedVolume const *>::const_iterator i = placed_volumes_.begin(); i != placed_volumes_.end(); ++i) {

    (*i)->CopyToGpu(LookupLogical((*i)->GetLogicalVolume()), LookupTransformation((*i)->GetTransformation()),
                    LookupPlaced(*i));

    // check (assert) that everything is ok concerning the order of placed volume objects
    // also asserts that sizeof(vecgeom::cxx::VPlacedVolume) == sizeof(vecgeom::cuda::VPlacedVolume)
    assert((size_t)(*i) ==
           (size_t)(&GeoManager::gCompactPlacedVolBuffer[0]) + sizeof(vecgeom::cxx::VPlacedVolume) * (*i)->id());
#ifdef VECGEOM_ENABLE_CUDA
    assert((size_t)(LookupPlaced(*i).GetPtr()) ==
           (size_t)(fPlacedVolumeBufferOnDevice.GetPtr()) + sizeof(vecgeom::cxx::VPlacedVolume) * (*i)->id());
#endif
  }
  timer.Stop();
  if (verbose_ > 2) std::cout << " OK; TIME NEEDED " << timer.Elapsed() << "s \n";

  if (verbose_ > 2) std::cout << "Copying daughter arrays...";
  timer.Start();
  std::vector<CudaDaughter_t> daughter_array;
  for (std::set<Vector<Daughter_t> *>::const_iterator i = daughters_.begin(); i != daughters_.end(); ++i) {

    // First handle C arrays that must now point to GPU locations
    const int daughter_count = (*i)->size();
    daughter_array.resize(daughter_count);
    int j = 0;
    for (Daughter_t *k = (*i)->begin(); k != (*i)->end(); ++k) {
      daughter_array[j] = LookupPlaced(*k);
      j++;
    }
    DevicePtr<CudaDaughter_t> daughter_array_gpu(LookupDaughterArray(*i));
    // daughter_array_gpu.Allocate( daughter_count );
    daughter_array_gpu.ToDevice(&(daughter_array[0]), daughter_count);
    // vecgeom::CopyToGpu(
    //    daughter_array_gpu, LookupDaughterArray(*i), daughter_count*sizeof(Daughter)
    // );

    // Create array object wrapping newly copied C arrays
    (*i)->CopyToGpu(LookupDaughterArray(*i), LookupDaughters(*i));
  }
  timer.Stop();
  if (verbose_ > 1) std::cout << " OK; TIME NEEDED " << timer.Elapsed() << "s \n";

  synchronized = true;

  world_gpu_ = LookupPlaced(world_);

  overalltimer.Stop();
  if (verbose_ > 0) std::cout << "Geometry synchronized to GPU in " << overalltimer.Elapsed() << " s.\n";

  return world_gpu_;
}

void CudaManager::LoadGeometry(VPlacedVolume const *const volume)
{

  if (world_ == volume) return;

  CleanGpu();

  logical_volumes_.clear();
  unplaced_volumes_.clear();
  placed_volumes_.clear();
  transformations_.clear();
  daughters_.clear();

  world_ = volume;
  ScanGeometry(volume);

  std::cerr << "ScanGeometry found pvolumes" << placed_volumes_.size() << "\n";

  // Already set by CleanGpu(), but keep it here for good measure
  synchronized = false;
}

void CudaManager::LoadGeometry()
{
  LoadGeometry(GeoManager::Instance().GetWorld());
}

void CudaManager::CleanGpu()
{

  if (memory_map.size() == 0 && world_gpu_ == NULL) return;

  if (verbose_ > 1) std::cout << "Cleaning GPU...";

  for (auto i = allocated_memory_.begin(), i_end = allocated_memory_.end(); i != i_end; ++i) {
    i->Deallocate();
  }
  allocated_memory_.clear();
  memory_map.clear();
  gpu_memory_map.clear();

  world_gpu_   = vecgeom::DevicePtr<vecgeom::cuda::VPlacedVolume>();
  synchronized = false;

  if (verbose_ > 1) std::cout << " OK\n";
}

// allocates space to transfer a collection/container to the GPU
// a typical collection is a set/vector of placed volume pointers etc.
template <typename Coll>
bool CudaManager::AllocateCollectionOnCoproc(const char *verbose_title, const Coll &data, bool isforplacedvol)
{
  // NOTE: Code need to be enhanced to propage the error correctly.

  if (verbose_ > 2) std::cout << "Allocating " << verbose_title << "...";

  size_t totalSize = 0;
  // calculate total size of buffer on GPU to hold the GPU copies of the collection
  for (auto i : data) {
    totalSize += i->DeviceSizeOf();
  }

  GpuAddress gpu_address;
  gpu_address.Allocate(totalSize);
  allocated_memory_.push_back(gpu_address);

  // record a GPU memory location for each object in the collection to be copied
  for (auto i : data) {
    memory_map[ToCpuAddress(i)]                                   = gpu_address;
    if (isforplacedvol) fGPUtoCPUmapForPlacedVolumes[gpu_address] = i;
    gpu_address += i->DeviceSizeOf();
  }

  if (verbose_ > 2) std::cout << " OK\n";

  return true;
}

// a special treatment for placed volumes to ensure same order of placed volumes in compact buffer
// as on CPU
bool CudaManager::AllocatePlacedVolumesOnCoproc()
{
  // check if geometry is closed
  if (!GeoManager::Instance().IsClosed()) {
    std::cerr << "Warning: Geometry on host side MUST be closed before copying to DEVICE\n";
  }

  // we start from the compact buffer on the CPU
  unsigned int size = placed_volumes_.size();

  //   if (verbose_ > 2) std::cout << "Allocating placed volume ";
  std::cerr << "Allocating placed volume ";
  size_t totalSize = 0;
  // calculate total size of buffer on GPU to hold the GPU copies of the collection
  for (unsigned int i = 0; i < size; ++i) {
    assert(&GeoManager::gCompactPlacedVolBuffer[i] != nullptr);
    totalSize += (&GeoManager::gCompactPlacedVolBuffer[i])->DeviceSizeOf();
  }

  GpuAddress gpu_address;
  gpu_address.Allocate(totalSize);

  // store this address for later access (on the host)
  fPlacedVolumeBufferOnDevice = DevicePtr<vecgeom::cuda::VPlacedVolume>(gpu_address);
  // this address has to be made known globally to the device side
  vecgeom::cuda::InitDeviceCompactPlacedVolBufferPtr(gpu_address.GetPtr());

  allocated_memory_.push_back(gpu_address);

  // record a GPU memory location for each object in the collection to be copied
  // since the pointers in GeoManager::gCompactPlacedVolBuffer are sorted by the volume id, we are
  // getting the same order on the GPU/device automatically
  for (unsigned int i = 0; i < size; ++i) {
    VPlacedVolume const *ptr                  = &GeoManager::gCompactPlacedVolBuffer[i];
    memory_map[ToCpuAddress(ptr)]             = gpu_address;
    fGPUtoCPUmapForPlacedVolumes[gpu_address] = ptr;
    gpu_address += ptr->DeviceSizeOf();
  }

  if (verbose_ > 2) std::cout << " OK\n";

  return true;
}

void CudaManager::AllocateGeometry()
{

  if (verbose_ > 1) std::cout << "Allocating geometry on GPU...";

  {
    if (verbose_ > 2) std::cout << "Allocating logical volumes...";

    DevicePtr<cuda::LogicalVolume> gpu_array;
    gpu_array.Allocate(logical_volumes_.size());
    allocated_memory_.push_back(DevicePtr<char>(gpu_array));

    for (std::set<LogicalVolume const *>::const_iterator i = logical_volumes_.begin(); i != logical_volumes_.end();
         ++i) {
      memory_map[ToCpuAddress(*i)] = DevicePtr<char>(gpu_array);

      ++gpu_array;
    }

    if (verbose_ > 2) std::cout << " OK\n";
  }

  AllocateCollectionOnCoproc("unplaced volumes", unplaced_volumes_);

  // the allocation for placed volumes is a bit different (due to compact buffer treatment), so we call a specialized
  // function
  AllocatePlacedVolumesOnCoproc(); // for placed volumes

  // this we should only do if not using inplace transformations
  AllocateCollectionOnCoproc("transformations", transformations_);

  {
    if (verbose_ > 2) std::cout << "Allocating daughter lists...";

    DevicePtr<cuda::Vector<CudaDaughter_t>> daughter_gpu_array;
    daughter_gpu_array.Allocate(daughters_.size());
    allocated_memory_.push_back(GpuAddress(daughter_gpu_array));

    DevicePtr<CudaDaughter_t> daughter_gpu_c_array;
    daughter_gpu_c_array.Allocate(total_volumes);
    allocated_memory_.push_back(GpuAddress(daughter_gpu_c_array));

    for (std::set<Vector<Daughter> *>::const_iterator i = daughters_.begin(); i != daughters_.end(); ++i) {

      memory_map[ToCpuAddress(*i)]                   = GpuAddress(daughter_gpu_array);
      gpu_memory_map[GpuAddress(daughter_gpu_array)] = GpuAddress(daughter_gpu_c_array);
      ++daughter_gpu_array;
      daughter_gpu_c_array += (*i)->size();
    }

    if (verbose_ > 2) std::cout << " OK\n";
  }

  if (verbose_ == 2) std::cout << " OK\n";

  fprintf(stderr, "NUMBER OF PLACED VOLUMES %ld\n", placed_volumes_.size());
  fprintf(stderr, "NUMBER OF UNPLACED VOLUMES %ld\n", unplaced_volumes_.size());
}

void CudaManager::ScanGeometry(VPlacedVolume const *const volume)
{

  if (placed_volumes_.find(volume) == placed_volumes_.end()) {
    placed_volumes_.insert(volume);
  }
  if (logical_volumes_.find(volume->GetLogicalVolume()) == logical_volumes_.end()) {
    logical_volumes_.insert(volume->GetLogicalVolume());
  }
  if (transformations_.find(volume->GetTransformation()) == transformations_.end()) {
    transformations_.insert(volume->GetTransformation());
  }
  if (unplaced_volumes_.find(volume->GetUnplacedVolume()) == unplaced_volumes_.end()) {
    unplaced_volumes_.insert(volume->GetUnplacedVolume());
  }
  if (daughters_.find(volume->GetLogicalVolume()->fDaughters) == daughters_.end()) {
    daughters_.insert(volume->GetLogicalVolume()->fDaughters);
  }

  if (auto v = dynamic_cast<PlacedBooleanVolume<kUnion> const *>(volume)) {
    ScanGeometry(v->GetUnplacedVolume()->GetLeft());
    ScanGeometry(v->GetUnplacedVolume()->GetRight());
  }
  if (auto v = dynamic_cast<PlacedBooleanVolume<kIntersection> const *>(volume)) {
    ScanGeometry(v->GetUnplacedVolume()->GetLeft());
    ScanGeometry(v->GetUnplacedVolume()->GetRight());
  }
  if (auto v = dynamic_cast<PlacedBooleanVolume<kSubtraction> const *>(volume)) {
    ScanGeometry(v->GetUnplacedVolume()->GetLeft());
    ScanGeometry(v->GetUnplacedVolume()->GetRight());
  }

  if (auto v = dynamic_cast<PlacedScaledShape const *>(volume)) {
    ScanGeometry(v->GetUnplacedVolume()->fScaled.fPlaced);
  }

  for (Daughter_t *i = volume->GetDaughters().begin(); i != volume->GetDaughters().end(); ++i) {
    ScanGeometry(*i);
  }

  total_volumes++;
}

template <typename Type>
typename CudaManager::GpuAddress CudaManager::Lookup(Type const *const key)
{
  const CpuAddress cpu_address = ToCpuAddress(key);
  GpuAddress output            = memory_map[cpu_address];
  assert(output != nullptr);
  return output;
}

template <typename Type>
typename CudaManager::GpuAddress CudaManager::Lookup(DevicePtr<Type> key)
{
  GpuAddress gpu_address(key);
  GpuAddress output = gpu_memory_map[gpu_address];
  assert(output != nullptr);
  return output;
}

DevicePtr<cuda::VUnplacedVolume> CudaManager::LookupUnplaced(VUnplacedVolume const *const host_ptr)
{
  return DevicePtr<cuda::VUnplacedVolume>(Lookup(host_ptr));
}

DevicePtr<cuda::LogicalVolume> CudaManager::LookupLogical(LogicalVolume const *const host_ptr)
{
  return DevicePtr<cuda::LogicalVolume>(Lookup(host_ptr));
}

DevicePtr<cuda::VPlacedVolume> CudaManager::LookupPlaced(VPlacedVolume const *const host_ptr)
{
  return DevicePtr<cuda::VPlacedVolume>(Lookup(host_ptr));
}

DevicePtr<cuda::Transformation3D> CudaManager::LookupTransformation(Transformation3D const *const host_ptr)
{
  return DevicePtr<cuda::Transformation3D>(Lookup(host_ptr));
}

DevicePtr<cuda::Vector<CudaManager::CudaDaughter_t>> CudaManager::LookupDaughters(Vector<Daughter> *const host_ptr)
{
  return DevicePtr<cuda::Vector<CudaManager::CudaDaughter_t>>(Lookup(host_ptr));
}

DevicePtr<CudaManager::CudaDaughter_t> CudaManager::LookupDaughterArray(Vector<Daughter> *const host_ptr)
{
  GpuAddress daughters_(LookupDaughters(host_ptr));
  return DevicePtr<CudaManager::CudaDaughter_t>(Lookup(daughters_));
}

void CudaManager::PrintGeometry() const
{
  CudaManagerPrintGeometry(world_gpu());
}

// template <typename TrackContainer>
// void CudaManager::LocatePointsTemplate(TrackContainer const &container,
//                                        const int n, const int depth,
//                                        int *const output) const {
//   CudaManagerLocatePoints(world_gpu(), container, n, depth, output);
// }

// void CudaManager::LocatePoints(SOA3D<Precision> const &container,
//                                const int depth, int *const output) const {
//   Precision *const x_gpu =
//       AllocateOnGpu<Precision>(sizeof(Precision)*container.size());
//   Precision *const y_gpu =
//       AllocateOnGpu<Precision>(sizeof(Precision)*container.size());
//   Precision *const z_gpu =
//       AllocateOnGpu<Precision>(sizeof(Precision)*container.size());
//   SOA3D<Precision> *const soa3d_gpu = container.CopyToGpu(x_gpu, y_gpu, z_gpu);
//   LocatePointsTemplate(soa3d_gpu, container.size(), depth, output);
//   CudaFree(x_gpu);
//   CudaFree(y_gpu);
//   CudaFree(z_gpu);
//   CudaFree(soa3d_gpu);
// }
}
} // End namespace vecgeom
