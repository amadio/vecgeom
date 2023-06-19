/// \file CudaManager.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "VecGeom/management/CudaManager.h"

#include "VecGeom/backend/cuda/Interface.h"
#include "VecGeom/base/Array.h"
#include "VecGeom/base/Stopwatch.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/management/VolumeFactory.h"
#include "VecGeom/management/NavIndexTable.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/PlacedBooleanVolume.h"
#include "VecGeom/volumes/PlacedScaledShape.h"

#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/UnplacedPolyhedron.h"
#include "VecGeom/volumes/UnplacedPolycone.h"

#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <set>
#include <unordered_map>

namespace vecgeom {

namespace cuda {
// forward declare a global function
extern __global__ void InitDeviceCompactPlacedVolBufferPtr(void *gpu_ptr);
extern __global__ void InitDeviceNavIndexPtr(void *gpu_ptr, int maxdepth);
} // namespace cuda

inline namespace cxx {

CudaManager::CudaManager() : world_gpu_(), fGPUtoCPUmapForPlacedVolumes_()
{
  synchronized_  = true;
  world_         = NULL;
  verbose_       = 0;
  total_volumes_ = 0;
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

namespace {
/**
 * Bulk-copy unplaced volume instances to the GPU.
 * Some of these volumes have very costly copy functions that have to allocate memory.
 * This function sorts volumes by type, and calls a specialised bulk-copy function for the slowest
 * volume types.
 */
void CopyUnplacedVolumes(std::vector<vecgeom::cxx::VUnplacedVolume const *> &&volumesToCopy,
                         std::vector<vecgeom::cxx::DevicePtr<vecgeom::cuda::VUnplacedVolume>> &&devPtrs)
{
  using vecgeom::cxx::VUnplacedVolume;
  assert(volumesToCopy.size() == devPtrs.size());

  std::unordered_map<std::type_index, std::pair<std::vector<VUnplacedVolume const *>,
                                                std::vector<vecgeom::cxx::DevicePtr<vecgeom::cuda::VUnplacedVolume>>>>
      typesToCopy;
  for (auto i = 0u; i < volumesToCopy.size(); ++i) {
    const std::type_index tidx{typeid(*volumesToCopy[i])};
    typesToCopy[tidx].first.push_back(volumesToCopy[i]);
    typesToCopy[tidx].second.push_back(std::move(devPtrs[i]));
  }

  for (auto const &typeAndVolumes : typesToCopy) {
    const std::type_index &tid = typeAndVolumes.first;
    const auto &volumeData     = typeAndVolumes.second;

    if (tid == std::type_index(typeid(UnplacedPolyhedron))) {
      UnplacedPolyhedron::CopyToGpu(volumeData.first, volumeData.second);
    } else if (tid == typeid(UnplacedPolycone) || tid == typeid(GenericUnplacedPolycone)) {
      UnplacedPolycone::CopyToGpu(volumeData.first, volumeData.second);
    } else {
      for (auto i = 0u; i < volumeData.first.size(); ++i) {
        volumeData.first[i]->CopyToGpu(volumeData.second[i]);
      }
    }
  }
}

template <class T>
unsigned int SizeFromHighestId(std::set<T const *> const &items)
{
  if (items.empty()) return 0;
  auto iter =
      std::max_element(items.begin(), items.end(), [](T const *lhs, T const *rhs) { return lhs->id() < rhs->id(); });
  return (*iter)->id() + 1;
}

} // namespace

vecgeom::DevicePtr<const vecgeom::cuda::VPlacedVolume> CudaManager::Synchronize()
{
  Stopwatch timer, overalltimer;
  overalltimer.Start();
  if (verbose_ > 0) std::cout << "Starting synchronization to GPU.\n";

#ifdef VECGEOM_USE_NAVINDEX
  if (NavIndexTable::Instance()->GetTableSize() == 0)
    throw std::runtime_error("VECGEOM_USE_NAVINDEX is defined but navigation index table is not built");
#endif

  // Will return null if no geometry is loaded
  if (synchronized_) return vecgeom::DevicePtr<const vecgeom::cuda::VPlacedVolume>(world_gpu_);

  CleanGpu();

  // Populate the memory map with GPU addresses

  AllocateGeometry();

  // Create new objects with pointers adjusted to point to GPU memory, then
  // copy them to the allocated memory locations on the GPU.

  if (verbose_ > 1) std::cout << "Copying geometry to GPU..." << std::endl;

  if (verbose_ > 2) std::cout << "\nCopying logical volumes...";
  timer.Start();
  for (std::set<LogicalVolume const *>::const_iterator i = logical_volumes_.begin(); i != logical_volumes_.end(); ++i) {

    (*i)->CopyToGpu(LookupUnplaced((*i)->GetUnplacedVolume()), (*i)->id(), LookupDaughters((*i)->fDaughters),
                    LookupLogical(*i));
  }
  timer.Stop();
  if (verbose_ > 2) std::cout << " OK;\tTIME NEEDED " << timer.Elapsed() << "s \n";

  if (verbose_ > 2) std::cout << "Copying unplaced volumes...";
  timer.Start();
  {
    std::vector<VUnplacedVolume const *> volumesToCopy;
    std::vector<vecgeom::cxx::DevicePtr<vecgeom::cuda::VUnplacedVolume>> devPtrs;
    for (VUnplacedVolume const *vol : unplaced_volumes_) {
      volumesToCopy.emplace_back(vol);
      devPtrs.emplace_back(LookupUnplaced(vol));
    }
    CopyUnplacedVolumes(std::move(volumesToCopy), std::move(devPtrs));
  }
  timer.Stop();
  if (verbose_ > 2) std::cout << " OK;\tTIME NEEDED " << timer.Elapsed() << "s \n";

  if (verbose_ > 2) std::cout << "Copying transformations_...";
  timer.Start();
  {
    std::vector<Transformation3D const *> trafos;
    std::vector<DevicePtr<cuda::Transformation3D>> devPtrs;
    for (Transformation3D const *trafo : transformations_) {
      trafos.push_back(trafo);
      devPtrs.push_back(LookupTransformation(trafo));
    }

    Transformation3D::CopyManyToGpu(trafos, devPtrs);
  }
  timer.Stop();
  if (verbose_ > 2) std::cout << " OK;\tTIME NEEDED " << timer.Elapsed() << "s \n";

  if (verbose_ > 2) std::cout << "Copying placed volumes...";
  timer.Start();

  CopyPlacedVolumes();

  timer.Stop();
  if (verbose_ > 2) std::cout << (verbose_ > 3 ? "\n\t" : " ") << "OK;\tTIME NEEDED " << timer.Elapsed() << "s \n";

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
  if (verbose_ > 2) std::cout << " OK;\tTIME NEEDED " << timer.Elapsed() << "s \n";

  if (verbose_ > 2) std::cout << "Copying bounding boxes...";
  timer.Start();
  {
    std::vector<VUnplacedVolume const *> volumes;
    std::vector<DevicePtr<cuda::VUnplacedVolume>> devPtrs;
    for (VUnplacedVolume const *vol : unplaced_volumes_) {
      volumes.push_back(vol);
      devPtrs.push_back(LookupUnplaced(vol));
    }

    VUnplacedVolume::CopyBBoxesToGpu(volumes, devPtrs);
  }
  timer.Stop();
  if (verbose_ > 2) std::cout << " OK;\tTIME NEEDED " << timer.Elapsed() << "s \n";

  synchronized_ = true;

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

  // Already set by CleanGpu(), but keep it here for good measure
  synchronized_ = false;
}

void CudaManager::LoadGeometry()
{
  LoadGeometry(GeoManager::Instance().GetWorld());
}

void CudaManager::CleanGpu()
{

  if (memory_map_.size() == 0 && world_gpu_ == NULL) return;

  if (verbose_ > 1) std::cout << "Cleaning GPU...";

  for (auto i = allocated_memory_.begin(), i_end = allocated_memory_.end(); i != i_end; ++i) {
    i->Deallocate();
  }
  allocated_memory_.clear();
  memory_map_.clear();
  gpu_memory_map_.clear();

  world_gpu_    = vecgeom::DevicePtr<vecgeom::cuda::VPlacedVolume>();
  synchronized_ = false;

  if (verbose_ > 1) std::cout << " OK\n";
}

void CudaManager::Clear()
{
  CleanGpu();

  world_         = nullptr;
  synchronized_  = false;
  total_volumes_ = 0;

  logical_volumes_.clear();
  unplaced_volumes_.clear();
  placed_volumes_.clear();
  transformations_.clear();
  daughters_.clear();
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
    memory_map_[ToCpuAddress(i)] = gpu_address;
    if (isforplacedvol) fGPUtoCPUmapForPlacedVolumes_[gpu_address] = i;
    gpu_address += i->DeviceSizeOf();
  }

  if (verbose_ > 2) {
    std::cout << " OK: #elems in alloc_mem=" << allocated_memory_.size() << ", mem_map=" << memory_map_.size() << "\n";
  }

  return true;
}

// Copy navigation index table on the coprocessor
bool CudaManager::AllocateNavIndexOnCoproc()
{
  if (!GeoManager::gNavIndex) return false;
  auto table_size = NavIndexTable::Instance()->GetTableSize();
  auto table      = NavIndexTable::Instance()->GetTable();

  if (verbose_ > 2) std::cout << "Allocating navigation index table...";

  GpuAddress gpu_address;
  gpu_address.Allocate(table_size);

  // store this address for later access (on the host)
  fNavTableOnDevice = DevicePtr<NavIndex_t>(gpu_address);
  // this address has to be made known globally to the device side
  vecgeom::cuda::InitDeviceNavIndexPtr(gpu_address.GetPtr(), GeoManager::Instance().getMaxDepth());

  allocated_memory_.push_back(gpu_address);

  // Copy the table
  CopyToGpu((char *)table, gpu_address.GetPtr(), table_size);

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

  // Allocate one past the highest maximum reachable element ID
  unsigned int size = SizeFromHighestId(placed_volumes_);

  if (verbose_ > 2) std::cout << "Allocating " << size << " placed volumes...";

  size_t totalSize = 0;
  // calculate total size of buffer on GPU to hold the GPU copies of the collection
  for (unsigned int i = 0; i < size; ++i) {
    assert(&GeoManager::gCompactPlacedVolBuffer[i] != nullptr);
    totalSize += (&GeoManager::gCompactPlacedVolBuffer[i])->DeviceSizeOf();
  }

  GpuAddress gpu_address;
  gpu_address.Allocate(totalSize);
  CudaAssertError();

  // store this address for later access (on the host)
  fPlacedVolumeBufferOnDevice = DevicePtr<vecgeom::cuda::VPlacedVolume>(gpu_address);
  // this address has to be made known globally to the device side
  vecgeom::cuda::InitDeviceCompactPlacedVolBufferPtr(gpu_address.GetPtr());
  CudaAssertError();

  allocated_memory_.push_back(gpu_address);

  // record a GPU memory location for each object in the collection to be copied
  // since the pointers in GeoManager::gCompactPlacedVolBuffer are sorted by the volume id, we are
  // getting the same order on the GPU/device automatically
  for (unsigned int i = 0; i < size; ++i) {
    VPlacedVolume const *ptr                   = &GeoManager::gCompactPlacedVolBuffer[i];
    memory_map_[ToCpuAddress(ptr)]             = gpu_address;
    fGPUtoCPUmapForPlacedVolumes_[gpu_address] = ptr;
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

    // Allocate one past the highest maximum reachable element ID
    unsigned int size = SizeFromHighestId(logical_volumes_);

    DevicePtr<cuda::LogicalVolume> gpu_array;
    gpu_array.Allocate(size);
    allocated_memory_.push_back(DevicePtr<char>(gpu_array));

    for (std::set<LogicalVolume const *>::const_iterator i = logical_volumes_.begin(); i != logical_volumes_.end();
         ++i) {
      memory_map_[ToCpuAddress(*i)] = DevicePtr<char>(gpu_array);

      ++gpu_array;
    }

    if (verbose_ > 2) std::cout << " OK\n";
  }

  AllocateCollectionOnCoproc("unplaced volumes", unplaced_volumes_);
  CudaAssertError();

  // the allocation for placed volumes is a bit different (due to compact buffer treatment), so we call a specialized
  // function
  AllocatePlacedVolumesOnCoproc(); // for placed volumes
  CudaAssertError();

  // allocate the navigation index table (if any) on the coprocessor
  AllocateNavIndexOnCoproc();
  CudaAssertError();

  // this we should only do if not using inplace transformations
  AllocateCollectionOnCoproc("transformations", transformations_);
  CudaAssertError();

  {
    if (verbose_ > 2) std::cout << "Allocating daughter lists...";

    DevicePtr<cuda::Vector<CudaDaughter_t>> daughter_gpu_array;
    daughter_gpu_array.Allocate(daughters_.size());
    allocated_memory_.push_back(GpuAddress(daughter_gpu_array));

    DevicePtr<CudaDaughter_t> daughter_gpu_c_array;
    daughter_gpu_c_array.Allocate(total_volumes_);
    allocated_memory_.push_back(GpuAddress(daughter_gpu_c_array));

    for (std::set<Vector<Daughter> *>::const_iterator i = daughters_.begin(); i != daughters_.end(); ++i) {

      memory_map_[ToCpuAddress(*i)]                   = GpuAddress(daughter_gpu_array);
      gpu_memory_map_[GpuAddress(daughter_gpu_array)] = GpuAddress(daughter_gpu_c_array);

      ++daughter_gpu_array;
      daughter_gpu_c_array += (*i)->size();
    }

    if (verbose_ > 2) std::cout << " OK\n";
  }

  CudaAssertError();

  if (verbose_ > 2) {
    std::cout << " geometry OK: #elems in alloc_mem=" << allocated_memory_.size() << ", mem_map=" << memory_map_.size()
              << ", dau_gpu_c_array=" << gpu_memory_map_.size() << "\n";
  }

  if (verbose_ > 0) {
    std::cout << "NUMBER OF PLACED VOLUMES: " << placed_volumes_.size() << '\n';
    std::cout << "NUMBER OF UNPLACED VOLUMES: " << unplaced_volumes_.size() << '\n';
  }
}

void CudaManager::ScanGeometry(VPlacedVolume const *const volume)
{

  placed_volumes_.insert(volume);
  logical_volumes_.insert(volume->GetLogicalVolume());
  transformations_.insert(volume->GetTransformation());
  unplaced_volumes_.insert(volume->GetUnplacedVolume());
  daughters_.insert(volume->GetLogicalVolume()->fDaughters);

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

  total_volumes_++;
}

template <typename Type>
typename CudaManager::GpuAddress CudaManager::Lookup(Type const *const key) const
{
  const CpuAddress cpu_address = ToCpuAddress(key);
  const auto iter              = memory_map_.find(cpu_address);
  assert(iter != memory_map_.end());
  return iter->second;
}

template <typename Type>
typename CudaManager::GpuAddress CudaManager::Lookup(DevicePtr<Type> key) const
{
  GpuAddress gpu_address(key);
  const auto iter = gpu_memory_map_.find(gpu_address);
  assert(iter != gpu_memory_map_.end());
  return iter->second;
}

DevicePtr<cuda::VUnplacedVolume> CudaManager::LookupUnplaced(VUnplacedVolume const *const host_ptr) const
{
  return DevicePtr<cuda::VUnplacedVolume>(Lookup(host_ptr));
}

DevicePtr<cuda::LogicalVolume> CudaManager::LookupLogical(LogicalVolume const *const host_ptr) const
{
  return DevicePtr<cuda::LogicalVolume>(Lookup(host_ptr));
}

DevicePtr<cuda::VPlacedVolume> CudaManager::LookupPlaced(VPlacedVolume const *const host_ptr) const
{
  return DevicePtr<cuda::VPlacedVolume>(Lookup(host_ptr));
}

DevicePtr<cuda::Transformation3D> CudaManager::LookupTransformation(Transformation3D const *const host_ptr) const
{
  return DevicePtr<cuda::Transformation3D>(Lookup(host_ptr));
}

DevicePtr<cuda::Vector<CudaManager::CudaDaughter_t>> CudaManager::LookupDaughters(
    Vector<Daughter> *const host_ptr) const
{
  return DevicePtr<cuda::Vector<CudaManager::CudaDaughter_t>>(Lookup(host_ptr));
}

DevicePtr<CudaManager::CudaDaughter_t> CudaManager::LookupDaughterArray(Vector<Daughter> *const host_ptr) const
{
  GpuAddress daughters_(LookupDaughters(host_ptr));
  return DevicePtr<CudaManager::CudaDaughter_t>(Lookup(daughters_));
}

void CudaManager::PrintGeometry() const
{
  CudaManagerPrintGeometry(world_gpu());
}

/**
 * Sort all placed volumes by type, and bulk-copy all instances of each type to the device.
 */
void CudaManager::CopyPlacedVolumes() const
{
  struct TypeInfoForPlaced {
    std::vector<vecgeom::cxx::VPlacedVolume const *> hostVol;
    std::vector<vecgeom::cxx::DevicePtr<vecgeom::cuda::LogicalVolume>> logical;
    std::vector<vecgeom::cxx::DevicePtr<vecgeom::cuda::Transformation3D>> trafo;
    std::vector<vecgeom::cxx::CudaManager::CudaDaughterPtr_t> gpuVol;
  };

  std::unordered_map<std::type_index, TypeInfoForPlaced> typesToCopy;
  for (VPlacedVolume const *pvol : placed_volumes_) {
    const std::type_index tidx{typeid(*pvol)};

    auto &typeInfo = typesToCopy[std::type_index(typeid(*pvol))];
    typeInfo.hostVol.push_back(pvol);
    typeInfo.logical.push_back(LookupLogical(pvol->GetLogicalVolume()));
    typeInfo.trafo.push_back(LookupTransformation(pvol->GetTransformation()));
    typeInfo.gpuVol.push_back(LookupPlaced(pvol));

    // check (assert) that everything is ok concerning the order of placed volume objects
    // also asserts that sizeof(vecgeom::cxx::VPlacedVolume) == sizeof(vecgeom::cuda::VPlacedVolume)
    assert((size_t)(pvol) ==
           (size_t)(&GeoManager::gCompactPlacedVolBuffer[0]) + sizeof(vecgeom::cxx::VPlacedVolume) * pvol->id());
#ifdef VECGEOM_ENABLE_CUDA
    assert((size_t)(LookupPlaced(pvol).GetPtr()) ==
           (size_t)(fPlacedVolumeBufferOnDevice.GetPtr()) + sizeof(vecgeom::cxx::VPlacedVolume) * pvol->id());
#endif
  }

  for (const auto &type_volInfo : typesToCopy) {
    const auto &volInfo                 = type_volInfo.second;
    const VPlacedVolume *const firstVol = volInfo.hostVol.front();
    if (verbose_ > 3) {
      std::cout << "\n\t" << volInfo.hostVol.size() << "\t" << type_volInfo.first.name();
    }
    firstVol->CopyManyToGpu(volInfo.hostVol, volInfo.logical, volInfo.trafo, volInfo.gpuVol);
  }
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
} // namespace cxx
} // End namespace vecgeom
