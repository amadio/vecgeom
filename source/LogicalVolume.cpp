/// \file LogicalVolume.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/LogicalVolume.h"

#include "backend/Backend.h"
#ifdef VECGEOM_CUDA
#include "backend/cuda/Interface.h"
#endif
#include "base/Array.h"
#include "base/Transformation3D.h"
#include "base/Vector.h"
#include "management/GeoManager.h"
#include "management/VolumeFactory.h"
#include "volumes/PlacedVolume.h"
#include "navigation/SimpleSafetyEstimator.h"
#include "navigation/NewSimpleNavigator.h"
#include <climits>
#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// NOTE: This is in the wrong place (but SimpleSafetyEstimator does not yet have a source file).
#ifdef VECGEOM_NVCC
__device__ SimpleSafetyEstimator *gSimpleSafetyEstimator = nullptr;

VECGEOM_CUDA_HEADER_DEVICE
VSafetyEstimator *SimpleSafetyEstimator::Instance()
{
  if (gSimpleSafetyEstimator == nullptr) gSimpleSafetyEstimator = new SimpleSafetyEstimator();
  return gSimpleSafetyEstimator;
}
#endif

#ifdef VECGEOM_NVCC
__device__ VNavigator *gSimpleNavigator = nullptr;

template <>
VECGEOM_CUDA_HEADER_DEVICE
VNavigator *NewSimpleNavigator<false>::Instance()
{
  if (gSimpleNavigator == nullptr) gSimpleNavigator = new NewSimpleNavigator();
  return gSimpleNavigator;
}
#endif

int LogicalVolume::gIdCount = 0;

#ifndef VECGEOM_NVCC
LogicalVolume::LogicalVolume(char const *const label, VUnplacedVolume const *const unplaced_volume)
    : fUnplacedVolume(unplaced_volume), fId(0), fLabel(nullptr), fUserExtensionPtr(nullptr),
      fTrackingMediumPtr(nullptr), fBasketManagerPtr(nullptr), fLevelLocator(nullptr),
      fSafetyEstimator(SimpleSafetyEstimator::Instance()), fNavigator(NewSimpleNavigator<>::Instance()), fDaughters()
{
  fId = gIdCount++;
  GeoManager::Instance().RegisterLogicalVolume(this);
  fLabel     = new std::string(label);
  fDaughters = new Vector<Daughter>();
}

#else
VECGEOM_CUDA_HEADER_DEVICE
LogicalVolume::LogicalVolume(VUnplacedVolume const *const unplaced_vol, Vector<Daughter> *GetDaughter)
    // Id for logical volumes is not needed on the device for CUDA
    : fUnplacedVolume(unplaced_vol),
      fId(-1),
      fLabel(nullptr),
      fUserExtensionPtr(nullptr),
      fTrackingMediumPtr(nullptr),
      fBasketManagerPtr(nullptr),
      fDaughters(GetDaughter),
      fLevelLocator(nullptr),
      fSafetyEstimator(SimpleSafetyEstimator::Instance()),
      fNavigator(NewSimpleNavigator<>::Instance())
{
}

#endif

LogicalVolume::~LogicalVolume()
{
  delete fLabel;
  for (Daughter *i = GetDaughters().begin(); i != GetDaughters().end(); ++i) {
    // delete *i;
  }
#ifndef VECGEOM_NVCC // this guard might have to be extended
  GeoManager::Instance().DeregisterLogicalVolume(fId);
#endif
  delete fDaughters;
}

#ifndef VECGEOM_NVCC

VPlacedVolume *LogicalVolume::Place(char const *const label, Transformation3D const *const transformation) const
{
  return GetUnplacedVolume()->PlaceVolume(label, this, transformation);
}

VPlacedVolume *LogicalVolume::Place(Transformation3D const *const transformation) const
{
  return Place(fLabel->c_str(), transformation);
}

VPlacedVolume *LogicalVolume::Place(char const *const label) const
{
  return Place(label, &Transformation3D::kIdentity);
}

VPlacedVolume *LogicalVolume::Place() const
{
  return Place(fLabel->c_str());
}

VPlacedVolume const *LogicalVolume::PlaceDaughter(char const *const label, LogicalVolume const *const volume,
                                                  Transformation3D const *const transformation)
{
  VPlacedVolume const *const placed = volume->Place(label, transformation);
  //  std::cerr << label <<" LogVol@"<< this <<" and placed@"<< placed << std::endl;
  fDaughters->push_back(placed);
  return placed;
}

VPlacedVolume const *LogicalVolume::PlaceDaughter(LogicalVolume const *const volume,
                                                  Transformation3D const *const transformation)
{
  return PlaceDaughter(volume->GetLabel().c_str(), volume, transformation);
}

void LogicalVolume::PlaceDaughter(VPlacedVolume const *const placed)
{
  fDaughters->push_back(placed);
}

// void LogicalVolume::SetDaughter(unsigned int i, VPlacedVolume const *pvol) { daughters_->operator[](i) = pvol; }

#endif

VECGEOM_CUDA_HEADER_BOTH
void LogicalVolume::Print(const int indent) const
{
  for (int i = 0; i < indent; ++i)
    printf("  ");
  printf("LogicalVolume [%i]", fId);
#ifndef VECGEOM_NVCC
  if (fLabel->size()) {
    printf(" \"%s\"", fLabel->c_str());
  }
#endif
  printf(":\n");
  for (int i = 0; i <= indent; ++i)
    printf("  ");
  fUnplacedVolume->Print();
  printf("\n");
  for (int i = 0; i <= indent; ++i)
    printf("  ");
  if (fDaughters->size() > 0) {
    printf("Contains %zu daughter", fDaughters->size());
    if (fDaughters->size() != 1) printf("s");
  }
}

VECGEOM_CUDA_HEADER_BOTH
void LogicalVolume::PrintContent(const int indent) const
{
  for (int i = 0; i < indent; ++i)
    printf("  ");
  Print(indent);
  if (fDaughters->size() > 0) {
    printf(":");
    for (Daughter *i = fDaughters->begin(), *i_end = fDaughters->end(); i != i_end; ++i) {
      (*i)->PrintContent(indent + 2);
    }
  }
}

std::ostream &operator<<(std::ostream &os, LogicalVolume const &vol)
{
  os << *vol.GetUnplacedVolume() << " [";
  for (Daughter *i = vol.GetDaughters().begin(); i != vol.GetDaughters().end(); ++i) {
    if (i != vol.GetDaughters().begin()) os << ", ";
    os << (**i);
  }
  os << "]";
  return os;
}

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::LogicalVolume> LogicalVolume::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const unplaced_vol,
                                                        DevicePtr<cuda::Vector<CudaDaughter_t>> GetDaughter,
                                                        DevicePtr<cuda::LogicalVolume> const gpu_ptr) const
{
  gpu_ptr.Construct(unplaced_vol, GetDaughter);
  CudaAssertError();
  return gpu_ptr;
}

DevicePtr<cuda::LogicalVolume> LogicalVolume::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const unplaced_vol,
                                                        DevicePtr<cuda::Vector<CudaDaughter_t>> daughter) const
{
  DevicePtr<cuda::LogicalVolume> gpu_ptr;
  gpu_ptr.Allocate();
  return this->CopyToGpu(unplaced_vol, daughter, gpu_ptr);
}

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::LogicalVolume>::SizeOf();
template void DevicePtr<cuda::LogicalVolume>::Construct(DevicePtr<cuda::VUnplacedVolume> const,
                                                        DevicePtr<cuda::Vector<cuda::VPlacedVolume const *>>) const;
}

#endif // VECGEOM_NVCC

} // End global namespace
