// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \file PlacedVolume.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/PlacedVolume.h"
#include "base/Vector3D.h"
#include "base/RNG.h"
#include "management/GeoManager.h"
#include "volumes/utilities/VolumeUtilities.h"
#include "base/SOA3D.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

unsigned int VPlacedVolume::g_id_count = 0;

#ifndef VECCORE_CUDA
VPlacedVolume::VPlacedVolume(char const *const label, LogicalVolume const *const logical_volume,
                             Transformation3D const *const transformation, PlacedBox const *const bounding_box)
    : id_(), label_(NULL), logical_volume_(logical_volume),
#ifdef VECGEOM_INPLACE_TRANSFORMATIONS
      fTransformation(*transformation),
#else
      fTransformation(transformation),
#endif
      bounding_box_(bounding_box)
{
  id_ = g_id_count++;
  GeoManager::Instance().RegisterPlacedVolume(this);
  label_ = new std::string(label);
}

VECCORE_ATT_HOST_DEVICE
VPlacedVolume::VPlacedVolume(VPlacedVolume const &other)
    : id_(), label_(NULL), logical_volume_(), fTransformation(), bounding_box_()
{
  assert(0 && "COPY CONSTRUCTOR FOR PlacedVolumes NOT IMPLEMENTED");
}

VECCORE_ATT_HOST_DEVICE
VPlacedVolume *VPlacedVolume::operator=(VPlacedVolume const &other)
{
  // deliberaty copy using memcpy to also copy the virtual table
  if (this != &other) {
    // overriding the vtable is exactly what I want
    // so I silence a compier warning via the void* cast
    std::memcpy((void *)this, (void *)&other, sizeof(VPlacedVolume));
  }
  return this;
  //    if (this != &other) // protect against invalid self-assignment
  //    {
  //        id_ = other.id_;
  //        label_ = other.label_;
  //        logical_volume_ = other.logical_volume_;
  //        transformation_ = other.transformation_;
  //        bounding_box_ = other.bounding_box_;
  //    }
  //    return this;
}
#endif

VECCORE_ATT_HOST_DEVICE
VPlacedVolume::~VPlacedVolume()
{
#ifndef VECCORE_CUDA
  GeoManager::Instance().DeregisterPlacedVolume(id_);
  delete label_;
#endif
}

VECCORE_ATT_HOST_DEVICE
void VPlacedVolume::Print(const int indent) const
{
  for (int i = 0; i < indent; ++i)
    printf("  ");
  PrintType();
  printf(" [%i]", id_);
#ifndef VECCORE_CUDA
  if (label_->size()) {
    printf(" \"%s\"", label_->c_str());
  }
#endif
  printf(": \n");
  for (int i = 0; i <= indent; ++i)
    printf("  ");
  GetTransformation()->Print();
  printf("\n");
  logical_volume_->Print(indent + 1);
}

VECCORE_ATT_HOST_DEVICE
void VPlacedVolume::PrintContent(const int indent) const
{
  Print(indent);
  if (GetDaughters().size() > 0) {
    printf(":");
    for (VPlacedVolume const **vol = GetDaughters().begin(), **volEnd = GetDaughters().end(); vol != volEnd; ++vol) {
      printf("\n");
      (*vol)->PrintContent(indent + 3);
    }
  }
}

VECCORE_ATT_HOST
std::ostream &operator<<(std::ostream &os, VPlacedVolume const &vol)
{
  os << "(" << (*vol.GetUnplacedVolume()) << ", " << (*vol.GetTransformation()) << ")";
  return os;
}

// implement a default function for SamplePointOnSurface
// based on contains + DistanceToOut

Precision VPlacedVolume::Capacity()
{
  throw std::runtime_error("Capacity not implemented");
  return 0.;
}

VECCORE_ATT_HOST_DEVICE
bool VPlacedVolume::Normal(Vector3D<Precision> const & /*point*/, Vector3D<Precision> & /*normal*/) const
{
#ifndef VECCORE_CUDA
  throw std::runtime_error("Normal not implemented for this volume.");
#endif
  return false;
}

VECCORE_ATT_HOST_DEVICE
void VPlacedVolume::Extent(Vector3D<Precision> & /* min */, Vector3D<Precision> & /* max */) const
{
#ifndef VECCORE_CUDA
  throw std::runtime_error("Extent() not implemented for this shape type.");
#endif
}

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::VPlacedVolume const *>::SizeOf();
template size_t DevicePtr<char>::SizeOf();
template size_t DevicePtr<Precision>::SizeOf();
// template void DevicePtr<cuda::PlacedBox>::Construct(
//    DevicePtr<cuda::LogicalVolume> const logical_volume,
//    DevicePtr<cuda::Transformation3D> const transform,
//    const int id) const;

} // namespace cxx

#endif // VECCORE_CUDA

} // namespace vecgeom
