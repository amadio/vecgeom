/// @file SpecializedHype.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDHYPE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDHYPE_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/kernel/HypeImplementation.h"
#include "VecGeom/volumes/PlacedHype.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename hypeTypeT>
using SpecializedHype = SpecializedVolImplHelper<HypeImplementation<hypeTypeT>>;

using SimpleHype = SpecializedHype<HypeTypes::UniversalHype>;

template <typename Type>
VECCORE_ATT_DEVICE VPlacedVolume *SUnplacedHype<Type>::Create(LogicalVolume const *const logical_volume,
                                                              Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                                              const int id, const int copy_no, const int child_id,
#endif
                                                              VPlacedVolume *const placement)
{
  (void)placement;
  return new SpecializedHype<Type>(logical_volume, transformation
#ifdef VECCORE_CUDA
                                   ,
                                   id, copy_no, child_id
#endif
  );
}

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDHYPE_H_
