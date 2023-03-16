// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of the specialized Trd volume
/// @file volumes/SpecializedTrd.h
/// @author Georgios Bitzes

#ifndef VECGEOM_VOLUMES_SPECIALIZEDTRD_H_
#define VECGEOM_VOLUMES_SPECIALIZEDTRD_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/TrdImplementation.h"
#include "VecGeom/volumes/PlacedTrd.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename TrdTypeT>
using SpecializedTrd = SpecializedVolImplHelper<TrdImplementation<TrdTypeT>>;

using SimpleTrd = SpecializedTrd<TrdTypes::UniversalTrd>;

template <typename Type>
VECCORE_ATT_DEVICE VPlacedVolume *SUnplacedTrd<Type>::Create(LogicalVolume const *const logical_volume,
                                                             Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                                             const int id, const int copy_no, const int child_id,
#endif
                                                             VPlacedVolume *const placement)
{
  (void)placement;
  return new SpecializedTrd<Type>(logical_volume, transformation
#ifdef VECCORE_CUDA
                                  ,
                                  id, copy_no, child_id
#endif
  );
}

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDTRD_H_
