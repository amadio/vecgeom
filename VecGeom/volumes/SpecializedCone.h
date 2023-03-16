/*
 * SpecializedCone.h
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_SPECIALIZEDCONE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDCONE_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/kernel/ConeImplementation.h"
#include "VecGeom/volumes/PlacedCone.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename coneTypeT>
using SpecializedCone = SpecializedVolImplHelper<ConeImplementation<coneTypeT>>;

using SimpleCone = SpecializedCone<ConeTypes::UniversalCone>;

template <typename Type>
VECCORE_ATT_DEVICE VPlacedVolume *SUnplacedCone<Type>::Create(LogicalVolume const *const logical_volume,
                                                              Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                                              const int id, const int copy_no, const int child_id,
#endif
                                                              VPlacedVolume *const placement)
{
  (void)placement;
  return new SpecializedCone<Type>(logical_volume, transformation
#ifdef VECCORE_CUDA
                                   ,
                                   id, copy_no, child_id
#endif
  );
}

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDCONE_H_
