#ifndef VECGEOM_VOLUMES_SPECIALIZEDTUBE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDTUBE_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/TubeImplementation.h"
#include "VecGeom/volumes/PlacedTube.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT, typename tubeTypeT>
using SpecializedTube = SIMDSpecializedVolImplHelper<TubeImplementation<tubeTypeT>, transCodeT, rotCodeT>;

using SimpleTube = SpecializedTube<translation::kGeneric, rotation::kGeneric, TubeTypes::UniversalTube>;

template <typename Type>
template <TranslationCode transCodeT, RotationCode rotCodeT>
VECCORE_ATT_DEVICE
VPlacedVolume *SUnplacedTube<Type>::Create(LogicalVolume const *const logical_volume,
                                           Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                           const int id, const int copy_no, const int child_id,
#endif
                                           VPlacedVolume *const placement)
{
  (void)placement;
  return new SpecializedTube<transCodeT, rotCodeT, Type>(logical_volume, transformation
#ifdef VECCORE_CUDA
                                                         ,
                                                         id, copy_no, child_id
#endif
  );
}

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDTUBE_H_
