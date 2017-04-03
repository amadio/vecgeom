#ifndef VECGEOM_VOLUMES_SPECIALIZEDTUBE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDTUBE_H_

#include "base/Global.h"

#include "volumes/kernel/TubeImplementation.h"
#include "volumes/PlacedTube.h"
#include "volumes/SpecializedPlacedVolImplHelper.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT, typename tubeTypeT>
using SpecializedTube = SIMDSpecializedVolImplHelper<TubeImplementation<tubeTypeT>, transCodeT, rotCodeT>;

using SimpleTube = SpecializedTube<translation::kGeneric, rotation::kGeneric, TubeTypes::UniversalTube>;

template <TranslationCode transCodeT, RotationCode rotCodeT>
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedTube::Create(LogicalVolume const *const logical_volume,
                                    Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
                                    const int id,
#endif
                                    VPlacedVolume *const placement)
{
  (void)placement;
  return new SimpleTube(logical_volume, transformation
#ifdef VECGEOM_NVCC
                        ,
                        id
#endif
                        );
}
}
} // End global namespace

#endif
