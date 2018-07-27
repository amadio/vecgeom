/// @file SpecializedTrd.h
/// @author Georgios Bitzes (georgios.bitzes@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDTRD_H_
#define VECGEOM_VOLUMES_SPECIALIZEDTRD_H_

#include "base/Global.h"

#include "volumes/kernel/TrdImplementation.h"
#include "volumes/PlacedTrd.h"
#include "volumes/SpecializedPlacedVolImplHelper.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT, typename TrdTypeT>
using SpecializedTrd = SIMDSpecializedVolImplHelper<TrdImplementation<TrdTypeT>, transCodeT, rotCodeT>;

using SimpleTrd = SpecializedTrd<translation::kGeneric, rotation::kGeneric, TrdTypes::UniversalTrd>;

template <typename Type>
template <TranslationCode transCodeT, RotationCode rotCodeT>
VECCORE_ATT_DEVICE
VPlacedVolume *SUnplacedTrd<Type>::Create(LogicalVolume const *const logical_volume,
                                          Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                          const int id,
#endif
                                          VPlacedVolume *const placement)
{
  (void)placement;
  return new SpecializedTrd<transCodeT, rotCodeT, Type>(logical_volume, transformation
#ifdef VECCORE_CUDA
                                                        ,
                                                        id
#endif
  );
}

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDTRD_H_
