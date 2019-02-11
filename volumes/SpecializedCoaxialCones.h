/// @file SpecializedCoaxialCones.h
/// @author Raman Sehgal (raman.sehgal@cern.ch), Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDCOAXIALCONES_H_
#define VECGEOM_VOLUMES_SPECIALIZEDCOAXIALCONES_H_

#include "base/Global.h"

#include "volumes/kernel/CoaxialConesImplementation.h"
#include "volumes/PlacedCoaxialCones.h"
#include "volumes/SpecializedPlacedVolImplHelper.h"
#include "volumes/UnplacedCoaxialCones.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedCoaxialCones = SIMDSpecializedVolImplHelper<CoaxialConesImplementation, transCodeT, rotCodeT>;

using SimpleCoaxialCones = SpecializedCoaxialCones<translation::kGeneric, rotation::kGeneric>;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDCOAXIALCONES_H_
