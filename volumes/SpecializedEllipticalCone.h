/// @file SpecializedEllipticalCone.h
/// @author Raman Sehgal (raman.sehgal@cern.ch), Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDELLIPTICALCONE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDELLIPTICALCONE_H_

#include "base/Global.h"

#include "volumes/kernel/EllipticalConeImplementation.h"
#include "volumes/PlacedEllipticalCone.h"
#include "volumes/SpecializedPlacedVolImplHelper.h"
#include "volumes/UnplacedEllipticalCone.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedEllipticalCone = SIMDSpecializedVolImplHelper<EllipticalConeImplementation, transCodeT, rotCodeT>;

using SimpleEllipticalCone = SpecializedEllipticalCone<translation::kGeneric, rotation::kGeneric>;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDELLIPTICALCONE_H_
