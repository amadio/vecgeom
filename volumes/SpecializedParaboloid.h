/// @file SpecializedParaboloid.h
/// Original Implementation Marilena Bandieramonte (marilena.bandieramonte@cern.ch)
///
/// revision + moving to new backend structure : Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDPARABOLOID_H_
#define VECGEOM_VOLUMES_SPECIALIZEDPARABOLOID_H_

#include "base/Global.h"

#include "volumes/kernel/ParaboloidImplementation.h"
#include "volumes/PlacedParaboloid.h"
#include "volumes/SpecializedPlacedVolImplHelper.h"
#include "volumes/UnplacedParaboloid.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedParaboloid = SIMDSpecializedVolImplHelper<ParaboloidImplementation, transCodeT, rotCodeT>;

using SimpleParaboloid = SpecializedParaboloid<translation::kGeneric, rotation::kGeneric>;
}
} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDPARABOLOID_H_
