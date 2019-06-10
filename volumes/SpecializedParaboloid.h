// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of the specialized paraboloid volume.
/// @file volumes/SpecializedParaboloid.h
/// @author Marilena Bandieramonte

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
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDPARABOLOID_H_
