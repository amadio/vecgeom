// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of the specialized parallelepiped volume.
/// @file volumes/SpecializedParallelepiped.h
/// @author Johannes de Fine Licht

#ifndef VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_
#define VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_

#include "base/Global.h"

#include "volumes/kernel/ParallelepipedImplementation.h"
#include "volumes/PlacedParallelepiped.h"
#include "volumes/SpecializedPlacedVolImplHelper.h"
#include "volumes/UnplacedParallelepiped.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedParallelepiped = SIMDSpecializedVolImplHelper<ParallelepipedImplementation, transCodeT, rotCodeT>;

using SimpleParallelepiped = SpecializedParallelepiped<translation::kGeneric, rotation::kGeneric>;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_
