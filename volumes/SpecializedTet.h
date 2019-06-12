// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of the specialized tetrahedron volume
/// @file volumes/SpecializedTet.h
/// @author Raman Sehgal, Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_SPECIALIZEDTET_H_
#define VECGEOM_VOLUMES_SPECIALIZEDTET_H_

#include "base/Global.h"

#include "volumes/kernel/TetImplementation.h"
#include "volumes/PlacedTet.h"
#include "volumes/SpecializedPlacedVolImplHelper.h"
#include "volumes/UnplacedTet.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedTet = SIMDSpecializedVolImplHelper<TetImplementation, transCodeT, rotCodeT>;

using SimpleTet = SpecializedTet<translation::kGeneric, rotation::kGeneric>;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDTET_H_
