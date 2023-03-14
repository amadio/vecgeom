// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of the specialized tetrahedron volume
/// @file volumes/SpecializedTet.h
/// @author Raman Sehgal, Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_SPECIALIZEDTET_H_
#define VECGEOM_VOLUMES_SPECIALIZEDTET_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/TetImplementation.h"
#include "VecGeom/volumes/PlacedTet.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedTet.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedTet = SpecializedVolImplHelper<TetImplementation, transCodeT, rotCodeT>;

using SimpleTet = SpecializedTet<translation::kGeneric, rotation::kGeneric>;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDTET_H_
