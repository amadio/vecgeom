// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of the specialized parallelepiped volume.
/// @file volumes/SpecializedParallelepiped.h
/// @author Johannes de Fine Licht

#ifndef VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_
#define VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/ParallelepipedImplementation.h"
#include "VecGeom/volumes/PlacedParallelepiped.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedParallelepiped.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

using SpecializedParallelepiped = SpecializedVolImplHelper<ParallelepipedImplementation>;

using SimpleParallelepiped = SpecializedParallelepiped;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDPARALLELEPIPED_H_
