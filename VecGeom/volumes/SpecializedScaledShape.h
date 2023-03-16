/// \file SpecializedScaledShape.h
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDSCALEDSHAPE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDSCALEDSHAPE_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/kernel/ScaledShapeImplementation.h"
#include "VecGeom/volumes/PlacedScaledShape.h"
#include "VecGeom/volumes/UnplacedScaledShape.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

using SpecializedScaledShape = SpecializedVolImplHelper<ScaledShapeImplementation>;

using SimpleScaledShape = SpecializedScaledShape;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDSCALEDSHAPE_H_
