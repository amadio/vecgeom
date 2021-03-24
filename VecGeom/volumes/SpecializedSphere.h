/// @file SpecializedSphere.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDSPHERE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDSPHERE_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/kernel/SphereImplementation.h"
#include "VecGeom/volumes/PlacedSphere.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedSphere.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedSphere = SpecializedVolImplHelper<SphereImplementation, transCodeT, rotCodeT>;
using SimpleSphere      = SpecializedSphere<translation::kGeneric, rotation::kGeneric>;
}
} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDSPHERE_H_
