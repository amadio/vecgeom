/// @file EllipticalConeImplementation.h
/// @author Raman Sehgal (raman.sehgal@cern.ch), Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_ELLIPTICALCONEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_ELLIPTICALCONEIMPLEMENTATION_H_

#include "base/Vector3D.h"
#include "volumes/EllipticalConeStruct.h"
#include "volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct EllipticalConeImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, EllipticalConeImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedEllipticalCone;
template <typename T>
struct EllipticalConeStruct;
class UnplacedEllipticalCone;

struct EllipticalConeImplementation {

  using PlacedShape_t    = PlacedEllipticalCone;
  using UnplacedStruct_t = EllipticalConeStruct<double>;
  using UnplacedVolume_t = UnplacedEllipticalCone;

  VECCORE_ATT_HOST_DEVICE
  static void PrintType()
  {
    //  printf("SpecializedEllipticalCone<%i, %i>", transCodeT, rotCodeT);
  }

  template <typename Stream>
  static void PrintType(Stream &st, int transCodeT = translation::kGeneric, int rotCodeT = rotation::kGeneric)
  {
    st << "SpecializedEllipticalCone<" << transCodeT << "," << rotCodeT << ">";
  }

  template <typename Stream>
  static void PrintImplementationType(Stream &st)
  {
    (void)st;
    // st << "EllipticalConeImplementation<" << transCodeT << "," << rotCodeT << ">";
  }

  template <typename Stream>
  static void PrintUnplacedType(Stream &st)
  {
    (void)st;
    // TODO: this is wrong
    // st << "UnplacedEllipticalCone";
  }

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void Contains(UnplacedStruct_t const &ellipticalcone, Vector3D<Real_v> const &point, Bool_v &inside)
  {
    Bool_v unused, outside;
    GenericKernelForContainsAndInside<Real_v, Bool_v, false>(ellipticalcone, point, unused, outside);
    inside = !outside;
  }

  // BIG QUESTION: DO WE WANT TO GIVE ALL 3 TEMPLATE PARAMETERS
  // -- OR -- DO WE WANT TO DEDUCE Bool_v, Index_t from Real_v???
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void Inside(UnplacedStruct_t const &ellipticalcone, Vector3D<Real_v> const &point, Inside_t &inside)
  {

    using Bool_v       = vecCore::Mask_v<Real_v>;
    using InsideBool_v = vecCore::Mask_v<Inside_t>;
    Bool_v completelyinside, completelyoutside;
    GenericKernelForContainsAndInside<Real_v, Bool_v, true>(ellipticalcone, point, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyoutside, Inside_t(EInside::kOutside));
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyinside, Inside_t(EInside::kInside));
  }

  template <typename Real_v, typename Bool_v, bool ForInside>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void GenericKernelForContainsAndInside(UnplacedStruct_t const &ellipticalcone, Vector3D<Real_v> const &point,
                                                Bool_v &completelyinside, Bool_v &completelyoutside)
  {
    /* TODO : Logic to check where the point is inside or not.
    **
    ** if ForInside is false then it will only check if the point is outside,
    ** and is used by Contains function
    **
    ** if ForInside is true then it will check whether the point is inside or outside,
    ** and if neither inside nor outside then it is on the surface.
    ** and is used by Inside function
    */

  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void DistanceToIn(UnplacedStruct_t const &ellipticalcone, Vector3D<Real_v> const &point,
                           Vector3D<Real_v> const &direction, Real_v const & /*stepMax*/, Real_v &distance)
  {
    /* TODO :  Logic to calculate Distance from outside point to the EllipticalCone surface */
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void DistanceToOut(UnplacedStruct_t const &ellipticalcone, Vector3D<Real_v> const &point,
                            Vector3D<Real_v> const &direction, Real_v const & /* stepMax */, Real_v &distance)
  {
    /* TODO :  Logic to calculate Distance from inside point to the EllipticalCone surface */
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void SafetyToIn(UnplacedStruct_t const &ellipticalcone, Vector3D<Real_v> const &point, Real_v &safety)
  {
    /* TODO :  Logic to calculate Safety from outside point to the EllipticalCone surface */
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static void SafetyToOut(UnplacedStruct_t const &ellipticalcone, Vector3D<Real_v> const &point, Real_v &safety)
  {
    /* TODO :  Logic to calculate Safety from inside point to the EllipticalCone surface */
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static Vector3D<Real_v> NormalKernel(UnplacedStruct_t const &ellipticalcone, Vector3D<Real_v> const &point,
                                       typename vecCore::Mask_v<Real_v> &valid)
  {
    // Computes the normal on a surface and returns it as a unit vector
	  Vector3D<Real_v> normal(0.,0.,0.);
    return normal;
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_ELLIPTICALCONEIMPLEMENTATION_H_
