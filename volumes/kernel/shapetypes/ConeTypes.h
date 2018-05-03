/*
 * @file ConeTypes.h
 * @purpose Contains all cone types
 * @author Sandro Wenzel
 *
 * 140514 Sandro Wenzel  - Created
 * 180303 Guilherme Lima - Adapted for specialization and unplaced shape factory
 *
 */

#ifndef VECGEOM_VOLUMES_KERNEL_SHAPETYPES_CONETYPES_H_
#define VECGEOM_VOLUMES_KERNEL_SHAPETYPES_CONETYPES_H_

//#include "volumes/UnplacedCone.h"
//#include "volumes/ConeStruct.h"

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes, struct, UniversalCone, UniversalCone);

#ifndef VECGEOM_NO_SPECIALIZATION

VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes, struct, NonHollowCone, UniversalCone);
VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes, struct, NonHollowConeWithSmallerThanPiSector, UniversalCone);
VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes, struct, NonHollowConeWithBiggerThanPiSector, UniversalCone);
VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes, struct, NonHollowConeWithPiSector, UniversalCone);

VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes, struct, HollowCone, UniversalCone);
VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes, struct, HollowConeWithSmallerThanPiSector, UniversalCone);
VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes, struct, HollowConeWithBiggerThanPiSector, UniversalCone);
VECGEOM_DEVICE_DECLARE_NS_CONV(ConeTypes, struct, HollowConeWithPiSector, UniversalCone);

#endif // VECGEOM_NO_SPECIALIZATION

inline namespace VECGEOM_IMPL_NAMESPACE {
namespace ConeTypes {

#define DEFINE_CONE_TYPE(name)                      \
  struct name {                                     \
    VECCORE_ATT_HOST_DEVICE                         \
    static char const *toString() { return #name; } \
  }

// A cone that encompasses all cases - not specialized and will do extra checks at runtime
DEFINE_CONE_TYPE(UniversalCone);

//#ifndef VECGEOM_NO_SPECIALIZATION

// A cone not having rmin or phi sector
DEFINE_CONE_TYPE(NonHollowCone);
// A cone without rmin but with a phi sector smaller than pi
DEFINE_CONE_TYPE(NonHollowConeWithSmallerThanPiSector);
// A cone without rmin but with a phi sector greater than pi
DEFINE_CONE_TYPE(NonHollowConeWithBiggerThanPiSector);
// A cone without rmin but with a phi sector equal to pi
DEFINE_CONE_TYPE(NonHollowConeWithPiSector);

// A cone with rmin and no phi sector
DEFINE_CONE_TYPE(HollowCone);
// A cone with rmin and a phi sector smaller than pi
DEFINE_CONE_TYPE(HollowConeWithSmallerThanPiSector);
// A cone with rmin and a phi sector greater than pi
DEFINE_CONE_TYPE(HollowConeWithBiggerThanPiSector);
// A cone with rmin and a phi sector equal to pi
DEFINE_CONE_TYPE(HollowConeWithPiSector);

//#endif // VECGEOM_NO_SPECIALIZATION

#undef DEFINE_CONE_TYPE

// Mapping of cone types to certain characteristics
enum ETreatmentType { kYes = 0, kNo, kUnknown };

// asking for phi treatment
template <typename T>
struct NeedsPhiTreatment {
  static const ETreatmentType value = kYes;
};

#ifndef VECGEOM_NO_SPECIALIZATION

template <>
struct NeedsPhiTreatment<NonHollowCone> {
  static const ETreatmentType value = kNo;
};
template <>
struct NeedsPhiTreatment<HollowCone> {
  static const ETreatmentType value = kNo;
};

#endif // VECGEOM_NO_SPECIALIZATION

template <>
struct NeedsPhiTreatment<UniversalCone> {
  static const ETreatmentType value = kUnknown;
};

template <typename T, typename UnplacedCone>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
bool checkPhiTreatment(const UnplacedCone &cone)
{
  if (NeedsPhiTreatment<T>::value != kUnknown)
    return NeedsPhiTreatment<T>::value == kYes;
  else
    return cone.fDPhi < vecgeom::kTwoPi;
}

// asking for rmin treatment
template <typename T>
struct NeedsRminTreatment {
  static const ETreatmentType value = kYes;
};

#ifndef VECGEOM_NO_SPECIALIZATION

template <>
struct NeedsRminTreatment<NonHollowCone> {
  static const ETreatmentType value = kNo;
};
template <>
struct NeedsRminTreatment<NonHollowConeWithSmallerThanPiSector> {
  static const ETreatmentType value = kNo;
};
template <>
struct NeedsRminTreatment<NonHollowConeWithBiggerThanPiSector> {
  static const ETreatmentType value = kNo;
};
template <>
struct NeedsRminTreatment<NonHollowConeWithPiSector> {
  static const ETreatmentType value = kNo;
};

#endif // VECGEOM_NO_SPECIALIZATION

template <>
struct NeedsRminTreatment<UniversalCone> {
  static const ETreatmentType value = kUnknown;
};

template <typename T, typename UnplacedCone>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
bool checkRminTreatment(const UnplacedCone &cone)
{
  if (NeedsRminTreatment<T>::value != kUnknown)
    return NeedsRminTreatment<T>::value == kYes;
  else
    // return cone.GetRmin1() > 0 || cone.GetRmin2() > 0;
    return cone.fRmin1 > 0 || cone.fRmin2 > 0;
}

// sector size
enum EAngleType { kNoAngle = 0, kSmallerThanPi, kOnePi, kBiggerThanPi, kUnknownAngle };

template <typename T>
struct SectorType {
  static const EAngleType value = kNoAngle;
};

template <>
struct SectorType<UniversalCone> {
  static const EAngleType value = kUnknownAngle;
};

#ifndef VECGEOM_NO_SPECIALIZATION

template <>
struct SectorType<NonHollowConeWithSmallerThanPiSector> {
  static const EAngleType value = kSmallerThanPi;
};

template <>
struct SectorType<NonHollowConeWithPiSector> {
  static const EAngleType value = kOnePi;
};

template <>
struct SectorType<NonHollowConeWithBiggerThanPiSector> {
  static const EAngleType value = kBiggerThanPi;
};
template <>
struct SectorType<HollowConeWithSmallerThanPiSector> {
  static const EAngleType value = kSmallerThanPi;
};

template <>
struct SectorType<HollowConeWithPiSector> {
  static const EAngleType value = kOnePi;
};

template <>
struct SectorType<HollowConeWithBiggerThanPiSector> {
  static const EAngleType value = kBiggerThanPi;
};

#endif // VECGEOM_NO_SPECIALIZATION

} // End namespace CONETYPES
} // End VECGEOM_IMPL_NAMESPACE
} // End global namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_SHAPETYPES_CONETYPES_H_
