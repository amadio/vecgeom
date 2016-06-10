#include "volumes/UnplacedVolume.h"
#include "volumes/PlacedVolume.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

using Real_v = vecgeom::VectorBackend::Real_v;

// trivial implementations for the interface functions
// (since we are moving to these interfaces only gradually)
// ---------------- Contains --------------------------------------------------------------------

VECGEOM_CUDA_HEADER_BOTH
bool VUnplacedVolume::Contains(Vector3D<Precision> const &p) const
{
#ifndef VECGEOM_NVCC
  throw std::runtime_error("unimplemented function called");
#endif
  return false;
}

VECGEOM_CUDA_HEADER_BOTH
EnumInside VUnplacedVolume::Inside(Vector3D<Precision> const &p) const
{
#ifndef VECGEOM_NVCC
  throw std::runtime_error("unimplemented function called");
#endif
  return EnumInside(0);
}

// ---------------- DistanceToOut functions -----------------------------------------------------

VECGEOM_CUDA_HEADER_BOTH
Precision VUnplacedVolume::DistanceToOut(Vector3D<Precision> const &p, Vector3D<Precision> const &d,
                                Precision const &step_max ) const
{
#ifndef VECGEOM_NVCC
  throw std::runtime_error("unimplemented function called");
#endif
  return -1.;
}

// the USolid/GEANT4-like interface for DistanceToOut (returning also exiting normal)
VECGEOM_CUDA_HEADER_BOTH
Precision VUnplacedVolume::DistanceToOut(Vector3D<Precision> const &p, Vector3D<Precision> const &d, Vector3D<Precision> &normal,
                                bool &convex, Precision const &step_max) const
{
#ifndef VECGEOM_NVCC
  throw std::runtime_error("unimplemented function called");
#endif
  return -1.;
}

// an explicit SIMD interface
VECGEOM_CUDA_HEADER_BOTH
Real_v VUnplacedVolume::DistanceToOutVec(Vector3D<Real_v> const &p, Vector3D<Real_v> const &d, Real_v const &step_max) const
{
#ifndef VECGEOM_NVCC
  throw std::runtime_error("unimplemented function called");
#endif
  return Real_v(-1.);
}

// the container/basket interface (possibly to be deprecated)
void VUnplacedVolume::DistanceToOut(SOA3D<Precision> const &points, SOA3D<Precision> const &directions,
                                    Precision const *const step_max, Precision *const output) const
{
#ifndef VECGEOM_NVCC
  throw std::runtime_error("unimplemented function called");
#endif
}

// ---------------- SafetyToOut functions -----------------------------------------------------

VECGEOM_CUDA_HEADER_BOTH
Precision VUnplacedVolume::SafetyToOut(Vector3D<Precision> const &p) const
{
#ifndef VECGEOM_NVCC
  throw std::runtime_error("unimplemented function called");
#endif
  return -1.;
}

// an explicit SIMD interface
Real_v VUnplacedVolume::SafetyToOutVec(Vector3D<Real_v> const &p) const
{
#ifndef VECGEOM_NVCC
  throw std::runtime_error("unimplemented function called");
#endif
  return Real_v(-1.);
}

// the container/basket interface (possibly to be deprecated)
void VUnplacedVolume::SafetyToOut(SOA3D<Precision> const &points, Precision *const output) const
{
#ifndef VECGEOM_NVCC
  throw std::runtime_error("unimplemented function called");
#endif
}

// ---------------- DistanceToIn functions -----------------------------------------------------

VECGEOM_CUDA_HEADER_BOTH
Precision VUnplacedVolume::DistanceToIn(Vector3D<Precision> const &position, Vector3D<Precision> const &direction,
                               const Precision &step_max ) const
{
#ifndef VECGEOM_NVCC
  throw std::runtime_error("unimplemented function called");
#endif
  return -1.;
}

VECGEOM_CUDA_HEADER_BOTH
Real_v VUnplacedVolume::DistanceToInVec(Vector3D<Real_v> const &position, Vector3D<Real_v> const &direction,
                               const Real_v &step_max ) const
{
#ifndef VECGEOM_NVCC
  throw std::runtime_error("unimplemented function called");
#endif
  return Real_v(-1.);
}

// ---------------- SafetyToIn functions -------------------------------------------------------

VECGEOM_CUDA_HEADER_BOTH
Precision VUnplacedVolume::SafetyToIn(Vector3D<Precision> const &position) const
{
#ifndef VECGEOM_NVCC
  throw std::runtime_error("unimplemented function called");
#endif
  return -1.;
}

// explicit SIMD interface
Real_v VUnplacedVolume::SafetyToInVec(Vector3D<Real_v> const &p) const
{
#ifndef VECGEOM_NVCC
  throw std::runtime_error("unimplemented function called");
#endif
  return Real_v(-1.);
}

// ---------------- Normal ---------------------------------------------------------------------

VECGEOM_CUDA_HEADER_BOTH
bool VUnplacedVolume::Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const
{
#ifndef VECGEOM_NVCC
  throw std::runtime_error("unimplemented function called");
#endif
  return false;
}

// ---------------- GetPointOnSurface ----------------------------------------------------------
Vector3D<Precision> VUnplacedVolume::GetPointOnSurface() const
{
  throw std::runtime_error("unimplemented function called");
  return Vector3D<Precision>();
}

// ----------------- Extent --------------------------------------------------------------------
VECGEOM_CUDA_HEADER_BOTH
void VUnplacedVolume::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
#ifndef VECGEOM_NVCC
  throw std::runtime_error("unimplemented function called");
#endif
}

std::ostream &operator<<(std::ostream &os, VUnplacedVolume const &vol)
{
  vol.Print(os);
  return os;
}

#ifndef VECGEOM_NVCC

VPlacedVolume *VUnplacedVolume::PlaceVolume(LogicalVolume const *const volume,
                                            Transformation3D const *const transformation,
                                            VPlacedVolume *const placement) const
{

  const TranslationCode trans_code = transformation->GenerateTranslationCode();
  const RotationCode rot_code = transformation->GenerateRotationCode();

  return SpecializedVolume(volume, transformation, trans_code, rot_code, placement);
}

VPlacedVolume *VUnplacedVolume::PlaceVolume(char const *const label, LogicalVolume const *const volume,
                                            Transformation3D const *const transformation,
                                            VPlacedVolume *const placement) const
{
  VPlacedVolume *const placed = PlaceVolume(volume, transformation, placement);
  placed->set_label(label);
  return placed;
}

#endif
}
} // End global namespace
