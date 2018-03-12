#include "volumes/UnplacedVolume.h"
#include "volumes/PlacedVolume.h"
#include "volumes/utilities/VolumeUtilities.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

using Real_v = vecgeom::VectorBackend::Real_v;

// trivial implementations for the interface functions
// (since we are moving to these interfaces only gradually)

// ---------------- DistanceToOut functions -----------------------------------------------------

// the USolid/GEANT4-like interface for DistanceToOut (returning also exiting normal)
VECCORE_ATT_HOST_DEVICE
Precision VUnplacedVolume::DistanceToOut(Vector3D<Precision> const &p, Vector3D<Precision> const &d,
                                         Vector3D<Precision> &normal, bool &convex, Precision step_max) const
{
#ifndef VECCORE_CUDA
  throw std::runtime_error("unimplemented function called");
#endif
  return -1.;
}

// an explicit SIMD interface
VECCORE_ATT_HOST_DEVICE
Real_v VUnplacedVolume::DistanceToOutVec(Vector3D<Real_v> const &p, Vector3D<Real_v> const &d,
                                         Real_v const &step_max) const
{
#ifndef VECCORE_CUDA
  throw std::runtime_error("unimplemented function called");
#endif
  return Real_v(-1.);
}

// the container/basket interface (possibly to be deprecated)
void VUnplacedVolume::DistanceToOut(SOA3D<Precision> const &points, SOA3D<Precision> const &directions,
                                    Precision const *const step_max, Precision *const output) const
{
#ifndef VECCORE_CUDA
  throw std::runtime_error("unimplemented function called");
#endif
}

// ---------------- SafetyToOut functions -----------------------------------------------------
// an explicit SIMD interface
Real_v VUnplacedVolume::SafetyToOutVec(Vector3D<Real_v> const &p) const
{
#ifndef VECCORE_CUDA
  throw std::runtime_error("unimplemented function called");
#endif
  return Real_v(-1.);
}

// the container/basket interface (possibly to be deprecated)
void VUnplacedVolume::SafetyToOut(SOA3D<Precision> const &points, Precision *const output) const
{
#ifndef VECCORE_CUDA
  throw std::runtime_error("unimplemented function called");
#endif
}

// ---------------- DistanceToIn functions -----------------------------------------------------
VECCORE_ATT_HOST_DEVICE
Real_v VUnplacedVolume::DistanceToInVec(Vector3D<Real_v> const &position, Vector3D<Real_v> const &direction,
                                        const Real_v &step_max) const
{
#ifndef VECCORE_CUDA
  throw std::runtime_error("unimplemented function called");
#endif
  return Real_v(-1.);
}

// ---------------- SafetyToIn functions -------------------------------------------------------
// explicit SIMD interface
Real_v VUnplacedVolume::SafetyToInVec(Vector3D<Real_v> const &p) const
{
#ifndef VECCORE_CUDA
  throw std::runtime_error("unimplemented function called");
#endif
  return Real_v(-1.);
}

// ---------------- Normal ---------------------------------------------------------------------

VECCORE_ATT_HOST_DEVICE
bool VUnplacedVolume::Normal(Vector3D<Precision> const &p, Vector3D<Precision> &normal) const
{
#ifndef VECCORE_CUDA
  throw std::runtime_error("unimplemented function called");
#endif
  return false;
}

// ---------------- SamplePointOnSurface ----------------------------------------------------------
Vector3D<Precision> VUnplacedVolume::SamplePointOnSurface() const
{
  throw std::runtime_error("unimplemented function called");
  return Vector3D<Precision>();
}

// ----------------- Extent --------------------------------------------------------------------
VECCORE_ATT_HOST_DEVICE
void VUnplacedVolume::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
#ifndef VECCORE_CUDA
  throw std::runtime_error("unimplemented function called");
#endif
}

// estimating the surface area by sampling
// based on the method of G4
Precision VUnplacedVolume::EstimateSurfaceArea(int nStat) const
{
  double ell = -1.;
  Vector3D<Precision> p;
  Vector3D<Precision> minCorner;
  Vector3D<Precision> maxCorner;
  Vector3D<Precision> delta;

  // min max extents of pSolid along X,Y,Z
  this->Extent(minCorner, maxCorner);

  // limits
  delta = maxCorner - minCorner;

  if (ell <= 0.) // Automatic definition of skin thickness
  {
    Precision minval = delta.x();
    if (delta.y() < delta.x()) {
      minval = delta.y();
    }
    if (delta.z() < minval) {
      minval = delta.z();
    }
    ell = .01 * minval;
  }

  Precision dd = 2 * ell;
  minCorner.x() -= ell;
  minCorner.y() -= ell;
  minCorner.z() -= ell;
  delta.x() += dd;
  delta.y() += dd;
  delta.z() += dd;

  int inside = 0;
  for (int i = 0; i < nStat; ++i) {
    p = minCorner + Vector3D<Precision>(delta.x() * RNG::Instance().uniform(), delta.y() * RNG::Instance().uniform(),
                                        delta.z() * RNG::Instance().uniform());
    if (this->Contains(p)) {
      if (this->SafetyToOut(p) < ell) {
        inside++;
      }
    } else {
      if (this->SafetyToIn(p) < ell) {
        inside++;
      }
    }
  }
  // @@ The conformal correction can be upgraded
  return delta.x() * delta.y() * delta.z() * inside / dd / nStat;
}

// estimating the cubic volume by sampling
// based on the method of G4
double VUnplacedVolume::EstimateCapacity(int nStat) const
{
  double epsilon = 1E-4;

  // limits
  if (nStat < 100) nStat = 100;

  Vector3D<Precision> lower, upper, offset;
  this->Extent(lower, upper);
  offset                        = 0.5 * (upper + lower);
  const Vector3D<Precision> dim = 0.5 * (upper - lower);

  int insidecounter = 0;
  for (int i = 0; i < nStat; i++) {
    auto p = offset + volumeUtilities::SamplePoint(dim);
    if (this->Contains(p)) insidecounter++;
  }
  return 8. * (dim[0] + epsilon) * (dim[1] + epsilon) * (dim[2] + epsilon) * insidecounter / nStat;
}

std::ostream &operator<<(std::ostream &os, VUnplacedVolume const &vol)
{
  vol.Print(os);
  return os;
}

#ifndef VECCORE_CUDA

VPlacedVolume *VUnplacedVolume::PlaceVolume(LogicalVolume const *const volume,
                                            Transformation3D const *const transformation,
                                            VPlacedVolume *const placement) const
{

  const TranslationCode trans_code = transformation->GenerateTranslationCode();
  const RotationCode rot_code      = transformation->GenerateRotationCode();

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
