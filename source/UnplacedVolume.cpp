#include "volumes/UnplacedVolume.h"
#include "volumes/PlacedVolume.h"
#include "base/SOA3D.h"
#include "volumes/utilities/VolumeUtilities.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

using Vec3D  = Vector3D<Precision>;
using Real_v = vecgeom::VectorBackend::Real_v;

// generic implementation for SamplePointOnSurface
Vector3D<Precision> VUnplacedVolume::SamplePointOnSurface() const
{
  Vector3D<Precision> surfacepoint;
  SOA3D<Precision> points(1);
  volumeUtilities::FillRandomPoints(*this, points);

  Vector3D<Precision> dir = volumeUtilities::SampleDirection();
  surfacepoint            = points[0] + DistanceToOut(points[0], dir) * dir;

  // assert( Inside(surfacepoint) == vecgeom::kSurface );
  return surfacepoint;
}

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

// ----------------- Extent --------------------------------------------------------------------
VECCORE_ATT_HOST_DEVICE
void VUnplacedVolume::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
#ifndef VECCORE_CUDA
  throw std::runtime_error("unimplemented function called");
#endif
}

// estimating the surface area by sampling
// based on the six-point method of G4
Precision VUnplacedVolume::EstimateSurfaceArea(int nStat) const
{
  static const double s2 = 1. / Sqrt(2.);
  static const double s3 = 1. / Sqrt(3.);

  // Predefined directions
  //
  static const Vec3D directions[64] = {
      Vec3D(0, 0, 0),      Vec3D(-1, 0, 0),    // (  ,  ,  ) ( -,  ,  )
      Vec3D(1, 0, 0),      Vec3D(-1, 0, 0),    // ( +,  ,  ) (-+,  ,  )
      Vec3D(0, -1, 0),     Vec3D(-s2, -s2, 0), // (  , -,  ) ( -, -,  )
      Vec3D(s2, -s2, 0),   Vec3D(0, -1, 0),    // ( +, -,  ) (-+, -,  )

      Vec3D(0, 1, 0),      Vec3D(-s2, s2, 0), // (  , +,  ) ( -, +,  )
      Vec3D(s2, s2, 0),    Vec3D(0, 1, 0),    // ( +, +,  ) (-+, +,  )
      Vec3D(0, -1, 0),     Vec3D(-1, 0, 0),   // (  ,-+,  ) ( -,-+,  )
      Vec3D(1, 0, 0),      Vec3D(-1, 0, 0),   // ( +,-+,  ) (-+,-+,  )

      Vec3D(0, 0, -1),     Vec3D(-s2, 0, -s2),   // (  ,  , -) ( -,  , -)
      Vec3D(s2, 0, -s2),   Vec3D(0, 0, -1),      // ( +,  , -) (-+,  , -)
      Vec3D(0, -s2, -s2),  Vec3D(-s3, -s3, -s3), // (  , -, -) ( -, -, -)
      Vec3D(s3, -s3, -s3), Vec3D(0, -s2, -s2),   // ( +, -, -) (-+, -, -)

      Vec3D(0, s2, -s2),   Vec3D(-s3, s3, -s3), // (  , +, -) ( -, +, -)
      Vec3D(s3, s3, -s3),  Vec3D(0, s2, -s2),   // ( +, +, -) (-+, +, -)
      Vec3D(0, 0, -1),     Vec3D(-s2, 0, -s2),  // (  ,-+, -) ( -,-+, -)
      Vec3D(s2, 0, -s2),   Vec3D(0, 0, -1),     // ( +,-+, -) (-+,-+, -)

      Vec3D(0, 0, 1),      Vec3D(-s2, 0, s2),   // (  ,  , +) ( -,  , +)
      Vec3D(s2, 0, s2),    Vec3D(0, 0, 1),      // ( +,  , +) (-+,  , +)
      Vec3D(0, -s2, s2),   Vec3D(-s3, -s3, s3), // (  , -, +) ( -, -, +)
      Vec3D(s3, -s3, s3),  Vec3D(0, -s2, s2),   // ( +, -, +) (-+, -, +)

      Vec3D(0, s2, s2),    Vec3D(-s3, s3, s3), // (  , +, +) ( -, +, +)
      Vec3D(s3, s3, s3),   Vec3D(0, s2, s2),   // ( +, +, +) (-+, +, +)
      Vec3D(0, 0, 1),      Vec3D(-s2, 0, s2),  // (  ,-+, +) ( -,-+, +)
      Vec3D(s2, 0, s2),    Vec3D(0, 0, 1),     // ( +,-+, +) (-+,-+, +)

      Vec3D(0, 0, -1),     Vec3D(-1, 0, 0),    // (  ,  ,-+) ( -,  ,-+)
      Vec3D(1, 0, 0),      Vec3D(-1, 0, 0),    // ( +,  ,-+) (-+,  ,-+)
      Vec3D(0, -1, 0),     Vec3D(-s2, -s2, 0), // (  , -,-+) ( -, -,-+)
      Vec3D(s2, -s2, 0),   Vec3D(0, -1, 0),    // ( +, -,-+) (-+, -,-+)

      Vec3D(0, 1, 0),      Vec3D(-s2, s2, 0), // (  , +,-+) ( -, +,-+)
      Vec3D(s2, s2, 0),    Vec3D(0, 1, 0),    // ( +, +,-+) (-+, +,-+)
      Vec3D(0, -1, 0),     Vec3D(-1, 0, 0),   // (  ,-+,-+) ( -,-+,-+)
      Vec3D(1, 0, 0),      Vec3D(-1, 0, 0),   // ( +,-+,-+) (-+,-+,-+)
  };

  // Get bounding box
  //
  Vec3D bmin, bmax;
  this->Extent(bmin, bmax);
  Vec3D bdim = bmax - bmin;

  // Define statistics and shell thickness
  //
  int npoints   = (nStat < 1000) ? 1000 : nStat;
  double coeff  = 0.5 / Cbrt(double(npoints));
  double eps    = coeff * bdim.Min(); // half thickness
  double twoeps = 2. * eps;
  double del    = 1.8 * eps; // six-point offset - should be more than sqrt(3.)

  // Enlarge bounding box by eps
  //
  bmin -= Vec3D(eps);
  bdim += Vec3D(twoeps);

  // Calculate surface area
  //
  int icount = 0;
  for (int i = 0; i < npoints; ++i) {
    double px = bmin.x() + bdim.x() * RNG::Instance().uniform();
    double py = bmin.y() + bdim.y() * RNG::Instance().uniform();
    double pz = bmin.z() + bdim.z() * RNG::Instance().uniform();
    Vec3D p(px, py, pz);
    EnumInside in = this->Inside(p);
    double dist   = 0;
    if (in == EInside::kInside) {
      if (this->SafetyToOut(p) >= eps) continue;
      int icase = 0;
      if (this->Inside(Vec3D(px - del, py, pz)) != EInside::kInside) icase += 1;
      if (this->Inside(Vec3D(px + del, py, pz)) != EInside::kInside) icase += 2;
      if (this->Inside(Vec3D(px, py - del, pz)) != EInside::kInside) icase += 4;
      if (this->Inside(Vec3D(px, py + del, pz)) != EInside::kInside) icase += 8;
      if (this->Inside(Vec3D(px, py, pz - del)) != EInside::kInside) icase += 16;
      if (this->Inside(Vec3D(px, py, pz + del)) != EInside::kInside) icase += 32;
      if (icase == 0) continue;
      Vec3D v = directions[icase];
      dist    = this->DistanceToOut(p, v);
      Vec3D n;
      this->Normal(p + v * dist, n);
      dist *= v.Dot(n);
    } else if (in == EInside::kOutside) {
      if (this->SafetyToIn(p) >= eps) continue;
      int icase = 0;
      if (this->Inside(Vec3D(px - del, py, pz)) != EInside::kOutside) icase += 1;
      if (this->Inside(Vec3D(px + del, py, pz)) != EInside::kOutside) icase += 2;
      if (this->Inside(Vec3D(px, py - del, pz)) != EInside::kOutside) icase += 4;
      if (this->Inside(Vec3D(px, py + del, pz)) != EInside::kOutside) icase += 8;
      if (this->Inside(Vec3D(px, py, pz - del)) != EInside::kOutside) icase += 16;
      if (this->Inside(Vec3D(px, py, pz + del)) != EInside::kOutside) icase += 32;
      if (icase == 0) continue;
      Vec3D v = directions[icase];
      dist    = this->DistanceToIn(p, v);
      if (dist == kInfLength) continue;
      Vec3D n;
      this->Normal(p + v * dist, n);
      dist *= -(v.Dot(n));
    }
    if (dist < eps) icount++;
  }
  return bdim.x() * bdim.y() * bdim.z() * icount / npoints / twoeps;
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
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom
