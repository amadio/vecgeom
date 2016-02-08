/// \file UnplacedGenTrap.cpp

//#include "base/Global.h"
//#include "backend/Backend.h"
#include "volumes/UnplacedGenTrap.h"
#include <ostream>
#include <iomanip>
#include <iostream>
#include "management/VolumeFactory.h"
#include "volumes/SpecializedGenTrap.h"
#ifndef VECGEOM_NVCC
#include "base/RNG.h"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

//______________________________________________________________________________
VECGEOM_CUDA_HEADER_BOTH
void UnplacedGenTrap::ComputeBoundingBox() {
  // Computes bounding box parameters
  Vector3D<Precision> aMin, aMax;
  Extent(aMin, aMax);
  fBoundingBoxOrig = 0.5 * (aMin + aMax);
  Vector3D<Precision> halfLengths = 0.5 * (aMax - aMin);
  fBoundingBox.SetX(halfLengths.x());
  fBoundingBox.SetY(halfLengths.y());
  fBoundingBox.SetZ(halfLengths.z());
}

//______________________________________________________________________________
VECGEOM_CUDA_HEADER_BOTH
void UnplacedGenTrap::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const {
  // Returns the full 3D cartesian extent of the solid.
  aMin = aMax = fVertices[0];
  aMin[2] = -fDz;
  aMax[2] = fDz;
  for (int i = 0; i < 4; ++i) {
    // lower -fDz vertices
    if (aMin[0] > fVertices[i].x())
      aMin[0] = fVertices[i].x();
    if (aMax[0] < fVertices[i].x())
      aMax[0] = fVertices[i].x();
    if (aMin[1] > fVertices[i].y())
      aMin[1] = fVertices[i].y();
    if (aMax[1] < fVertices[i].y())
      aMax[1] = fVertices[i].y();
    // upper fDz vertices
    if (aMin[0] > fVertices[i + 4].x())
      aMin[0] = fVertices[i + 4].x();
    if (aMax[0] < fVertices[i + 4].x())
      aMax[0] = fVertices[i + 4].x();
    if (aMin[1] > fVertices[i + 4].y())
      aMin[1] = fVertices[i + 4].y();
    if (aMax[1] < fVertices[i + 4].y())
      aMax[1] = fVertices[i + 4].y();
  }
}

//______________________________________________________________________________
VECGEOM_CUDA_HEADER_BOTH
bool UnplacedGenTrap::SegmentsCrossing(Vector3D<Precision> p, Vector3D<Precision> p1, Vector3D<Precision> q,
                                       Vector3D<Precision> q1) const {
  // Check if 2 segments defined by (p,p1) and (q,q1) are crossing.
  using Vector = Vector3D<Precision>;
  Vector r = p1 - p; // p1 = p+r
  Vector s = q1 - q; // q1 = q+s
  Vector r_cross_s = Vector::Cross(r, s);
  if (r_cross_s.Mag2() < kTolerance) // parallel or colinear - ignore crossing
    return false;
  Precision t = Vector::Cross(q - p, s) / r_cross_s;
  if (t < 0 || t > 1)
    return true;
  Precision u = Vector::Cross(q - p, r) / r_cross_s;
  if (u < 0 || u > 1)
    return true;
  return false;
}

//______________________________________________________________________________
// computes if this gentrap is twisted
VECGEOM_CUDA_HEADER_BOTH
bool UnplacedGenTrap::ComputeIsTwisted() {
  // Computes tangents of twist angles (angles between projections on XY plane
  // of corresponding -dz +dz edges).

  bool twisted = false;
  double dx1, dy1, dx2, dy2;
  const int nv = 4; // half the number of verices

  for (int i = 0; i < 4; ++i) {
    dx1 = fVertices[(i + 1) % nv].x() - fVertices[i].x();
    dy1 = fVertices[(i + 1) % nv].y() - fVertices[i].y();
    if ((dx1 == 0) && (dy1 == 0)) {
      continue;
    }

    dx2 = fVertices[nv + (i + 1) % nv].x() - fVertices[nv + i].x();
    dy2 = fVertices[nv + (i + 1) % nv].y() - fVertices[nv + i].y();

    if ((dx2 == 0 && dy2 == 0)) {
      continue;
    }
    double twist_angle = std::fabs(dy1 * dx2 - dx1 * dy2);
    // attention: this thing was a different tolerance: UGenTrap::tolerance
    if (twist_angle < kTolerance) {
      continue;
    }
    twisted = true;

    // SetTwistAngle(i, twist_angle);
    // Check on big angles, potentially navigation problem

    // calculate twist angle
    // twist_angle = std::acos((dx1 * dx2 + dy1 * dy2)
    //                        /(std::sqrt(dx1 * dx1 + dy1 * dy1)
    //                        *std::sqrt(dx2 * dx2 + dy2 * dy2)));

    //   if (std::fabs(twist_angle) > 0.5 * UUtils::kPi + VUSolid::fgTolerance)
    //   {
    //     std::ostringstream message;
    //     message << "Twisted Angle is bigger than 90 degrees - " << GetName()
    //             << std::endl
    //             << "     Potential problem of malformed Solid !" << std::endl
    //             << "     TwistANGLE = " << twist_angle
    //             << "*rad  for lateral plane N= " << i;
    //   }
  }
  return twisted;
}

//______________________________________________________________________________
// computes if this gentrap top and bottom quadrilaterals are convex
VECGEOM_CUDA_HEADER_BOTH
bool UnplacedGenTrap::ComputeIsConvexQuadrilaterals() {
  for (int i = 0; i < 4; ++i) {
    int j = (i + 1) % 4;
    Precision crossij = fVertices[i].x() * fVertices[j].y() - fVertices[j].x() * fVertices[i].y();
    if (crossij > 0)
      return false;
    crossij = fVertices[i + 4].x() * fVertices[j + 4].y() - fVertices[j + 4].x() * fVertices[i + 4].y();
    if (crossij > 0)
      return false;
  }
  return true;
}

//______________________________________________________________________________
Vector3D<Precision> UnplacedGenTrap::GetPointOnSurface() const {
  // Generate randomly a point on one of the surfaces
  // Select randomly a surface
  int i = int(RNG::Instance().uniform(0., 6.));
  Vector3D<Precision> point;
  if (i < 4) {
    int j = (i + 1) % 4;
    Vector3D<Precision> vi(fVertices[i + 4] - fVertices[i]);
    Vector3D<Precision> vj(fVertices[j + 4] - fVertices[j]);
    Vector3D<Precision> h0(fVertices[j] - fVertices[i]);
    // Random height
    Precision fz = RNG::Instance().uniform(0., 1.);
    // Random fraction along the horizontal hi at selected z
    Precision f = RNG::Instance().uniform(0., 1.);
    point = fVertices[i] + fz * vi + f * h0 + f * fz * (vj - vi);
    return point;
  }
  i -= 4; // now 0 (bottom surface) or 1 (top surface)
  // Select z position
  Precision z = (2 * i - 1) * fDz;
  i *= 4; // now matching the index of the start vertex
  // Compute min/max  in x and y for the selected surface
  // Avoid using the bounding box due to possible point-like top/bottom
  // which would be impossible to sample
  Precision xmin = fVertices[i].x();
  Precision xmax = xmin;
  Precision ymin = fVertices[i].y();
  Precision ymax = ymin;
  for (int j = i + 1; j < i + 4; ++j) {
    if (fVertices[j].x() < xmin)
      xmin = fVertices[j].x();
    if (fVertices[j].x() > xmax)
      xmax = fVertices[j].x();
    if (fVertices[j].y() < ymin)
      ymin = fVertices[j].y();
    if (fVertices[j].y() > ymax)
      ymax = fVertices[j].y();
  }
  Precision cross, x, y;
  bool inside = false;
  while (!inside) {
    inside = true;
    // Now generate randomly between (xmin,xmax) and (ymin,ymax)
    x = RNG::Instance().uniform(xmin, xmax);
    y = RNG::Instance().uniform(ymin, ymax);
    // Now make sure the point (x,y) is on the selected surface. Use same
    // algorithm as for Contains
    for (int j = i; j < i + 4; ++j) {
      int k = i + (j + 1) % 4;
      Precision dx = fVertices[k].x() - fVertices[j].x();
      Precision dy = fVertices[k].y() - fVertices[j].y();
      cross = (x - fVertices[j].x()) * dy - (y - fVertices[j].y()) * dx;
      if (cross < 0.) {
        inside = false;
        break;
      }
    }
  }
  // Now point inside
  point.Set(x, y, z);
  return point;
}

//______________________________________________________________________________
VECGEOM_CUDA_HEADER_BOTH
void UnplacedGenTrap::Print() const {
  printf("--------------------------------------------------------\n");
  printf("    =================================================== \n");
  printf(" Solid geometry type: UnplacedGenTrap \n");
  printf("   half length Z: %f mm \n", fDz);
  printf("   list of vertices:\n");

  for (int i = 0; i < 8; ++i) {
    printf("#%d", i);
    printf("   vx = %f mm", fVertices[i].x());
    printf("   vy = %f mm\n", fVertices[i].y());
  }
  printf("   planar: %s\n", IsPlanar() ? "true" : "false");
}

//______________________________________________________________________________
void UnplacedGenTrap::Print(std::ostream &os) const {
  int oldprc = os.precision(16);
  os << "--------------------------------------------------------\n"
     //     << "    *** Dump for solid - " << GetName() << " *** \n"
     << "    =================================================== \n"
     << " Solid geometry type: UnplacedGenTrap \n"
     << "   half length Z: " << fDz << " mm \n"
     << "   list of vertices:\n";

  for (int i = 0; i < 8; ++i) {
    os << std::setw(5) << "#" << i << "   vx = " << fVertices[i].x() << " mm"
       << "   vy = " << fVertices[i].y() << " mm\n";
  }
  os << "   planar: " << IsPlanar() << std::endl;
  os.precision(oldprc);
}

//
#ifdef VECGEOM_NVCC
VECGEOM_CUDA_HEADER_DEVICE
#endif
VPlacedVolume *UnplacedGenTrap::SpecializedVolume(LogicalVolume const *const volume,
                                                  Transformation3D const *const transformation,
                                                  const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
                                                  const int id,
#endif
                                                  VPlacedVolume *const placement) const {
  return VolumeFactory::CreateByTransformation<UnplacedGenTrap>(volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
                                                                id,
#endif
                                                                placement);
}

//______________________________________________________________________________
template <TranslationCode trans_code, RotationCode rot_code>
#ifdef VECGEOM_NVCC
VECGEOM_CUDA_HEADER_DEVICE
#endif
    VPlacedVolume *
    UnplacedGenTrap::Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
                            const int id,
#endif
                            VPlacedVolume *const placement) {
  if (placement) {
    new (placement) SpecializedGenTrap<trans_code, rot_code>(logical_volume, transformation
#ifdef VECGEOM_NVCC
                                                             ,
                                                             id
#endif
                                                             );
    return placement;
  }
  return new SpecializedGenTrap<trans_code, rot_code>(logical_volume, transformation
#ifdef VECGEOM_NVCC
                                                      ,
                                                      id
#endif
                                                      );
}

//______________________________________________________________________________
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume *UnplacedGenTrap::CreateSpecializedVolume(LogicalVolume const *const volume,
                                                        Transformation3D const *const transformation,
                                                        const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
                                                        const int id,
#endif
                                                        VPlacedVolume *const placement) {
  return VolumeFactory::CreateByTransformation<UnplacedGenTrap>(volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
                                                                id,
#endif
                                                                placement);
}

#ifdef VECGEOM_CUDA_INTERFACE

//______________________________________________________________________________
DevicePtr<cuda::VUnplacedVolume> UnplacedGenTrap::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const {
  return CopyToGpuImpl<UnplacedGenTrap>(in_gpu_ptr, fVertices, fDz);
}

//______________________________________________________________________________
DevicePtr<cuda::VUnplacedVolume> UnplacedGenTrap::CopyToGpu() const { return CopyToGpuImpl<UnplacedGenTrap>(); }

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedGenTrap>::SizeOf();
template void DevicePtr<cuda::UnplacedGenTrap>::Construct(Vector3D<Precision> vertices[], Precision halfzheight) const;

} // End cxx namespace

#endif

} // End global namespace
