/// \file UnplacedGenTrap.cpp
/// \author: swenzel
///  Modified and completed: mihaela.gheata@cern.ch

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
UnplacedGenTrap::UnplacedGenTrap(const Precision verticesx[], const Precision verticesy[], Precision halfzheight)
    : fBBdimensions(0., 0., 0.), fBBorigin(0., 0., 0.), fVertices(), fVerticesX(), fVerticesY(), fDz(halfzheight),
      fInverseDz(1. / halfzheight), fHalfInverseDz(0.5 / halfzheight), fIsTwisted(false), fConnectingComponentsX(),
      fConnectingComponentsY(), fDeltaX(), fDeltaY(), fSurfaceShell(verticesx, verticesy, halfzheight) {
  // Constructor
  
  // Set vertices in Vector3D form
  for (int i = 0; i < 4; ++i) {
    fVertices[i].operator[](0) = verticesx[i];
    fVertices[i].operator[](1) = verticesy[i];
    fVertices[i].operator[](2) = -halfzheight;
  }
  for (int i = 4; i < 8; ++i) {
    fVertices[i].operator[](0) = verticesx[i];
    fVertices[i].operator[](1) = verticesy[i];
    fVertices[i].operator[](2) = halfzheight;
  }

  // Make sure vertices are defined clockwise
  Precision sum1 = 0.;
  Precision sum2 = 0.;
  for (int i = 0; i < 4; ++i) {
    int j = (i + 1) % 4;
    sum1 += fVertices[i].x() * fVertices[j].y() - fVertices[j].x() * fVertices[i].y();
    sum2 += fVertices[i + 4].x() * fVertices[j + 4].y() - fVertices[j + 4].x() * fVertices[i + 4].y();
  }

  // Me should generate an exception here
  if (sum1 * sum2 < -kTolerance) {
    printf("ERROR: Unplaced generic trap defined with opposite clockwise\n");
    Print();
    return;
  }

  // Revert sequence of vertices to have them clockwise
  if (sum1 > kTolerance) {
    printf("INFO: Reverting to clockwise vertices of GenTrap shape:\n");
    Print();
    Vertex_t vtemp;
    vtemp = fVertices[1];
    fVertices[1] = fVertices[3];
    fVertices[3] = vtemp;
    vtemp = fVertices[5];
    fVertices[5] = fVertices[7];
    fVertices[7] = vtemp;
  }

  // Check that opposite segments are not crossing -> fatal exception
  if (SegmentsCrossing(fVertices[0], fVertices[1], fVertices[3], fVertices[2]) ||
      SegmentsCrossing(fVertices[1], fVertices[2], fVertices[0], fVertices[3]) ||
      SegmentsCrossing(fVertices[4], fVertices[5], fVertices[7], fVertices[6]) ||
      SegmentsCrossing(fVertices[5], fVertices[6], fVertices[4], fVertices[7])) {
    printf("ERROR: Unplaced generic trap defined with crossing opposite segments\n");
    Print();
    return;
  }

  // Check that top and bottom quadrilaterals are convex
  if (!ComputeIsConvexQuadrilaterals()) {
    printf("ERROR: Unplaced generic trap defined with top/bottom quadrilaterals not convex\n");
    Print();
    return;
  }

  // Initialize the vertices components and connecting components
  for (int i = 0; i < 4; ++i) {
    fConnectingComponentsX[i] = (fVertices[i] - fVertices[i + 4]).x();
    fConnectingComponentsY[i] = (fVertices[i] - fVertices[i + 4]).y();
    fVerticesX[i] = fVertices[i].x();
    fVerticesX[i + 4] = fVertices[i + 4].x();
    fVerticesY[i] = fVertices[i].y();
    fVerticesY[i + 4] = fVertices[i + 4].y();
  }
    
  // Initialize components of horizontal connecting vectors
  for (int i = 0; i < 4; ++i) {
    int j = (i + 1) % 4;
    fDeltaX[i] = fVerticesX[j] - fVerticesX[i];
    fDeltaX[i + 4] = fVerticesX[j + 4] - fVerticesX[i + 4];
    fDeltaY[i] = fVerticesY[j] - fVerticesY[i];
    fDeltaY[i + 4] = fVerticesY[j + 4] - fVerticesY[i + 4];
  }
  fIsTwisted = ComputeIsTwisted();
  ComputeBoundingBox();
}
  
//______________________________________________________________________________
VECGEOM_CUDA_HEADER_BOTH
void UnplacedGenTrap::ComputeBoundingBox() {
  // Computes bounding box parameters
  Vertex_t aMin, aMax;
  Extent(aMin, aMax);
  fBBorigin = 0.5 * (aMin + aMax);
  fBBdimensions = 0.5 * (aMax - aMin);
}

//______________________________________________________________________________
VECGEOM_CUDA_HEADER_BOTH
void UnplacedGenTrap::Extent(Vertex_t &aMin, Vertex_t &aMax) const {
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
bool UnplacedGenTrap::SegmentsCrossing(Vertex_t p, Vertex_t p1, Vertex_t q, Vertex_t q1) const {
  // Check if 2 segments defined by (p,p1) and (q,q1) are crossing.
  // See: http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
  using Vector = Vertex_t;
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
  // Check if the trapezoid is twisted. A lateral face is twisted if the top and
  // bottom segments are not parallel (cross product not null)

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
  }
  return twisted;
}

//______________________________________________________________________________
VECGEOM_CUDA_HEADER_BOTH
bool UnplacedGenTrap::ComputeIsConvexQuadrilaterals() {
  // Computes if this gentrap top and bottom quadrilaterals are convex. The vertices
  // have to be pre-ordered clockwise in the XY plane.
 
  // The cross product of all vector pairs corresponding to ordered consecutive
  // segments has to be positive.
  for (int i = 0; i < 4; ++i) {
    int j = (i + 1) % 4;
    // Bottom face
    Precision crossij = fVertices[i].x() * fVertices[j].y() - fVertices[j].x() * fVertices[i].y();
    if (crossij > 0)
      return false;
    // Top face  
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
  Vertex_t point;
#ifndef VECGEOM_NVCC  // CUDA does not support RNG:: for now
  int i = int(RNG::Instance().uniform(0., 6.));
  if (i < 4) {
    int j = (i + 1) % 4;
    Vertex_t vi(fVertices[i + 4] - fVertices[i]);
    Vertex_t vj(fVertices[j + 4] - fVertices[j]);
    Vertex_t h0(fVertices[j] - fVertices[i]);
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
  // Set point coordinates
  point.Set(x, y, z);
#endif // VECGEOM_NVCC
  return point;
}

//______________________________________________________________________________
Precision UnplacedGenTrap::SurfaceArea() const {
  // Computes analytically the surface area of the trapezoid. The formula is
  // computed by integrating along Z axis the sum of areas for infinitezimal
  // trapezoids of each lateral surface. Since this can be twisted, the area 
  // of each such mini-surface is computed separately for the top/bottom parts 
  // separated by the diagonal.
  //    vi, vj = vectors bottom->top for each lateral surface // j=(i+1)%4 
  //    hi0 = vector connecting consecutive bottom vertices
  Vertex_t vi, vj, hi0, vres;
  Precision surfTop = 0.;
  Precision surfBottom = 0.;
  Precision surfLateral = 0;
  for (int i = 0; i < 4; ++i) {
    int j = (i + 1) % 4;
    surfBottom += 0.5 * (fVerticesX[i] * fVerticesY[j] - fVerticesX[j] * fVerticesY[i]);
    surfTop += 0.5 * (fVerticesX[i + 4] * fVerticesY[j + 4] - fVerticesX[j + 4] * fVerticesY[i + 4]);
    vi.Set(fVerticesX[i + 4] - fVerticesX[i], fVerticesY[i + 4] - fVerticesY[i], 2 * fDz);
    vj.Set(fVerticesX[j + 4] - fVerticesX[j], fVerticesY[j + 4] - fVerticesY[j], 2 * fDz);
    hi0.Set(fVerticesX[j] - fVerticesX[i], fVerticesY[j] - fVerticesY[i], 0.);
    vres = 0.5 * (Vertex_t::Cross(vi + vj, hi0) + Vertex_t::Cross(vi, vj));
    surfLateral += vres.Mag();
  }
  return (Abs(surfTop) + Abs(surfBottom) + surfLateral);
}

//______________________________________________________________________________
Precision UnplacedGenTrap::volume() const {
  // Computes analytically the capacity of the trapezoid
  int i, j;
  Precision capacity = 0;
  for (i = 0; i < 4; i++) {
    j = (i + 1) % 4;
    capacity += 0.25 * fDz * ((fVerticesX[i] + fVerticesX[i + 4]) * (fVerticesY[j] + fVerticesY[j + 4]) -
                              (fVerticesX[j] + fVerticesX[j + 4]) * (fVerticesY[i] + fVerticesY[i + 4]) +
                              (1. / 3) * ((fVerticesX[i + 4] - fVerticesX[i]) * (fVerticesY[j + 4] - fVerticesY[j]) -
                                          (fVerticesX[j] - fVerticesX[j + 4]) * (fVerticesY[i] - fVerticesY[i + 4])));
  }
  return Abs(capacity);
}

//______________________________________________________________________________
VECGEOM_CUDA_HEADER_BOTH
void UnplacedGenTrap::Print() const {
  printf("--------------------------------------------------------\n");
  printf("    =================================================== \n");
  printf(" Solid type: UnplacedGenTrap \n");
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
     << " Solid type: UnplacedGenTrap \n"
     << "   half length Z: " << fDz << " mm \n"
     << "   list of vertices:\n";

  for (int i = 0; i < 8; ++i) {
    os << std::setw(5) << "#" << i << "   vx = " << fVertices[i].x() << " mm"
       << "   vy = " << fVertices[i].y() << " mm\n";
  }
  os << "   planar: " << IsPlanar() << std::endl;
  os.precision(oldprc);
}

#if defined(VECGEOM_USOLIDS)
//______________________________________________________________________________
VECGEOM_CUDA_HEADER_BOTH
std::ostream &UnplacedGenTrap::StreamInfo(std::ostream &os) const {
  int oldprc = os.precision(16);
  os << "--------------------------------------------------------\n"
     //     << "    *** Dump for solid - " << GetName() << " *** \n"
     << "    =================================================== \n"
     << " Solid type: UnplacedGenTrap \n"
     << "   half length Z: " << fDz << " mm \n"
     << "   list of vertices:\n";

  for (int i = 0; i < 8; ++i) {
    os << std::setw(5) << "#" << i << "   vx = " << fVertices[i].x() << " mm"
       << "   vy = " << fVertices[i].y() << " mm\n";
  }
  os << "   planar: " << IsPlanar() << std::endl;
  os.precision(oldprc);
  return os;
}
#endif

//______________________________________________________________________________
VECGEOM_CUDA_HEADER_DEVICE
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
VECGEOM_CUDA_HEADER_DEVICE VPlacedVolume *UnplacedGenTrap::Create(LogicalVolume const *const logical_volume,
                                                                  Transformation3D const *const transformation,
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
/*
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
*/
#ifdef VECGEOM_CUDA_INTERFACE

//______________________________________________________________________________
DevicePtr<cuda::VUnplacedVolume> UnplacedGenTrap::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const {
  // Copy vertices on GPU, then create the object
  Precision *xv_gpu_ptr = AllocateOnGpu<Precision>(8 * sizeof(Precision));
  Precision *yv_gpu_ptr = AllocateOnGpu<Precision>(8 * sizeof(Precision));

  vecgeom::CopyToGpu(fVerticesX, xv_gpu_ptr, 8 * sizeof(Precision));
  vecgeom::CopyToGpu(fVerticesY, yv_gpu_ptr, 8 * sizeof(Precision));

  DevicePtr<cuda::VUnplacedVolume> gpugentrap =
      CopyToGpuImpl<UnplacedGenTrap>(in_gpu_ptr, xv_gpu_ptr, yv_gpu_ptr, GetDZ());
  FreeFromGpu(xv_gpu_ptr);
  FreeFromGpu(yv_gpu_ptr);
  return gpugentrap;
}

//______________________________________________________________________________
DevicePtr<cuda::VUnplacedVolume> UnplacedGenTrap::CopyToGpu() const { return CopyToGpuImpl<UnplacedGenTrap>(); }

#endif // VECGEOM_CUDA_INTERFACE

} // End impl namespace

#ifdef VECGEOM_NVCC

namespace cxx {

template size_t DevicePtr<cuda::UnplacedGenTrap>::SizeOf();
template void DevicePtr<cuda::UnplacedGenTrap>::Construct(Precision *, Precision *, Precision) const;

} // End cxx namespace

#endif

} // End global namespace
