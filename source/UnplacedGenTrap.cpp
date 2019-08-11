/// \file UnplacedGenTrap.cpp
/// \author: swenzel
///  Modified and completed: mihaela.gheata@cern.ch

#include "volumes/UnplacedGenTrap.h"
#include <ostream>
#include <iomanip>
#include <iostream>
#include "management/VolumeFactory.h"
#include "volumes/SpecializedGenTrap.h"
#ifndef VECCORE_CUDA
#include "base/RNG.h"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

//______________________________________________________________________________
Vector3D<Precision> UnplacedGenTrap::SamplePointOnSurface() const
{
  // Generate randomly a point on one of the surfaces
  // Select randomly a surface
  Vertex_t point;
#ifndef VECCORE_CUDA // CUDA does not support RNG:: for now
  // Avoid using the bounding box due to possible point-like top/bottom
  // which would be impossible to sample
  bool degenerate[6] = {false};
  int nvertices      = 4; // by default 4 vertices on top/bottom faces
  // bottom
  for (unsigned int j = 0; j < 4; ++j) {
    if ((Abs(fGenTrap.fDeltaX[j]) < kTolerance) && (Abs(fGenTrap.fDeltaY[j]) < kTolerance)) nvertices--;
  }
  if (nvertices < 3) degenerate[4] = true;
  nvertices = 4;
  // top
  for (unsigned int j = 0; j < 4; ++j) {
    if ((Abs(fGenTrap.fDeltaX[j + 4]) < kTolerance) && (Abs(fGenTrap.fDeltaY[j + 4]) < kTolerance)) nvertices--;
  }
  if (nvertices < 3) degenerate[5] = true;
  for (unsigned int j = 0; j < 4; ++j) {
    if ((Abs(fGenTrap.fDeltaX[j]) < kTolerance) && (Abs(fGenTrap.fDeltaY[j]) < kTolerance) &&
        (Abs(fGenTrap.fDeltaX[j + 4]) < kTolerance) && (Abs(fGenTrap.fDeltaY[j + 4]) < kTolerance))
      degenerate[j] = true;
  }
  // Shoot on non-degenerate surface
  unsigned int i = 0;
  while (1) {
    i = int(RNG::Instance().uniform(0., 6.));
    if (!degenerate[i]) break;
  }
  // Generate point on lateral surface
  if (i < 4) {
    unsigned int j = (i + 1) % 4;
    Vertex_t vi(fGenTrap.fVertices[i + 4] - fGenTrap.fVertices[i]);
    Vertex_t vj(fGenTrap.fVertices[j + 4] - fGenTrap.fVertices[j]);
    Vertex_t h0(fGenTrap.fVertices[j] - fGenTrap.fVertices[i]);
    // Random height
    Precision fz = RNG::Instance().uniform(0., 1.);
    // Random fraction along the horizontal hi at selected z
    Precision f = RNG::Instance().uniform(0., 1.);
    point       = fGenTrap.fVertices[i] + fz * vi + f * h0 + f * fz * (vj - vi);
    return point;
  }
  i -= 4; // now 0 (bottom surface) or 1 (top surface)
  // Select z position
  Precision cross, x, y;
  Precision z = (2. * i - 1.) * fGenTrap.fDz;
  i *= 4; // now matching the index of the start vertex
          // Compute min/max  in x and y for the selected surface

  // Consider degenerate cases (if we would like to generate points also on these)
  /*
    if (nvertices <= 1) {
      // A single vertex. Generate the point identical to the vertex
      point.Set(fVertices[i].x(), fVertices[i].y(), z);
      return point;
    } else if (nvertices == 2) {
      for (int j = i; j< i + 4 ; ++j) {
        if ( (Abs(fDeltaX[j]) < kTolerance) && (Abs(fDeltaY[j]) < kTolerance) ) continue;
        // We have found two different points. Generate a random x:
        x = RNG::Instance().uniform(fVertices[j].x(), fVertices[j+1].x());
        // Calculate corresponding y
        if (Abs(fDeltaX[j]) < kTolerance)
          y = RNG::Instance().uniform(fVertices[j].y(), fVertices[j+1].y());
        else
          y = fVertices[j].y() + (x - fVertices[j].x())*fDeltaY[j]/fDeltaX[j];
        point.Set(x,y,z);
        return point;
      }
    }
  */
  // Generate point on top/bottom surfaces
  Precision xmin = fGenTrap.fVertices[i].x();
  Precision xmax = xmin;
  Precision ymin = fGenTrap.fVertices[i].y();
  Precision ymax = ymin;
  for (unsigned int j = i + 1; j < i + 4; ++j) {
    if (fGenTrap.fVertices[j].x() < xmin) xmin = fGenTrap.fVertices[j].x();
    if (fGenTrap.fVertices[j].x() > xmax) xmax = fGenTrap.fVertices[j].x();
    if (fGenTrap.fVertices[j].y() < ymin) ymin = fGenTrap.fVertices[j].y();
    if (fGenTrap.fVertices[j].y() > ymax) ymax = fGenTrap.fVertices[j].y();
  }
  bool inside = false;
  while (!inside) {
    inside = true;
    // Now generate randomly between (xmin,xmax) and (ymin,ymax)
    x = RNG::Instance().uniform(xmin, xmax);
    y = RNG::Instance().uniform(ymin, ymax);
    // Now make sure the point (x,y) is on the selected surface. Use same
    // algorithm as for Contains
    for (unsigned int j = i; j < i + 4; ++j) {
      unsigned int k = i + (j + 1) % 4;
      Precision dx   = fGenTrap.fVertices[k].x() - fGenTrap.fVertices[j].x();
      Precision dy   = fGenTrap.fVertices[k].y() - fGenTrap.fVertices[j].y();
      cross          = (x - fGenTrap.fVertices[j].x()) * dy - (y - fGenTrap.fVertices[j].y()) * dx;
      if (cross < -kTolerance) {
        inside = false;
        break;
      }
    }
  }
  // Set point coordinates
  point.Set(x, y, z);
#endif // VECCORE_CUDA
  return point;
}

//______________________________________________________________________________
Precision UnplacedGenTrap::SurfaceArea() const
{
  // Computes analytically the surface area of the trapezoid. The formula is
  // computed by integrating along Z axis the sum of areas for infinitezimal
  // trapezoids of each lateral surface. Since this can be twisted, the area
  // of each such mini-surface is computed separately for the top/bottom parts
  // separated by the diagonal.
  //    vi, vj = vectors bottom->top for each lateral surface // j=(i+1)%4
  //    hi0 = vector connecting consecutive bottom vertices
  Vertex_t vi, vj, hi0, vres;
  Precision surfTop     = 0.;
  Precision surfBottom  = 0.;
  Precision surfLateral = 0;
  for (int i = 0; i < 4; ++i) {
    int j = (i + 1) % 4;
    surfBottom +=
        0.5 * (fGenTrap.fVerticesX[i] * fGenTrap.fVerticesY[j] - fGenTrap.fVerticesX[j] * fGenTrap.fVerticesY[i]);
    surfTop += 0.5 * (fGenTrap.fVerticesX[i + 4] * fGenTrap.fVerticesY[j + 4] -
                      fGenTrap.fVerticesX[j + 4] * fGenTrap.fVerticesY[i + 4]);
    vi.Set(fGenTrap.fVerticesX[i + 4] - fGenTrap.fVerticesX[i], fGenTrap.fVerticesY[i + 4] - fGenTrap.fVerticesY[i],
           2 * fGenTrap.fDz);
    vj.Set(fGenTrap.fVerticesX[j + 4] - fGenTrap.fVerticesX[j], fGenTrap.fVerticesY[j + 4] - fGenTrap.fVerticesY[j],
           2 * fGenTrap.fDz);
    hi0.Set(fGenTrap.fVerticesX[j] - fGenTrap.fVerticesX[i], fGenTrap.fVerticesY[j] - fGenTrap.fVerticesY[i], 0.);
    vres = 0.5 * (Vertex_t::Cross(vi + vj, hi0) + Vertex_t::Cross(vi, vj));
    surfLateral += vres.Mag();
  }
  return (Abs(surfTop) + Abs(surfBottom) + surfLateral);
}

//______________________________________________________________________________
VECCORE_ATT_HOST_DEVICE
bool UnplacedGenTrap::Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const
{
  bool valid;
  GenTrapImplementation::NormalKernel<double>(fGenTrap, point, normal, valid);
  return valid;
}

//______________________________________________________________________________
Precision UnplacedGenTrap::volume() const
{
  // Computes analytically the capacity of the trapezoid
  int i, j;
  Precision capacity = 0;
  for (i = 0; i < 4; i++) {
    j = (i + 1) % 4;
    capacity +=
        0.25 * fGenTrap.fDz *
        ((fGenTrap.fVerticesX[i] + fGenTrap.fVerticesX[i + 4]) * (fGenTrap.fVerticesY[j] + fGenTrap.fVerticesY[j + 4]) -
         (fGenTrap.fVerticesX[j] + fGenTrap.fVerticesX[j + 4]) * (fGenTrap.fVerticesY[i] + fGenTrap.fVerticesY[i + 4]) +
         (1. / 3) * ((fGenTrap.fVerticesX[i + 4] - fGenTrap.fVerticesX[i]) *
                         (fGenTrap.fVerticesY[j + 4] - fGenTrap.fVerticesY[j]) -
                     (fGenTrap.fVerticesX[j] - fGenTrap.fVerticesX[j + 4]) *
                         (fGenTrap.fVerticesY[i] - fGenTrap.fVerticesY[i + 4])));
  }
  return Abs(capacity);
}

//______________________________________________________________________________
void UnplacedGenTrap::Print(std::ostream &os) const
{
  int oldprc = os.precision(16);
  os << "--------------------------------------------------------\n"
     //     << "    *** Dump for solid - " << GetName() << " *** \n"
     << "    =================================================== \n"
     << " Solid type: UnplacedGenTrap \n"
     << "   half length Z: " << fGenTrap.fDz << " mm \n"
     << "   list of vertices:\n";

  for (int i = 0; i < 8; ++i) {
    os << std::setw(5) << "#" << i << "   vx = " << fGenTrap.fVertices[i].x() << " mm"
       << "   vy = " << fGenTrap.fVertices[i].y() << " mm\n";
  }
  os << "   planar: " << IsPlanar() << std::endl;
  os.precision(oldprc);
}

//______________________________________________________________________________
std::ostream &UnplacedGenTrap::StreamInfo(std::ostream &os) const
{
  int oldprc = os.precision(16);
  os << "--------------------------------------------------------\n"
     //     << "    *** Dump for solid - " << GetName() << " *** \n"
     << "    =================================================== \n"
     << " Solid type: UnplacedGenTrap \n"
     << "   half length Z: " << fGenTrap.fDz << " mm \n"
     << "   list of vertices:\n";

  for (int i = 0; i < 8; ++i) {
    os << std::setw(5) << "#" << i << "   vx = " << fGenTrap.fVertices[i].x() << " mm"
       << "   vy = " << fGenTrap.fVertices[i].y() << " mm\n";
  }
  os << "   planar: " << IsPlanar() << std::endl;
  os.precision(oldprc);
  return os;
}

#ifndef VECCORE_CUDA
SolidMesh *UnplacedGenTrap::CreateMesh3D(Transformation3D const &trans, size_t nSegments) const
{

  SolidMesh *sm = new SolidMesh();
  if (IsPlanar()) {
    sm->SetVertices(GetVertices(), 8);
    sm->TransformVertices(trans);

    sm->AddPolygon(4, {0, 1, 2, 3}, true);
    sm->AddPolygon(4, {4, 7, 6, 5}, true);
    sm->AddPolygon(4, {0, 4, 5, 1}, true);
    sm->AddPolygon(4, {1, 5, 6, 2}, true);
    sm->AddPolygon(4, {2, 6, 7, 3}, true);
    sm->AddPolygon(4, {3, 7, 4, 0}, true);
  } else {
    typedef Vector3D<double> Vec_t;
    Vec_t *const vertices = new Vec_t[4 * (nSegments + 1)];

    size_t idx = 0;
    for (int j = 0; j < 4; j++) {

      Vertex_t v1 = GetVertex(j);
      Vertex_t v3 = GetVertex(j + 4);
      Vec_t step  = (v3 - v1) / nSegments;

      for (size_t i = 0; i <= nSegments; i++) {
        v1 += step;
        vertices[idx++] = v1;
      }
    }

    sm->SetVertices(vertices, 4 * (nSegments + 1));
    delete[] vertices;
    sm->TransformVertices(trans);

    for (size_t s0 = 0, s1 = nSegments + 1, s2 = 2 * s1, s3 = 3 * s1; s0 < nSegments; s0++, s1++, s2++, s3++) {
      sm->AddPolygon(4, {s0, s0 + 1, s1 + 1, s1}, true); // lateral surface 0
      sm->AddPolygon(4, {s1, s1 + 1, s2 + 1, s2}, true); // lateral surface 1
      sm->AddPolygon(4, {s2, s2 + 1, s3 + 1, s3}, true); // lateral surface 2
      sm->AddPolygon(4, {s3, s3 + 1, s0 + 1, s0}, true); // lateral surface 3
    }
    sm->AddPolygon(4, {0, nSegments + 1, 2 * (nSegments  + 1), 3 * (nSegments + 1)}, true);
    sm->AddPolygon(4, {3 * (nSegments + 1) + nSegments,2 * (nSegments  + 1) + nSegments, nSegments + 1 + nSegments,  0 + nSegments}, true);
    sm->InitPolygons();
  }

  return sm;
}
#endif

//______________________________________________________________________________
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedGenTrap::SpecializedVolume(LogicalVolume const *const volume,
                                                  Transformation3D const *const transformation,
                                                  const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECCORE_CUDA
                                                  const int id,
#endif
                                                  VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedGenTrap>(volume, transformation, trans_code, rot_code,
#ifdef VECCORE_CUDA
                                                                id,
#endif
                                                                placement);
}

//______________________________________________________________________________
template <TranslationCode trans_code, RotationCode rot_code>
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedGenTrap::Create(LogicalVolume const *const logical_volume,
                                       Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                       const int id,
#endif
                                       VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedGenTrap<trans_code, rot_code>(logical_volume, transformation
#ifdef VECCORE_CUDA
                                                             ,
                                                             id
#endif
    );
    return placement;
  }
  return new SpecializedGenTrap<trans_code, rot_code>(logical_volume, transformation
#ifdef VECCORE_CUDA
                                                      ,
                                                      id
#endif
  );
}

//______________________________________________________________________________
/*
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedGenTrap::CreateSpecializedVolume(LogicalVolume const *const volume,
                                                        Transformation3D const *const transformation,
                                                        const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECCORE_CUDA
                                                        const int id,
#endif
                                                        VPlacedVolume *const placement) {
  return VolumeFactory::CreateByTransformation<UnplacedGenTrap>(volume, transformation, trans_code, rot_code,
#ifdef VECCORE_CUDA
                                                                id,
#endif
                                                                placement);
}
*/
#ifdef VECGEOM_CUDA_INTERFACE

//______________________________________________________________________________
DevicePtr<cuda::VUnplacedVolume> UnplacedGenTrap::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  // Copy vertices on GPU, then create the object
  Precision *xv_gpu_ptr = AllocateOnGpu<Precision>(8 * sizeof(Precision));
  Precision *yv_gpu_ptr = AllocateOnGpu<Precision>(8 * sizeof(Precision));

  vecgeom::CopyToGpu(fGenTrap.fVerticesX, xv_gpu_ptr, 8 * sizeof(Precision));
  vecgeom::CopyToGpu(fGenTrap.fVerticesY, yv_gpu_ptr, 8 * sizeof(Precision));

  DevicePtr<cuda::VUnplacedVolume> gpugentrap =
      CopyToGpuImpl<UnplacedGenTrap>(in_gpu_ptr, xv_gpu_ptr, yv_gpu_ptr, GetDZ());
  FreeFromGpu(xv_gpu_ptr);
  FreeFromGpu(yv_gpu_ptr);
  return gpugentrap;
}

//______________________________________________________________________________
DevicePtr<cuda::VUnplacedVolume> UnplacedGenTrap::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedGenTrap>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedGenTrap>::SizeOf();
template void DevicePtr<cuda::UnplacedGenTrap>::Construct(Precision *, Precision *, Precision) const;

} // namespace cxx

#endif

} // namespace vecgeom
