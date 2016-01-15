/// \file UnplacedGenTrap.cpp

//#include "base/Global.h"
//#include "backend/Backend.h"
#include "volumes/UnplacedGenTrap.h"
#include <ostream>
#include <iomanip>
#include <iostream>
#include "management/VolumeFactory.h"
#include "volumes/SpecializedGenTrap.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

//______________________________________________________________________________
 void UnplacedGenTrap::ComputeBoundingBox()
 {
    // Computes bounding box parameters
   Vector3D<Precision> aMin, aMax;
   Extent(aMin, aMax);
   fBoundingBoxOrig = 0.5 * (aMin + aMax);
   Vector3D<Precision> halfLengths =  0.5 * (aMax - aMin);
   fBoundingBox.SetX(halfLengths.x());
   fBoundingBox.SetY(halfLengths.y());
   fBoundingBox.SetZ(halfLengths.z());
 }

//______________________________________________________________________________
void UnplacedGenTrap::Extent(Vector3D<Precision> & aMin, Vector3D<Precision> & aMax) const
{
  // Returns the full 3D cartesian extent of the solid.
  aMin = aMax = fVertices[0];
  aMin[2] = -fDz;
  aMax[2] = fDz;
  for (int  i=0; i<4; ++i) {
     // lower -fDz vertices
     if (aMin[0] > fVertices[i].x()) aMin[0] = fVertices[i].x();
     if (aMax[0] < fVertices[i].x()) aMax[0] = fVertices[i].x();
     if (aMin[1] > fVertices[i].y()) aMin[1] = fVertices[i].y();
     if (aMax[1] < fVertices[i].y()) aMax[1] = fVertices[i].y();
     // upper fDz vertices
     if (aMin[0] > fVertices[i+4].x()) aMin[0] = fVertices[i+4].x();
     if (aMax[0] < fVertices[i+4].x()) aMax[0] = fVertices[i+4].x();
     if (aMin[1] > fVertices[i+4].y()) aMin[1] = fVertices[i+4].y();
     if (aMax[1] < fVertices[i+4].y()) aMax[1] = fVertices[i+4].y();
  }
}

//______________________________________________________________________________
bool UnplacedGenTrap::SegmentsCrossing(Vector3D<Precision> p, Vector3D<Precision> p1,
                 Vector3D<Precision> q, Vector3D<Precision> q1) const
{
// Check if 2 segments defined by (p,p1) and (q,q1) are crossing.
  using Vector = Vector3D<Precision>;
  Vector r = p1 - p; // p1 = p+r
  Vector s = q1 - q; // q1 = q+s
  Vector r_cross_s = Vector::Cross(r,s);
  if ( r_cross_s.Mag2() < kTolerance ) // parallel or colinear - ignore crossing
    return false;
  Precision t = Vector::Cross(q-p,s)/r_cross_s;
  if (t < 0 || t > 1) return true;
  Precision u = Vector::Cross(q-p,r)/r_cross_s;
  if (u < 0 || u > 1) return true;
  return false;
}

//______________________________________________________________________________
 // computes if this gentrap is twisted
 bool UnplacedGenTrap::ComputeIsTwisted()
 {
   // Computes tangents of twist angles (angles between projections on XY plane
   // of corresponding -dz +dz edges).

   bool twisted = false;
   double  dx1, dy1, dx2, dy2;
   const int nv = 4; // half the number of verices

   for (int  i = 0; i < 4; ++i)
   {
     dx1 = fVertices[(i + 1) % nv].x() - fVertices[i].x();
     dy1 = fVertices[(i + 1) % nv].y() - fVertices[i].y();
     if ((dx1 == 0) && (dy1 == 0))
     {
       continue;
     }

     dx2 = fVertices[nv + (i + 1) % nv].x() - fVertices[nv + i].x();
     dy2 = fVertices[nv + (i + 1) % nv].y() - fVertices[nv + i].y();

     if ((dx2 == 0 && dy2 == 0))
     {
       continue;
     }
     double  twist_angle = std::fabs(dy1 * dx2 - dx1 * dy2);
     // attention: this thing was a different tolerance: UGenTrap::tolerance
     if (twist_angle < kTolerance)
     {
       continue;
     }
     twisted = true;

     // SetTwistAngle(i, twist_angle);
     // Check on big angles, potentially navigation problem

     // calculate twist angle
     //twist_angle = std::acos((dx1 * dx2 + dy1 * dy2)
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
// computes if this gentrap is convex
bool UnplacedGenTrap::ComputeIsConvex()
{
// Convexity is assured if the generic trap is planar and if one of the top or
// bottom faces is a convex quadrilateral.
  if (fIsTwisted) return false;
  // All cross products of consecutive segments should be negative or zero
  for (int i=0; i<4; ++i) {
    int j = (i+1)%4;
    Precision crossij = fVertices[i].x()*fVertices[j].y()-fVertices[j].x()*fVertices[i].y();
    if (crossij > 0) return false;
  }
  return true;  
}

//______________________________________________________________________________
VECGEOM_CUDA_HEADER_BOTH
void UnplacedGenTrap::Print() const {
//    printf("UnplacedGenTrap; more precise print to be implemented");
    UnplacedGenTrap::Print(std::cout);
}

//______________________________________________________________________________
void UnplacedGenTrap::Print(std::ostream &os) const {
    int  oldprc = os.precision(16);
    os << "--------------------------------------------------------\n"
//     << "    *** Dump for solid - " << GetName() << " *** \n"
       << "    =================================================== \n"
       << " Solid geometry type: UnplacedGenTrap \n"
       << "   half length Z: " << fDz << " mm \n"
       << "   list of vertices:\n";

      for (int  i = 0; i < 8; ++i)
      {
        os << std::setw(5) << "#" << i
           << "   vx = " << fVertices[i].x() << " mm"
           << "   vy = " << fVertices[i].y() << " mm\n";
      }
    os << "   planar: " << IsPlanar() << "  convex: " << fIsConvex << std::endl;
    os.precision(oldprc);
}

template <TranslationCode transCodeT, RotationCode rotCodeT>
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedGenTrap::Create(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) {
      if (placement) {
       return new(placement) SpecializedGenTrap<transCodeT, rotCodeT>(
#ifdef VECGEOM_NVCC
       logical_volume, transformation, NULL, id); // TODO: add bounding box?
#else
       logical_volume, transformation);
#endif
      }
      return new SpecializedGenTrap<transCodeT, rotCodeT>(
#ifdef VECGEOM_NVCC
      logical_volume, transformation, NULL, id); // TODO: add bounding box?
#else
      logical_volume, transformation);
#endif
}

//
VECGEOM_CUDA_HEADER_DEVICE
VPlacedVolume* UnplacedGenTrap::SpecializedVolume(
    LogicalVolume const *const volume,
    Transformation3D const *const transformation,
    const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
    const int id,
#endif
    VPlacedVolume *const placement) const {
    return VolumeFactory::CreateByTransformation<
         UnplacedGenTrap>(volume, transformation, trans_code, rot_code,
#ifdef VECGEOM_NVCC
         id,
#endif
         placement);
}

} } // End global namespace

//namespace vecgeom {
//
//#ifdef VECGEOM_CUDA_INTERFACE
//
//void UnplacedParaboloid_CopyToGpu(VUnplacedVolume *const gpu_ptr);
//
//VUnplacedVolume* UnplacedParaboloid::CopyToGpu(
//    VUnplacedVolume *const gpu_ptr) const {
//  UnplacedParaboloid_CopyToGpu(gpu_ptr);
//  CudaAssertError();
//  return gpu_ptr;
//}
//
//VUnplacedVolume* UnplacedParaboloid::CopyToGpu() const {
//  VUnplacedVolume *const gpu_ptr = AllocateOnGpu<UnplacedParaboloid>();
//  return this->CopyToGpu(gpu_ptr);
//}
//
//#endif
//
//#ifdef VECGEOM_NVCC
//
//class VUnplacedVolume;
//
//__global__
//void UnplacedParaboloid_ConstructOnGpu(VUnplacedVolume *const gpu_ptr) {
//  new(gpu_ptr) vecgeom_cuda::UnplacedParaboloid();
//}
//
//void UnplacedParaboloid_CopyToGpu(VUnplacedVolume *const gpu_ptr) {
//  UnplacedParaboloid_ConstructOnGpu<<<1, 1>>>(gpu_ptr);
//}
//
//#endif
//
// } // End namespace vecgeom
