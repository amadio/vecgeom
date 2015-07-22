/*
 * ABBoxNavigator.cpp
 *
 *  Created on: 24.04.2015
 *      Author: swenzel
 */

#include "management/ABBoxManager.h"
#include "navigation/ABBoxNavigator.h"
#include "volumes/UnplacedBox.h"

#ifdef VECGEOM_VC
//#include "backend/vc/Backend.h"
#include "backend/vcfloat/Backend.h"
#else
#include "backend/scalarfloat/Backend.h"
#endif

#include <cassert>

namespace vecgeom {
inline namespace cxx {

int ABBoxNavigator::GetHitCandidates(LogicalVolume const *lvol, Vector3D<Precision> const &point,
                                     Vector3D<Precision> const &dir, ABBoxManager::ABBoxContainer_t const &corners,
                                     int size, ABBoxManager::HitContainer_t &hitlist) const {

  Vector3D<Precision> invdir(1. / dir.x(), 1. / dir.y(), 1. / dir.z());
  int vecsize = size;
  int hitcount = 0;
  int sign[3];
  sign[0] = invdir.x() < 0;
  sign[1] = invdir.y() < 0;
  sign[2] = invdir.z() < 0;
  // interpret as binary number and do a switch statement
  // do a big switch statement here
  // int code = 2 << size[0] + 2 << size[1] + 2 << size[2];
  for (auto box = 0; box < vecsize; ++box) {
    double distance =
        BoxImplementation<translation::kIdentity, rotation::kIdentity>::IntersectCachedKernel2<kScalar, double>(
            &corners[2 * box], point, invdir, sign[0], sign[1], sign[2], 0, vecgeom::kInfinity);
    if (distance < vecgeom::kInfinity) {
      hitcount++;
      hitlist.push_back(ABBoxManager::BoxIdDistancePair_t(box, distance));
    }
  }

//    switch( size[0] + size[1] + size[2] ){
//    case 0: {
//        for( auto box = 0; box < vecsize; ++box ){
//        double distance = BoxImplementation<translation::kIdentity,
//        rotation::kIdentity>::IntersectCachedKernel<kScalar,0,0,0>(
//           &corners[2*box],
//           point,
//           invdir,
//           0, vecgeom::kInfinity );
//           if( distance < vecgeom::kInfinity ) hitcount++;
//         }       break; }
//    case 3: {
//        for( auto box = 0; box < vecsize; ++box ){
//                double distance = BoxImplementation<translation::kIdentity,
//                rotation::kIdentity>::IntersectCachedKernel<kScalar,1,1,1>(
//                   &corners[2*box],
//                   point,
//                   invdir,
//                   0, vecgeom::kInfinity );
//                   if( distance < vecgeom::kInfinity ) hitcount++;
//                 }       break; }
//    default : std::cerr << "DEFAULT CALLED\n";
//    }
#ifdef INNERTIMER
  timer.Stop();
  std::cerr << "# CACHED hitting " << hitcount << "\n";
  std::cerr << "# CACHED timer " << timer.Elapsed() << "\n";
#endif
  return hitcount;
}

// vector version
int ABBoxNavigator::GetHitCandidates_v(LogicalVolume const *lvol, Vector3D<Precision> const &point,
                                       Vector3D<Precision> const &dir, ABBoxManager::ABBoxContainer_v const &corners,
                                       int size, ABBoxManager::HitContainer_t &hitlist) const {

#ifdef VECGEOM_VC
  Vector3D<float> invdirfloat(1.f / (float)dir.x(), 1.f / (float)dir.y(), 1.f / (float)dir.z());
  Vector3D<float> pfloat((float)point.x(), (float)point.y(), (float)point.z());

  int vecsize = size;
  int hitcount = 0;
  int sign[3];
  sign[0] = invdirfloat.x() < 0;
  sign[1] = invdirfloat.y() < 0;
  sign[2] = invdirfloat.z() < 0;
  for (auto box = 0; box < vecsize; ++box) {
    ABBoxManager::Real_v distance =
        BoxImplementation<translation::kIdentity, rotation::kIdentity>::IntersectCachedKernel2<kVcFloat,
                                                                                               ABBoxManager::Real_t>(
            &corners[2 * box], pfloat, invdirfloat, sign[0], sign[1], sign[2], 0,
            static_cast<float>(vecgeom::kInfinity));
    ABBoxManager::Bool_v hit = distance < static_cast<float>(vecgeom::kInfinity);
    // this is Vc specific
    // a little tricky: need to iterate over the mask -- this does not easily work with scalar types
    if (Any(hit)) {
      for (auto i = hit.firstOne(); i < kVcFloat::precision_v::Size; ++i) {
        if (hit[i])
          hitlist.push_back(ABBoxManager::BoxIdDistancePair_t(box * kVcFloat::precision_v::Size + i, distance[i]));
      }
    }
  }
  return hitcount;
#else
  Vector3D<float> invdirfloat(1.f / (float)dir.x(), 1.f / (float)dir.y(), 1.f / (float)dir.z());
  Vector3D<float> pfloat((float)point.x(), (float)point.y(), (float)point.z());

  int vecsize = size;
  int hitcount = 0;
  int sign[3];
  sign[0] = invdirfloat.x() < 0;
  sign[1] = invdirfloat.y() < 0;
  sign[2] = invdirfloat.z() < 0;
  for (auto box = 0; box < vecsize; ++box) {
    float distance =
        BoxImplementation<translation::kIdentity, rotation::kIdentity>::IntersectCachedKernel2<kScalarFloat, float>(
            &corners[2 * box], pfloat, invdirfloat, sign[0], sign[1], sign[2], 0,
            static_cast<float>(vecgeom::kInfinity));
    bool hit = distance < static_cast<float>(vecgeom::kInfinity);
    if (hit)
      hitlist.push_back(ABBoxManager::BoxIdDistancePair_t(box, distance));
  }
  return hitcount;
#endif
}

}}
