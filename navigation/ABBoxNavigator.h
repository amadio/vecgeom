/*
 * ABBoxNavigator.h
 *
 *  Created on: 24.04.2015
 *      Author: swenzel
 */

#pragma once

#include "base/Global.h"

#include "volumes/PlacedVolume.h"
#include "base/Vector3D.h"
#include "management/GeoManager.h"
#include "navigation/NavigationState.h"
#include "base/Transformation3D.h"
#include "volumes/kernel/BoxImplementation.h"
#include "navigation/SimpleNavigator.h"

#ifdef VECGEOM_VC
#include "backend/vc/Backend.h"
#endif

#include <vector>
#include <cassert>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {


// A navigator using aligned bounding box = ABBox (hierarchies) to quickly find
// potential hit targets.
// This navigator goes into the direction of "voxel" navigators used in Geant4
// and ROOT. Checking single-rays against a set of aligned bounding boxes can be done
// in a vectorized fashion.
class ABBoxNavigator
{

public:
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  ABBoxNavigator(){}


  int GetHitCandidates( LogicalVolume const * lvol,
          Vector3D<Precision> const & point,
          Vector3D<Precision> const & dir,
          ABBoxManager::ABBoxContainer_t const & corners, int size,
          ABBoxManager::BoxIdDistancePair_t * /* hitlist */
  ) const;

  int GetHitCandidates_v( LogicalVolume const * lvol,
            Vector3D<Precision> const & point,
            Vector3D<Precision> const & dir,
            ABBoxManager::ABBoxContainer_v const & corners, int size,
            ABBoxManager::BoxIdDistancePair_t * /* hitlist */
  ) const;


  size_t GetSafetyCandidates_v( Vector3D<Precision> const & /* point */,
                            ABBoxManager::ABBoxContainer_v const & /* corners */,
                            int size,
                            ABBoxManager::BoxIdDistancePair_t * /* boxsafetypairs */,
                            Precision upperlimit ) const;


  // convert index to physical daugher
  VPlacedVolume const * LookupDaughter( LogicalVolume const *lvol, int id ) const {
      assert( id >= 0 && "access with negative index");
      assert( id < lvol->GetDaughtersp()->size() && "access beyond size of daughterlist ");
      return lvol->GetDaughtersp()->operator []( id );
  }

   /**
   * A function to navigate ( find next boundary and/or the step to do )
   */
   VECGEOM_CUDA_HEADER_BOTH
   void FindNextBoundaryAndStep( Vector3D<Precision> const & /* global point */,
                          Vector3D<Precision> const & /* global dir */,
                          NavigationState const & /* currentstate */,
                          NavigationState & /* newstate */,
                          Precision const & /* proposed physical step */,
                          Precision & /*step*/
                         ) const;

   /**
    * A function to get back the safe distance; given a NavigationState object and a current global point
    * point
    */
   VECGEOM_CUDA_HEADER_BOTH
   Precision GetSafety( Vector3D<Precision> const & /*global_point*/,
               NavigationState const & /* currentstate */
   ) const;

   // NOTE: there is no vector interface here yet --> this is part of SimpleNavigator


}; // end of class declaration




} } // End global namespace
