/*
 * UnplacedCone.h
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 */



#ifndef VECGEOM_VOLUMES_UNPLACEDCONE_H_
#define VECGEOM_VOLUMES_UNPLACEDCONE_H_

#include "base/Global.h"

#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"
#include <cmath>

namespace VECGEOM_NAMESPACE {

/**
 * Class representing an unplaced cone; Encapsulated parameters of a cone and
 * functions that do not depend on how the cone is placed in a reference frame
 *
 * The unplaced cone is represented by the following parameters
 *
 * Member Data:
 *
 *  fDz half length in z direction;  ( the cone has height 2*fDz )
 *  fRmin1  inside radius at  -fDz ( in internal coordinate system )
 *  fRmin2  inside radius at  +fDz
 *  fRmax1  outside radius at -fDz
 *  fRmax2  outside radius at +fDz
 *  fSPhi starting angle of the segment in radians
 *  fDPhi delta angle of the segment in radiansdz
 */
class UnplacedCone : public VUnplacedVolume, AlignedBase {

private:
    VECGEOM_CUDA_HEADER_BOTH
      static void GetAlongVectorToPhiSector(Precision phi, Precision &x, Precision &y) {
        x = std::cos(phi);
        y = std::sin(phi);
      }

  Precision fRmin1;
  Precision fRmax1;
  Precision fRmin2;
  Precision fRmax2;
  Precision fDz;
  Precision fSPhi;
  Precision fDPhi;

  // vectors characterizing the normals of phi planes
  // makes task to detect phi sektors very efficient
  Vector3D<Precision> fNormalPhi1;
  Vector3D<Precision> fNormalPhi2;
  Precision fAlongPhi1x;
  Precision fAlongPhi1y;
  Precision fAlongPhi2x;
  Precision fAlongPhi2y;

  // Some precomputed values to avoid divisions etc
  Precision fInnerSlope; // "gradient" of inner surface in z direction
  Precision fOuterSlope; // "gradient" of outer surface in z direction
  Precision fInnerOffset;
  Precision fOuterOffset;
  Precision fOuterSlopeSquare;
  Precision fInnerSlopeSquare;
  Precision fOuterOffsetSquare;
  Precision fInnerOffsetSquare;

public:

  VECGEOM_CUDA_HEADER_BOTH
  // should be implemented in source file
  UnplacedCone(Precision rmin1,
          Precision rmax1, Precision rmin2, Precision rmax2,
          Precision dz, Precision phimin, Precision phimax) :
        fRmin1(rmin1),
        fRmax1(rmax1),
        fRmin2(rmin2),
        fRmax2(rmax2),
        fDz(dz),
        fSPhi(phimin),
        fDPhi(phimax),
        fInnerSlope(), // "gradient" of inner surface in z direction
        fOuterSlope(), // "gradient" of outer surface in z direction
        fInnerOffset(),
        fOuterOffset(),
        fOuterSlopeSquare(),
        fInnerSlopeSquare(),
        fOuterOffsetSquare(),
        fInnerOffsetSquare() {

        // check this very carefully
       fInnerSlope = -(fRmin1 - fRmin2)/(2.*fDz);
       fOuterSlope = -(fRmax1 - fRmax2)/(2.*fDz);
       fInnerOffset = fRmin2 - fInnerSlope*fDz;
       fOuterOffset = fRmax2 - fOuterSlope*fDz;
       fOuterSlopeSquare = fOuterSlope*fOuterSlope;
       fInnerSlopeSquare = fInnerSlope*fInnerSlope;
       fOuterOffsetSquare = fOuterOffset*fOuterOffset;
       fInnerOffsetSquare = fInnerOffset*fInnerOffset;

       GetAlongVectorToPhiSector(fSPhi, fAlongPhi1x, fAlongPhi1y);
       GetAlongVectorToPhiSector(fSPhi + fDPhi, fAlongPhi2x, fAlongPhi2y);
       // calculate caches
       // the possible caches are one major difference between tube and cone

       // calculate caches
//       cacheRminSqr=dRmin1*dRmin1;
//       cacheRmaxSqr=dRmax1*dRmax1;

//       if ( dRmin1 > Utils::GetRadHalfTolerance() )
//           {
//              // CHECK IF THIS CORRECT ( this seems to be inversed with tolerance for ORmax
//              cacheTolORminSqr = (dRmin1 - Utils::GetRadHalfTolerance()) * (dRmin1 - Utils::GetRadHalfTolerance());
//              cacheTolIRminSqr = (dRmin1 + Utils::GetRadHalfTolerance()) * (dRmin1 + Utils::GetRadHalfTolerance());
//           }
//           else
//           {
//              cacheTolORminSqr = 0.0;
//              cacheTolIRminSqr = 0.0;
//           }
//
//           cacheTolORmaxSqr = (dRmax1 + Utils::GetRadHalfTolerance()) * (dRmax1 + Utils::GetRadHalfTolerance());
//           cacheTolIRmaxSqr = (dRmax1 - Utils::GetRadHalfTolerance()) * (dRmax1 - Utils::GetRadHalfTolerance());
//
//           // calculate normals
//           GeneralPhiUtils::GetNormalVectorToPhiPlane(dSPhi, normalPhi1, true);
//           GeneralPhiUtils::GetNormalVectorToPhiPlane(dSPhi + dDPhi, normalPhi2, false);
//
//           // get alongs
//           GeneralPhiUtils::GetAlongVectorToPhiPlane(dSPhi, alongPhi1);
//           GeneralPhiUtils::GetAlongVectorToPhiPlane(dSPhi + dDPhi, alongPhi2);
//
//           // normalPhi1.print();
//           // normalPhi2.print();
//        };
    }

    // public interfaces
    Precision GetRmin1() const {return fRmin1;}
    Precision GetRmax1() const {return fRmax1;}
    Precision GetRmin2() const {return fRmin2;}
    Precision GetRmax2() const {return fRmax2;}
    Precision GetDz() const {return fDz;}
    Precision GetSPhi() const {return fSPhi;}
    Precision GetDPhi() const {return fDPhi;}
    Precision dphi() const {return fDPhi;}
    Precision GetInnerSlope() const {return fInnerSlope;}
    Precision GetOuterSlope() const {return fOuterSlope;}
    Precision GetInnerOffset() const {return fInnerOffset;}
    Precision GetOuterOffset() const {return fOuterOffset;}
    // these values could be cached
    Precision GetInnerSlopeSquare() const {return fInnerSlope*fInnerSlope;}
    Precision GetOuterSlopeSquare() const {return fOuterSlope*fOuterSlope;}
    Precision GetInnerOffsetSquare() const {return fInnerOffset*fInnerOffset;}
    Precision GetOuterOffsetSquare() const {return fOuterOffset*fOuterOffset;}

    Precision alongPhi1x() const { return fAlongPhi1x; }
    Precision alongPhi1y() const { return fAlongPhi1y; }
    Precision alongPhi2x() const { return fAlongPhi2x; }
    Precision alongPhi2y() const { return fAlongPhi2y; }

    Precision Capacity() const {
        return (2.*fDz* kPiThird)*(fRmax1*fRmax1+fRmax2*fRmax2+fRmax1*fRmax2-
                fRmin1*fRmin1-fRmin2*fRmin2-fRmin1*fRmin2);
    }

    virtual int memory_size() const { return sizeof(*this); }

    VECGEOM_CUDA_HEADER_BOTH
    virtual void Print() const;
    virtual void Print(std::ostream &os) const;

    VECGEOM_CUDA_HEADER_DEVICE
     virtual VPlacedVolume* SpecializedVolume(
         LogicalVolume const *const volume,
         Transformation3D const *const transformation,
         const TranslationCode trans_code, const RotationCode rot_code,
   #ifdef VECGEOM_NVCC
         const int id,
   #endif
         VPlacedVolume *const placement = NULL) const;

    template <TranslationCode transCodeT, RotationCode rotCodeT>
     VECGEOM_CUDA_HEADER_DEVICE
     static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                                  Transformation3D const *const transformation,
   #ifdef VECGEOM_NVCC
                                  const int id,
   #endif
                                  VPlacedVolume *const placement = NULL);

};


} // end namespace

#endif
