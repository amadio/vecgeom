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
                                     int size, ABBoxManager::BoxIdDistancePair_t *hitlist) const {

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
      hitlist[hitcount]=ABBoxManager::BoxIdDistancePair_t(box, distance);
      hitcount++;
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
                                       int size, ABBoxManager::BoxIdDistancePair_t *hitlist) const {

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
        if (hit[i]){
          hitlist[hitcount]=(ABBoxManager::BoxIdDistancePair_t(box * kVcFloat::precision_v::Size + i, distance[i]));
          hitcount++;
	}}
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
    if (hit){
      hitlist[hitcount]=(ABBoxManager::BoxIdDistancePair_t(box, distance));
      hitcount++;
    }
  }
  return hitcount;
#endif
}

size_t ABBoxNavigator::GetSafetyCandidates_v(Vector3D<Precision> const &point, ABBoxManager::ABBoxContainer_v const &corners, int size,
                                         ABBoxManager::BoxIdDistancePair_t *boxsafetypairs, Precision upper_squared_limit) const {
    Vector3D<float> pointfloat((float)point.x(), (float)point.y(), (float)point.z());
int candidatecount=0;
    #ifdef VECGEOM_VC
    int vecsize = size;
    for( auto box = 0; box < vecsize; ++box ){
         ABBoxManager::Real_v safetytoboxsqr =  ABBoxImplementation::ABBoxSafetySqr<kVcFloat, ABBoxManager::Real_t>(
                        corners[2*box], corners[2*box+1], pointfloat );

         ABBoxManager::Bool_v hit = safetytoboxsqr < ABBoxManager::Real_t(upper_squared_limit);
         if (Any(hit)) {
           for (auto i = 0; i < kVcFloat::precision_v::Size; ++i) {
             if (hit[i]){
               boxsafetypairs[candidatecount]=(ABBoxManager::BoxIdDistancePair_t(box * kVcFloat::precision_v::Size + i, safetytoboxsqr[i]));
             candidatecount++;}
           }
         }
    }
#else
    int vecsize = size;
    for( auto box = 0; box < vecsize; ++box ){
         ABBoxManager::Real_v safetytoboxsqr =  ABBoxImplementation::ABBoxSafetySqr<kScalarFloat, float>(
                        corners[2*box], corners[2*box+1], pointfloat );

         bool hit = safetytoboxsqr < ABBoxManager::Real_t(upper_squared_limit);
         if ( hit ){
               boxsafetypairs[candidatecount]=(ABBoxManager::BoxIdDistancePair_t(box, safetytoboxsqr));
        candidatecount++;}
    }
#endif
  return candidatecount;
}

//#define VERBOSE
Precision ABBoxNavigator::GetSafety(Vector3D<Precision> const & globalpoint,
                            NavigationState const & currentstate) const
{
    // a stack based workspace array
  static __thread ABBoxManager::BoxIdDistancePair_t boxsafetylist[VECGEOM_MAXDAUGHTERS]={};


    // this information might have been cached already ??
   Transformation3D m;
   currentstate.TopMatrix(m);
   Vector3D<Precision> localpoint=m.Transform(globalpoint);

   VPlacedVolume const *currentvol = currentstate.Top();
   double safety = currentvol->SafetyToOut( localpoint );
   double safetysqr = safety*safety;

   // safety to bounding boxes
   LogicalVolume const *lvol = currentvol->GetLogicalVolume();
   if( safety > 0. && lvol->GetDaughtersp()->size() > 0 ){
       ABBoxManager &instance = ABBoxManager::Instance();
       //ABBoxManager::HitContainer_t &boxsafetylist = instance.GetAllocatedHitContainer();


       int size;

       ABBoxManager::ABBoxContainer_v bboxes =  instance.GetABBoxes_v( lvol , size );
       // calculate squared bounding box safeties in vectorized way
       auto ncandidates = GetSafetyCandidates_v(localpoint, bboxes, size, boxsafetylist, safetysqr);
#ifdef SORT
       // sorting the list
       ABBoxManager::sort(boxsafetylist, ABBoxManager::HitBoxComparatorFunctor());

// at this moment boxsafetylist only contains
// elements whose "bounding box" safetysqr is smaller than safetytooutsqr and which hence have to be checked
#ifdef VERBOSE
    std::cerr << "boxsafetylist has " << boxsafetylist.size() << " candidates \n";
#endif
    for (auto boxsafetypair : boxsafetylist) {
      if (boxsafetypair.second < safetysqr) {
        //   std::cerr << " id " << boxsafetypair.first << " safetysqr " << boxsafetypair.second << "\n";
        VPlacedVolume const *candidate = LookupDaughter(lvol, boxsafetypair.first);
        auto candidatesafety = candidate->SafetyToIn(localpoint);
#ifdef VERBOSE
        if (candidatesafety * candidatesafety > boxsafetypair.second && boxsafetypair.second > 0)
          std::cerr << "real safety smaller than boxsafety \n";
#endif
        if (candidatesafety < safety) {
          safety = candidatesafety;
          safetysqr = safety * safety;
        } else { // this box has a safety which is larger than the best known safety so we can stop here
#ifdef VERBOSE
          std::cerr << "early return active \n";
#endif
          break;
        }
      }
    }
#else // not sorting the final list
    for (unsigned int candidate = 0; candidate < ncandidates; ++candidate){
        auto boxsafetypair = boxsafetylist[candidate];
        if (boxsafetypair.second < safetysqr) {
        //   std::cerr << " id " << boxsafetypair.first << " safetysqr " << boxsafetypair.second << "\n";
        VPlacedVolume const *candidate = LookupDaughter(lvol, boxsafetypair.first);
        if(boxsafetypair.first  > lvol->GetDaughtersp()->size()) break;
        auto candidatesafety = candidate->SafetyToIn(localpoint);
#ifdef VERBOSE
        if (candidatesafety * candidatesafety > boxsafetypair.second && boxsafetypair.second > 0)
          std::cerr << "real safety smaller than boxsafety \n";
#endif
        if (candidatesafety < safety) {
          safety = candidatesafety;
          safetysqr = safety * safety;
        }
      }
    }
#endif
   }
   return safety;
 }



// a simple sort class (based on insertionsort)
//template <typename T, typename Cmp>
void insertionsort( ABBoxManager::BoxIdDistancePair_t * arr, unsigned int N ){
    for (unsigned short i = 1; i < N; ++i) {
           ABBoxManager::BoxIdDistancePair_t value = arr[i];
           short hole = i;

           for (; hole > 0 && value.second < arr[hole - 1].second; --hole)
               arr[hole] = arr[hole - 1];

           arr[hole] = value;
       }
}


//#define VERBOSE
void
ABBoxNavigator::FindNextBoundaryAndStep( Vector3D<Precision> const & globalpoint,
                                          Vector3D<Precision> const & globaldir,
                                          NavigationState     const & currentstate,
                                          NavigationState           & newstate,
                                          Precision           const & pstep,
                                          Precision                 & step
                                        ) const
{
  static __thread ABBoxManager::BoxIdDistancePair_t hitlist[VECGEOM_MAXDAUGHTERS]={};

    // this information might have been cached in previous navigators??
#ifdef VERBOSE
    static int counter = 0;
    if( counter % 1 == 0 )
    std::cerr << counter << " " << globalpoint << " \n";
    counter++;
#endif

   Transformation3D m;
   currentstate.TopMatrix(m);
   Vector3D<Precision> localpoint=m.Transform(globalpoint);
   Vector3D<Precision> localdir=m.TransformDirection(globaldir);

   VPlacedVolume const * currentvolume = currentstate.Top();
   int nexthitvolume = -1; // means mother

   // StepType st = kPhysicsStep; // physics or geometry step
   step = currentvolume->DistanceToOut( localpoint, localdir, pstep );

   // NOTE: IF STEP IS NEGATIVE HERE, SOMETHING IS TERRIBLY WRONG. WE CAN TRY TO HANDLE THE SITUATION
   // IN TRYING TO PROPOSE THE RIGHT LOCATION IN NEWSTATE AND RETURN
   // I WOULD MUCH FAVOUR IF THIS WAS DONE OUTSIDE OF THIS FUNCTION BY THE USER
    if( step < 0. )
    {
       // TODO: instead of directly exiting we could see whether we hit a daughter
       // which is usally a logic thing to do
      // std::cerr << "negative DO\n";
     //  step = 0.;
     //  currentstate.CopyTo(&newstate);
     //  newstate.Pop();
     //  SimpleNavigator nav;
     //  nav.RelocatePointFromPath( localpoint, newstate );
      // return;
        step = kInfinity;
    }

   // if( step > 1E20 )
   //     std::cerr << "infinite DO\n";
   // TODO: compare steptoout and physics step and take minimum



   // do a quick and vectorized search using aligned bounding boxes
   // obtains a sorted container ( vector or list ) of hitboxstructs
   LogicalVolume const * currentlvol = currentstate.Top()->GetLogicalVolume();

#ifdef VERBOSE
   std::cerr << " I am in " << currentlvol->GetLabel() << "\n";
#endif
   if( currentlvol->GetDaughtersp()->size() > 0 ){
#ifdef VERBOSE
       std::cerr << " searching through " << currentlvol->GetDaughtersp()->size() << " daughters\n";
#endif
//     ABBoxManager::HitContainer_t & hitlist = ABBoxManager::Instance().GetAllocatedHitContainer();

       //       hitlist.clear();
       int size;
//       ABBoxManager::ABBoxContainer_t bboxes1 =  ABBoxManager::Instance().GetABBoxes( currentlvol , size );
//       GetHitCandidates( currentlvol,
//                         localpoint,
//                         localdir,
//                         bboxes1,
//                        size, hitlist );
#ifdef VERBOSE
       int c1 = hitlist.size();
      std::cerr << hitlist << "\n";
#endif
  //     hitlist.clear();
       ABBoxManager::ABBoxContainer_v bboxes =  ABBoxManager::Instance().GetABBoxes_v( currentlvol , size );
       auto ncandidates = GetHitCandidates_v( currentlvol,
                          localpoint,
                          localdir,
                          bboxes,
                          size, hitlist );
#ifdef VERBOSE
            int c2 = hitlist.size();
        std::cerr << hitlist << "\n";
        std::cerr << " hitting scalar " << c1 << " vs vector " << c2 << "\n";
 if( c1 != c2 )
     std::cerr << "HUHU " << c1 << " " << c2;
        #endif

        // sorting the histlist
//        ABBoxManager::sort( hitlist, ABBoxManager::HitBoxComparatorFunctor() );
        insertionsort( hitlist, ncandidates );

        // assumption: here hitlist is sorted in ascending distance order
#ifdef VERBOSE
        std::cerr << " hitting " << hitlist.size() << " boundary boxes\n";
#endif
        //for( auto hitbox : hitlist )
        for(int index=0;index < ncandidates;++index)
        {
            auto hitbox = hitlist[index];
            VPlacedVolume const * candidate = LookupDaughter( currentlvol, hitbox.first );

            // only consider those hitboxes which are within potential reach of this step
            if( ! ( step < hitbox.second )) {
            //      std::cerr << "checking id " << hitbox.first << " at box distance " << hitbox.second << "\n";
             if( hitbox.second < 0 ){
                bool checkindaughter = candidate->Contains( localpoint );
                if( checkindaughter == true ){
                    // need to relocate
                    step = 0;
                    nexthitvolume = hitbox.first;
                    // THE ALTERNATIVE WOULD BE TO PUSH THE CURRENT STATE AND RETURN DIRECTLY
                    break;
                }
            }
            Precision ddistance = candidate->DistanceToIn( localpoint, localdir, step );
#ifdef VERBOSE
            std::cerr << "distance to " << candidate->GetLabel() << " is " << ddistance << "\n";
#endif
            nexthitvolume = (ddistance < step) ? hitbox.first : nexthitvolume;
            step      = (ddistance < step) ? ddistance  : step;
        }
      else
      {
          break;
      }
   }
   }

   // now we have the candidates
   // try
   currentstate.CopyTo(&newstate);

   // is geometry further away than physics step?
   // not necessarily true
   if(step > pstep)
   {
       assert( true && "impossible state");
       // don't need to do anything
       step = pstep;
       newstate.SetBoundaryState( false );
       return;
   }
   newstate.SetBoundaryState( true );

   //assert( step >= 0 && "step negative");

   if( step > 1E30 )
     {
      //std::cout << "WARNING: STEP INFINITY; should never happen unless outside\n";
           //InspectEnvironmentForPointAndDirection( globalpoint, globaldir, currentstate );

           // set step to zero and retry one level higher
           step = 0;
           newstate.Pop();
           return;
      }

      if( step < 0. )
      {
        //std::cout << "WARNING: STEP NEGATIVE\n";
        //InspectEnvironmentForPointAndDirection( globalpoint, globaldir, currentstate );
         step = 0.;
      }

   // TODO: this is tedious, please provide operators in Vector3D!!
   // WE SHOULD HAVE A FUNCTION "TRANSPORT" FOR AN OPERATION LIKE THIS
   Vector3D<Precision> newpointafterboundary = localdir;
   newpointafterboundary*=(step + 1e-6);
   newpointafterboundary+=localpoint;

   if( nexthitvolume != -1 ) // not hitting mother
   {
      // continue directly further down
      VPlacedVolume const * nextvol = LookupDaughter( currentlvol, nexthitvolume );
      Transformation3D const * trans = nextvol->GetTransformation();

      SimpleNavigator nav;
      nav.LocatePoint( nextvol, trans->Transform(newpointafterboundary), newstate, false );
      assert( newstate.Top() != currentstate.Top() && " error relocating when entering ");
      return;
   }
   else // hitting mother
   {
      SimpleNavigator nav;
      nav.RelocatePointFromPath( newpointafterboundary, newstate );

      // can I push particle ?
      // int correctstep = 0;
      while( newstate.Top() == currentstate.Top() )
      {
     //     newstate.Print();
     //     step+=1E-6;
     //     SimpleNavigator nav;
     //     newstate.Clear();
     //     nav.LocatePoint( GeoManager::Instance().GetWorld(), globalpoint + (step)*globaldir, newstate, true );
     //     std::cerr << "correcting " << correctstep << " remaining dist to out "
      //              << currentvolume->DistanceToOut( localpoint + step*localdir, localdir, pstep )
      //              << " " << currentvolume->Contains( localpoint + step*localdir )
      //    << " " << currentvolume->SafetyToIn( localpoint + step*localdir )
      //    << " " << currentvolume->SafetyToOut( localpoint + step*localdir ) << "\n";
      //    currentvolume->PrintType();

      //    correctstep++;
       //   std::cerr << "Matrix error " << const_cast<NavigationState &> ( currentstate ).CalcTransformError( globalpoint, globaldir );
        newstate.Pop();
      }
//      if( newstate.Top() == currentstate.Top() )
//      {
//         std::cerr << "relocate failed; trying to locate from top for step " << step << "\n";
//         newstate.Clear();
//         SimpleNavigator nav;
//         nav.LocatePoint( GeoManager::Instance().GetWorld(), globalpoint + (step+1E-6)*globaldir, newstate, true );
//         //  std::cerr << "newstate top " << newstate.Top()->GetLabel() << "\n";
//      }
//      if( newstate.Top() == currentstate.Top() )
//      {
//         SimpleNavigator nav;
//         nav.InspectEnvironmentForPointAndDirection( globalpoint, globaldir, currentstate );
//      }
      assert( newstate.Top() != currentstate.Top() && " error relocating when leaving ");
   }
}


}}
