/*
 * TestExportToROOT.cpp
 *
 *  Created on: 28.10.2014
 *      Author: swenzel
 */

#include "volumes/PlacedVolume.h"
#include "base/Global.h"
#include "base/Transformation3D.h"
#include "volumes/UnplacedBox.h"
#include "volumes/UnplacedTube.h"
#include "volumes/UnplacedCone.h"
#include "volumes/UnplacedTube.h"
#include "volumes/UnplacedTrd.h"
#include "volumes/UnplacedOrb.h"
#include "volumes/UnplacedParaboloid.h"
#include "volumes/UnplacedParallelepiped.h"
#include "volumes/UnplacedBooleanMinusVolume.h"
#include "volumes/utilities/VolumeUtilities.h"
#include "base/SOA3D.h"
#include "navigation/NavigationState.h"
#include "navigation/SimpleNavigator.h"
#include "management/RootGeoManager.h"
#include "management/GeoManager.h"
#include "TGeoManager.h"
#include "TGeoBranchArray.h"
#include <iostream>

using namespace vecgeom;


// create a VecGeom geometry
LogicalVolume const * make3LevelBooleanSubtraction()
{
    // components for boolean solid
    UnplacedBox const * motherbox = new UnplacedBox(5.,5.,5.);
    UnplacedTube const * subtractedtube = new UnplacedTube(0.5,2.,2.,0,kTwoPi);

    // translation for boolean solid right shape
    Transformation3D const * translation= new Transformation3D(-2.5,0,0);

    // we will also subtract another small box
    UnplacedBox const * subtractedbox = new UnplacedBox(1, 1, 1);
    Transformation3D  const * translation2 = new Transformation3D( 4, 4, 4);

    VPlacedVolume * placedsubtractedtube
        = (new LogicalVolume("",subtractedtube))->Place(translation);
    VPlacedVolume * placedmotherbox = (new LogicalVolume("",motherbox))->Place();

    VPlacedVolume * placedsubtractedbox
        = ( new LogicalVolume("",subtractedbox))->Place(translation2);

    // now make the unplaced boolean solid
    UnplacedBooleanMinusVolume const *booleansolid =
            new UnplacedBooleanMinusVolume(placedmotherbox, placedsubtractedtube);
    LogicalVolume const *  booleanlogical = new LogicalVolume("booleanL",booleansolid);

    UnplacedBooleanMinusVolume const * booleansolid2 = new UnplacedBooleanMinusVolume(
             booleanlogical->Place(),
             placedsubtractedbox);

    LogicalVolume const * booleanlogical2 = new LogicalVolume("booleanL2", booleansolid2);
    return booleanlogical2;
}


VPlacedVolume* SetupGeometry() {
  UnplacedBox *worldUnplaced = new UnplacedBox(40, 10, 10);
  UnplacedBox *boxUnplaced = new UnplacedBox(2.5, 2.5, 2.5);

  UnplacedTube *tube1Unplaced = new UnplacedTube( 0.5, 1., 0.5, 0., kTwoPi);
  UnplacedTube *tube2Unplaced = new UnplacedTube( 0.5, 1., 0.5, 0., kPi);
  UnplacedCone *cone1Unplaced = new UnplacedCone( 0.5, 1., 0.6, 1.2, 0.5, 0., kTwoPi);
  UnplacedCone *cone2Unplaced = new UnplacedCone( 0.5, 1., 0.6, 1.2,0.5, kPi/4., kPi);


  UnplacedTrd *trdUnplaced = new UnplacedTrd( 0.1, 0.2, 0.15, 0.05 );

  UnplacedOrb *orbUnplaced = new UnplacedOrb( 0.1 );
  UnplacedParaboloid *paraUnplaced = new UnplacedParaboloid( 0.1, 0.2, 0.1 );
  UnplacedParallelepiped *epipedUnplaced =  new UnplacedParallelepiped( 0.1, 0.05, 0.05, 0.2, 0.4, 0.1 );

  Transformation3D *placement1 = new Transformation3D( 5,  5,  5,  0,  0,  0);
  Transformation3D *placement2 = new Transformation3D(-5,  5,  5, 45,  0,  0);
  Transformation3D *placement3 = new Transformation3D( 5, -5,  5,  0, 45,  0);
  Transformation3D *placement4 = new Transformation3D( 5,  5, -5,  0,  0, 45);
  Transformation3D *placement5 = new Transformation3D(-5, -5,  5, 45, 45,  0);
  Transformation3D *placement6 = new Transformation3D(-5,  5, -5, 45,  0, 45);
  Transformation3D *placement7 = new Transformation3D( 5, -5, -5,  0, 45, 45);
  Transformation3D *placement8 = new Transformation3D(-5, -5, -5, 45, 45, 45);

  Transformation3D *placement9 = new Transformation3D(-0.5,-0.5,-0.5,0,0,0);
  Transformation3D *placement10 = new Transformation3D(0.5,0.5,0.5,0,45,0);
  Transformation3D *idendity    = new Transformation3D();

  LogicalVolume *world = new LogicalVolume("world",worldUnplaced);
  LogicalVolume *box =   new LogicalVolume("lbox1",boxUnplaced);
  LogicalVolume *tube1 = new LogicalVolume("ltube1", tube1Unplaced);
  LogicalVolume *tube2 = new LogicalVolume("ltube2", tube2Unplaced);
  LogicalVolume *cone1 = new LogicalVolume("lcone1", cone1Unplaced);
  LogicalVolume *cone2 = new LogicalVolume("lcone2", cone2Unplaced);

  LogicalVolume *trd1  = new LogicalVolume("ltrd", trdUnplaced);

  LogicalVolume *orb1 = new LogicalVolume("lorb1", orbUnplaced);
  LogicalVolume *parab1 = new LogicalVolume("lparab1", paraUnplaced);
  LogicalVolume *epip1 = new LogicalVolume("lepip1", epipedUnplaced);

  world->PlaceDaughter(orb1, idendity);
  trd1->PlaceDaughter(parab1, idendity);
  world->PlaceDaughter(epip1, idendity);

  tube1->PlaceDaughter( trd1, idendity );
  box->PlaceDaughter( tube1, placement9 );
  box->PlaceDaughter( tube2, placement10 );

  world->PlaceDaughter(box, placement1);
  world->PlaceDaughter(box, placement2);
  world->PlaceDaughter(box, placement3);
  world->PlaceDaughter(box, placement4);
  world->PlaceDaughter(box, placement5);
  world->PlaceDaughter(box, placement6);
  world->PlaceDaughter(box, placement7);
  world->PlaceDaughter(box, placement8);

  world->PlaceDaughter(cone1, new Transformation3D(8,0,0));
  world->PlaceDaughter(cone2, new Transformation3D(-8,0,0));

  // add a subtraction
  world->PlaceDaughter( make3LevelBooleanSubtraction(), new Transformation3D(-30,0,0) );

  return world->Place();
}



//LocatePoints1()
//LocatePointsRoot()


int main()
{
    VPlacedVolume const * world = SetupGeometry();
    GeoManager::Instance().SetWorld(world);
    GeoManager::Instance().CloseGeometry();
    int md1 = GeoManager::Instance().getMaxDepth();
    int mpv1 = GeoManager::Instance().GetPlacedVolumesCount();
    int mlv1 = GeoManager::Instance().GetLogicalVolumesCount();
    int ntotalnodes1 = GeoManager::Instance().GetTotalNodeCount();

    // test one million points
    const int NPOINTS = 1000000;
    SOA3D<Precision> testpoints(NPOINTS);

    volumeUtilities::FillRandomPoints( *GeoManager::Instance().GetWorld(),testpoints);

    // create one million navigation state objects
    NavigationState ** states1 = new NavigationState*[NPOINTS];
    NavigationState ** states2 = new NavigationState*[NPOINTS];
    TGeoBranchArray ** rootstates = new TGeoBranchArray*[NPOINTS];

    SimpleNavigator nav;
    for( int i=0;i<NPOINTS;++i )
    {
        states1[i] = new NavigationState( GeoManager::Instance().getMaxDepth() );
        states2[i] = new NavigationState( GeoManager::Instance().getMaxDepth() );
        nav.LocatePoint( GeoManager::Instance().GetWorld(), testpoints[i], *states1[i], true );
    }

    // exporting to ROOT file
    RootGeoManager::Instance().ExportToROOTGeometry( world, "geom1.root" );

    /*** locate in ROOT geometry        **/
    for( int i=0; i<NPOINTS; ++i){
        rootstates[i] = states1[i]->ToTGeoBranchArray();

        TGeoNavigator * nav = ::gGeoManager->GetCurrentNavigator();
        nav->FindNode( testpoints[i][0], testpoints[i][1], testpoints[i][2] );
        // save state in a ROOT state
        rootstates[i]->InitFromNavigator(nav);
    }
    /*** end locate in ROOT geometry ***/

    assert( ::gGeoManager->GetNNodes() == ntotalnodes1 );
//    assert( ::gGeoManager->GetListOfVolumes()->GetEntries() == mlv1 );

    //
    RootGeoManager::Instance().Clear();

    // now try to read back in
    RootGeoManager::Instance().set_verbose(1);
    RootGeoManager::Instance().LoadRootGeometry("geom1.root");

    for( int i=0;i<NPOINTS;++i )
    {
        nav.LocatePoint( GeoManager::Instance().GetWorld(), testpoints[i], *states2[i], true );
    }

    for( int i=0;i<NPOINTS;++i )
    {
        // we cannot compare pointers here; they are different before and after the reload
        // we need names
        if( states1[i]->GetCurrentLevel() != states2[i]->GetCurrentLevel()
            || rootstates[i]->GetLevel() != states1[i]->GetCurrentLevel()-1
            || rootstates[i]->GetLevel() != states2[i]->GetCurrentLevel()-1
          ){
            // I SUSPECT THAT THIS MIGHT HAPPEN WHEN THERE IS AN OVERLAP
            std::cerr << "### PROBLEM " << i << " s1 "
                    << states1[i]->GetCurrentLevel() << " s2 " << states2[i]->GetCurrentLevel()
                    << " r " << rootstates[i]->GetLevel()+1 << "\n";
        }
    }

    //// see if everything was restored
    // RootGeoManager::Instance().world()->logical_volume()->PrintContent(0);

    int md2 = GeoManager::Instance().getMaxDepth();
    int mpv2 = GeoManager::Instance().GetPlacedVolumesCount();
    int mlv2 = GeoManager::Instance().GetLogicalVolumesCount();
    int ntotalnodes2 = GeoManager::Instance().GetTotalNodeCount();

    assert( md2 == md1 );
    assert( mpv2 == mpv1 );
    assert( mlv2 == mlv1 );
    assert( mpv2 > 0);
    assert( mlv2 > 0);
    assert( ntotalnodes1 == ntotalnodes2 );

    return 0;
}
