/// @file HyperboloidTest.cpp
/// @author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)

#include "volumes/LogicalVolume.h"
#include "volumes/Box.h"
#include "volumes/Hype.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"

#include "TGeoShape.h"
#include "TGraph2D.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"
#include "TGeoParaboloid.h"
#include "TGeoHype.h"
#include "TGeoVolume.h"
#include "TPolyMarker3D.h"
#include "TRandom3.h"
#include "TColor.h"
#include "TROOT.h"
#include "TAttMarker.h"
#include "TMath.h"
#include "TF1.h"

using namespace vecgeom;

int main( int argc,  char *argv[]) {
    
    TApplication theApp("App",&argc,argv);
    
    if (argc<8)
	{
    	std::cout << "usage " << argv[0] << " <rMin[0-10]> <stIn[0-10]> <rMax[0-10]> <stOut[0-10]> <dZ[0-10]> <Npoints> [0-10^8] <GenerationType> [0-12]\n";
    	return 0;
 	}
   	
    double rMin=atoi(argv[1]), stIn=atoi(argv[2]), rMax=atoi(argv[3]), stOut=atoi(argv[4]), dz=atoi(argv[5]);
    
    
    double tIn=TMath::Tan(stIn*kDegToRad);
    double tOut=TMath::Tan(stOut*kDegToRad);
    double EndInnerRadius2=tIn*tIn*dz*dz+rMin*rMin;
    double EndOuterRadius2=tOut*tOut*dz*dz+rMax*rMax;
    double EndInnerRadius=TMath::Sqrt(EndInnerRadius2);
    double EndOuterRadius=TMath::Sqrt(EndOuterRadius2);
   
    int np=atoi(argv[6]);
    int generation=atoi(argv[7]);
    
    std::cout<<"EndInnerRadius== "<<EndInnerRadius<<"\n";
    std::cout<<"EndOuterRadius== "<<EndOuterRadius<<"\n";
    std::cout<<"EndInnerRadius2== "<<EndInnerRadius2<<"\n";
    std::cout<<"EndOuterRadius2== "<<EndOuterRadius2<<"\n";
    std::cout<<"Generation== "<<generation<<"\n";
    std::cout<<"Setting up VecGeom geometry\n";
    UnplacedBox worldUnplaced = UnplacedBox(10., 10., 10.);
    UnplacedHype hypeUnplaced = UnplacedHype(rMin, stIn, rMax, stOut, dz);
    std::cout<<"World and unplaced hype created\n";
    LogicalVolume world = LogicalVolume("MBworld", &worldUnplaced);
    LogicalVolume myhype = LogicalVolume("MBhype", &hypeUnplaced);
    world.PlaceDaughter(&myhype, &Transformation3D::kIdentity);
    VPlacedVolume *worldPlaced = world.Place();
    //worldPlaced->PrintContent();
    
    GeoManager::Instance().set_world(worldPlaced);
    //Vector<Daughter> dau=worldPlaced->daughters();
    std::cout<<"World and hype placed\n";
    
    //My placed volume
    //dau[0]->PrintContent();
    
    VPlacedVolume *hypePlaced=myhype.Place();
    hypePlaced->PrintContent();
   

    std::cout<<"Setting up ROOT geometry\n";
    new TGeoManager("world", "the simplest geometry");
    TGeoMaterial *mat = new TGeoMaterial("Vacuum",0,0,0);
    TGeoMedium *med = new TGeoMedium("Vacuum",1,mat);
    TGeoVolume *top = gGeoManager->MakeBox("Top",med,10,10,10);
    
    gGeoManager->SetTopVolume(top);
    gGeoManager->CloseGeometry();
    top->SetLineColor(kMagenta);
    gGeoManager->SetTopVisible();


    TGeoVolume *someVolume = gGeoManager->MakeHype("myHyperboloid", med, rMin, stIn, rMax, stOut, dz);
    TGeoHype *hype=new TGeoHype("myHype",rMin, stIn, rMax, stOut, dz);
    
    top->AddNode(someVolume,1);
    std::cout<<"Displaying ROOT geometry\n";
    TCanvas *c=new TCanvas();
    top->Draw();
    sleep(3);
    c->Update();
    sleep(3);
    
    
    int myCountIn=0;
    int myCountOut=0;
    int rootCountIn=0;
    int rootCountOut=0;
    int mismatchDistToIn=0;
    int mismatchDistToOut=0;
    int mismatchSafetyToIn=0;
    int mismatchSafetyToOut=0;
    int unvalidatedSafetyToIn=0;
    int unvalidatedSafetyToOut=0;
    int notValidSafetyToIn=0;
    int notValidSafetyToOut=0;
    
    float mbDistToIn;
    float rootDistToIn;
//    float mbDistToOut;
//    float rootDistToOut;
//    float mbSafetyToOut;
//    float rootSafetyToOut;
//    float mbSafetyToIn;
//    float rootSafetyToIn;
    
    double coord[3], direction[3], new_coord[3], module;
    double x=worldUnplaced.x();
    double y=worldUnplaced.y();
    double z=worldUnplaced.z();
    
    bool inside;
        
    Vector3D <Precision> *points = new Vector3D<Precision>[np];
    Vector3D <Precision> *dir = new Vector3D<Precision>[np];
    TRandom3 r3;
    r3.SetSeed(time(NULL));
    
  
    
    for(int i=0; i<np; i++) // points inside world volume
    {
        
        //generation=i%11;
        
        
        //generic generation
        if (generation==0) {
            //std::cout<<"0\n";
            points[i].x()=r3.Uniform(-x, x);
            points[i].y()=r3.Uniform(-y, y);
            points[i].z()=r3.Uniform(-z, z);
            
            dir[i].x()=r3.Uniform(-1, 1);
            dir[i].y()=r3.Uniform(-1, 1);
            dir[i].z()=r3.Uniform(-1, 1);
                
        }
        
            
        //points generated everywhere and directions pointing to the origin
        if (generation==1) {
            //std::cout<<"1\n";
            points[i].x()=r3.Uniform(-x, x);
            points[i].y()=r3.Uniform(-y, y);
            points[i].z()=r3.Uniform(-z, z);
            
            dir[i].x()=-points[i].x();
            dir[i].y()=-points[i].y();
            dir[i].z()=-points[i].z();
            
        }
            
        //points generated everywhere and directions perpendicular to the z-axix
        if (generation==2) {
            //std::cout<<"2\n";
            points[i].x()=r3.Uniform(-x, x);
            points[i].y()=r3.Uniform(-y, y);
            points[i].z()=r3.Uniform(-z, z);
                
            dir[i].x()=-points[i].x();
            dir[i].y()=-points[i].y();
            dir[i].z()=0;
        }
            
        //points generated in -dZ<z<dZ and directions pointing to the origin --> approaching the Hype from the hyperbolic surface
        if (generation==3) {
            //std::cout<<"3\n";
            points[i].x()=r3.Uniform(-x, x);
            points[i].y()=r3.Uniform(-y, y);
            points[i].z()=r3.Uniform(-dz, dz);
            
            dir[i].x()=-points[i].x();
            dir[i].y()=-points[i].y();
            dir[i].z()=-points[i].z();
        }
            
            
        //points generated in -dZ<z<dZ and directions perpendicular to the z-axix --> approaching the Hype from the hyperbolic surface
        if (generation==4) {
            //std::cout<<"4\n";
            points[i].x()=r3.Uniform(-x, x);
            points[i].y()=r3.Uniform(-y, y);
            points[i].z()=r3.Uniform(-dz, dz);
            
            dir[i].x()=-points[i].x();
            dir[i].y()=-points[i].y();
            dir[i].z()=0;
        }
        
        
        //points generated outside the bounding cylinder and with -Dz<z<Dz and pointing to the origin
        if (generation==5) {
            //std::cout<<"5\n";
            do{
                points[i].x()=r3.Uniform(-EndOuterRadius, EndOuterRadius);
                points[i].y()=r3.Uniform(-EndOuterRadius, EndOuterRadius);
                points[i].z()=r3.Uniform(-dz, dz);
            }while (points[i].x()*points[i].x()+points[i].y()*points[i].y()<EndOuterRadius2);
            
            dir[i].x()=-points[i].x();
            dir[i].y()=-points[i].y();
            dir[i].z()=-points[i].z();
            
        }
        //points generated outside the bounding cylinder and with -Dz<z<Dz and pointing everywhere
        if (generation==6) {
            //std::cout<<"6\n";
            do{
                points[i].x()=r3.Uniform(-EndOuterRadius, EndOuterRadius);
                points[i].y()=r3.Uniform(-EndOuterRadius, EndOuterRadius);
                points[i].z()=r3.Uniform(-dz, dz);
            }while (points[i].x()*points[i].x()+points[i].y()*points[i].y()<EndOuterRadius2);
            
            dir[i].x()=r3.Uniform(-1, 1);
            dir[i].y()=r3.Uniform(-1, 1);
            dir[i].z()=r3.Uniform(-1, 1);
            
        }
        
        //points that eventually hit the hyperboloid surface
        if (generation==7) {
            //std::cout<<"7\n";
            Float_t distZ;
            Float_t xHit;
            Float_t yHit;
            Float_t rhoHit;
            
            do{
                points[i].x()=r3.Uniform(-x, x);
                points[i].y()=r3.Uniform(-y, y);
                points[i].z()=r3.Uniform(dz, z);
                
                dir[i].x()=r3.Uniform(-1, 1);
                dir[i].y()=r3.Uniform(-1, 1);
                dir[i].z()=r3.Uniform(-1, 1);
                
                distZ = (Abs(points[i].z())-dz) / (Abs(dir[i].z()));
                xHit = points[i].x()+distZ*dir[i].x();
                yHit = points[i].y()+distZ*dir[i].y();
                rhoHit=TMath::Sqrt(xHit*xHit+yHit*yHit);
            }
            while(rhoHit<EndOuterRadius);
        }
        
        
        //points that hit the upper end surface
        if (generation==8) {
           
            //std::cout<<"8\n";
            Float_t distZ;
            Float_t xHit;
            Float_t yHit;
            Float_t rhoHit;
            
            do{
                points[i].x()=r3.Uniform(-x, x);
                points[i].y()=r3.Uniform(-y, y);
                points[i].z()=r3.Uniform(dz, z);
                
                dir[i].x()=r3.Uniform(-1, 1);
                dir[i].y()=r3.Uniform(-1, 1);
                dir[i].z()=r3.Uniform(-1, 1);
                
                distZ = (Abs(points[i].z())-dz) / (Abs(dir[i].z()));
                xHit = points[i].x()+distZ*dir[i].x();
                yHit = points[i].y()+distZ*dir[i].y();
                rhoHit=TMath::Sqrt(xHit*xHit+yHit*yHit);
                
            }
            while(rhoHit>EndOuterRadius || rhoHit<EndInnerRadius);
        }
       
        //points generated in -dZ<z<dZ and directions perpendicular to the z-axix --> approaching the Hype from the hyperbolic surface
        if (generation==9) {
            //std::cout<<"9\n";
            points[i].x()=r3.Uniform(-x, x);
            points[i].y()=r3.Uniform(-y, y);
            points[i].z()=r3.Uniform(-dz, dz);
            
            dir[i].x()=0;
            dir[i].y()=0;
            dir[i].z()=-points[i].z();
        }

        //points generated in -dZ<z<dZ and directions pointing to the origin --> approaching the Hype from the hyperbolic surface
        if (generation==10) {
            //std::cout<<"10\n";
            points[i].x()=r3.Uniform(-x, x);
            points[i].y()=r3.Uniform(-y, y);
            points[i].z()=r3.Uniform(-dz, dz);
            
            dir[i].x()=-points[i].x();
            dir[i].y()=-points[i].y();
            dir[i].z()=-points[i].z();
        }
        
        //points generated in -dZ<z<dZ and directions leaving the volume --> approaching the Hype from the hyperbolic surface
        if (generation==11) {
            //std::cout<<"11\n";
            points[i].x()=r3.Uniform(-x, x);
            points[i].y()=r3.Uniform(-y, y);
            points[i].z()=r3.Uniform(-dz, dz);
            
            dir[i].x()=r3.Uniform(-1, 1);
            dir[i].y()=r3.Uniform(-1, 1);
            dir[i].z()=r3.Uniform(-1, 1);
        }
        
        //NB: 12 e 13 cannot be used if Rmin=0
        
        //points that hit the inner surface of the hype
        if (generation==12) {
            std::cout<<"12\n";
            Float_t distZ;
            Float_t xHit;
            Float_t yHit;
            Float_t rhoHit;
            
            do{
                points[i].x()=r3.Uniform(-x, x);
                points[i].y()=r3.Uniform(-y, y);
                points[i].z()=r3.Uniform(dz, z);
                
                dir[i].x()=-points[i].x();
                dir[i].y()=-points[i].y();
                dir[i].z()=-points[i].z();
                
                distZ = (Abs(points[i].z())-dz) / (Abs(dir[i].z()));
                xHit = points[i].x()+distZ*dir[i].x();
                yHit = points[i].y()+distZ*dir[i].y();
                rhoHit=TMath::Sqrt(xHit*xHit+yHit*yHit);
            }
            while(rhoHit>EndInnerRadius);
        }
        
        //points generated in -dZ<z<dZ inside the inner part and directions pointing to the origin --> approaching the Hype from the inner hyperbolic surface
        if (generation==13) {
            std::cout<<"13\n";
            do{
                points[i].x()=r3.Uniform(-x, x);
                points[i].y()=r3.Uniform(-y, y);
                points[i].z()=r3.Uniform(-dz, dz);
            }
            while( (points[i].x()*points[i].x()+points[i].y()*points[i].y())>( tIn*tIn*points[i].z()*points[i].z() + rMin*rMin ));
            dir[i].x()=-points[i].x();
            dir[i].y()=-points[i].y();
            dir[i].z()=-points[i].z();
        }

        
        module=Sqrt(dir[i].x()*dir[i].x()+dir[i].y()*dir[i].y()+dir[i].z()*dir[i].z());
        dir[i].x()=dir[i].x()/module;
        dir[i].y()=dir[i].y()/module;
        dir[i].z()=dir[i].z()/module;
    }
    std::cout<<"\n";
    
    //Marker for inside points
    TPolyMarker3D *markerInside=0;
    TObjArray *pm = new TObjArray(128);
    markerInside = (TPolyMarker3D*)pm->At(4);
    markerInside = new TPolyMarker3D();
    markerInside->SetMarkerColor(kYellow);
    markerInside->SetMarkerStyle(8);
    markerInside->SetMarkerSize(0.4);
    pm->AddAt(markerInside, 4);
    
    //Marker for outside points
    TPolyMarker3D *markerOutside=0;
    markerOutside = (TPolyMarker3D*)pm->At(4);
    markerOutside = new TPolyMarker3D();
    markerOutside->SetMarkerColor(kGreen+1);
    markerOutside->SetMarkerStyle(8);
    markerOutside->SetMarkerSize(0.1);
    pm->AddAt(markerOutside, 4);
    
        
    //Marker for sphere outside points
    TPolyMarker3D *markerSphereOutside=0;
    markerSphereOutside = (TPolyMarker3D*)pm->At(4);
    
    //Marker for sphere inside points
    TPolyMarker3D *markerSphereInside=0;
    markerSphereInside = (TPolyMarker3D*)pm->At(4);
    int counter;
    
    for(int i=0; i<np; i++)
    {
        inside=hypePlaced->Inside(points[i]);
        if(inside!=0){ //Enum-inside give back 0 if the point is inside
            myCountOut++;
            
        }
        else{
            myCountIn++;
            markerInside->SetNextPoint(points[i].x(), points[i].y(), points[i].z());
        }
        coord[0]=points[i].x();
        coord[1]=points[i].y();
        coord[2]=points[i].z();
        
        direction[0]=dir[i].x();
        direction[1]=dir[i].y();
        direction[2]=dir[i].z();
            
            
        inside=someVolume->Contains(coord);
        //inside=par->Contains(coord);
        
        //the point is outside!
        if(inside==0){
            rootCountOut++;
            
            //mbDistToIn=dau[0]->DistanceToIn(points[i], dir[i]);
            
            //DISTANCE TO IN
            mbDistToIn=hypePlaced->DistanceToIn(points[i], dir[i]);
            rootDistToIn=hype->DistFromOutside(coord, direction);
            markerOutside->SetNextPoint(points[i].x(), points[i].y(), points[i].z());
            
            if( (mbDistToIn!=rootDistToIn) && !(mbDistToIn == kInfinity && rootDistToIn>1e+29))
            {
                std::cout<<"mbDistToIn: "<<mbDistToIn;
                std::cout<<" rootDistToIn: "<<rootDistToIn<<"\n";
                mismatchDistToIn++;
                
            }
            
//            //SAFETY TO IN
//            mbSafetyToIn=hypePlaced->SafetyToIn(points[i]);
//            rootSafetyToIn=par->Safety(coord, false);
//            
//            //validation of SafetyToIn
//            //I shoot random point belonging to the sphere with radious mbSafetyToIn and
//            //then I see it they are all still outside the volume
//            
//            markerSphereOutside = new TPolyMarker3D();
//            markerSphereOutside->SetMarkerColor(kGreen+i);
//            counter=0;
//            for (int j=0; j<100000; j++) //10^5
//            {
//                
//                double v=r3.Uniform(0, 1);
//                double theta=r3.Uniform(0, 2*kPi);
//                double phi=TMath::ACos(2*v-1);
//                
//                double r= mbSafetyToIn*TMath::Power(r3.Uniform(0, 1), 1./3);
//                //std::cout<<"r: "<<r<<"\n";
//                    
//                    
//                double x_offset=r*TMath::Cos(theta)*TMath::Sin(phi);
//                double y_offset=r*TMath::Sin(theta)*TMath::Sin(phi);
//                
//                double z_offset=r*TMath::Cos(phi);
//                
//                new_coord[0]=coord[0]+x_offset;
//                new_coord[1]=coord[1]+y_offset;
//                new_coord[2]=coord[2]+z_offset;
//                
//                double safety2=mbSafetyToIn*mbSafetyToIn;
//            
//                if(x_offset*x_offset+y_offset*y_offset+z_offset*z_offset<=safety2)
//                {
//                    counter++;
//                    markerSphereOutside->SetNextPoint(new_coord[0], new_coord[1], new_coord[2]);
//                    inside=someVolume->Contains(new_coord);
//                    if(inside) notValidSafetyToIn++;
//                }
//                    
//            }
//            //if (markerSphereOutside) markerSphereOutside->Draw("SAME");
//            //c->Update();
//            if( (mbSafetyToIn!=rootSafetyToIn))
//            {
//                //std::cout<<"mbSafetyToIn: "<<mbSafetyToIn;
//                //std::cout<<" rootSafetyToIn: "<<rootSafetyToIn<<"\n";
//                mismatchSafetyToIn++;
//            }
//            if( (mbSafetyToIn>rootSafetyToIn))
//            {
//                //std::cout<<"mbSafetyToIn: "<<mbSafetyToIn;
//                //std::cout<<" rootSafetyToIn: "<<rootSafetyToIn<<"\n";
//                unvalidatedSafetyToIn++;
//            }
            
        }
        else{
            
            //POINT IS INSIDE
            rootCountIn++;
            
//            //DISTANCE TO OUT
//            mbDistToOut=hypePlaced->DistanceToOut(points[i], dir[i]);
//            rootDistToOut=par->DistFromInside(coord, direction);
//            if( (mbDistToOut!=rootDistToOut))
//            {
//                //markerOutside->SetNextPoint(points[i].x(), points[i].y(), points[i].z());
//                std::cout<<"mbDistToOut: "<<mbDistToOut;
//                std::cout<<" rootDistToOut: "<<rootDistToOut<<"\n";
//                mismatchDistToOut++;
//            }
//            
//            //SAFETY TO OUT
//            mbSafetyToOut=hypePlaced->SafetyToOut(points[i]);
//            rootSafetyToOut=par->Safety(coord, true);
//            if( (mbSafetyToOut!=rootSafetyToOut))
//            {
//                //std::cout<<"mbSafetyToOut: "<<mbSafetyToOut;
//                //std::cout<<" rootSafetyToOut: "<<rootSafetyToOut<<"\n";
//                mismatchSafetyToOut++;
//            }
//            if( (mbSafetyToOut>rootSafetyToOut))
//            {
//                unvalidatedSafetyToOut++;
//            }
//                
//            //validation of SafetyToOut
//            //I shoot random point belonging to the sphere with radious mbSafetyToOut and
//            //then I see it they are all still outside the volume
//            
//            markerSphereInside = new TPolyMarker3D();
//            markerSphereInside->SetMarkerColor(kGreen+i);
//            for (int j=0; j<10000; j++)
//            {
//                
//                double v=r3.Uniform(0, 1);
//                double theta=r3.Uniform(0, 2*kPi);
//                double phi=TMath::ACos(2*v-1);
//                
//                double r= mbSafetyToOut*TMath::Power(r3.Uniform(0, 1), 1./3);
//                
//                double x_offset=r*TMath::Cos(theta)*TMath::Sin(phi);
//                double y_offset=r*TMath::Sin(theta)*TMath::Sin(phi);
//                
//                double z_offset=r*TMath::Cos(phi);
//                
//                new_coord[0]=coord[0]+x_offset;
//                new_coord[1]=coord[1]+y_offset;
//                new_coord[2]=coord[2]+z_offset;
//                
//                double safety2=mbSafetyToOut*mbSafetyToOut;
//                
//                if(x_offset*x_offset+y_offset*y_offset+z_offset*z_offset<=safety2)
//                {
//                    markerSphereInside->SetNextPoint(new_coord[0], new_coord[1], new_coord[2]);
//                    inside=someVolume->Contains(new_coord);
//                    if(!inside) notValidSafetyToOut++;
//                }
//                
//            }
        }
    }
    
    if (markerInside) markerInside->Draw("SAME");
    c->Update();
    sleep(3);
    
    if (markerOutside) markerOutside->Draw("SAME");
    c->Update();
    sleep(3);
    
    std::cout<<"MB: NPointsInside: "<<myCountIn<<" NPointsOutside: "<<myCountOut<<" \n";
    std::cout<<"Root: NPointsInside: "<<rootCountIn<<" NPointsOutside: "<<rootCountOut<<" \n";
    std::cout<<"DistToIn mismatches: "<<mismatchDistToIn<<" \n";
    std::cout<<"DistToOut mismatches: "<<mismatchDistToOut<<" \n";
    std::cout<<"SafetyToIn mismatches: "<<mismatchSafetyToIn<<" \n";
    std::cout<<"SafetyToOut mismatches: "<<mismatchSafetyToOut<<" \n";
    std::cout<<"Against ROOT unvalidated SafetyToIn: "<<unvalidatedSafetyToIn<<" \n";
    std::cout<<"Against ROOT Unvalidated SafetyToOut: "<<unvalidatedSafetyToOut<<" \n";
    std::cout<<"Not valid SafetyToIn: "<<notValidSafetyToIn<<" \n";
    std::cout<<"Not valid SafetyToOut: "<<notValidSafetyToOut<<" \n";
//    std::cout<<"Counter: "<<counter<<" \n";
    
    theApp.Run();
    return 0;
    }



