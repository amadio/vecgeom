#include "G4Hype.hh"
#include "G4Sphere.hh"
#include "G4ThreeVector.hh"
#include "G4Types.hh"
#define PI 3.14159265358979323846
int main(){

//G4Hype hype("testHype",10,15,PI/4,PI/4,50);
G4Hype hype("testHype",5,15,PI/4,PI/4,50);
std::cout<<"Volume : "<<hype.GetCubicVolume()<<std::endl;
std::cout<<"SurfaceArea : "<<hype.GetSurfaceArea()<<std::endl;
G4ThreeVector testPoint(18,16,0);
G4ThreeVector testDir(-1,0,0);
G4ThreeVector normal = hype.SurfaceNormal(testPoint);
std::cout<< "Normal Calculated from Geant4 : "<< normal <<std::endl;
double Dist = hype.DistanceToIn(testPoint,testDir);
std::cout<<"DistanceToIn from G4 : "<<Dist<<std::endl;


G4bool calcNorm;
G4bool *validNorm;
G4ThreeVector *norm;
G4ThreeVector testSPoint(1.3022779695533808209, -14.943430098582158649, -0.045066952023698494956);
G4ThreeVector testSDir(-0.12055468236486513445, 0.91364927195065248622, -0.38821588893803588016);
Dist = hype.DistanceToOut(testSPoint,testSDir,calcNorm,validNorm,norm);
std::cout<<"Distance To Out from G4 : "<<Dist<<std::endl;

return 0;
} 
