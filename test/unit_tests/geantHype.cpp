#include "G4Hype.hh"
#include "G4Sphere.hh"
#include "G4ThreeVector.hh"
#define PI 3.14159265358979323846
int main(){

G4Hype hype("testHype",10,15,PI/4,PI/4,50);
std::cout<<"Volume : "<<hype.GetCubicVolume()<<std::endl;
std::cout<<"SurfaceArea : "<<hype.GetSurfaceArea()<<std::endl;
G4ThreeVector testPoint(18,0,0);
G4ThreeVector normal = hype.SurfaceNormal(testPoint);
std::cout<< "Normal Calculated from Geant4 : "<< normal <<std::endl;

return 0;
}
