#include "G4Hype.hh"
#include "G4Sphere.hh"
#include "G4ThreeVector.hh"
#include "G4Types.hh"
#include <iomanip>
#include "G4VSolid.hh"
#include "G4GeometryTolerance.hh"

#define PI 3.14159265358979323846

int main(){

typedef G4Hype Hype_t;
typedef G4ThreeVector Vec_t;
double Dist=0.;
Vec_t *norm;
	G4bool calcNorm;
	G4bool *validNorm;
/*
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
G4ThreeVector testSPoint(-44.450479843527674007, 18.584586574030911521, 45.784626421394953866);
G4ThreeVector testSDir(-0.65138175733324088501 ,-0.27513920805728042662 ,-0.70710693844924998874);

double mag = std::sqrt(testSPoint.x() * testSPoint.x() + testSPoint.y() * testSPoint.y()  );
std::cout<<std::setprecision(15)<<std::endl;
std::cout<<"\n Circular Magnitude of Point : "<<mag<<std::endl;
double kCarTolerance = G4GeometryTolerance::GetInstance()->GetSurfaceTolerance();
std::cout<<"kCarTolerance : "<<kCarTolerance<<std::endl;
double xR2 = 44.450479843527674007*44.450479843527674007 + 18.584586574030911521*18.584586574030911521;
double oRad2 = 15. * 15. + 45.78462642139495386*45.78462642139495386;
double iRad2 = 5. * 5. + 45.78462642139495386*45.78462642139495386;

double endOuterRadius = std::sqrt(15. * 15. + 50. * 50.);
double endInnerRadius = std::sqrt(5. * 5. + 50. * 50.);

std::cout<<"xR2 : "<<xR2 << std::endl;
std::cout<<"OuterCompRadius : "<< (oRad2 + kCarTolerance*endOuterRadius)<<std::endl;
std::cout<<"InnerCompRadius : "<< (iRad2 - kCarTolerance*endInnerRadius)<<std::endl;

Dist = hype.DistanceToOut(testSPoint,testSDir,calcNorm,validNorm,norm);
std::cout<<"Distance To Out from G4 : "<<Dist<<std::endl;

std::cout<<"Location of Point from G4  : " << hype.Inside(testSPoint)<<std::endl;
std::cout<<"==========================================================\n";
*/
/*
	std::cout<<"------------ Debugging SafetyToIn ---------------"<<std::endl;

	//Hype_t testStoIHype("SafetyToInHype",10,15,PI/6,PI/6,50);
	//Vec_t testStoIPoint(4.278980, 28.387627, 55.518894);

	Hype_t testStoIHype("SafetyToInHype",10,15,PI/6,PI/6,50);
	Vec_t testStoIPoint(8.0358966686930024537, 14.260110268024778435,-11.636981024237805116);


	double Dist = testStoIHype.DistanceToIn(testStoIPoint);
	std::cout<<"SafetyToin Distance is : "<<Dist<<std::endl;

	Hype_t testSurfHype("DistToOutHype",5.,15,PI/4,PI/3,50); 
	Vec_t testSurfPoint(-27.923606564455180745, -18.925923051335679759, -33.36043115143591109);
	Vec_t testSurfDir(0.68594939680747823996, 0.17161750750547485889, -0.70712152854871068719);
	Vec_t *norm;
	G4bool calcNorm;
	G4bool *validNorm;
    //convex = true;
	Dist = testSurfHype.DistanceToOut(testSurfPoint,testSurfDir,calcNorm,validNorm,norm);
	std::cout<<"Distance To Out from Geant4 : "<<Dist<<std::endl;
	Vec_t newPt(testSurfPoint + Dist*testSurfDir);
	std::cout<<"location of point from Geant4 : "<<testSurfHype.Inside(newPt)<<std::endl;
	Dist = testSurfHype.DistanceToOut(newPt,testSurfDir,calcNorm,validNorm,norm);
	std::cout<<"Distance To Out of newPt from Geant4 : "<<Dist<<std::endl;

	Vec_t testDToInPoint(-26.70205919484012469 ,1.4445546782040044054, 70.437702749904943289);
	Vec_t testDToInDir(-0.53884309954166709211 ,-0.0071101770891100739974, -0.8423761389415601597);
	Dist = testSurfHype.DistanceToIn(testDToInPoint,testDToInDir);
	std::cout<<"DistanceToIn from Geant4 : "<<Dist<<std::endl;
	std::cout<<"SafetyDistance from Geant4 : "<< testSurfHype.DistanceToIn(testDToInPoint)<<std::endl;

	Vec_t normal;
	Vec_t testNormPoint(0.42373546604079592726, -5.005169952185724469, 0.48091370891575024871);
	Vec_t testNormDir(0.24792999855963759881, -0.67289006656080596613, -0.69695744069348475325);
	normal = testSurfHype.SurfaceNormal(testNormPoint);
	std::cout<<"Normal Calculated from Geant4 is  : "<<normal<<std::endl;
	Dist = testSurfHype.DistanceToIn(testNormPoint,testNormDir);
	std::cout<<"DistanceToIn from Geant4 : "<<Dist<<std::endl;
	*/

	//********************************************************************************************
	//std::cout<<"***************************************************************************************"<<std::endl;
	//Hype_t testSurfHype("DistToOutHype",5.,20,PI/4,PI/3,50); 
    //}
	std::cout<<std::setprecision(15);
	std::cout<<"***************************************************************************************"<<std::endl;
	Hype_t testSurfHype("DistToOutHype",5.,20,PI/4,PI/3,50); 
	Vec_t testDToOutPoint(76.8105482935058, 4.2227289090123, 42.8862522583373);
	Vec_t testDToOutDir(0.725823123403194, -0.686644810231805, -0.0412273951998107);

	Dist = testSurfHype.DistanceToOut(testDToOutPoint,testDToOutDir,calcNorm,validNorm,norm);
	std::cout<<"Distance To Out from Geant4 : "<<Dist<<std::endl;
	std::cout<<"location of point from Geant4 : "<<testSurfHype.Inside(testDToOutPoint)<<std::endl;
	Dist = testSurfHype.DistanceToIn(testDToOutPoint,testDToOutDir);
	std::cout<<"Distance To In from Geant4 : "<<Dist<<std::endl;

	std::cout<<"SafetyToOut from Geant4 : "<<testSurfHype.DistanceToOut(testDToOutPoint)<<std::endl;

	std::cout<<"------------------------------------------\n";
	Hype_t midHype("Mid Hype",10,20,PI/4,PI/3,50);
	Vec_t midPoint(0,15,0);
	Vec_t midDir(1,0,0);
	Dist = midHype.DistanceToOut(midPoint,midDir,calcNorm,validNorm,norm);
        std::cout<<"Distance To Out of MID point from Geant4 : "<<Dist<<std::endl;
	
/*
	Vec_t normal2 = testStoIHype.SurfaceNormal(testStoIPoint);
	std::cout<< "Normal Calculated from Geant4 : "<< normal2 <<std::endl;
	//Vec_t normal2(0,0,-1);
	Dist = testStoIHype.DistanceToIn(testStoIPoint,normal2);
	std::cout<<"DistanceToIn using normal : "<<Dist<<std::endl;

*/
return 0;
} 
