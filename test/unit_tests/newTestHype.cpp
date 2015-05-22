//
//
// TestBox
//             Ensure asserts are compiled in

#undef NDEBUG
#include "base/Global.h"
#include "base/Vector3D.h"
#include "volumes/Box.h"
#include "volumes/Hype.h"
#include "ApproxEqual.h"
//#ifdef VECGEOM_USOLIDS
//#include "UBox.hh"
//#include "G4Hype.hh"
//#include "UVector3.hh"
//#endif

#include <cassert>
#include <cmath>
#include <iomanip> 

#define PI 3.14159265358979323846


template <class Hype_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision> >
bool TestHype() {
	double Dist=0.;
	Vec_t norm;
	bool convex;
    	convex = true;
    /*
    
    //Hype_t test("Solid VecGeomHype #test",10,15, PI/4,PI/4, 50); //5,6,0.523598775598298,0.523598775598298,10);
	Vec_t norm;
	bool convex;
    convex = true;
	Hype_t test("Solid VecGeomHype #test",5,15, PI/4,PI/4, 50); 
    Vec_t testOutPoint(18,0,0);
	//double Dist = test.SafetyFromOutside(testOutPoint);
	double Dist = test.SafetyToIn(testOutPoint);
    std::cout<<"SafetyFromOutside : "<<Dist<<std::endl;
	
	Vec_t testInPoint(12,0,0);
//	Dist = test.SafetyFromInside(testInPoint);
	Dist = test.SafetyToOut(testInPoint);
    std::cout<<"SafetyFromInside : "<<Dist<<std::endl;

    std::cout<<"Capacity is : "<<test.Capacity()<<std::endl;
    std::cout<<"SurfaceArea is : "<<test.SurfaceArea()<<std::endl;

	Vec_t testPoint(18,16,0);
	Vec_t testDir(-1,0,0);
	Vec_t normal;
	test.Normal(testPoint,normal);
	
	std::cout<<"Normal Calculated from VecGeom is  : "<<normal<<std::endl;
	
	Dist = test.DistanceToIn(testPoint,testDir);
	std::cout<<"DisanceToIn  : "<<Dist<<std::endl;

	std::cout<<test.GetEntityType()<<std::endl;

	Vec_t testSPoint(-44.450479843527674007, 18.584586574030911521, 45.784626421394953866);
	Vec_t testSDir(-0.65138175733324088501 ,-0.27513920805728042662 ,-0.70710693844924998874);

	double mag = std::sqrt(testSPoint.x() * testSPoint.x() + testSPoint.y() * testSPoint.y() );
	std::cout<<std::setprecision(15)<<std::endl;
	std::cout<<"Circular Magnitude of Point : "<<mag<<std::endl;
	
	Dist = test.DistanceToOut(testSPoint,testSDir,norm,convex);
	std::cout<<"Distance To Out : "<<Dist<<std::endl;

	std::cout<<"location of point VecGEom : "<<test.Inside(testSPoint)<<std::endl;
*/
/*
	std::cout<<"------------ Debugging SafetyToIn ---------------"<<std::endl;

	Hype_t testStoIHype("SafetyToInHype",10,15,PI/6,PI/6,50);
	Vec_t testStoIPoint(8.0358966686930024537, 14.260110268024778435,-11.636981024237805116);

   
	double Dist = testStoIHype.SafetyToIn(testStoIPoint);
	std::cout<<"SafetyToin Distance is : "<<Dist<<std::endl;

		
	//Vec_t normal;
	//testStoIHype.Normal(testStoIPoint,normal);
	//std::cout<<"Normal Calculated from VecGeom is  : "<<normal<<std::endl;
	

 	Hype_t testSurfHype("DistToOutHype",5.,15,PI/4,PI/3,50); 
	Vec_t testSurfPoint(-27.923606564455180745, -18.925923051335679759, -33.36043115143591109);
	Vec_t testSurfDir(0.68594939680747823996, 0.17161750750547485889, -0.70712152854871068719);
	Vec_t norm;
	bool convex;
    convex = true;
	Dist = testSurfHype.DistanceToOut(testSurfPoint,testSurfDir,norm,convex);
	std::cout<<"Distance To Out from VecGeom : "<<Dist<<std::endl;
	Vec_t newPt(testSurfPoint + Dist*testSurfDir);
	std::cout<<"location of point from VecGEom : "<<testSurfHype.Inside(newPt)<<std::endl;
	Dist = testSurfHype.DistanceToOut(newPt,testSurfDir,norm,convex);
	std::cout<<"Distance To Out of newPt from VecGeom : "<<Dist<<std::endl;


	Vec_t testDToInPoint(-26.70205919484012469 ,1.4445546782040044054, 70.437702749904943289);
	Vec_t testDToInDir(-0.53884309954166709211 ,-0.0071101770891100739974, -0.8423761389415601597);
	Dist = testSurfHype.DistanceToIn(testDToInPoint,testDToInDir);
	std::cout<<"DistanceToIn from VecGeom : "<<Dist<<std::endl;
	std::cout<<"SafetyDistance from Geant4 : "<< testSurfHype.SafetyToIn(testDToInPoint)<<std::endl;

	Vec_t normal;
	Vec_t testNormPoint(0.42373546604079592726, -5.005169952185724469, 0.48091370891575024871);
	Vec_t testNormDir(0.24792999855963759881, -0.67289006656080596613, -0.69695744069348475325);
	testSurfHype.Normal(testNormPoint,normal);
	std::cout<<"Normal Calculated from VecGeom is  : "<<normal<<std::endl;
	Dist = testSurfHype.DistanceToIn(testNormPoint,testNormDir);
	std::cout<<"DistanceToIn from VecGeom : "<<Dist<<std::endl;

	//********************************************************************************************
*/
	std::cout<<std::setprecision(15);
	Hype_t testSurfHype("DistToOutHype",5.,20,PI/4,PI/3,50); 
	std::cout<<"***************************************************************************************"<<std::endl;
	Vec_t testDToOutPoint(76.8105482935058, 4.2227289090123, 42.8862522583373);
	Vec_t testDToOutDir(0.725823123403194, -0.686644810231805, -0.0412273951998107);
	Dist = testSurfHype.DistanceToOut(testDToOutPoint,testDToOutDir,norm,convex);
	std::cout<<"Distance To Out from VecGeom : "<<Dist<<std::endl;
	std::cout<<"location of point from VecGEom : "<<testSurfHype.Inside(testDToOutPoint)<<std::endl;
	Dist = testSurfHype.DistanceToIn(testDToOutPoint,testDToOutDir);
	std::cout<<"Distance To In from VecGeom : "<<Dist<<std::endl;
	std::cout<<"SafetyToOut from VecGeom : "<<testSurfHype.SafetyToOut(testDToOutPoint)<<std::endl;

	std::cout<<"------------------------------------------\n";
	Hype_t midHype("Mid Hype",10,20,PI/4,PI/3,50);
	Vec_t midPoint(15,0,0);
	Vec_t midDir(0,1,0);
	Dist = midHype.DistanceToOut(midPoint,midDir,norm,convex);
        std::cout<<"Distance To Out of MID point from VecGeom : "<<Dist<<std::endl;

	std::cout<<"Volume : "<<testSurfHype.Capacity()<<std::endl;
    return true;
}

int main() {
    
//#ifdef VECGEOM_USOLIDS
//  assert(TestHype<UHype>());
//  std::cout << "UHype passed\n";
//#endif
  std::cout<<"-------------------------------------------------------------------------------------------------"<<std::endl;
  //std::cout<<"*************************************************************************************************"<<std::endl;
  assert(TestHype<vecgeom::SimpleHype>());
  //vecgeom::SimpleHype test("Solid VecGeomHype #test",5, 30, 6, 30, 10); //5,6,0.523598775598298,0.523598775598298,10);


  std::cout << "VecGeomHype passed\n";
  return 0;
}

