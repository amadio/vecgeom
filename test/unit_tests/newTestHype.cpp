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

	Vec_t testSPoint(1.3022779695533808209, -14.943430098582158649, -0.045066952023698494956);
	Vec_t testSDir(-0.12055468236486513445, 0.91364927195065248622, -0.38821588893803588016);
	Dist = test.DistanceToOut(testSPoint,testSDir,norm,convex);
	std::cout<<"Distance To Out : "<<Dist<<std::endl;

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

