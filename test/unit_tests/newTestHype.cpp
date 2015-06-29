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
	Hype_t hypeD("test_VecGeomHype",5.,20,PI/6,PI/4,50);
	Vec_t testP(-4.08987645483561, -43.9168827288048, 50.0000008049304);
	Vec_t testD(-0.301030608699687, -0.527155667813519, 0.794661861748743);
	std::cout<<"PDOTV3D : "<<testP.Dot(testD)<<std::endl;
	Dist = hypeD.DistanceToIn(testP,testD);
	std::cout<<"DistToIn : "<<Dist<<std::endl;
	Dist = hypeD.DistanceToOut(testP,testD,norm,convex);
    std::cout<<"DistToOut : "<<Dist<<std::endl;
	std::cout<<"Location of Point from VecGeom  : " << hypeD.Inside(testP)<<std::endl;
	*/

	std::cout<<"**************************************************************\n";
	Hype_t hypeD("test_VecGeomHype",5.,20,PI/6,PI/4,50);
	Vec_t testP(28.4073618747969, -14.7583198850071, -49.8361524659377);
	Vec_t testD(-0.573820272751705, 0.0885439480090447, 0.814180731686848);
	Dist = hypeD.DistanceToIn(testP,testD);
	std::cout<<"DistToIn for Corner point: "<<Dist<<std::endl;
	double DistanceOut2 = hypeD.DistanceToOut(testP,testD,norm,convex);
	//Vec_t newPt
	//double DistanceOut2 = volumeUSolids->DistanceToOut(point, dir ,norm,convex2);


    Vec_t pointInAct = testP+testD*DistanceOut2*(1.-10*1e-9);
	Vec_t pointIn(-41.955404037290094,-3.9009183909538354 , 49.99999999185846);
	std::cout<<"=== Second Time ===\n";
   double DistanceOut =  hypeD.DistanceToOut(pointIn,testD,norm,convex);


	std::cout<<"============================================"<<std::endl;
	std::cout<<"Point : "<<testP<<"   :: Dir : "<<testD<<std::endl;
	std::cout<<"DistanceOut2 : "<<DistanceOut2<<std::endl;
	std::cout<<"   PointIn : "<<pointIn<<std::endl;
	std::cout<<"PointInAct : "<<pointInAct<<std::endl;
	std::cout<<"DistanceOut : "<<DistanceOut<<std::endl;
    //ReportError( &nError,pointIn, dir, DistanceOut, "SD: DistanceToOut is not precise");
	std::cout<<"Location of Point from VecGeom  : " << hypeD.Inside(pointIn)<<std::endl;
	std::cout<<"============================================"<<std::endl;

	Vec_t transP(-39.0527636295821, 20.032697606176, 49.9999990365669);
	std::cout<<"Location of Point from VecGeom  : " << hypeD.Inside(transP)<<std::endl;

    std::cout<<"============================================"<<std::endl;
    Hype_t hypeS("test_VecGeomHype",5.,20,PI/6,PI/3,50);
    Vec_t pointS(51.3317803344263, 6.208742448696, -15.9749535016947);
    Vec_t dirS(-0.886749038154024 ,-0.0747924453665009, 0.456160315513102);
    Dist = hypeS.DistanceToIn(pointS,dirS);
    double safeDist = hypeS.SafetyToIn(pointS);
    std::cout<<"Dist : "<<Dist<<" :: SafetyDist : "<<safeDist<<std::endl;

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

