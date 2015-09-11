/*Unit test to test the Convexity function for various shapes
 *
 */

#include "base/Global.h"
#include "base/Vector3D.h"
#include "volumes/Box.h"
#include "volumes/Sphere.h"
#include "ApproxEqual.h"

#include "volumes/Orb.h"
#include "volumes/Sphere.h"
#include "volumes/Paraboloid.h"
#include "volumes/Cone.h"
#include "volumes/Torus2.h"
#include "volumes/Tube.h"
#include "volumes/Parallelepiped.h"
#include "volumes/Trd.h"
#include "volumes/Polycone.h"
#include "volumes/LogicalVolume.h"
//#include "volumes/UnplacedSphere.h"

//Added just for debugging, should be removed later
#include "USphere.hh"
#include "UCons.hh"

#include <cmath>
#include <iomanip>

#undef NDEBUG
#include <cassert>

#define PI 3.14159265358979323846
#define deg PI/180.

using namespace VECGEOM_NAMESPACE;

bool test_ConvexityOrb() {

	vecgeom::SimpleOrb b1("Solid VecGeomOrb #1",5.);
	assert(b1.IsConvex());
	return true;
}


bool test_ConvexitySphere() {

	double rmin=0., rmax=5., sphi=0., dphi=2*PI, stheta=0., dtheta=PI;
	vecgeom::SimpleSphere b1("Solide VecGeomSphere #1", rmin, rmax, sphi, dphi, stheta, dtheta);
	assert(b1.IsConvex());
	vecgeom::SimpleSphere b2("Solide VecGeomSphere #2", rmin, rmax, 0. , PI, stheta, dtheta);
	assert(b2.IsConvex());
	vecgeom::SimpleSphere b3("Solide VecGeomSphere #3", rmin, rmax, 0. , PI/3, stheta, dtheta);
	assert(b3.IsConvex());
	vecgeom::SimpleSphere b4("Solide VecGeomSphere #4", rmin, rmax, 0. , 4*PI/3, stheta, dtheta);
	assert(!b4.IsConvex());
	vecgeom::SimpleSphere b5("Solide VecGeomSphere #5", rmin, rmax, PI/3. , PI/3, stheta, dtheta);
	assert(b5.IsConvex());
	vecgeom::SimpleSphere b6("Solide VecGeomSphere #6", rmin, rmax, PI/3. , 2*PI/3, stheta, dtheta);
	assert(b6.IsConvex());
	vecgeom::SimpleSphere b7("Solide VecGeomSphere #7", rmin, rmax, PI/3. , PI, stheta, dtheta);
	assert(b7.IsConvex());
	vecgeom::SimpleSphere b8("Solide VecGeomSphere #8", rmin, rmax, PI/3. , 7*PI/6, stheta, dtheta);
	assert(!b8.IsConvex());

	//checking proper dphi calculation if specified dphi>2PI
	//Should be accepted by Wedge
	//Convention used for dPhi is if(dPhi>2PI) dPhi=2PI //needs a relook
	vecgeom::SimpleSphere b9("Solide VecGeomSphere #9", rmin, rmax, PI/3. , 4*PI, stheta, dtheta);
	assert(b9.IsConvex());
	//std::cerr<<"Newly Calcuated DPHi of b9 : "<<b9.GetDPhi()<<std::endl;

	vecgeom::SimpleSphere b10("Solide VecGeomSphere #10", rmin, rmax, PI/3. , 5*PI, stheta, dtheta);
	assert(b10.IsConvex());
	//std::cerr<<"Newly Calcuated DPHi of b10 : "<<b10.GetDPhi()<<std::endl;

	//vecgeom::SimpleSphere b2("Solide VecGeomSphere #1", 3, rmax, sphi, dphi, stheta, dtheta);

	//This case should be discussed
	vecgeom::SimpleSphere b11("Solide VecGeomSphere #11", rmin, rmax, PI/3. , ((2*PI) + (7*PI/6)), stheta, dtheta);
	assert(b11.IsConvex());
	//std::cerr<<"Newly Calcuated DPHi of b11 : "<<b10.GetDPhi()<<std::endl;

	vecgeom::SimpleSphere b12("Solide VecGeomSphere #12", rmin, rmax, 0. , 2*PI, stheta, PI/2);
	assert(b12.IsConvex());
	vecgeom::SimpleSphere b13("Solide VecGeomSphere #13", rmin, rmax, 0. , 2*PI, stheta, 2*PI/3);
	assert(!b13.IsConvex());
	vecgeom::SimpleSphere b14("Solide VecGeomSphere #14", rmin, rmax, 0. , 2*PI, stheta, PI/3);
	assert(b14.IsConvex());
	vecgeom::SimpleSphere b15("Solide VecGeomSphere #15", rmin, rmax, 0. , 2*PI, PI/6, PI/6);
	assert(!b15.IsConvex());
	vecgeom::SimpleSphere b16("Solide VecGeomSphere #16", rmin, rmax, 0. , 2*PI, PI/6, PI/3);
	assert(!b16.IsConvex());
	vecgeom::SimpleSphere b17("Solide VecGeomSphere #17", rmin, rmax, 0. , 2*PI, PI/2, PI/2);
	assert(b17.IsConvex());
	vecgeom::SimpleSphere b18("Solide VecGeomSphere #18", rmin, rmax, 0. , 2*PI, PI/2, PI/6);
	assert(!b18.IsConvex());
	vecgeom::SimpleSphere b19("Solide VecGeomSphere #19", rmin, rmax, 0. , 2*PI, 2*PI/3, PI/3);
	assert(b19.IsConvex());
	vecgeom::SimpleSphere b20("Solide VecGeomSphere #20", rmin, rmax, 0. , 2*PI, 2*PI/3, PI/6);
	assert(!b20.IsConvex());
	vecgeom::SimpleSphere b21("Solide VecGeomSphere #21", rmin, rmax, 0. , 2*PI, PI/3, 2*PI/3);
	assert(!b21.IsConvex());

	vecgeom::SimpleSphere b22("Solide VecGeomSphere #22", rmin, rmax, 0. , 2*PI, PI/3, PI/3);
	assert(!b22.IsConvex());

	vecgeom::SimpleSphere b23("Solide VecGeomSphere #23", 3, rmax, 0. , 2*PI, 0, PI);
	assert(!b23.IsConvex());

	vecgeom::SimpleSphere b24("Solide VecGeomSphere #24", 3, rmax, 0. , 2*PI/3, PI/3, PI/3);
	assert(!b24.IsConvex());


	return true;
}

bool test_ConvexityParaboloid() {

	vecgeom::SimpleParaboloid b1("VecGeomParaboloid", 5., 8., 10.);
	assert(b1.IsConvex());
	vecgeom::SimpleParaboloid b2("VecGeomParaboloid", 0., 8., 10.);
	assert(b1.IsConvex());
	return true;
}

bool test_ConvexityCone() {

	double rmin1=0., rmax1=5., rmin2=0., rmax2=7., dz=10., sphi=0., dphi=2*PI;
	vecgeom::SimpleCone b1("VecGeomCone1",rmin1,rmax1,rmin2,rmax2,dz,sphi,dphi);
	assert(b1.IsConvex());
	vecgeom::SimpleCone b2("VecGeomCone2",2.,rmax1,rmin2,rmax2,dz,sphi,dphi);
	assert(!b2.IsConvex());
	vecgeom::SimpleCone b3("VecGeomCone3",rmin1,rmax1,4.,rmax2,dz,sphi,dphi);
	assert(!b3.IsConvex());
	vecgeom::SimpleCone b4("VecGeomCone4",2.,rmax1,4.,rmax2,dz,sphi,dphi);
	assert(!b4.IsConvex());
	vecgeom::SimpleCone b5("VecGeomCone5",rmin1,rmax1,rmin2,rmax2,dz,sphi,PI);
	assert(b5.IsConvex());
	vecgeom::SimpleCone b6("VecGeomCone6",rmin1,rmax1,rmin2,rmax2,dz,sphi,PI/3);
	assert(b6.IsConvex());
	vecgeom::SimpleCone b7("VecGeomCone7",rmin1,rmax1,rmin2,rmax2,dz,sphi,4*PI/3);
	assert(!b7.IsConvex());

	vecgeom::SimpleCone b8("VecGeomCone8",rmin1,rmax1,rmin2,rmax2,dz,sphi,4*PI);
	//std::cout<<"DPhi : "<<b8.GetDPhi()<<std::endl;

	//should work after correction in the vecgeom cone so that
	//DPhi=2Pi in case where Dphi>2Pi
	//assert(b8.IsConvex());


	return true;
}

bool test_ConvexityTorus() {

	double rmin=0., rmax=5., rtor=0., sphi=0., dphi=2*PI;
	vecgeom::SimpleTorus2 b1("VecGeomTorus1",rmin,rmax,rtor,sphi,dphi);//Torus becomes Orb
	assert(b1.IsConvex());
	vecgeom::SimpleTorus2 b2("VecGeomTorus2",3,rmax,rtor,sphi,dphi);//Torus becomes SphericalShell
	assert(!b2.IsConvex());
	vecgeom::SimpleTorus2 b3("VecGeomTorus3",3,rmax,15,sphi,dphi);//Real Complete Torus
	assert(!b3.IsConvex());
	vecgeom::SimpleTorus2 b4("VecGeomTorus4",rmin,rmax,rtor,sphi,PI);
	assert(b4.IsConvex());
	vecgeom::SimpleTorus2 b5("VecGeomTorus5",rmin,rmax,rtor,sphi,PI/3);
	assert(b5.IsConvex());
	vecgeom::SimpleTorus2 b6("VecGeomTorus6",rmin,rmax,rtor,sphi,4*PI/3);
	assert(!b6.IsConvex());
	vecgeom::SimpleTorus2 b7("VecGeomTorus7",rmin,rmax,rtor,sphi,3*PI);

	//if dPhi>2PI then dPhi=? //below assertion depends on this
	//may be we have to set to 2Phi as it is for Sphere and should be for Cone
	//assert(b7.IsConvex());


	return true;
}

bool test_ConvexityTube() {

	double rmin=0., rmax=5., dz=10., sphi=0., dphi=2*PI;
	vecgeom::SimpleTube b1("VecgeomTube1",rmin,rmax,dz,sphi,dphi); //Solid Cylinder
	assert(b1.IsConvex());
	vecgeom::SimpleTube b2("VecgeomTube2",3,rmax,dz,sphi,dphi); //Hollow Cylinder
	assert(!b2.IsConvex());
	vecgeom::SimpleTube b3("VecgeomTube3",rmin,rmax,dz,sphi,PI);
	assert(b3.IsConvex());
	vecgeom::SimpleTube b4("VecgeomTube4",rmin,rmax,dz,sphi,2*PI/3);
	assert(b4.IsConvex());
	vecgeom::SimpleTube b5("VecgeomTube5",rmin,rmax,dz,sphi,4*PI/3);
	assert(!b5.IsConvex());
	vecgeom::SimpleTube b6("VecgeomTube6",rmin,rmax,dz,sphi,3*PI);

	//if dPhi>2PI then dPhi=? //below assertion depends on this
	//may be we have to set to 2Phi as it is for Sphere and should be for Cone and Torus
	//assert(b6.IsConvex());

	return true;
}


bool test_ConvexityParallelepiped() {
	double dx=20., dy=30., dz=40., alpha=30., theta=15., phi=30. ;
	vecgeom::SimpleParallelepiped b1("VecGeomParallelepiped1", dx, dy, dz, alpha, theta, phi);
	assert(b1.IsConvex());

	return true;
}

bool test_ConvexityTrd() {
	double xlower=20., xupper=10., ylower=15., yupper=15, dz=40.;
	vecgeom::SimpleTrd b1("VecGeomParallelepiped1", xlower, xupper, ylower, yupper, dz);
	assert(b1.IsConvex());

	return true;
}




//_________________________________________________________________________________________


//_________________________________________________________________________________________
//Temporary tests used for debugging, may be removed later on
//_________________________________________________________________________________________
template <class Sphere_t>
bool test_Sphere(){

	double rmin=0., rmax=5., sphi=0., dphi=2*PI, stheta=0., dtheta=PI;
	Sphere_t b9("Solide VecGeomSphere #9", rmin, rmax, PI/3. , 4*PI, stheta, dtheta);
	std::cout<<"New Calculate value of DPHI - B9  : "<<b9.GetDPhi()<<std::endl;
	Sphere_t b10("Solide VecGeomSphere #10", rmin, rmax, PI/3. , 5*PI, stheta, dtheta);
	std::cout<<"New Calculate value of DPHI - B10 : "<<b10.GetDPhi()<<std::endl;
	Sphere_t b11("Solide VecGeomSphere #11", rmin, rmax, 0. , 3*PI, stheta, dtheta);
	std::cout<<"New Calculate value of DPHI - B11 : "<<b11.GetDPhi()<<std::endl;
	Sphere_t b12("Solide VecGeomSphere #12", rmin, rmax, 0. , 4*PI, stheta, dtheta);
	std::cout<<"New Calculate value of DPHI - B12 : "<<b12.GetDPhi()<<std::endl;

}

template <class Torus_t>
bool test_Torus(){
	double rmin=0., rmax=5., rtor=0., sphi=0., dphi=2*PI;
	Torus_t b1("VecGeomTorus1",rmin,rmax,rtor,sphi,1.5*PI);
	std::cout<<b1.dphi()<<std::endl;
}

template <class Tube_t>
bool test_Tube(){
	double rmin=0., rmax=5., dz=10., sphi=0., dphi=2*PI;
	Tube_t b1("VecGeomTube1",rmin,rmax,dz,sphi,3*PI);
	std::cout<<b1.dphi()<<std::endl;
}

template <class Cone_t>
bool test_Cone(){
	double rmin1=0., rmax1=5., rmin2=0., rmax2=7., dz=10., sphi=0., dphi=2*PI;
	Cone_t b8("VecGeomCone8",rmin1,rmax1,rmin2,rmax2,dz,sphi,1.5*PI);
	std::cout<<b8.GetDPhi()<<std::endl;
}


void test_phiCheck(){

	//std::cout<<"--- USphere ---"<<std::endl;
	//test_Sphere<USphere>();
	//std::cout<<"--- VecGeom Sphere ---"<<std::endl;
	//test_Sphere<vecgeom::SimpleSphere>();
	std::cout<<"UCons Cone : ";
	test_Cone<UCons>();
	std::cout<<"VecGeom Cone ";
	test_Cone<vecgeom::SimpleCone>();
	std::cout<<"VecGeom Torus : ";
	test_Torus<vecgeom::SimpleTorus2>();
	std::cout<<"VecGeom Tube : ";
	test_Tube<vecgeom::SimpleTube>();

}
//_________________________________________________________________________________________

int main(){

	assert(test_ConvexityOrb());
	assert(test_ConvexitySphere());
	assert(test_ConvexityParaboloid());
	assert(test_ConvexityCone());
	assert(test_ConvexityTorus());
	assert(test_ConvexityTube());
	assert(test_ConvexityParallelepiped());
	assert(test_ConvexityTrd());
	//assert(test_Convexity_Y());
	std::cout<<"------------------------------"<<std::endl;
	std::cout<<"--- Convexity Tests Passed ---"<<std::endl;
	std::cout<<"------------------------------"<<std::endl;
	test_phiCheck();

	return 0;
}
