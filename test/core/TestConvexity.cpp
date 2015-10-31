/*Unit test to test the Convexity function for various shapes
 *
 */

#include "base/Global.h"
#include "base/Vector3D.h"
#include "volumes/Box.h"
#include "volumes/Sphere.h"
//#include "ApproxEqual.h"

#include "volumes/Orb.h"
#include "volumes/Sphere.h"
#include "volumes/Paraboloid.h"
#include "volumes/Cone.h"
#include "volumes/Torus2.h"
#include "volumes/Tube.h"
#include "volumes/Parallelepiped.h"
#include "volumes/Trd.h"
#include "volumes/Polycone.h"
#include "volumes/Trapezoid.h"
#include "volumes/Polyhedron.h"
#include "volumes/LogicalVolume.h"
#include "volumes/ScaledShape.h"
#include "volumes/utilities/VolumeUtilities.h"

#include <cmath>
#include <iomanip>

#undef NDEBUG
#include <cassert>

#define PI 3.14159265358979323846
#define deg PI/180.

using namespace VECGEOM_NAMESPACE;

bool test_ConvexityOrb() {

	vecgeom::SimpleOrb b1("Solid VecGeomOrb #1",5.);
	assert(b1.GetUnplacedVolume()->IsConvex());
	return true;
}

bool test_ConvexitySphere() {

	double rmin=0., rmax=5., sphi=0., dphi=2*PI, stheta=0., dtheta=PI;
	vecgeom::SimpleSphere b1("Solide VecGeomSphere #1", rmin, rmax, sphi, dphi, stheta, dtheta);
	assert(b1.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleSphere b2("Solide VecGeomSphere #2", rmin, rmax, 0. , PI, stheta, dtheta);
	assert(b2.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleSphere b3("Solide VecGeomSphere #3", rmin, rmax, 0. , PI/3, stheta, dtheta);
	assert(b3.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleSphere b4("Solide VecGeomSphere #4", rmin, rmax, 0. , 4*PI/3, stheta, dtheta);
	assert(!b4.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleSphere b5("Solide VecGeomSphere #5", rmin, rmax, PI/3. , PI/3, stheta, dtheta);
	assert(b5.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleSphere b6("Solide VecGeomSphere #6", rmin, rmax, PI/3. , 2*PI/3, stheta, dtheta);
	assert(b6.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleSphere b7("Solide VecGeomSphere #7", rmin, rmax, PI/3. , PI, stheta, dtheta);
	assert(b7.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleSphere b8("Solide VecGeomSphere #8", rmin, rmax, PI/3. , 7*PI/6, stheta, dtheta);
	assert(!b8.GetUnplacedVolume()->IsConvex());

	//checking proper dphi calculation if specified dphi>2PI
	//Should be accepted by Wedge
	//Convention used for dPhi is if(dPhi>2PI) dPhi=2PI //needs a relook
	vecgeom::SimpleSphere b9("Solide VecGeomSphere #9", rmin, rmax, PI/3. , 4*PI, stheta, dtheta);
	assert(b9.GetUnplacedVolume()->IsConvex());
	//std::cerr<<"Newly Calcuated DPHi of b9 : "<<b9.GetDPhi()<<std::endl;

	vecgeom::SimpleSphere b10("Solide VecGeomSphere #10", rmin, rmax, PI/3. , 5*PI, stheta, dtheta);
	assert(b10.GetUnplacedVolume()->IsConvex());
	//std::cerr<<"Newly Calcuated DPHi of b10 : "<<b10.GetDPhi()<<std::endl;

	//vecgeom::SimpleSphere b2("Solide VecGeomSphere #1", 3, rmax, sphi, dphi, stheta, dtheta);

	//This case should be discussed
	vecgeom::SimpleSphere b11("Solide VecGeomSphere #11", rmin, rmax, PI/3. , ((2*PI) + (7*PI/6)), stheta, dtheta);
	assert(b11.GetUnplacedVolume()->IsConvex());
	//std::cerr<<"Newly Calcuated DPHi of b11 : "<<b10.GetDPhi()<<std::endl;

	vecgeom::SimpleSphere b12("Solide VecGeomSphere #12", rmin, rmax, 0. , 2*PI, stheta, PI/2);
	assert(b12.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleSphere b13("Solide VecGeomSphere #13", rmin, rmax, 0. , 2*PI, stheta, 2*PI/3);
	assert(!b13.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleSphere b14("Solide VecGeomSphere #14", rmin, rmax, 0. , 2*PI, stheta, PI/3);
	assert(b14.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleSphere b15("Solide VecGeomSphere #15", rmin, rmax, 0. , 2*PI, PI/6, PI/6);
	assert(!b15.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleSphere b16("Solide VecGeomSphere #16", rmin, rmax, 0. , 2*PI, PI/6, PI/3);
	assert(!b16.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleSphere b17("Solide VecGeomSphere #17", rmin, rmax, 0. , 2*PI, PI/2, PI/2);
	assert(b17.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleSphere b18("Solide VecGeomSphere #18", rmin, rmax, 0. , 2*PI, PI/2, PI/6);
	assert(!b18.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleSphere b19("Solide VecGeomSphere #19", rmin, rmax, 0. , 2*PI, 2*PI/3, PI/3);
	assert(b19.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleSphere b20("Solide VecGeomSphere #20", rmin, rmax, 0. , 2*PI, 2*PI/3, PI/6);
	assert(!b20.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleSphere b21("Solide VecGeomSphere #21", rmin, rmax, 0. , 2*PI, PI/3, 2*PI/3);
	assert(!b21.GetUnplacedVolume()->IsConvex());

	vecgeom::SimpleSphere b22("Solide VecGeomSphere #22", rmin, rmax, 0. , 2*PI, PI/3, PI/3);
	assert(!b22.GetUnplacedVolume()->IsConvex());

	vecgeom::SimpleSphere b23("Solide VecGeomSphere #23", 3, rmax, 0. , 2*PI, 0, PI);
	assert(!b23.GetUnplacedVolume()->IsConvex());

	vecgeom::SimpleSphere b24("Solide VecGeomSphere #24", 3, rmax, 0. , 2*PI/3, PI/3, PI/3);
	assert(!b24.GetUnplacedVolume()->IsConvex());


	return true;
}

bool test_ConvexityParaboloid() {

	vecgeom::SimpleParaboloid b1("VecGeomParaboloid", 5., 8., 10.);
	assert(b1.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleParaboloid b2("VecGeomParaboloid", 0., 8., 10.);
	assert(b1.GetUnplacedVolume()->IsConvex());
	return true;
}

bool test_ConvexityCone() {

	double rmin1=0., rmax1=5., rmin2=0., rmax2=7., dz=10., sphi=0., dphi=2*PI;
	vecgeom::SimpleCone b1("VecGeomCone1",rmin1,rmax1,rmin2,rmax2,dz,sphi,dphi);
	assert(b1.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleCone b2("VecGeomCone2",2.,rmax1,rmin2,rmax2,dz,sphi,dphi);
	assert(!b2.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleCone b3("VecGeomCone3",rmin1,rmax1,4.,rmax2,dz,sphi,dphi);
	assert(!b3.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleCone b4("VecGeomCone4",2.,rmax1,4.,rmax2,dz,sphi,dphi);
	assert(!b4.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleCone b5("VecGeomCone5",rmin1,rmax1,rmin2,rmax2,dz,sphi,PI);
	assert(b5.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleCone b6("VecGeomCone6",rmin1,rmax1,rmin2,rmax2,dz,sphi,PI/3);
	assert(b6.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleCone b7("VecGeomCone7",rmin1,rmax1,rmin2,rmax2,dz,sphi,4*PI/3);
	assert(!b7.GetUnplacedVolume()->IsConvex());

	vecgeom::SimpleCone b8("VecGeomCone8",rmin1,rmax1,rmin2,rmax2,dz,sphi,4*PI);
	//std::cout<<"DPhi : "<<b8.GetDPhi()<<std::endl;

	//should work after correction in the vecgeom cone so that
	//DPhi=2Pi in case where Dphi>2Pi
	//assert(b8.GetUnplacedVolume()->IsConvex());


	return true;
}

bool test_ConvexityTorus() {

	double rmin=0., rmax=5., rtor=0., sphi=0., dphi=2*PI;
	vecgeom::SimpleTorus2 b1("VecGeomTorus1",rmin,rmax,rtor,sphi,dphi);//Torus becomes Orb
	assert(b1.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleTorus2 b2("VecGeomTorus2",3,rmax,rtor,sphi,dphi);//Torus becomes SphericalShell
	assert(!b2.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleTorus2 b3("VecGeomTorus3",3,rmax,15,sphi,dphi);//Real Complete Torus
	assert(!b3.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleTorus2 b4("VecGeomTorus4",rmin,rmax,rtor,sphi,PI);
	assert(b4.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleTorus2 b5("VecGeomTorus5",rmin,rmax,rtor,sphi,PI/3);
	assert(b5.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleTorus2 b6("VecGeomTorus6",rmin,rmax,rtor,sphi,4*PI/3);
	assert(!b6.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleTorus2 b7("VecGeomTorus7",rmin,rmax,rtor,sphi,3*PI);

	//if dPhi>2PI then dPhi=? //below assertion depends on this
	//may be we have to set to 2Phi as it is for Sphere and should be for Cone
	//assert(b7.GetUnplacedVolume()->IsConvex());


	return true;
}

bool test_ConvexityTube() {

	double rmin=0., rmax=5., dz=10., sphi=0., dphi=2*PI;
	vecgeom::SimpleTube b1("VecgeomTube1",rmin,rmax,dz,sphi,dphi); //Solid Cylinder
	assert(b1.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleTube b2("VecgeomTube2",3,rmax,dz,sphi,dphi); //Hollow Cylinder
	assert(!b2.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleTube b3("VecgeomTube3",rmin,rmax,dz,sphi,PI);
	assert(b3.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleTube b4("VecgeomTube4",rmin,rmax,dz,sphi,2*PI/3);
	assert(b4.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleTube b5("VecgeomTube5",rmin,rmax,dz,sphi,4*PI/3);
	assert(!b5.GetUnplacedVolume()->IsConvex());
	vecgeom::SimpleTube b6("VecgeomTube6",rmin,rmax,dz,sphi,3*PI);

	//if dPhi>2PI then dPhi=? //below assertion depends on this
	//may be we have to set to 2Phi as it is for Sphere and should be for Cone and Torus
	//assert(b6.GetUnplacedVolume()->IsConvex());

	return true;
}


bool test_ConvexityParallelepiped() {
	double dx=20., dy=30., dz=40., alpha=30., theta=15., phi=30. ;
	vecgeom::SimpleParallelepiped b1("VecGeomParallelepiped1", dx, dy, dz, alpha, theta, phi);
	assert(b1.GetUnplacedVolume()->IsConvex());

	return true;
}

bool test_ConvexityTrd() {
	double xlower=20., xupper=10., ylower=15., yupper=15, dz=40.;
	vecgeom::SimpleTrd b1("VecGeomParallelepiped1", xlower, xupper, ylower, yupper, dz);
	assert(b1.GetUnplacedVolume()->IsConvex());

	return true;
}

bool test_ConvexityPolycone() {

	double phiStart=0., deltaPhi=kTwoPi/3;
	int nZ=10;
	//double rmin[10]={0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
	double rmin[10]={6.,0.,0.,0.,15.,0.,0.,0.,3.,15.};
	double rmax[10]={10.,10.,10.,20.,20.,10.,10.,5.,5.,20.};
	double z[10]={-20.,0.,0.,20.,20.,40.,45.,50.,50.,60.};

	//double rmin[4]={0.,0.,0.,0.};
	//double rmax[4]={15.,15.,15.,10.};
	//double z[4]={0.,20.,30.,40.};

	vecgeom::SimplePolycone b1("VecGeomPolycone", phiStart, deltaPhi, nZ, z, rmin, rmax);
	assert(!b1.GetUnplacedVolume()->IsConvex());

	return true;
}

bool test_ConvexityTrapezoid() {

	vecgeom::SimpleTrapezoid b1("trap3",50,0,0,50,50,50,PI/4,50,50,50,PI/4) ;
	assert(b1.GetUnplacedVolume()->IsConvex());

	return true;
}

bool test_ConvexityPolyhedron() {

	double phiStart=0., deltaPhi=170;
	int sides=4;//, nZ=10;
	constexpr int nPlanes = 4;
	double zPlanes[nPlanes] = {-2, -1, 1, 2};
	double rInner[nPlanes] = {0, 0, 0, 0};
	double rOuter[nPlanes] = {2, 2, 2, 2};

	vecgeom::SimplePolyhedron b1("Vecgeom Polyhedron", phiStart, deltaPhi, sides, nPlanes, zPlanes, rInner, rOuter);
	assert(b1.GetUnplacedVolume()->IsConvex());

	vecgeom::SimplePolyhedron b2("Vecgeom Polyhedron", phiStart, 60, sides, nPlanes, zPlanes, rInner, rOuter);
	assert(b2.GetUnplacedVolume()->IsConvex());

	vecgeom::SimplePolyhedron b3("Vecgeom Polyhedron", phiStart, 200, sides, nPlanes, zPlanes, rInner, rOuter);
	assert(!b3.GetUnplacedVolume()->IsConvex());

	rOuter[1]=1.;
	rOuter[2]=1.;
	vecgeom::SimplePolyhedron b5("Vecgeom Polyhedron", phiStart, 120, sides, nPlanes, zPlanes, rInner, rOuter);
	assert(!b5.GetUnplacedVolume()->IsConvex());

	rOuter[0]=1.;
	rOuter[1]=2.;
	rOuter[2]=2.;
	rOuter[3]=1.;
	vecgeom::SimplePolyhedron b6("Vecgeom Polyhedron", phiStart, 120, sides, nPlanes, zPlanes, rInner, rOuter);
	assert(!b6.GetUnplacedVolume()->IsConvex());

	rInner[1]=1.;
	rInner[2]=0.5;
	vecgeom::SimplePolyhedron b4("Vecgeom Polyhedron", phiStart, 60, sides, nPlanes, zPlanes, rInner, rOuter);
	assert(!b4.GetUnplacedVolume()->IsConvex());

	return true;
}

//_________________________________________________________________________________________
//Convexity test for scaled Shapes

bool test_ConvexityScaledOrb() {
  vecgeom::SimpleOrb orb("Visualizer Orb", 3);
  vecgeom::SimpleScaledShape scaledOrb("Scaled Orb", orb.GetUnplacedVolume(), 0.5, 1.2, 1.);
  assert(scaledOrb.GetUnplacedVolume()->IsConvex());

  return true;
}


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

	return true;
}

template <class Torus_t>
bool test_Torus(){
	double rmin=0., rmax=5., rtor=0., sphi=0., dphi=1.5*PI;
	Torus_t b1("VecGeomTorus1",rmin,rmax,rtor,sphi,dphi);
	std::cout<<b1.dphi()<<std::endl;
	return true;
}

template <class Tube_t>
bool test_Tube(){
	double rmin=0., rmax=5., dz=10., sphi=0., dphi=3*PI;
	Tube_t b1("VecGeomTube1",rmin,rmax,dz,sphi,dphi);
	std::cout<<b1.dphi()<<std::endl;
	return true;
}

template <class Cone_t>
bool test_Cone(){
	double rmin1=0., rmax1=5., rmin2=0., rmax2=7., dz=10., sphi=0., dphi=1.5*PI;
	Cone_t b8("VecGeomCone8",rmin1,rmax1,rmin2,rmax2,dz,sphi,dphi);
	std::cout<<b8.GetDPhi()<<std::endl;

	return true;
}


void test_phiCheck(){
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
	assert(test_ConvexityPolycone());
	assert(test_ConvexityTrapezoid());
	assert(test_ConvexityPolyhedron());

	//Test for ScaledShapes
	test_ConvexityScaledOrb();

	std::cout<<"------------------------------"<<std::endl;
	std::cout<<"--- Convexity Tests Passed ---"<<std::endl;
	std::cout<<"------------------------------"<<std::endl;
	test_phiCheck();

	return 0;
}
