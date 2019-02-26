/// @file TestCoaxialCones.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch)

// ensure asserts are compiled in
#undef NDEBUG

#include <iomanip>
#include "base/Global.h"
#include "base/Vector3D.h"
#include "volumes/EllipticUtilities.h"
#include "volumes/CoaxialCones.h"
#include "ApproxEqual.h"

bool testvecgeom = false;

using vecgeom::kInfLength;
using vecgeom::kTolerance;

template <class CoaxialCones_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision>>
bool TestCoaxialCones()
{
  /*
   * Add the require unit test
   *
   */

	Vec_t pt(0., 0., 0.);
	//vecgeom::Vector<vecgeom::ConeParam> coneParamVector;
	int numOfCones = 2;
	double rmin1[]={0.,15.};
	double rmax1[]={5.,20.};
	double rmin2[]={0.,15.};
	double rmax2[]={8.,20.};
	double dz = 5.;

	CoaxialCones_t Simple("CoaxialCones", numOfCones, rmin1, rmax1, rmin2,
			rmax2, dz, 0., 2 * vecgeom::kPi);
	std::cout << "Location : " << Simple.Inside(pt) << " : "
			<< vecgeom::EInside::kInside << std::endl;

	//Inside
	assert(Simple.Inside(pt) == vecgeom::EInside::kInside);
	Vec_t surfPt1(7., 0., 5.);
	assert(Simple.Inside(surfPt1) == vecgeom::EInside::kSurface);
	Vec_t outPt1(7., 0., -5.);
	assert(Simple.Inside(outPt1) == vecgeom::EInside::kOutside);
	Vec_t surfPt2(15., 0., 0.);
	std::cout << "Location of SurfPt2 (15.,0.,0.) : " << Simple.Inside(surfPt2)
			<< std::endl;
	assert(Simple.Inside(surfPt2) == vecgeom::EInside::kSurface);
	Vec_t inPt2(16., 0., 0.);
	std::cout << "Location of inPt2 (16.,0.,0.) : " << Simple.Inside(inPt2)
			<< std::endl;
	assert(Simple.Inside(inPt2) == vecgeom::EInside::kInside);
	Vec_t outPt2(21., 0., 0.);
	assert(Simple.Inside(outPt2) == vecgeom::EInside::kOutside);
	Vec_t surfPt3(20., 0., 0.);
	assert(Simple.Inside(surfPt3) == vecgeom::EInside::kSurface);

	  // DistanceToIn
	  assert(Simple.DistanceToIn(Vec_t(7., 0., 10.), Vec_t(0., 0., -1.)) == 5);
	  assert(Simple.DistanceToIn(Vec_t(16., 0., 10.), Vec_t(0., 0., -1.)) == 5);
	  assert(Simple.DistanceToIn(Vec_t(25., 0., 0.), Vec_t(-1., 0., 0.)) == 5);
	  std::cout << "Distance of Trajectory that do not intersection any cone : "
	            << Simple.DistanceToIn(Vec_t(21., 0., 10.), Vec_t(0., 0., -1.)) << std::endl;
	  assert(Simple.DistanceToIn(Vec_t(-20., 0., 0.), Vec_t(1., 0., 0.)) == 0.);
	  assert(Simple.DistanceToIn(Vec_t(20., 0., 0.), Vec_t(-1., 0., 0.)) == 0.);
	  std::cout << "Distance : " << Simple.DistanceToIn(Vec_t(-10., 0., 0.), Vec_t(1., 0., 0.)) << std::endl;

	  std::cout << "DistanceToIn of Inside point , (Should be negative) : "
	            << Simple.DistanceToIn(inPt2, Vec_t(-1., 0., 0.)) << std::endl;
	  assert(Simple.DistanceToIn(inPt2, Vec_t(-1., 0., 0.)) < 0.);

	  // DistanceToOut
	  // std::cout << "DistanceToOut of Inside point (Should be non zero positive number) : " <<
	  // Simple.DistanceToOut(Vec_t(0.,0.,0.), Vec_t(0., 0., -1.)) << std::endl;
	  assert(Simple.DistanceToOut(Vec_t(0., 0., 0.), Vec_t(0., 0., -1.)) == 5.);
	  assert(Simple.DistanceToOut(Vec_t(16., 0., 0.), Vec_t(0., 0., -1.)) == 5.);
	  std::cout << "DistanceToOut of Outside point , (Should be negative) : "
	            << Simple.DistanceToOut(outPt2, Vec_t(-1., 0., 0.)) << std::endl;
	  assert(Simple.DistanceToOut(Vec_t(10., 0., 0.), Vec_t(1., 0., 0.)) < 0.);
	  assert(Simple.DistanceToOut(Vec_t(20., 0., 0.), Vec_t(1., 0., 0.)) == 0.);
	  assert(Simple.DistanceToOut(Vec_t(15., 0., 0.), Vec_t(-1., 0., 0.)) == 0.);
	  assert(Simple.DistanceToOut(surfPt1, Vec_t(0., 0., 1.)) == 0.);
	  assert(Simple.DistanceToOut(Vec_t(16., 0., 5.), Vec_t(0., 0., 1.)) == 0.);
	  assert(Simple.DistanceToOut(Vec_t(16., 0., 5.), Vec_t(0., 0., -1.)) == 10.);
	  // Edge point (15.,0.,5.)
	  assert(Simple.DistanceToOut(Vec_t(15., 0., 5.), Vec_t(0., 0., -1.)) == 10.);

	  // SafetyToOut
	  std::cout << "============= Safety Unit Tests =============" << std::endl;
	  std::cout << "Location of (0.,0.,0.) : " << Simple.Inside(Vec_t(0., 0., 0.)) << std::endl;
	  double safetyDist = Simple.SafetyToOut(Vec_t(0., 0., 0.));
	  std::cout << "SafetyToOut Dist : " << safetyDist << std::endl;
	  safetyDist = Simple.SafetyToIn(Vec_t(0., 0., 0.));
	  std::cout << "SafetyToIn Dist : " << safetyDist << std::endl;

	  // Normal
	  std::cout << "============ Normal Unit Tests =============" << std::endl;
	  bool valid = false;
	  Vec_t normal(0., 0., 0.);
	  valid = Simple.Normal(Vec_t(1., 0., 5.), normal);
	  assert(valid && ApproxEqual(normal, Vec_t(0., 0., 1.)));
	  valid = Simple.Normal(Vec_t(15., 0., 0.), normal);
	  assert(valid && ApproxEqual(normal, Vec_t(-1., 0., 0.)));
	  valid = Simple.Normal(Vec_t(20., 0., 0.), normal);
	  assert(valid && ApproxEqual(normal, Vec_t(1., 0., 0.)));

	  // Generating point on conical surface and checking its location and normal validity
	  Vec_t temp1(5., 0., -5), temp2(8., 0., 5);
	  Vec_t tempDir       = (temp2 - temp1).Unit();
	  double dist         = (temp2 - temp1).Mag();
	  Vec_t tempSurfPoint = temp1 + dist * 0.3 * tempDir;
	  assert(Simple.Normal(tempSurfPoint, normal) && (Simple.Inside(tempSurfPoint) == vecgeom::EInside::kSurface));



  return true;
}

int main(int argc, char *argv[])
{
  assert(TestCoaxialCones<vecgeom::SimpleCoaxialCones>());
  std::cout << "VecGeomCoaxialCones passed\n";

  return 0;
}
