/// @file TestGenericPolycone.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch)

// ensure asserts are compiled in
#undef NDEBUG

#include <iomanip>
#include "base/Global.h"
#include "base/Vector3D.h"
//#include "volumes/EllipticUtilities.h"
#include "volumes/CoaxialCones.h"
#include "volumes/GenericPolycone.h"
#include "ApproxEqual.h"
#include "base/Vector.h"
#include "volumes/GenericPolyconeStruct.h"

bool testvecgeom = false;

using vecgeom::kInfLength;
using vecgeom::kTolerance;

template <class GenericPolycone_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision>>
bool TestGenericPolycone()
{
  /*
   * Add the required unit test
   *
   */

  /* Creating a test Simple GenericPolycone by rotating the following contour
   *       ------------------------
   *       |                      |
   *       | /------------------\ |
   *       |/                    \|
   */

#if (0)
  /* Keeping this block of commented code for reference, that demonstrates
   * the use of other Constructor provided in UnplacedGenericPolycone.
   */
  vecgeom::Vector<vecgeom::Vector<vecgeom::ConeParam>> genPolyConeParamVector;
  vecgeom::Vector<vecgeom::ConeParam> coneParamVector1;
  coneParamVector1.push_back(vecgeom::ConeParam(1., 1., 1., 2., 0.5, 0., 2 * vecgeom::kPi));
  coneParamVector1.push_back(vecgeom::ConeParam(5., 5., 4., 5., 0.5, 0., 2 * vecgeom::kPi));
  genPolyConeParamVector.push_back(coneParamVector1);

  vecgeom::Vector<vecgeom::ConeParam> coneParamVector2;
  coneParamVector2.push_back(vecgeom::ConeParam(1., 5., 1., 5., 0.5, 0., 2 * vecgeom::kPi));
  genPolyConeParamVector.push_back(coneParamVector2);

  vecgeom::Vector<vecgeom::Precision> zS;
  zS.push_back(1.);
  zS.push_back(2.);
  zS.push_back(3.);

  // All the above manually specified stuff should come from
  // constructor with RZ corners
  GenericPolycone_t Simple("GenericPolycone", genPolyConeParamVector, zS);
#endif

#if (1)
  double startPhi = 0., deltaPhi = 2. * M_PI; /// 3.;
  constexpr int numRz = 6;
  double r[numRz]     = {1., 2., 4., 5., 5., 1.};
  double z[numRz]     = {1., 2., 2., 1., 3., 3.};
  GenericPolycone_t Simple("GenericPolycone", startPhi, deltaPhi, numRz, r, z);
  Vec_t aMin, aMax;
  Simple.GetUnplacedVolume()->Extent(aMin, aMax);
  std::cout << "Extent : " << aMin << " : " << aMax << std::endl;

  std::cout << "Capacity : " << Simple.GetUnplacedVolume()->Capacity() << std::endl;
  std::cout << "SurfaceArea : " << Simple.GetUnplacedVolume()->SurfaceArea() << std::endl;

  assert(Simple.Inside(Vec_t(2., 0., 2.5)) == vecgeom::EInside::kInside);
  std::cout << "Location of Inside point (2.,0.,2.5) using Contains : " << Simple.Contains(Vec_t(2., 0., 2.5))
            << std::endl;
  assert(Simple.Contains(Vec_t(2., 0., 2.5)));

  assert(Simple.Inside(Vec_t(1.5, 0., 1.9999)) == vecgeom::EInside::kInside);
  assert(Simple.Inside(Vec_t(4.5, 0., 1.9999)) == vecgeom::EInside::kInside);
  assert(Simple.Inside(Vec_t(2.1, 0., 1.9999)) == vecgeom::EInside::kOutside);
  assert(Simple.Inside(Vec_t(5.1, 0., 1.9999)) == vecgeom::EInside::kOutside);

  std::cout << "Location using Contains : " << Simple.Contains(Vec_t(5.1, 0., 1.9999)) << std::endl;
  assert(!Simple.Contains(Vec_t(5.1, 0., 1.9999)));
  assert(Simple.Inside(Vec_t(2., 0., 3.)) == vecgeom::EInside::kSurface);
  assert(Simple.Inside(Vec_t(5., 0., 2.1)) == vecgeom::EInside::kSurface);

  std::cout << "Buggy Surface Point (1.,0.,2.) : Location of point which is on edge of both Cones Sections : "
            << Simple.Inside(Vec_t(1., 0., 2.)) << std::endl;
  // assert(Simple.Inside(Vec_t(1.,0.,2.)) == vecgeom::EInside::kSurface);
  std::cout << "Buggy Surface Point (5.,0.,2.) : Location of point which is on edge of both Cones Sections : "
            << Simple.Inside(Vec_t(5., 0., 2.)) << std::endl;
  // assert(Simple.Inside(Vec_t(5.,0.,2.)) == vecgeom::EInside::kSurface);

  // Corner Point
  assert(Simple.Inside(Vec_t(1., 0., 1.)) == vecgeom::EInside::kSurface);
  assert(Simple.Inside(Vec_t(5., 0., 1.)) == vecgeom::EInside::kSurface);
  assert(Simple.Inside(Vec_t(1., 0., 3.)) == vecgeom::EInside::kSurface);
  assert(Simple.Inside(Vec_t(5., 0., 3.)) == vecgeom::EInside::kSurface);

  // DistanceToIn tests
  assert(Simple.DistanceToIn(Vec_t(3., 0., 1.5), Vec_t(0., 0., 1.)) == 0.5);

  Vec_t outPt1(8., 0., 1.5);
  Vec_t dir(-1., 0., 0.);
  double Dist = Simple.DistanceToIn(outPt1, dir);
  std::cout << std::setprecision(20) << "DistanceToIn of (8.,0.,1.5) : " << Dist << std::endl;
  assert(ApproxEqual(Dist, 3));
  assert(Simple.Inside(outPt1 + dir * Dist) == vecgeom::EInside::kSurface);

  Vec_t outPt2(3., 0., 0.);
  assert(Simple.Inside(outPt2) == vecgeom::EInside::kOutside);
  Vec_t dir2(Vec_t(4., 0., 2) - outPt2);
  // dir /= dir.Mag();//.Normalize();
  dir2.Normalize();
  Dist = Simple.DistanceToIn(outPt2, dir2);
  std::cout << "Distance : " << Dist << std::endl;
  std::cout << "Moved Point : " << (outPt2 + dir2 * Dist) << std::endl;
  std::cout << "Location of Moved point which is actually a corner point and again having a common edge of both Cones "
               "Sections (AN EXPECTED BUG) : "
            << Simple.Inside(outPt2 + dir2 * Dist) << std::endl;
  // assert(Simple.Inside(outPt2+dir2*Dist) == vecgeom::EInside::kSurface);

  // DistanceToIn test for Inside points
  Dist = Simple.DistanceToIn(Vec_t(2., 0., 2.5), dir2);
  std::cout << "DistanceToIn of inside Point (2.,0.,2.5) : (SHOULD BE NEGATIVE) : " << Dist << std::endl;
  assert(Dist < 0.);

  Dist = Simple.DistanceToIn(Vec_t(1.5, 0., 1.99999), dir2);
  std::cout << "DistanceToIn of inside Point (1.5,0.,1.99999) : (SHOULD BE NEGATIVE) : " << Dist << std::endl;
  assert(Dist < 0.);

  Vec_t outPt3(6., 0., 5.);
  Vec_t dir3(Vec_t(5., 0., 3.) - outPt3);
  dir3.Normalize();
  Dist = Simple.DistanceToIn(outPt3, dir3);
  assert(Simple.Inside(outPt3 + dir3 * Dist) == vecgeom::EInside::kSurface);

  // DistanceToOut tests
  assert(Simple.DistanceToOut(Vec_t(3., 0., 2.5), Vec_t(0., 0., 1.)) == 0.5);
  assert(Simple.DistanceToOut(Vec_t(3., 0., 2.5), Vec_t(0., 0., -1.)) == 0.5);
  assert(Simple.DistanceToOut(Vec_t(5., 0., 2.5), Vec_t(1., 0., 0.)) == 0.);
  assert(Simple.DistanceToOut(Vec_t(5., 0., 2.5), Vec_t(-1., 0., 0.)) == 4.);
  assert(Simple.DistanceToOut(Vec_t(4.5, 0., 2.), Vec_t(0., 0., 1.)) == 1.);
  assert(Simple.DistanceToOut(Vec_t(4.5, 0., 1.9), Vec_t(0., 0., 1.)) == 1.1);
  assert(Simple.DistanceToOut(outPt1, Vec_t(0., 0., 1.)) < 0.);
  assert(Simple.DistanceToOut(outPt2, Vec_t(0., 0., 1.)) < 0.);
  assert(Simple.DistanceToOut(outPt3, Vec_t(0., 0., 1.)) < 0.);
#if (0)
  {
    // Test from CoaxialCones which is basically a specialized case of GenericPolycone with only one section
    vecgeom::Vector<vecgeom::Vector<vecgeom::ConeParam>> genPolyConeParamVector;
    Vec_t pt(0., 0., 0.);
    vecgeom::Vector<vecgeom::ConeParam> coneParamVector;
    coneParamVector.push_back(vecgeom::ConeParam(0., 5., 0., 8., 5., 0., 2 * vecgeom::kPi));
    coneParamVector.push_back(vecgeom::ConeParam(15., 20., 15., 20., 5., 0., 2 * vecgeom::kPi));
    genPolyConeParamVector.push_back(coneParamVector);
    vecgeom::Vector<vecgeom::Precision> zS;
    zS.push_back(-5.);
    zS.push_back(5.);

    // CoaxialCones_t Simple("CoaxialCones", coneParamVector);
    GenericPolycone_t Simple("GenericPolycone", genPolyConeParamVector, zS);

    std::cout << "Location : " << Simple.Inside(pt) << " : " << vecgeom::EInside::kInside << std::endl;
    assert(Simple.Inside(pt) == vecgeom::EInside::kInside);

    Vec_t surfPt1(7., 0., 5.);
    assert(Simple.Inside(surfPt1) == vecgeom::EInside::kSurface);
    Vec_t outPt1(7., 0., -5.);
    assert(Simple.Inside(outPt1) == vecgeom::EInside::kOutside);
    Vec_t surfPt2(15., 0., 0.);
    std::cout << "Location of SurfPt2 (15.,0.,0.) : " << Simple.Inside(surfPt2) << std::endl;
    assert(Simple.Inside(surfPt2) == vecgeom::EInside::kSurface);
    Vec_t inPt2(16., 0., 0.);
    std::cout << "Location of inPt2 (16.,0.,0.) : " << Simple.Inside(inPt2) << std::endl;
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
  }
#endif
  {
    std::cout << "===================================================================" << std::endl;
    double sphi         = 0.;
    double dphi         = vecgeom::kTwoPi;
    const int numRZ1    = 10;
    double polycone_r[] = {1, 5, 3, 4, 9, 9, 3, 3, 2, 1};
    double polycone_z[] = {0, 1, 2, 3, 0, 5, 4, 3, 2, 1};
    auto poly2          = new GenericPolycone_t("GenericPoly", sphi, dphi, numRZ1, polycone_r, polycone_z);
    Vec_t ptIn(1.342000296513003121390284, 7.419050450955836595312576, 0.8763312698342673456863849);
    Vec_t dir(0.2092127200968610933884406, 0.7654562532663932161725029, 0.6085283576671245420186551);
    // Vec_t ptIn(7.447734221488879313710640417412, 4.828097455012545502484044845914, 4.814542288666542546593518636655);
    // Vec_t dir(0.570048353320007694655657815019, 0.214335135200638571273401566941, 0.793161600618481732460907096538);
    std::cout << "Location : " << poly2->Inside(ptIn) << std::endl;
    Dist = poly2->DistanceToOut(ptIn, dir);

    std::cout << "Calculated DistToOut : " << Dist << std::endl;
    // Dist = 0.20875007502700274054;
    std::cout << "Geant4     DistToOut : " << Dist << std::endl;
    Vec_t MovedPt = (ptIn + Dist * dir);
    std::cout << "Moved POInt : " << MovedPt << std::endl;
    std::cout << "Radius of Moved Point : " << MovedPt.Perp() << std::endl;

    std::cout << "Location of Moved Point : " << poly2->Inside(MovedPt) << std::endl;

    // assert(poly2->Inside(ptIn) == vecgeom::EInside::kInside);

    Vec_t safetyPt(2.53568647643333200392135, 7.381743158896469481078384, 0.9386758016241848467942077);
    std::cout << "location of SafetyPoint : " << poly2->Inside(safetyPt) << std::endl;
    std::cout << "SafetyToOut : " << poly2->SafetyToOut(safetyPt) << std::endl;

    // Safety Distance mismatch (for Surface points) detected by shapetester
    Vec_t safetyPt2(-8.992832946941797800377572, -0.3591054026977471558268462, 0.5);
    Vec_t dirSafetyPt2(0.0760655128324308066334325, -0.2131854768574358571786576, 0.9740461951132538542807993);
    std::cout << "location of SafetyPoint2 : " << poly2->Inside(safetyPt2) << std::endl;
    std::cout << "SafetyToIn Dist : " << poly2->SafetyToIn(safetyPt2) << std::endl;
    std::cout << "DistanceToIn Dist : " << poly2->DistanceToIn(safetyPt2, dirSafetyPt2) << std::endl;

    Vec_t safetyPt3(-6.28455157823346866052816, -6.442391799981537658936759, 0.1422521521363713237207094);
    Vec_t dirSafetyPt3(-0.04592037192788539501364653, 0.3504705316940292525451639, 0.9354473399695512059182079);
    std::cout << "location of SafetyPoint3 : " << poly2->Inside(safetyPt3) << std::endl;
    std::cout << "DistanceToOut Dist : " << poly2->DistanceToOut(safetyPt3, dirSafetyPt3) << std::endl;

    std::cout << "=========================================================" << std::endl;
    // ptIn.Set(-8.245033420713934191326188738458, 0.825740410462747709274822227599, 4.027595002605417917607155686710);
    // dir.Set(-0.760512717959938133738262422412, -0.494851077560996444049123965669, 0.420407917216015947214913239804);
    ptIn.Set(-42.046543375136415932047384558246, 2.876852248775723985829699813621, 41.956705401227381457829324062914);
    dir.Set(0.712814657374801763367599960475, -0.207692873805265298958744324409, -0.669894718894061602654232956411);

    Dist = poly2->DistanceToOut(ptIn, dir);
    std::cout << "Location of Actual point : " << poly2->Inside(ptIn) << std::endl;
    std::cout << "DistancetoOut : 		" << Dist << std::endl;
    std::cout << "Location of Moved point : " << poly2->Inside(ptIn + Dist * dir) << std::endl;
    std::cout << "Moved Point : 		" << (ptIn + Dist * dir) << std::endl;
    Dist = poly2->DistanceToIn(ptIn, dir);
    std::cout << "DistancetoIn : 		" << Dist << std::endl;
    std::cout << "Moved Point : 		" << (ptIn + Dist * dir) << std::endl;
    std::cout << "Radius of Moved Point : " << (ptIn + Dist * dir).Perp() << std::endl;
    std::cout << "Location of Moved point : " << poly2->Inside(ptIn + Dist * dir) << std::endl;

    std::cout << "===========================================================" << std::endl;

    ptIn.Set(10., 0., 4.895);
    Vec_t tempPt(0., 0., 0.);
    Vec_t dirtemp = tempPt - ptIn;
    dir           = dirtemp.Unit();

    std::cout << "Location of Actual point : " << poly2->Inside(ptIn) << std::endl;
    Dist = poly2->DistanceToIn(ptIn, dir);
    std::cout << "DistancetoIn : 		" << Dist << std::endl;
    std::cout << "Moved Point : 		" << (ptIn + Dist * dir) << std::endl;
    std::cout << "Radius of Moved Point : " << (ptIn + Dist * dir).Perp() << std::endl;
    std::cout << "Location of Moved point : " << poly2->Inside(ptIn + Dist * dir) << std::endl;

    std::cout << "========= Some more mismatches from benchmark =========== " << std::endl;
    Vec_t ptToIn(-29.402024989716856850918702548370, -28.599908958628695643255923641846,
                 -35.130380012379767151742271380499);
    Vec_t dirPtToIn(0.605765083102027368511244276306, 0.371690655275148829073117440203,
                    0.703487541379038683331259562692);
    Dist = poly2->DistanceToIn(ptToIn, dirPtToIn);
    std::cout << "DistancetoIn : 		" << Dist << std::endl;
    std::cout << "Moved Point : 		" << (ptToIn + Dist * dirPtToIn) << std::endl;
    std::cout << "Radius of Moved Point : " << (ptToIn + Dist * dirPtToIn).Perp() << std::endl;
    std::cout << "Location of Moved point : " << poly2->Inside(ptToIn + Dist * dirPtToIn) << std::endl;
  }
#endif

  return true;
}

int main(int argc, char *argv[])
{
  assert(TestGenericPolycone<vecgeom::SimpleGenericPolycone>());
  std::cout << "VecGeomGenericPolycone passed\n";

  return 0;
}
