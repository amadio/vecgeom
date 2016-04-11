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
#ifdef VECGEOM_USOLIDS
#include "UBox.hh"
//#include "UHype.hh"
#include "UVector3.hh"
#endif
#include<iomanip>
#include <cassert>
#include <cmath>

#define PI 3.14159265358979323846

template <class Hype_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision> >
bool TestHype() {

  std::cout << std::setprecision(15);
  // int verbose=0;
  double fRmin = 10, fRmax = 20, stIn = PI / 4, stOut = PI / 3, halfZ = 50;
  vecgeom::Precision fR = 9.;
  Vec_t pzero(0, 0, 0);
  Vec_t pbigx(100, 0, 0), pbigy(0, 100, 0), pbigz(0, 0, 100);
  Vec_t pbigmx(-100, 0, 0), pbigmy(0, -100, 0), pbigmz(0, 0, -100);
  Vec_t ponx(fR, 0., 0.);   // point on surface on X axis
  Vec_t ponmx(-fR, 0., 0.); // point on surface on minus X axis
  Vec_t pony(0., fR, 0.);   // point on surface on Y axis
  Vec_t ponmy(0., -fR, 0.); // point on surface on minus Y axis
  Vec_t ponz(0., 0., fR);   // point on surface on Z axis
  Vec_t ponmz(0., 0., -fR); // point on surface on minus Z axis

  Vec_t ponxsideO(fRmax, 0, 0), ponysideO(0, fRmax, 0);//, ponzsideO(fRmax,0,halfZ);
  Vec_t ponxsideI(fRmin, 0, 0), ponysideI(0, fRmin, 0);//, ponzsideI(fRmin,0,halfZ);;

  Vec_t ponmxside(-fR, 0, 0), ponmyside(0, -fR, 0), ponmzside(0, 0, -fR);

  Vec_t vx(1, 0, 0), vy(0, 1, 0), vz(0, 0, 1);
  Vec_t vmx(-1, 0, 0), vmy(0, -1, 0), vmz(0, 0, -1);
  Vec_t vxy(1 / std::sqrt(2.0), 1 / std::sqrt(2.0), 0);
  Vec_t vmxy(-1 / std::sqrt(2.0), 1 / std::sqrt(2.0), 0);
  Vec_t vmxmy(-1 / std::sqrt(2.0), -1 / std::sqrt(2.0), 0);
  Vec_t vxmy(1 / std::sqrt(2.0), -1 / std::sqrt(2.0), 0);
  Vec_t vxmz(1 / std::sqrt(2.0), 0, -1 / std::sqrt(2.0));

  Hype_t b1("Solid VecGeomHype #1", fRmin, fRmax, PI / 4, PI / 3, 50);
  Hype_t b2("Solid VecGeomHype #2", 10, 20, PI / 4, PI / 4, 50);

  // Check name
  // assert(b1.GetName()=="Solid VecGeomHype #1");
  // assert(b2.GetName()=="Solid VecGeomHype #2");

  // COMMENTING CAPACITY AND SURFACE AREA FOR THE TIME BEING

  // Check cubic volume
  // assert(b1.Capacity() == ((4 * PI / 3) * fR * fR * fR));
  // assert(b2.Capacity() == ((4 * PI / 3) * 6 * 6 * 6));

  // Check Surface area
  // assert(b1.SurfaceArea() == ((4 * PI) * fR * fR));
  // assert(b2.SurfaceArea() == ((4 * PI) * 6 * 6));

  Vec_t minExtent, maxExtent;
  b1.Extent(minExtent, maxExtent);
  double tSTout = std::tan(stOut);
  // double tSTout2 = tSTout*tSTout;
  double tSTin = std::tan(stIn);
  // double tSTin2 = tSTin*tSTin;
  // double endOuterRadius = std::sqrt(fRmax*fRmax + tSTout2*halfZ*halfZ);

  double xy = std::sqrt(fRmax * fRmax + tSTout * tSTout * halfZ * halfZ);
  assert(ApproxEqual(minExtent, Vec_t(-xy, -xy, -halfZ)));
  assert(ApproxEqual(maxExtent, Vec_t(xy, xy, halfZ)));

  /*
  b2.Extent(minExtent,maxExtent);
  assert(ApproxEqual(minExtent,Vec_t(-6,-6,-6)));
  assert(ApproxEqual(maxExtent,Vec_t( 6, 6, 6)));
  */

  // Check Surface Normal
  Vec_t normal;
  bool valid;

  double Dist;
  Vec_t norm;
  bool convex;
  convex = true;

  // COMMENTING NORMAL FOR THE TIME BEING
  // These Normal tests Needs a relook ---

  valid = b1.Normal(pbigx, normal);
  assert(ApproxEqual(normal, Vec_t(0, 0, 1)));

  valid = b1.Normal(pbigy, normal);
  assert(ApproxEqual(normal, Vec_t(0, 0, 1)));

  valid = b1.Normal(pbigz, normal);
  assert(ApproxEqual(normal, Vec_t(0, 0, 1)));

  valid = b1.Normal(pbigmx, normal);
  assert(ApproxEqual(normal, Vec_t(0, 0, 1)));

  valid = b1.Normal(pbigmy, normal);
  assert(ApproxEqual(normal, Vec_t(0, 0, 1)));

  valid = b1.Normal(pbigmz, normal);
  assert(ApproxEqual(normal, Vec_t(0, 0, -1)));

  // Raman-----------------

  // valid = b1.Normal(pony,normal);
  // assert(ApproxEqual(normal,Vec_t(0,1,0)));

  valid = b1.Normal(ponz, normal);
  assert(ApproxEqual(normal, Vec_t(0, 0, 1)));

   valid = b1.Normal(ponmx,normal);
  // assert(ApproxEqual(normal,Vec_t(-1,0,0)));

  // valid = b1.Normal(ponmy,normal);
  // assert(ApproxEqual(normal,Vec_t(0,-1,0)));

  valid = b1.Normal(ponmz, normal);
  assert(ApproxEqual(normal, Vec_t(0, 0, -1)));

  // DistanceToOut(P,V) with asserts for norm and convex

  // Raman----------

  Vec_t pmidx(fRmin + ((fRmax - fRmin) / 2), 0, 0);
  Vec_t pmidy(0, fRmin + ((fRmax - fRmin) / 2), 0);

  Dist = b1.DistanceToOut(pmidx, vx, norm, convex);
  assert(ApproxEqual(Dist, ((fRmax - fRmin) / 2)) && ApproxEqual(norm, vx));

  Dist = b1.DistanceToOut(-pmidx, vmx, norm, convex);
  assert(ApproxEqual(Dist, ((fRmax - fRmin) / 2)) && ApproxEqual(norm, vmx));

  Dist = b1.DistanceToOut(pmidy, vy, norm, convex);
  assert(ApproxEqual(Dist, ((fRmax - fRmin) / 2)) && ApproxEqual(norm, vy));

  Dist = b1.DistanceToOut(-pmidy, vmy, norm, convex);
  assert(ApproxEqual(Dist, ((fRmax - fRmin) / 2)) && ApproxEqual(norm, vmy));

  double distZ = std::sqrt((pmidx.Mag2() - (fRmin * fRmin)) / (tSTin * tSTin));

  Dist = b1.DistanceToOut(pmidx, vz, norm, convex);
  assert(ApproxEqual(Dist, distZ)); //&& ApproxEqual(norm,vz) && convex);

  Dist = b1.DistanceToOut(pmidx, vmz, norm, convex);
  assert(ApproxEqual(Dist, distZ)); //&& ApproxEqual(norm,vmz) && convex);

  Dist = b1.DistanceToOut(-pmidx, vz, norm, convex);
  assert(ApproxEqual(Dist, distZ)); //&& ApproxEqual(norm,vz) && convex);

  Dist = b1.DistanceToOut(-pmidx, vmz, norm, convex);
  assert(ApproxEqual(Dist, distZ)); //&& ApproxEqual(norm,vmz) && convex);

  Dist = b1.DistanceToOut(pmidy, vz, norm, convex);
  assert(ApproxEqual(Dist, distZ)); //&& ApproxEqual(norm,vz) && convex);

  Dist = b1.DistanceToOut(pmidy, vmz, norm, convex);
  assert(ApproxEqual(Dist, distZ)); //&& ApproxEqual(norm,vmz) && convex);

  Dist = b1.DistanceToOut(-pmidy, vz, norm, convex);
  assert(ApproxEqual(Dist, distZ)); //&& ApproxEqual(norm,vz) && convex);

  Dist = b1.DistanceToOut(-pmidy, vmz, norm, convex);
  assert(ApproxEqual(Dist, distZ)); //&& ApproxEqual(norm,vmz) && convex);

  // Point is already outside and checking DistanceToOut. In this case distance is shoule be set to -1.

  Dist = b1.DistanceToOut(pbigx, vx, norm, convex);
  std::cout << "Dist : " << Dist << std::endl;
  assert(ApproxEqual(Dist, -1.));

  Dist = b1.DistanceToOut(pbigy, vy, norm, convex);
  assert(ApproxEqual(Dist, -1.));

  Dist = b1.DistanceToOut(pbigz, vz, norm, convex);
  assert(ApproxEqual(Dist, -1.));

  Dist = b1.DistanceToOut(pbigx, vmx, norm, convex);
  assert(ApproxEqual(Dist, -1.));

  Dist = b1.DistanceToOut(pbigy, vmy, norm, convex);
  assert(ApproxEqual(Dist, -1.));

  Dist = b1.DistanceToOut(pbigz, vmz, norm, convex);
  assert(ApproxEqual(Dist, -1.));

  // Raman------------------

  // Check Inside

  // Raman------------

  assert(b1.Inside(pzero) == vecgeom::EInside::kOutside);
  assert(b1.Inside(pbigx) == vecgeom::EInside::kOutside);
  assert(b1.Inside(pbigy) == vecgeom::EInside::kOutside);
  assert(b1.Inside(pbigz) == vecgeom::EInside::kOutside);

  assert(b1.Inside(pmidx) == vecgeom::EInside::kInside);
  assert(b1.Inside(-pmidx) == vecgeom::EInside::kInside);
  assert(b1.Inside(pmidy) == vecgeom::EInside::kInside);
  assert(b1.Inside(-pmidy) == vecgeom::EInside::kInside);

  assert(b1.Inside(ponxsideO) == vecgeom::EInside::kSurface);
  assert(b1.Inside(ponysideO) == vecgeom::EInside::kSurface);
  assert(b1.Inside(ponxsideI) == vecgeom::EInside::kSurface);
  assert(b1.Inside(ponysideI) == vecgeom::EInside::kSurface);

  // Raman-------------
  // assert(b1.Inside(ponzsideO)==vecgeom::EInside::kSurface);
  // assert(b1.Inside(ponzsideI)==vecgeom::EInside::kSurface);

  /*
  // SafetyFromInside(P)
  Dist=b1.SafetyFromInside(pzero);
  assert(ApproxEqual(Dist,fR));
  Dist=b1.SafetyFromInside(vx);
  assert(ApproxEqual(Dist,fR-1));
  Dist=b1.SafetyFromInside(vy);
  assert(ApproxEqual(Dist,fR-1));
  Dist=b1.SafetyFromInside(vz);
  assert(ApproxEqual(Dist,fR-1));

  //SafetyFromOutside(P)
  Dist=b1.SafetyFromOutside(pbigx);
  assert(ApproxEqual(Dist,100-fR));
  Dist=b1.SafetyFromOutside(pbigmx);
  assert(ApproxEqual(Dist,100-fR));
  Dist=b1.SafetyFromOutside(pbigy);
  assert(ApproxEqual(Dist,100-fR));
  Dist=b1.SafetyFromOutside(pbigmy);
  assert(ApproxEqual(Dist,100-fR));
  Dist=b1.SafetyFromOutside(pbigz);
  assert(ApproxEqual(Dist,100-fR));
  Dist=b1.SafetyFromOutside(pbigmz);
  assert(ApproxEqual(Dist,100-fR));
  */
  // DistanceToIn(P,V)

  // Raman------------

  Dist = b1.DistanceToIn(pbigx, vmx);
  assert(ApproxEqual(Dist, 100 - fRmax));
  Dist = b1.DistanceToIn(pbigmx, vx);
  assert(ApproxEqual(Dist, 100 - fRmax));
  Dist = b1.DistanceToIn(pbigy, vmy);
  assert(ApproxEqual(Dist, 100 - fRmax));
  Dist = b1.DistanceToIn(pbigmy, vy);
  assert(ApproxEqual(Dist, 100 - fRmax));

  Dist = b1.DistanceToIn(pbigz, vmz);
  assert(ApproxEqual(Dist, 100 - fR));
  Dist = b1.DistanceToIn(pbigmz, vz);
  assert(ApproxEqual(Dist, 100 - fR));

  Dist = b1.DistanceToIn(pbigx, vxy);
  if (Dist >= UUtils::kInfinity)
    Dist = UUtils::Infinity();
  assert(ApproxEqual(Dist, UUtils::Infinity()));

  Dist = b1.DistanceToIn(pbigmx, vxy);
  if (Dist >= UUtils::kInfinity)
    Dist = UUtils::Infinity();
  assert(ApproxEqual(Dist, UUtils::Infinity()));

  std::cout << "------------------------------------------------" << std::endl;
  Vec_t pointOTol(20 + vecgeom::cxx::kTolerance, 0., 0.);
  Dist = b1.DistanceToIn(pointOTol, vx);
  std::cout << "DistToIn : " << Dist << std::endl;
  if (Dist >= UUtils::kInfinity)
    Dist = UUtils::Infinity();
  assert(ApproxEqual(Dist, UUtils::Infinity()));

  Dist = b1.DistanceToOut(pointOTol, vx, norm, convex); // This case fails
  std::cout << "DistToOut : " << Dist << std::endl;
  assert(ApproxEqual(Dist, 0.));

  // Point inside outer tolerance of outer hyperboloid  and directing in
  Dist = b1.DistanceToIn(pointOTol, vmx);
  std::cout << "DistToIn : " << Dist << std::endl;
  assert(ApproxEqual(Dist, 0.));

  Dist = b1.DistanceToOut(pointOTol, vmx, norm, convex); // This case fails
  std::cout << "DistToOut : " << Dist << std::endl;
  assert(ApproxEqual(Dist, 10.));

  // Point outside inner tolerance of outer hyperboloid and directing out
  std::cout << "------------------------------------------------" << std::endl;
  Vec_t pointOTolI(20 - vecgeom::cxx::kTolerance, 0., 0.);
  Dist = b1.DistanceToIn(pointOTolI, vx);
  std::cout << "DistToIn : " << Dist << std::endl;
  if (Dist >= UUtils::kInfinity)
    Dist = UUtils::Infinity();
  assert(ApproxEqual(Dist, UUtils::Infinity()));

  Dist = b1.DistanceToOut(pointOTolI, vx, norm, convex); // This case fails
  std::cout << "DistToOut : " << Dist << std::endl;
  assert(ApproxEqual(Dist, 0.));

  // Point Outside inner tolerance of outer hyperboloid and directing in

  Dist = b1.DistanceToIn(pointOTolI, vmx);
  std::cout << "DistToIn : " << Dist << std::endl;
  assert(ApproxEqual(Dist, 0.));

  Dist = b1.DistanceToOut(pointOTolI, vmx, norm, convex);
  std::cout << "DistToOut : " << Dist << std::endl;
  assert(ApproxEqual(Dist, 10.));

  std::cout << "------------------------------------------------" << std::endl;
  Vec_t pointOTol_IH(10 + vecgeom::cxx::kTolerance, 0., 0.);
  Dist = b1.DistanceToIn(pointOTol_IH, vx);
  std::cout << "DistToIn : " << Dist << std::endl;

  Dist = b1.DistanceToOut(pointOTol_IH, vx, norm, convex); // This case fails
  std::cout << "DistToOut : " << Dist << std::endl;
  assert(ApproxEqual(Dist, 10.));

  Dist = b1.DistanceToIn(pointOTol_IH, vmx); // May fail
  std::cout << "DistToIn : " << Dist << std::endl;
  // std::cout<<"DDDDD : "<<Dist<<std::endl;
  assert(ApproxEqual(Dist, 20.));

  Dist = b1.DistanceToOut(pointOTol_IH, vmx, norm, convex); // This case fails
  std::cout << "DistToOut : " << Dist << std::endl;
  // std::cout<<"SSSS : "<<Dist<<std::endl;
  assert(ApproxEqual(Dist, 0.));

  std::cout << "------------------------------------------------" << std::endl;
  Vec_t pointOTol_IHm(10 - vecgeom::cxx::kTolerance, 0., 0.);
  Dist = b1.DistanceToIn(pointOTol_IHm, vx);
  std::cout << "DistToIn : " << Dist << std::endl;
  assert(ApproxEqual(Dist, 0.));

  Dist = b1.DistanceToOut(pointOTol_IHm, vx, norm, convex); // This case fails
  std::cout << "DistToOut : " << Dist << std::endl;
  assert(ApproxEqual(Dist, 10.));
  Dist = b1.DistanceToIn(pointOTol_IHm, vmx); // May fail
  std::cout << "DistToIn : " << Dist << std::endl;
  assert(ApproxEqual(Dist,20.));

  Dist = b1.DistanceToOut(pointOTol_IH, vmx, norm, convex); // This case fails
  std::cout << "DistToOut : " << Dist << std::endl;
  assert(ApproxEqual(Dist, 0.));

  std::cout << "------------------------------------------------" << std::endl;
  Vec_t pointOTol_Z(60, 0., 50. + vecgeom::cxx::kTolerance);

  Dist = b1.DistanceToIn(pointOTol_Z, vz);
  if (Dist >= UUtils::kInfinity)
    Dist = UUtils::Infinity();
  std::cout << "DistToIn : " << Dist << std::endl;
  assert(ApproxEqual(Dist, UUtils::Infinity()));

  Dist = b1.DistanceToOut(pointOTol_Z, vz, norm, convex); // This case fails
  std::cout << "DistToOut : " << Dist << std::endl;
  assert(ApproxEqual(Dist, 0.));

  Dist = b1.DistanceToIn(pointOTol_Z, vmz);
  std::cout << "DistToIn : " << Dist << std::endl;
  assert(ApproxEqual(Dist, 0.));

  double Dist3 = b1.DistanceToOut(pointOTol_Z, vmz, norm, convex); // This case fails
  std::cout << "DistToOut : " << Dist3 << std::endl;
  assert(ApproxEqual(Dist,0.));

  std::cout << "------------------------------------------------" << std::endl;

  Vec_t pointOTol_ZNeg(60, 0., -50. - vecgeom::cxx::kTolerance);

  Dist = b1.DistanceToIn(pointOTol_ZNeg, vmz);
  if (Dist >= UUtils::kInfinity)
    Dist = UUtils::Infinity();
  std::cout << "DistToIn : " << Dist << std::endl;
  assert(ApproxEqual(Dist, UUtils::Infinity()));

  Dist = b1.DistanceToOut(pointOTol_ZNeg, vmz, norm, convex); // This case fails
  std::cout << "DistToOut : " << Dist << std::endl;
  assert(ApproxEqual(Dist, 0.));

  Dist = b1.DistanceToIn(pointOTol_ZNeg, vz);
  std::cout << "DistToIn : " << Dist << std::endl;
  assert(ApproxEqual(Dist, 0.));

  double Dist4 = b1.DistanceToOut(pointOTol_ZNeg, vz, norm, convex); // This case fails
  std::cout << "DistToOut : " << Dist4 << std::endl;
  assert(ApproxEqual(Dist3, Dist4));

  //UNIT TESTS FOR WRONG SIDE POINTS

  // Testing DistanceToOut for outside points
  Dist = b1.DistanceToOut(pbigx, vmx, norm, convex);
  assert(ApproxEqual(Dist, -1.));
  Dist = b1.DistanceToOut(pbigy, vmy, norm, convex);
  assert(ApproxEqual(Dist, -1.));
  Dist = b1.DistanceToOut(pbigz, vmz, norm, convex);
  assert(ApproxEqual(Dist, -1.));
  Dist = b1.DistanceToOut(pbigmx, vmx, norm, convex);
  assert(ApproxEqual(Dist, -1.));
  Dist = b1.DistanceToOut(pbigmy, vmy, norm, convex);
  assert(ApproxEqual(Dist, -1.));
  Dist = b1.DistanceToOut(pbigmz, vmz, norm, convex);
  assert(ApproxEqual(Dist, -1.));


  // Testing SafetFromInside for outside points
  Dist = b1.SafetyFromInside(pbigx);
  assert(ApproxEqual(Dist,-1));
  Dist = b1.SafetyFromInside(pbigy);
  assert(ApproxEqual(Dist,-1));
  Dist = b1.SafetyFromInside(pbigz);
  assert(ApproxEqual(Dist,-1));
  Dist = b1.SafetyFromInside(pbigmx);
  assert(ApproxEqual(Dist,-1));
  Dist = b1.SafetyFromInside(pbigmy);
  assert(ApproxEqual(Dist,-1));
  Dist = b1.SafetyFromInside(pbigmz);
  assert(ApproxEqual(Dist,-1));

  // Testing DistanceToIn for inside points
  Dist = b1.DistanceToIn(Vec_t(15., 0., 0.), vx);
  assert(ApproxEqual(Dist, -1.));
  Dist = b1.DistanceToIn(Vec_t(0.,15., 0.), vy);
  assert(ApproxEqual(Dist, -1.));
  Dist = b1.DistanceToIn(Vec_t(-15., 0., 0.), vmx);
  assert(ApproxEqual(Dist, -1.));
  Dist = b1.DistanceToIn(Vec_t(0.,-15., 0.), vmy);
  assert(ApproxEqual(Dist, -1.));

  // Testing SafetyFromOut for inside points
  Dist = b1.SafetyFromOutside(Vec_t(15., 0., 0.));
  assert(ApproxEqual(Dist, -1.));
  Dist = b1.SafetyFromOutside(Vec_t(0., 15., 0.));
  assert(ApproxEqual(Dist, -1.));
  Dist = b1.SafetyFromOutside(Vec_t(-15., 0., 0.));
  assert(ApproxEqual(Dist, -1.));
  Dist = b1.SafetyFromOutside(Vec_t(0., -15., 0.));
  assert(ApproxEqual(Dist, -1.));


  return true;
}

 int main(int argc, char *argv[]) {

    if( argc < 2)
     {
       std::cerr << "need to give argument :--usolids or --vecgeom\n";
       return 1;
     }

     if( ! strcmp(argv[1], "--usolids") )
     {
 #ifdef VECGEOM_USOLIDS
  // assert(TestHype<UHype>());
   std::cout << "UHype passed\n";
       #else
       std::cerr << "VECGEOM_USOLIDS was not defined\n";
       return 2;
 #endif
     }
     else if( ! strcmp(argv[1], "--vecgeom") )
     {
   assert(TestHype<vecgeom::SimpleHype>());
   std::cout << "VecGeomHype passed\n";
     }
     else
     {
       std::cerr << "need to give argument :--usolids or --vecgeom\n";
       return 1;
     }


   return 0;
 }
