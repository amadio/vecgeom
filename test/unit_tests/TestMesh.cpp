///
/// file:    TestMesh.cpp
/// purpose: Unit tests for the meshes of 3D models.
///

//-- ensure asserts are compiled in
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "base/Utils3D.h"
#include "ApproxEqual.h"
#include "volumes/Box.h"
#include "test/benchmark/ArgParser.h"

#include "volumes/SolidMesh.h"
#include "volumes/UnplacedParallelepiped.h"
#include "volumes/UnplacedTrapezoid.h"
#include "volumes/UnplacedTet.h"
#include "volumes/UnplacedTrd.h"
#include "management/GeoManager.h"
#include "volumes/UnplacedSExtruVolume.h"
#include "volumes/UnplacedEllipticalTube.h"
#include "volumes/UnplacedEllipticalCone.h"
#include "volumes/UnplacedOrb.h"
#include "volumes/UnplacedParaboloid.h"
#include "volumes/UnplacedCone.h"
#include "volumes/UnplacedParaboloid.h"
#include "volumes/UnplacedPolycone.h"
#include "volumes/UnplacedPolyhedron.h"
#include "volumes/UnplacedGenTrap.h"
#include "volumes/UnplacedEllipsoid.h"
#include "volumes/UnplacedCutTube.h"
#include "volumes/UnplacedTube.h"
#include "volumes/UnplacedTorus2.h"
#include "volumes/UnplacedHype.h"
#include "volumes/UnplacedSphere.h"
#include "volumes/UnplacedExtruded.h"

#ifdef VECGEOM_ROOT
#include "utilities/Visualizer.h"
#include "TPolyMarker3D.h"
#include "TPolyLine3D.h"
#endif

#ifdef VECGEOM_ROOT
void DrawPolygon(vecgeom::Utils3D::Polygon const &poly, vecgeom::Visualizer &visualizer, size_t color)
{
  using namespace vecgeom;
  using Vec_t = Vector3D<double>;
  TPolyLine3D pl(poly.fN + 1);
  pl.SetLineColor(color);
  for (size_t i = 0; i < poly.fN; ++i)
    pl.SetNextPoint(poly.GetVertex(i).x(), poly.GetVertex(i).y(), poly.GetVertex(i).z());
  pl.SetNextPoint(poly.GetVertex(0).x(), poly.GetVertex(0).y(), poly.GetVertex(0).z());
  visualizer.AddLine(pl);

  // Compute center of polygon
  Vec_t center;
  for (size_t i = 0; i < poly.fN; ++i)
    center += poly.GetVertex(i);
  center *= 1. / poly.fN;
  TPolyLine3D plnorm(2);
  plnorm.SetLineColor(color);
  plnorm.SetNextPoint(center[0], center[1], center[2]);
  plnorm.SetNextPoint(center[0] + poly.fNorm[0], center[1] + poly.fNorm[1], center[2] + poly.fNorm[2]);
  visualizer.AddLine(plnorm);
}

void DrawPolyhedron(const vecgeom::Utils3D::Polyhedron &polyh, vecgeom::Visualizer &visualizer, size_t color)
{
  using namespace vecgeom;

  for (size_t i = 0; i < polyh.GetNpolygons(); ++i)
    DrawPolygon(polyh.GetPolygon(i), visualizer, color);
}
#endif

int main(int argc, char *argv[])
{
  using namespace vecgeom;
  using namespace vecCore::math;

  using vecgeom::Utils3D::Line;
  using vecgeom::Utils3D::Plane;
  using vecgeom::Utils3D::Polygon;
  using vecgeom::Utils3D::Polyhedron;

#ifdef VECGEOM_ROOT
  OPTION_STRING(v, "noVolume");
  // OPTION_BOOL(t, false);
#endif


#ifdef VECGEOM_ROOT
  VUnplacedVolume *unplacedvolume = nullptr;
  using Vec_t = Vector3D<double>;
#define WORLDSIZE 10

  if (!v.compare("noVolume")) {
    std::cout << "\nUsage:\n"
                 "./TestMesh -v [str] -t [bool]\n"
                 "\n"
                 "Available volumes: \"box\", \"parallelepiped\", \"sextruvolume\", \"tet\", \"trapezoid\", "
                 "\"trd\", \"ellipticaltube\", \"ellipticalcone\", \"orb\".\nUse -t to apply a random transformation.\n"
                 "\n"
                 "\n"
                 "";
    return 0;
  } else if (!v.compare("box")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedBox>(3., 3., 3.);
  } else if (!v.compare("parallelepiped")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedParallelepiped>(2., 2., 3., 90., 90., 90.);
  } else if (!v.compare("sextruvolume")) {
#define N 10
    double dx = 5;
    double dy = 5;
    double dz = 3;

    double x[N], y[N];
    for (size_t i = 0; i < (size_t)N; ++i) {
      x[i] = dx * std::sin(i * (2. * M_PI) / N);
      y[i] = dy * std::cos(i * (2. * M_PI) / N);
    }
    unplacedvolume = GeoManager::MakeInstance<UnplacedSExtruVolume>(N, x, y, -dz, dz);
  } else if (!v.compare("trapezoid")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedTrapezoid>(5., 0., 0., 3., 4., 5., 0., 3., 4., 5., 0.);
  } else if (!v.compare("trd")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedTrd>(2., 3., 2., 5.);
  } else if (!v.compare("tet")) {
    Vec_t p0(0., 0., 5.), p1(-5., -5., -5.), p2(5., -5., -5.), p3(-5., 5., -5.);
    unplacedvolume = GeoManager::MakeInstance<UnplacedTet>(p0, p1, p2, p3);
  } else if (!v.compare("ellipticaltube")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedEllipticalTube>(2, 4, 5);
  } else if (!v.compare("ellipticalcone")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedEllipticalCone>(1., 1., 5., 3.);
  } else if (!v.compare("orb")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedOrb>(8.);
  } else if (!v.compare("cone")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedCone>(3., 5, 2., 4., 5, 0, kPi);
  } else if (!v.compare("paraboloid")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedParaboloid>(0., 7., 5.);
  } else if (!v.compare("polycone")) {
    int Nz         = 4;
    double rmin[]  = {0., 3., 7., 8};
    double rmax[]  = {5., 4., 8., 9};
    double z[]     = {-3, 0, 3, 8};
    unplacedvolume = GeoManager::MakeInstance<UnplacedPolycone>(0, kPi, Nz, z, rmin, rmax);
  } else if (!v.compare("polyhedron")) {
    constexpr int nPlanes      = 5;
    Precision zPlanes[nPlanes] = {-7, -1, -1, 1, 7};
    Precision rInner[nPlanes]  = {3, 4, 5, 5, 7};
    Precision rOuter[nPlanes]  = {8, 9, 9, 9, 10};

    unplacedvolume = GeoManager::MakeInstance<UnplacedPolyhedron>(15 * kDegToRad, 180 * kDegToRad, 5, nPlanes, zPlanes,
                                                                  rInner, rOuter);
  } else if (!v.compare("gentrap")) {
    // twisted

    Precision verticesx[8] = {-3, -3, 3, 3, -3.889087296526, 0.35355339059327, 3.889087296526, -0.35355339059327};
    Precision verticesy[8] = {-3, 3, 3, -3, 0.35355339059327, 3.889087296526, -0.35355339059327, -3.889087296526};

    // no twist
    // Precision verticesx1[8] = {-3, -3, 3, 3, -2, -2, 2, 2};
    // Precision verticesy1[8] = {-3, 3, 3, -3, -2, 2, 2, -2};

    unplacedvolume = GeoManager::MakeInstance<UnplacedGenTrap>(verticesx, verticesy, 8);
  } else if (!v.compare("ellipsoid")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedEllipsoid>(3, 4, 5, -3, 3);
  } else if (!v.compare("cuttube")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedCutTube>(3, 5, 7, 0, kPi, Vec_t(1, 0, -1), Vec_t(1, 0, 1));
  } else if (!v.compare("tube")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedTube>(2., 10., 5., 0., 2*kPi);
  } else if (!v.compare("torus")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedTorus2>(0., 5., 5., 0, 2*kPi);
  } else if (!v.compare("hype")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedHype>(0., 4., 45., 45., 5.);
  } else if(!v.compare("sphere")) {
	  unplacedvolume = GeoManager::MakeInstance<UnplacedSphere>(3, 5, 0, kPi, kPi/4 , kPi );
  } else if(!v.compare("polyx")){

	  /*
	  Utils3D::Line l1, l2;
	  l1.fPts[0] = Vec_t(1,0,0);l1.fPts[1] = Vec_t(2,0,0);
	  l2.fPts[0] = Vec_t(-1,0,-1);l2.fPts[1] = Vec_t(5,0,1);

	  std::cout << l1.Intersect(l2)->fA << ' ' <<  l1.Intersect(l2)->fB << ' ' << l1.Intersect(l2)->fObjectType << '\n';
	  */


	  std::vector<Vec_t> v;
	  v.push_back(Vec_t(2,2,0));
	  v.push_back(Vec_t(-2,2,0));
	  v.push_back(Vec_t(-2,-2,0));
	  v.push_back(Vec_t(2,-2,0));
	  Utils3D::Polygon p{4, v, false};
	  p.fInd = {0,1,2,3};
	  p.Init();


	  /*
	  v.push_back(Vec_t(-3,0,0));
	  v.push_back(Vec_t(-2.5,1,0));
	  v.push_back(Vec_t(-2,0,0));
	  */

	  v.push_back(Vec_t(3,2,0));
	  v.push_back(Vec_t(-1,2,0));
	  v.push_back(Vec_t(-1,-2,0));
	  v.push_back(Vec_t(3,-2,0));


	  //v.push_back(Vec_t(3,-2,0));
	  Utils3D::Polygon p2{4, v, false};
	  p2.fInd = {4,5,6,7};
	  p2.Init();






	  /*
	  Utils3D::Polygon p2{5,v,false};
	  v.push_back(Vec_t(5, 2,-1));
	  v.push_back(Vec_t(5,2,1));
	  v.push_back(Vec_t(0,0,0));
	  v.push_back(Vec_t(-5, 2,1));
	  v.push_back(Vec_t(-5, 2,-1));
*/


	  /*
	  Utils3D::Polygon p2{4,v,false};
	  v.push_back(Vec_t(3,2,0));
	  v.push_back(Vec_t(-2,2,0));
	  v.push_back(Vec_t(-2,-3,0));
	  //v.push_back(Vec_t(1,1,0));
	  v.push_back(Vec_t(3,-3,0));


	  p2.fInd = {5,6,7,8};
	  p2.Init();
*/


	  Utils3D::PolygonIntersection *pi = p.Intersect(p2);

	  Visualizer visualizer;
	  SimpleBox boxshape("box", 2, 2, 2);
	  visualizer.AddVolume(boxshape);
	  //DrawPolyhedron(unplacedvolume->CreateMesh3D(Transformation3D(), 20)->GetMesh(), visualizer, kBlue);
	  DrawPolygon(p, visualizer, kGreen);
	  DrawPolygon(p2, visualizer, kRed);

	  TPolyLine3D pl(2* (pi->fLines.size()));
	  pl.SetLineColor(kGreen);
	  for (size_t i = 0; i < pi->fLines.size(); ++i){
		  visualizer.AddLine((pi->fLines[i].fPts[0]), (pi->fLines[i].fPts[1]));
	  }

	  for(auto poly: pi->fPolygons){
		  DrawPolygon(poly, visualizer, kBlue);
	  }


	  /*Line l1{Vec_t(), Vec_t(1,0,0)};
	  Line l2{Vec_t(1,1,1), Vec_t(1,1,2)};
	  Utils3D::LineIntersection* li = l1.Intersect(l2);
	  std::cout << li->fA << ' ' << li-> fB << ' ' << li ->fType << '\n';
*/



	  visualizer.Show();


  }else if(!v.compare("extruded")){
#define nvert 10
#define nsect 5

	  double rmin = 3.;
	    double rmax = 5.;
	    bool convex = true;


	  vecgeom::XtruVertex2 *vertices = new vecgeom::XtruVertex2[nvert];
	  vecgeom::XtruSection *sections = new vecgeom::XtruSection[nsect];
	  double *x                      = new double[nvert];
	  double *y                      = new double[nvert];

	  double phi = 2. * kPi / nvert;
	  double r;
	  for (int i = 0; i < nvert; ++i) {
	    r = rmax;
	    if (i % 2 > 0 && !convex) r = rmin;
	    vertices[i].x = r * vecCore::math::Cos(i * phi);
	    vertices[i].y = r * vecCore::math::Sin(i * phi);
	    x[i]          = vertices[i].x;
	    y[i]          = vertices[i].y;
	  }
	  for (int i = 0; i < nsect; ++i) {
	    sections[i].fOrigin.Set(0, 0, -2. + i * 4. / (nsect - 1));
	    sections[i].fScale = 1;
	    if(i == 0)
	    	sections[0].fScale = 0.5;
	  }

	  unplacedvolume = GeoManager::MakeInstance<UnplacedExtruded>(nvert, vertices, nsect, sections);
  }
  else if(!v.compare("polytri")){
	  std::cout <<"hahaha\n";
  	  Visualizer visualizer;
  	  SimpleBox boxshape("box", 2, 2, 2);
  	visualizer.AddVolume(boxshape);


  	  std::vector<Vec_t> v;
  	  v.push_back(Vec_t(2,2,0));
  	  v.push_back(Vec_t(-2,2,0));
  	  v.push_back(Vec_t(-2,-2,0));
  	  v.push_back(Vec_t(0, 0, 0));
  	  v.push_back(Vec_t(2,-2,0));
  	  Utils3D::Polygon p{5, v, false};
  	  p.fInd = {0,1,2,3,4};
  	  p.Init();

  	  //DrawPolygon(p, visualizer, kBlue);
  	  std::vector<Polygon> polys;
  	  p.TriangulatePolygon(polys);
	  for(auto poly: polys){

		  DrawPolygon(poly, visualizer, kRed);
	  }

  	  v.push_back(Vec_t(2,2,1));
  	  v.push_back(Vec_t(-2,2,1));
  	  v.push_back(Vec_t(-2,-2,1));
  	  v.push_back(Vec_t(0, 0, 1));
  	  v.push_back(Vec_t(2,-2,1));
  	  Utils3D::Polygon p2{5, v, false};
  	  p2.fInd = {5,6,7,8,9};
  	  p2.Init();
  	DrawPolygon(p2, visualizer, kBlue);



  	 visualizer.Show();
    }





  Visualizer visualizer;
  SimpleBox boxshape("box", WORLDSIZE, WORLDSIZE, WORLDSIZE);
  visualizer.AddVolume(boxshape);
  DrawPolyhedron(unplacedvolume->CreateMesh3D(Transformation3D(), 10)->GetMesh(), visualizer, kBlue);
  //DrawPolygon(p, visualizer, kBlue);
  visualizer.Show();




#endif



  return 0;
}
