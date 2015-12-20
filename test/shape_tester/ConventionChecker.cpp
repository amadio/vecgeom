/// @file ConventionChecker.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch)

/* This file contains implementation of additional functions added to ShapeTester,
 * to have the shape convention checking feature.
 */

#include "base/RNG.h"
#include <iomanip>
#include <sstream>
#include <ctime>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "ShapeTester.h"
#include "VUSolid.hh"
#include "UTransform3D.hh"

#include "base/Vector3D.h"
#include "volumes/Box.h"

#ifdef VECGEOM_ROOT
#include "TGeoShape.h"
#include "TGeoParaboloid.h"
#include "TGeoBBox.h"
#include "TGraph2D.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"
#include "TGeoParaboloid.h"
#include "TGeoVolume.h"
#include "TPolyMarker3D.h"
#include "TRandom3.h"
#include "TColor.h"
#include "TROOT.h"
#include "TAttMarker.h"
#include "TH1D.h"
#include "TH2F.h"
#include "TF1.h"
#include "TVirtualPad.h"
#include "TView3D.h"
#endif

#undef NDEBUG
#include <cassert>

//Function to set the number of Points to be displayed in case of convention not followed
void ShapeTester::SetNumDisp(int num) {
  fNumDisp = num;
}

// Helper function taken from ApproxEqual.h
bool ShapeTester::ApproxEqual(const double x, const double y) {
  if (x == y) {
    return true;
  } else if (x * y == 0.0) {
    double diff = std::fabs(x - y);
    return diff < kApproxEqualTolerance;
  } else {
    double diff = std::fabs(x - y);
    double abs_x = std::fabs(x), abs_y = std::fabs(y);
    return diff / (abs_x + abs_y) < kApproxEqualTolerance;
  }
}

// Return true if the 3vector check is approximately equal to target
template <class Vec_t>
bool ShapeTester::ApproxEqual(const Vec_t &check, const Vec_t &target) {
  return (ApproxEqual(check.x(), target.x()) && ApproxEqual(check.y(), target.y()) &&
          ApproxEqual(check.z(), target.z()))
             ? true
             : false;
}

/* Function to Setup all the convention messages
 * With this interface it will be easy, if we want to put
 * some more conventions in future
 */
void ShapeTester::SetupConventionMessages() {
    // For Surface Points
  fScore = 0;                                                                                  // index
  fConventionMessage.push_back("DistanceToIn()  : For Point On Surface and Entering the Shape"); // 0
  fConventionMessage.push_back("DistanceToIn()  : For Point On Surface and Exiting the Shape");  // 1
  fConventionMessage.push_back("DistanceToOut() : For Point On Surface and Exiting the Shape");  // 2
  fConventionMessage.push_back("DistanceToOut() : For Point On Surface and Entering the Shape"); // 3
  fConventionMessage.push_back("SafetyToIn()    : For Point On Surface ");                       // 4
  fConventionMessage.push_back("SafetyToOut()   : For Point On Surface ");                       // 5

  // For Inside Points
  fConventionMessage.push_back("DistanceToIn()  : For Inside Point"); // 6
  fConventionMessage.push_back("DistanceToOut() : For Inside Point"); // 7
  fConventionMessage.push_back("SafetyToIn()    : For Inside Point"); // 8
  fConventionMessage.push_back("SafetyToOut()   : For Inside Point"); // 9

  // Outside Points
  fConventionMessage.push_back("DistanceToIn()  : For Outside Point"); // 10
  fConventionMessage.push_back("DistanceToOut() : For Outside Point"); // 11
  fConventionMessage.push_back("SafetyToIn()    : For Outside Point"); // 12
  fConventionMessage.push_back("SafetyToOut()   : For Outside Point"); // 13

  fNumDisp = 1;
}

//Funtion to check conventions for Surface Points
bool ShapeTester::ShapeConventionSurfacePoint() {
  int nError=0;
  bool surfPointConventionPassed = true;
  for (int i = 0; i < fMaxPointsSurface + fMaxPointsEdge; i++) { // test GetPointOnSurface()
    UVector3 point = fPoints[fOffsetSurface + i];
    UVector3 direction = fDirections[fOffsetSurface + i];
    if (fVolumeUSolids->Inside(point) != vecgeom::EInside::kSurface) {
      assert(false && "For Surface point, Inside says that the Point is not on the Surface");
    }

    // Point on Surface and moving inside
    UVector3 normal(0., 0., 0.);
    // bool valid =
    fVolumeUSolids->Normal(point, normal);

    double Dist = fVolumeUSolids->DistanceToIn(point, direction);
    int indx = 0;

    // Conventions Check for DistanceToIn
    if (Dist >= UUtils::kInfinity)
      Dist = UUtils::Infinity();

    if (direction.Dot(normal) < 0.) // particle is entering into the shape
    {
      // assert(Dist == 0.);
      if (Dist != 0.) {
        ReportError(&nError, point, direction, Dist, "DistanceToIn for Surface Point entering into the Shape should be 0.");
        fScore |= (1 << indx);
        surfPointConventionPassed &= false;
      }
    }

    indx = 1;
    // Consider all the shapes as "Not convex" even if it is !!.
    bool convexShape = false; // convexShape = IsConvex()
    // Point on Surface and moving outside
    if (direction.Dot(normal) > 0.) // particle is exiting from the shape
    {
      // assert(ApproxEqual(Dist,UUtils::Infinity()));
      if (convexShape) {
        // assert(ApproxEqual(Dist, UUtils::Infinity()));
        if (!ApproxEqual(Dist, UUtils::Infinity())) {
          fScore |= (1 << indx);
          surfPointConventionPassed &= false;
        }
      } else {
        // If the shape is not convex then DistanceIn is distance to sNext Intersection
        // It may possible that it will not hit the shape again, in that case, Distance should be infinity
        // So overall distance must be greater than zero.
        // assert(Dist > 0.);
        if (!(Dist > 0.)) {
          ReportError(&nError, point, direction, Dist,
                      "DistanceToIn for Surface Point exiting into the Shape should be > 0.");
          fScore |= (1 << indx);
          surfPointConventionPassed &= false;
        }
      }
    }

    // Conventions check for DistanceToOut
    indx = 2;
    UVector3 norm(0., 0., 0.);
    bool convex = false;
    Dist = fVolumeUSolids->DistanceToOut(point, direction, norm, convex);
    if (Dist >= UUtils::kInfinity)
      Dist = UUtils::Infinity();
    if (direction.Dot(normal) > 0.) // particle is exiting from the shape
    {
      // assert((Dist == 0.) && "DistanceToOut for surface point moving outside should be equal to 0.");
      if (!(Dist == 0.)) {
        ReportError(&nError, point, direction, Dist,
                    "DistanceToOut for Surface Point exiting the Shape should be 0.");

        fScore |= (1 << indx);
        surfPointConventionPassed &= false;
      }
    }

    indx = 3;
    if (direction.Dot(normal) < 0.) // particle is entering from the shape
    {
      // assert(Dist > 0.);
      if (!(Dist > 0.)) {
        ReportError(&nError, point, direction, Dist,
                    "DistanceToOut for Surface Point entering into the Shape should be > 0.");

        fScore |= (1 << indx);
        surfPointConventionPassed &= false;
      }
    }

    indx = 4;
    // Conventions check for SafetyFromOutside
    Dist = fVolumeUSolids->SafetyFromOutside(point);
    if (Dist >= UUtils::kInfinity)
      Dist = UUtils::Infinity();
    // assert(Dist == 0.);
    if (!(Dist == 0.)) {
      ReportError(&nError, point, direction, Dist,
                  "SafetyFromOutside for Surface Point should be 0.");
      fScore |= (1 << indx);
      surfPointConventionPassed &= false;
    }

    indx = 5;
    // Conventions check for SafetyFromInside
    Dist = fVolumeUSolids->SafetyFromInside(point);
    if (Dist >= UUtils::kInfinity)
      Dist = UUtils::Infinity();
    // assert(Dist == 0.);
    if (!(Dist == 0.)) {
      ReportError(&nError, point, direction, Dist, "SafetyFromInside for Surface Point should be 0.");

      fScore |= (1 << indx);
      surfPointConventionPassed &= false;
    }
  }

  return surfPointConventionPassed;
}

//Function to check conventions for Inside points
bool ShapeTester::ShapeConventionInsidePoint() {

  int nError=0;
  double Dist;

  bool insidePointConventionPassed = true;

  for (int i = 0; i < fMaxPointsInside; i++) { // test GetPointOnSurface()
    UVector3 point = fPoints[fOffsetInside + i];
    UVector3 direction = fDirections[fOffsetInside + i];
    if (fVolumeUSolids->Inside(point) != vecgeom::EInside::kInside) {
      assert(false && "For Inside point, Inside function says that the Point is not on inside");
    }

    // Convention Check for DistanceToIn
    int indx = 6;
    Dist = fVolumeUSolids->DistanceToIn(point, direction);
    if (Dist >= UUtils::kInfinity)
      Dist = UUtils::Infinity();
    // assert((Dist < 0.) && "DistanceToIn for Inside Point can never >= 0. (Wrong Side Point)");
    if (!(Dist < 0.)) {
      ReportError(&nError, point, direction, Dist, "DistanceToIn for Inside Point should be Negative (-1.) (Wrong side point)");

      fScore |= (1 << indx);
      insidePointConventionPassed &= false;
    }

    indx = 7;
    // Convention Check for DistanceToOut
    UVector3 norm(0., 0., 0.);
    bool convex = false;
    Dist = fVolumeUSolids->DistanceToOut(point, direction, norm, convex);
    if (Dist >= UUtils::kInfinity)
      Dist = UUtils::Infinity();
    // assert(Dist != UUtils::Infinity() && "DistanceToOut can never be Infinity for Inside Point.");
    if (!(Dist != UUtils::Infinity())) {
      ReportError(&nError, point, direction, Dist,
                  "DistanceToOut for Inside Point can never be Infinity");
      fScore |= (1 << indx);
      insidePointConventionPassed &= false;
    }

    indx = 8;
    // Conventions Check for SafetyFromOutside
    Dist = fVolumeUSolids->SafetyFromOutside(point);
    if (Dist >= UUtils::kInfinity)
      Dist = UUtils::Infinity();
    // assert((Dist < 0.) && "SafetyFromOutside can never be >= 0. for Inside Point (Wrong side point).");
    if (!(Dist < 0.)) {
      ReportError(&nError, point, direction, Dist,
                  "SafetyFromOutside for Inside Point should be Negative (-1.) (Wrong side point)");
      fScore |= (1 << indx);
      insidePointConventionPassed &= false;
    }

    indx = 9;
    // Conventions Check for SafetyFromInside
    Dist = fVolumeUSolids->SafetyFromInside(point);
    if (Dist >= UUtils::kInfinity)
      Dist = UUtils::Infinity();
    // assert((Dist > 0.) && "SafetyFromInside can never be <= 0. for Inside Point.");
    if (!(Dist > 0.)) {
      ReportError(&nError, point, direction, Dist,
                  "SafetyFromInside for Inside Point should be > 0.");

      fScore |= (1 << indx);
      insidePointConventionPassed &= false;
    }
  }

  return insidePointConventionPassed;
}

//Function to check conventions for outside points
bool ShapeTester::ShapeConventionOutsidePoint() {
  int nError=0;
  double Dist;

  bool outsidePointConventionPassed = true;

  for (int i = 0; i < fMaxPointsOutside; i++) { // test GetPointOnSurface()
    UVector3 point = fPoints[fOffsetOutside + i];
    UVector3 direction = fDirections[fOffsetOutside + i];
    if (fVolumeUSolids->Inside(point) != vecgeom::EInside::kOutside) {
      assert(false && "For Outside point, Inside function says that the Point is not Outside");
    }

    int indx = 10;
    // Convention Check for DistanceToIn
    Dist = fVolumeUSolids->DistanceToIn(point, direction);
    if (Dist >= UUtils::kInfinity)
      Dist = UUtils::Infinity();
    // assert((Dist > 0.) && "DistanceToIn for Outside point can never be <= 0.");
    if (!(Dist > 0.)) {
      ReportError(&nError, point, direction, Dist, "DistanceToIn for Outside Point should be > 0.");
      fScore |= (1 << indx);
      outsidePointConventionPassed &= false;
    }

    indx = 11;
    // Convention Check for DistanceToOut
    UVector3 norm(0., 0., 0.);
    bool convex = false;
    Dist = fVolumeUSolids->DistanceToOut(point, direction, norm, convex);
    if (Dist >= UUtils::kInfinity)
      Dist = UUtils::Infinity();
    // assert((Dist < 0.) && "DistanceToOut for Outside Point can never >= 0. (Wrong Side Point)");
    if (!(Dist < 0.)) {
      ReportError(&nError, point, direction, Dist, "DistanceToOut for Outside Point should be Negative (-1.) (Wrong side point)");
      fScore |= (1 << indx);
      outsidePointConventionPassed &= false;
    }

    indx = 12;
    // Conventions Check for SafetyFromOutside
    Dist = fVolumeUSolids->SafetyFromOutside(point);
    if (Dist >= UUtils::kInfinity)
      Dist = UUtils::Infinity();
    // assert((Dist > 0.) && "SafetyFromOutside can never be <= 0. for Outside Point.");
    if (!(Dist > 0.)) {
    	ReportError(&nError, point, direction, Dist,
    	                  "SafetyFromOutside for Outside Point should be > 0.");
      fScore |= (1 << indx);
      outsidePointConventionPassed &= false;
    }

    indx = 13;
    // Conventions Check for SafetyFromInside
    Dist = fVolumeUSolids->SafetyFromInside(point);
    if (Dist >= UUtils::kInfinity)
      Dist = UUtils::Infinity();
    if (Dist >= UUtils::kInfinity)
      Dist = UUtils::Infinity();
    // assert((Dist < 0.) && "SafetyFromInside for Outside point can never be >= 0.(Wrong Side Point).");
    if (!(Dist < 0.)) {
      ReportError(&nError, point, direction, Dist, "SafetyFromInside for Outside Point should be Negative (-1) (Wrong side point).");
      fScore |= (1 << indx);
      outsidePointConventionPassed &= false;
    }
  }

  return outsidePointConventionPassed;
}

//Function that will call the above three functions to do the convention check
bool ShapeTester::ShapeConventionChecker() {

  // Setting up Convention sMessages
  SetupConventionMessages();

  // Generating Points and direction for
  // Inside, Surface, Outside fPoints
  CreatePointsAndDirections();

  bool surfacePointConventionResult = ShapeConventionSurfacePoint();
  bool insidePointnConventionResult = ShapeConventionInsidePoint();
  bool outsidePointConventionResult = ShapeConventionOutsidePoint();
  std::cout << "-------------------------------------------------" << std::endl;
  std::cout << "Generated Score : " << fScore << std::endl;
  std::cout << "-------------------------------------------------" << std::endl;

  //assert(surfacePointConventionResult && insidePointnConventionResult && outsidePointConventionResult &&
    //     "Shape Conventions NOT passed");

  if (surfacePointConventionResult && insidePointnConventionResult && outsidePointConventionResult) {
    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "---------- Shape Conventions Passed -------------" << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
  }

  GenerateConventionReport();


  return true;
}

//Function to print all the conventions messages
void ShapeTester::PrintConventionMessages() {

  for (auto i : fConventionMessage)
    std::cout << i << std::endl;

}

//Functions to generate Convention Report at the end
void ShapeTester::GenerateConventionReport() {

  int n = fScore;
  int index = -1;
  if (fScore) {
	std::cout<<"\033[1;39m";
    std::cout << "---------------------------------------------------------------" << std::endl;
    std::cout << "--------- Following ShapeConventions are Not Followed ---------" << std::endl;
    std::cout << "---------------------------------------------------------------" << std::endl;
    std::cout<<"\033[0m";
    while (n > 0) {
      index++;
      if (n % 2) {
        // std::cout << index << "  ";

        std::cout << "\033[1;31m " << fConventionMessage[index] << "\033[0m" << std::endl;

      }
      n /= 2;
    }
    std::cout << "---------------------------------------------------------------" << std::endl;
    std::cout << "--- Please refer to convention document on the repository -----" << std::endl;
    std::cout << "---------------------------------------------------------------" << std::endl;
    std::cout << "-------------- Continuing Shape Tester tests ------------------" << std::endl;
    std::cout << "---------------------------------------------------------------" << std::endl;
    std::cout << std::endl;
  }
}

/* Public interface to run convention checker.
 * This interface is intentionally left public, so as to allow, if one want to call 
 * just the convention checker without the ShapeTester's tests.
 */
bool ShapeTester::RunConventionChecker(VUSolid *testVolume) {
  fVolumeUSolids = testVolume;
  ShapeConventionChecker();

  return true;
}
