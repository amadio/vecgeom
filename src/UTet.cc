//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the  intellectual property  of the *
// * Vanderbilt University Free Electron Laser Center                 *
// * Vanderbilt University, Nashville, TN, USA                        *
// * Development supported by:                                        *
// * United States MFEL program  under grant FA9550-04-1-0045         *
// * and NASA under contract number NNG04CT05P                        *
// * Written by Marcus H. Mendenhall and Robert A. Weller.            *
// *                                                                  *
// * Contributed to the Geant4 Core, January, 2005.                   *
// *                                                                  *
// ********************************************************************
//
// $Id: UTet.cc 66356 2012-12-18 09:02:32Z gcosmo $
//
// class UTet
//
// Implementation for UTet class
//
// History:
//
//  20040903 - Marcus Mendenhall, created G4Tet
//  20041101 - Marcus Mendenhall, optimized constant dot products with
//             fCdotNijk values
//  20041101 - MHM removed tracking error by clipping DistanceToOut to 0
//             for surface cases
//  20041101 - MHM many speed optimizations in if statements
//  20041101 - MHM changed vdotn comparisons to 1e-12 instead of 0.0 to
//             avoid nearly-parallel problems
//  20041102 - MHM Added extra distance into solid to DistanceToIn(p,v)
//             hit testing
//  20041102 - MHM added ability to check for degeneracy without throwing
//             G4Exception
//  20041103 - MHM removed many unused variables from class
//  20040803 - Dionysios Anninos, added GetPointOnSurface() method
//  20061112 - MHM added code for G4VSolid GetSurfaceArea()
//  20100920 - Gabriele Cosmo added copy-ctor and operator=()
//  20120308 - Tatiana Nikitina change from G4Tet to UTet
//
// --------------------------------------------------------------------


//const char UTet::CVSVers[]="$Id: G4Tet.cc 66356 2012-12-18 09:02:32Z gcosmo $";
#include <cmath>
#include <iostream>
#include <sstream>
#include "UTet.hh"
#include "UUtils.hh"

using namespace std;


////////////////////////////////////////////////////////////////////////
//
// Constructor - create a tetrahedron
// This class is implemented separately from general polyhedra,
// because the simplex geometry can be computed very quickly,
// which may become important in situations imported from mesh generators,
// in which a very large number of G4Tets are created.
// A Tet has all of its geometrical information precomputed

UTet::UTet(const std::string& name,
                   UVector3& anchor,
                   UVector3& p2,
                   UVector3& p3,
                   UVector3& p4, bool *degeneracyFlag)
  : VUSolid(name), fpPolyhedron(0), warningFlag(0)
{
  // fV<x><y> is vector from vertex <y> to vertex <x>
  //
  UVector3 fV21=p2-anchor;
  UVector3 fV31=p3-anchor;
  UVector3 fV41=p4-anchor;

  // make sure this is a correctly oriented set of points for the tetrahedron
  //
  double signed_vol=fV21.Cross(fV31).Dot(fV41);
  if(signed_vol<0.0)
  {
    UVector3 temp(p4);
    p4=p3;
    p3=temp;
    temp=fV41;
    fV41=fV31;
    fV31=temp; 
  }
  fCubicVolume = std::fabs(signed_vol) / 6.;

  UVector3 fV24=p2-p4;
  UVector3 fV43=p4-p3;
  UVector3 fV32=p3-p2;

  fXMin=std::min(std::min(std::min(anchor.x, p2.x),p3.x),p4.x);
  fXMax=std::max(std::max(std::max(anchor.x, p2.x),p3.x),p4.x);
  fYMin=std::min(std::min(std::min(anchor.y, p2.y),p3.y),p4.y);
  fYMax=std::max(std::max(std::max(anchor.y, p2.y),p3.y),p4.y);
  fZMin=std::min(std::min(std::min(anchor.z, p2.z),p3.z),p4.z);
  fZMax=std::max(std::max(std::max(anchor.z, p2.z),p3.z),p4.z);

  fDx=(fXMax-fXMin)*0.5; fDy=(fYMax-fYMin)*0.5; fDz=(fZMax-fZMin)*0.5;

  fMiddle=UVector3(fXMax+fXMin, fYMax+fYMin, fZMax+fZMin)*0.5;
  fMaxSize=std::max(std::max(std::max((anchor-fMiddle).Mag(),
                                      (p2-fMiddle).Mag()),
                             (p3-fMiddle).Mag()),
                    (p4-fMiddle).Mag());

  bool degenerate=std::fabs(signed_vol) < 1e-9*fMaxSize*fMaxSize*fMaxSize;

  if(degeneracyFlag) *degeneracyFlag=degenerate;
  else if (degenerate)
  {
    UUtils::Exception("UTet::UTet()", "GeomSolids0002", FatalErrorInArguments,1,
              "Degenerate tetrahedron not allowed.");
  }

  fTol=1e-9*(std::fabs(fXMin)+std::fabs(fXMax)+std::fabs(fYMin)
            +std::fabs(fYMax)+std::fabs(fZMin)+std::fabs(fZMax));
  //fTol=kCarTolerance;

  fAnchor=anchor;
  fP2=p2;
  fP3=p3;
  fP4=p4;

  UVector3 fCenter123=(anchor+p2+p3)*(1.0/3.0); // face center
  UVector3 fCenter134=(anchor+p4+p3)*(1.0/3.0);
  UVector3 fCenter142=(anchor+p4+p2)*(1.0/3.0);
  UVector3 fCenter234=(p2+p3+p4)*(1.0/3.0);

  // compute area of each triangular face by cross product
  // and sum for total surface area

  UVector3 normal123=fV31.Cross(fV21);
  UVector3 normal134=fV41.Cross(fV31);
  UVector3 normal142=fV21.Cross(fV41);
  UVector3 normal234=fV32.Cross(fV43);

  fSurfaceArea=(
      normal123.Mag()+
      normal134.Mag()+
      normal142.Mag()+
      normal234.Mag()
  )/2.0;

  fNormal123=normal123.Unit();
  fNormal134=normal134.Unit();
  fNormal142=normal142.Unit();
  fNormal234=normal234.Unit();

  fCdotN123=fCenter123.Dot(fNormal123);
  fCdotN134=fCenter134.Dot(fNormal134);
  fCdotN142=fCenter142.Dot(fNormal142);
  fCdotN234=fCenter234.Dot(fNormal234);
 }


//////////////////////////////////////////////////////////////////////////
//
// Destructor

UTet::~UTet()
{
  delete fpPolyhedron;
}

///////////////////////////////////////////////////////////////////////////////
//
// Copy constructor

UTet::UTet(const UTet& rhs)
  : VUSolid(rhs),
    fCubicVolume(rhs.fCubicVolume), fSurfaceArea(rhs.fSurfaceArea),
    fpPolyhedron(0), fAnchor(rhs.fAnchor),
    fP2(rhs.fP2), fP3(rhs.fP3), fP4(rhs.fP4), fMiddle(rhs.fMiddle),
    fNormal123(rhs.fNormal123), fNormal142(rhs.fNormal142),
    fNormal134(rhs.fNormal134), fNormal234(rhs.fNormal234),
    warningFlag(rhs.warningFlag), fCdotN123(rhs.fCdotN123),
    fCdotN142(rhs.fCdotN142), fCdotN134(rhs.fCdotN134),
    fCdotN234(rhs.fCdotN234), fXMin(rhs.fXMin), fXMax(rhs.fXMax),
    fYMin(rhs.fYMin), fYMax(rhs.fYMax), fZMin(rhs.fZMin), fZMax(rhs.fZMax),
    fDx(rhs.fDx), fDy(rhs.fDy), fDz(rhs.fDz), fTol(rhs.fTol),
    fMaxSize(rhs.fMaxSize)
{
}


///////////////////////////////////////////////////////////////////////////////
//
// Assignment operator

UTet& UTet::operator = (const UTet& rhs) 
{
   // Check assignment to self
   //
   if (this == &rhs)  { return *this; }

   // Copy base class data
   //
   VUSolid::operator=(rhs);

   // Copy data
   //
   fCubicVolume = rhs.fCubicVolume; fSurfaceArea = rhs.fSurfaceArea;
   fpPolyhedron = 0; fAnchor = rhs.fAnchor;
   fP2 = rhs.fP2; fP3 = rhs.fP3; fP4 = rhs.fP4; fMiddle = rhs.fMiddle;
   fNormal123 = rhs.fNormal123; fNormal142 = rhs.fNormal142;
   fNormal134 = rhs.fNormal134; fNormal234 = rhs.fNormal234;
   warningFlag = rhs.warningFlag; fCdotN123 = rhs.fCdotN123;
   fCdotN142 = rhs.fCdotN142; fCdotN134 = rhs.fCdotN134;
   fCdotN234 = rhs.fCdotN234; fXMin = rhs.fXMin; fXMax = rhs.fXMax;
   fYMin = rhs.fYMin; fYMax = rhs.fYMax; fZMin = rhs.fZMin; fZMax = rhs.fZMax;
   fDx = rhs.fDx; fDy = rhs.fDy; fDz = rhs.fDz; fTol = rhs.fTol;
   fMaxSize = rhs.fMaxSize;

   return *this;
}

//////////////////////////////////////////////////////////////////////////
//
// CheckDegeneracy

/*bool UTet::CheckDegeneracy( UVector3 anchor,
                               UVector3 p2,
                               UVector3 p3,
                               UVector3 p4 )
{
  bool result;
  UTet *object=new UTet("temp",anchor,p2,p3,p4,&result);
  delete object;
  return result;
}
*/

/////////////////////////////////////////////////////////////////////////
//
// Return whether point inside/outside/on surface, using tolerance

VUSolid::EnumInside UTet::Inside(const UVector3 &p) const
{
  double r123, r134, r142, r234;

  // this is written to allow if-statement truncation so the outside test
  // (where most of the world is) can fail very quickly and efficiently

  if ( (r123=p.Dot(fNormal123)-fCdotN123) > fTol ||
       (r134=p.Dot(fNormal134)-fCdotN134) > fTol ||
       (r142=p.Dot(fNormal142)-fCdotN142) > fTol ||
       (r234=p.Dot(fNormal234)-fCdotN234) > fTol )
  {
    return eOutside; // at least one is out!
  }
  else if( (r123 < -fTol)&&(r134 < -fTol)&&(r142 < -fTol)&&(r234 < -fTol) )
  {
    return eInside; // all are definitively inside
  }
  else
  {
    return eSurface; // too close to tell
  }
}

///////////////////////////////////////////////////////////////////////
//
// Calculate side nearest to p, and return normal
// If two sides are equidistant, normal of first side (x/y/z) 
// encountered returned.
// This assumes that we are looking from the inside!
bool UTet::Normal( const UVector3& p, UVector3 &n) const
{
  double r123=std::fabs(p.Dot(fNormal123)-fCdotN123);
  double r134=std::fabs(p.Dot(fNormal134)-fCdotN134);
  double r142=std::fabs(p.Dot(fNormal142)-fCdotN142);
  double r234=std::fabs(p.Dot(fNormal234)-fCdotN234);

   static const double delta = 0.5*fTol;
  UVector3 sumnorm(0., 0., 0.);
  int noSurfaces=0; 

  if (r123 <= delta)         
  {
     noSurfaces ++; 
     sumnorm= fNormal123; 
  }

  if (r134 <= delta)    
  {
     noSurfaces ++; 
     sumnorm += fNormal134; 
  }
 
  if (r142 <= delta)    
  {
     noSurfaces ++; 
     sumnorm += fNormal142;
  }
  if (r234 <= delta)    
  {
     noSurfaces ++; 
     sumnorm += fNormal234;
  }
  
  if( noSurfaces > 0 )
  { 
     if( noSurfaces == 1 )
     { 
       n=sumnorm;return true; 
     }
     else
     {
       n=sumnorm.Unit();return true;
     }
  }
  else // Approximative Surface Normal
  {

    if( (r123<=r134) && (r123<=r142) && (r123<=r234) ) { n = fNormal123; }
    else if ( (r134<=r142) && (r134<=r234) )           { n = fNormal134; }
    else if (r142 <= r234)                             { n = fNormal142; }
    n = fNormal234;
    return false;
  }

  
}

///////////////////////////////////////////////////////////////////////////
//
// Calculate distance to box from an outside point
// - return kInfinity if no intersection.
// All this is very unrolled, for speed.

double UTet::DistanceToIn(const UVector3& p,
			  const UVector3& v, double /*aPstep*/ ) const
{
    UVector3 vu(v.Unit()), hp;
    double vdotn, t, tmin=UUtils::kInfinity;

    double extraDistance=10.0*fTol; // a little ways into the solid

    vdotn=-vu.Dot(fNormal123);
    if(vdotn > 1e-12)
    { // this is a candidate face, since it is pointing at us
      t=(p.Dot(fNormal123)-fCdotN123)/vdotn; // #  distance to intersection
      if( (t>=-fTol) && (t<tmin) )
      { // if not true, we're going away from this face or it's not close
        hp=p+vu*(t+extraDistance); // a little beyond point of intersection
        if ( ( hp.Dot(fNormal134)-fCdotN134 < 0.0 ) &&
             ( hp.Dot(fNormal142)-fCdotN142 < 0.0 ) &&
             ( hp.Dot(fNormal234)-fCdotN234 < 0.0 ) )
        {
          tmin=t;
        }
      }
    }

    vdotn=-vu.Dot(fNormal134);
    if(vdotn > 1e-12)
    { // # this is a candidate face, since it is pointing at us
      t=(p.Dot(fNormal134)-fCdotN134)/vdotn; // #  distance to intersection
      if( (t>=-fTol) && (t<tmin) )
      { // if not true, we're going away from this face
        hp=p+vu*(t+extraDistance); // a little beyond point of intersection
        if ( ( hp.Dot(fNormal123)-fCdotN123 < 0.0 ) && 
             ( hp.Dot(fNormal142)-fCdotN142 < 0.0 ) &&
             ( hp.Dot(fNormal234)-fCdotN234 < 0.0 ) )
        {
          tmin=t;
        }
      }
    }

    vdotn=-vu.Dot(fNormal142);
    if(vdotn > 1e-12)
    { // # this is a candidate face, since it is pointing at us
      t=(p.Dot(fNormal142)-fCdotN142)/vdotn; // #  distance to intersection
      if( (t>=-fTol) && (t<tmin) )
      { // if not true, we're going away from this face
        hp=p+vu*(t+extraDistance); // a little beyond point of intersection
        if ( ( hp.Dot(fNormal123)-fCdotN123 < 0.0 ) &&
             ( hp.Dot(fNormal134)-fCdotN134 < 0.0 ) &&
             ( hp.Dot(fNormal234)-fCdotN234 < 0.0 ) )
        {
          tmin=t;
        }
      }
    }

    vdotn=-vu.Dot(fNormal234);
    if(vdotn > 1e-12)
    { // # this is a candidate face, since it is pointing at us
      t=(p.Dot(fNormal234)-fCdotN234)/vdotn; // #  distance to intersection
      if( (t>=-fTol) && (t<tmin) )
      { // if not true, we're going away from this face
        hp=p+vu*(t+extraDistance); // a little beyond point of intersection
        if ( ( hp.Dot(fNormal123)-fCdotN123 < 0.0 ) &&
             ( hp.Dot(fNormal134)-fCdotN134 < 0.0 ) &&
             ( hp.Dot(fNormal142)-fCdotN142 < 0.0 ) )
        {
          tmin=t;
        }
      }
    }

  return std::max(0.0,tmin);
}

//////////////////////////////////////////////////////////////////////////
// 
// Approximate distance to tet.
// returns distance to sphere centered on bounding box
// - If inside return 0
double UTet::SafetyFromOutside ( const UVector3 &p, bool /*aAccurate*/) const

{
  double dd=(p-fMiddle).Mag() - fMaxSize - fTol;
  return std::max(0.0, dd);
}

/////////////////////////////////////////////////////////////////////////
//
// Calcluate distance to surface of box from inside
// by calculating distances to box's x/y/z planes.
// Smallest distance is exact distance to exiting.
double UTet::DistanceToOut( const UVector3  &p, const UVector3 &v,
	UVector3 &n, bool &convex, double /*aPstep*/) const
{
    UVector3 vu(v.Unit());
    double t1=UUtils::kInfinity,t2=UUtils::kInfinity,t3=UUtils::kInfinity,t4=UUtils::kInfinity, vdotn, tt;

    vdotn=vu.Dot(fNormal123);
    if(vdotn > 1e-12)  // #we're heading towards this face, so it is a candidate
    {
      t1=(fCdotN123-p.Dot(fNormal123))/vdotn; // #  distance to intersection
    }

    vdotn=vu.Dot(fNormal134);
    if(vdotn > 1e-12) // #we're heading towards this face, so it is a candidate
    {
      t2=(fCdotN134-p.Dot(fNormal134))/vdotn; // #  distance to intersection
    }

    vdotn=vu.Dot(fNormal142);
    if(vdotn > 1e-12) // #we're heading towards this face, so it is a candidate
    {
      t3=(fCdotN142-p.Dot(fNormal142))/vdotn; // #  distance to intersection
    }

    vdotn=vu.Dot(fNormal234);
    if(vdotn > 1e-12) // #we're heading towards this face, so it is a candidate
    {
      t4=(fCdotN234-p.Dot(fNormal234))/vdotn; // #  distance to intersection
    }

    tt=std::min(std::min(std::min(t1,t2),t3),t4);

    if (warningFlag && (tt == UUtils::kInfinity || tt < -fTol))
    {
      // DumpInfo();
       std::ostringstream message;
       message << "No good intersection found or already outside!?" << std::endl
	       << "p = " << p  << std::endl
	       << "v = " << v  << std::endl
              << "t1, t2, t3, t4  "
              << t1 << ", " << t2 << ", " << t3 << ", " << t4;
      
      UUtils::Exception("UTet::DistanceToOut(p,v,...)", "GeomSolids1002",
		  Warning,1, message.str().c_str());
      if(convex)
      {
        convex=false; // flag normal as meaningless
      }
    }
    else 
    {
      static UVector3 normal;
      if(tt==t1)        { normal=fNormal123; }
      else if (tt==t2)  { normal=fNormal134; }
      else if (tt==t3)  { normal=fNormal142; }
      else if (tt==t4)  { normal=fNormal234; }
      n=normal;
      if(convex) { convex=true; }
    }

    return std::max(tt,0.0); // avoid tt<0.0 by a tiny bit
                             // if we are right on a face
}

////////////////////////////////////////////////////////////////////////////
//
// Calculate exact shortest distance to any boundary from inside
// - If outside return 0
double UTet::SafetyFromInside ( const UVector3 &p, bool /*aAccurate*/) const
{
  double t1,t2,t3,t4;
  t1=fCdotN123-p.Dot(fNormal123); //  distance to plane, positive if inside
  t2=fCdotN134-p.Dot(fNormal134); //  distance to plane
  t3=fCdotN142-p.Dot(fNormal142); //  distance to plane
  t4=fCdotN234-p.Dot(fNormal234); //  distance to plane

  // if any one of these is negative, we are outside,
  // so return zero in that case

  double tmin=std::min(std::min(std::min(t1,t2),t3),t4);
  return (tmin < fTol)? 0:tmin;
}


//////////////////////////////////////////////////////////////////////////
//
// Stream object contents to an output stream

std::ostream& UTet::StreamInfo(std::ostream& os) const
{
  int oldprc = os.precision(16);
  os << "-----------------------------------------------------------\n"
  << "    *** Dump for solid - " << GetName() << " ***\n"
  << "    ===================================================\n"
  << " Solid type: UTet\n"
  << " Parameters: \n"
  << "    anchor: " << fAnchor << "  \n"
  << "    p2: " << fP2 << "  \n"
  << "    p3: " << fP3 << "  \n"
  << "    p4: " << fP4 << "  \n"
  << "    normal123: " << fNormal123 << " \n"
  << "    normal134: " << fNormal134 << " \n"
  << "    normal142: " << fNormal142 << " \n"
  << "    normal234: " << fNormal234 << " \n"
  << "-----------------------------------------------------------\n";
  os.precision(oldprc);

  return os;
}


////////////////////////////////////////////////////////////////////////
//
// GetPointOnFace
//
// Auxiliary method for get point on surface

UVector3 UTet::GetPointOnFace(UVector3 p1, UVector3 p2,
                                    UVector3 p3, double& area) const
{
  double lambda1,lambda2;
  UVector3 v, w;

  v = p3 - p1;
  w = p1 - p2;

  lambda1 = UUtils::Random(0.,1.);
  lambda2 = UUtils::Random(0.,lambda1);

  area = 0.5*(v.Cross(w)).Mag();
  return (p2 + lambda1*w + lambda2*v);
}

////////////////////////////////////////////////////////////////////////////
//
// GetPointOnSurface

UVector3 UTet::GetPointOnSurface() const
{
  double chose,aOne,aTwo,aThree,aFour;
  UVector3 p1, p2, p3, p4;
  
  p1 = GetPointOnFace(fAnchor,fP2,fP3,aOne);
  p2 = GetPointOnFace(fAnchor,fP4,fP3,aTwo);
  p3 = GetPointOnFace(fAnchor,fP4,fP2,aThree);
  p4 = GetPointOnFace(fP4,fP3,fP2,aFour);
  
  chose = UUtils::Random(0.,aOne+aTwo+aThree+aFour);
  if( (chose>=0.) && (chose <aOne) ) {return p1;}
  else if( (chose>=aOne) && (chose < aOne+aTwo) ) {return p2;}
  else if( (chose>=aOne+aTwo) && (chose<aOne+aTwo+aThree) ) {return p3;}
  return p4;
}

////////////////////////////////////////////////////////////////////////
//
// GetVertices

std::vector<UVector3> UTet::GetVertices() const 
{
  std::vector<UVector3> vertices(4);
  vertices[0] = fAnchor;
  vertices[1] = fP2;
  vertices[2] = fP3;
  vertices[3] = fP4;

  return vertices;
}


////////////////////////////////////////////////////////////////////////
//
// CreatePolyhedron

UPolyhedron* UTet::CreatePolyhedron () const 
{
  UPolyhedron *ph=new UPolyhedron;
  double xyz[4][3];
  static int faces[4][4]={{1,3,2,0},{1,4,3,0},{1,2,4,0},{2,3,4,0}};
  xyz[0][0]=fAnchor.x; xyz[0][1]=fAnchor.y; xyz[0][2]=fAnchor.z;
  xyz[1][0]=fP2.x; xyz[1][1]=fP2.y; xyz[1][2]=fP2.z;
  xyz[2][0]=fP3.x; xyz[2][1]=fP3.y; xyz[2][2]=fP3.z;
  xyz[3][0]=fP4.x; xyz[3][1]=fP4.y; xyz[3][2]=fP4.z;

  ph->createPolyhedron(4,4,xyz,faces);

  return ph;
}
//______________________________________________________________________________
void UTet::Extent ( UVector3 &aMin, UVector3 &aMax) const
{
	// Returns the full 3D cartesian extent of the solid.
	aMin.x = -fDx; aMax.x = fDx;
	aMin.y = -fDy; aMax.y = fDy;
	aMin.z = -fDz; aMax.z = fDz;
}



