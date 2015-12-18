//
// File:    TestGenTrap.cpp
// Purpose: Unit tests for the generic trapezoid
//

#include "base/Vector3D.h"
#include "volumes/Box.h"
#include "ApproxEqual.h"

#include "volumes/GenTrap.h"

//.. ensure asserts are compiled in
#undef NDEBUG
#include <cassert>

bool testvecgeom=false;

bool TestGenTrap() {
  using Vec_t = vecgeom::Vector3D<vecgeom::Precision>;
  using namespace vecgeom;
  std::vector<Vec_t> vertexlist1;
  vertexlist1.push_back( Vec_t(-3,-3, 0 ) );
  vertexlist1.push_back( Vec_t(-3, 3, 0 ) );
  vertexlist1.push_back( Vec_t( 3, 3, 0 ) );
  vertexlist1.push_back( Vec_t( 3,-3, 0 ) );
  vertexlist1.push_back( Vec_t(-2,-2, 0 ) );
  vertexlist1.push_back( Vec_t(-2, 2, 0 ) );
  vertexlist1.push_back( Vec_t( 2, 2, 0 ) );
  vertexlist1.push_back( Vec_t( 2,-2, 0 ) );
  std::vector<Vec_t> vertexlist2;
  vertexlist2.push_back( Vec_t(-3,-3, 0 ) );
  vertexlist2.push_back( Vec_t(-3, 3, 0 ) );
  vertexlist2.push_back( Vec_t( 3, 3, 0 ) );
  vertexlist2.push_back( Vec_t( 3,-3, 0 ) );
  vertexlist2.push_back( Vec_t(-2,-1.9, 0 ) );
  vertexlist2.push_back( Vec_t(-2, 2, 0 ) );
  vertexlist2.push_back( Vec_t( 2, 2, 0 ) );
  vertexlist2.push_back( Vec_t( 2,-2, 0 ) );
  SimpleGenTrap trap1("planar_trap", &vertexlist1[0], 5 );
  SimpleGenTrap trap2("twisted_trap", &vertexlist2[0], 5 );

  Precision vol, volCheck1, volCheck2;
  Precision surf, surfCheck1, surfCheck2;
// Check cubic volume

    vol = trap1.Capacity();
    volCheck1 = (1./3)*(6*6+4*6+4*4)*5;
    std::cout << "volume= " << vol << "   volCheck= " << volCheck1 << std::endl;
 //   assert(ApproxEqual(vol,volCheck1));

    vol = trap2.Capacity();
    volCheck2 = (1./3)*(6*6+3.9*6+3.9*3.9)*5;
    std::cout << "volume= " << vol << "   volCheck= " << volCheck2 << std::endl;
    assert(vol<volCheck1 && vol>volCheck2);

// Check surface area

    surf = trap1.SurfaceArea();
    surfCheck1 = 6*6+4*4+4*0.5*(6+4)*std::sqrt(10*10+1*1);
    assert(ApproxEqual(surf,surfCheck1));

    surf = trap2.SurfaceArea();
    surfCheck2 =  6*6+4*3.9+4*0.5*(6+4)*std::sqrt(10*10+1*1);
    assert(surf<surfCheck1 && surf>surfCheck2);


// Check Inside

/*
    assert(trap1.Inside(pzero)==vecgeom::EInside::kInside);
    assert(trap1.Inside(pbigz)==vecgeom::EInside::kOutside);
    assert(trap1.Inside(ponxside)==vecgeom::EInside::kSurface);
    assert(trap1.Inside(ponyside)==vecgeom::EInside::kSurface);
    assert(trap1.Inside(ponzside)==vecgeom::EInside::kSurface);

    assert(trap2.Inside(pzero)==vecgeom::EInside::kInside);
    assert(trap2.Inside(pbigz)==vecgeom::EInside::kOutside);
    assert(trap2.Inside(ponxside)==vecgeom::EInside::kSurface);
    assert(trap2.Inside(ponyside)==vecgeom::EInside::kSurface);
    assert(trap2.Inside(ponzside)==vecgeom::EInside::kSurface);
*/

// Check Surface Normal
   
/*
    valid=trap1.Normal(ponxside,normal);
    assert(ApproxEqual(normal,Vec_t(1.,0.,0.)));
    valid=trap1.Normal(ponmxside,normal);
    assert(ApproxEqual(normal,Vec_t(-1.,0.,0.)));
    valid=trap1.Normal(ponyside,normal);
    assert(ApproxEqual(normal,Vec_t(0.,1.,0.)));
    valid=trap1.Normal(ponmyside,normal);
    assert(ApproxEqual(normal,Vec_t(0.,-1.,0.)));
    valid=trap1.Normal(ponzside,normal);
    assert(ApproxEqual(normal,Vec_t(0.,0.,1.)));
    valid=trap1.Normal(ponmzside,normal);
    assert(ApproxEqual(normal,Vec_t(0.,0.,-1.)));
    valid=trap1.Normal(ponzsidey,normal);
    assert(ApproxEqual(normal,Vec_t(0.,0.,1.)));
    valid=trap1.Normal(ponmzsidey,normal);
    assert(ApproxEqual(normal,Vec_t(0.,0.,-1.)));
*/
    // Normals on Edges
/*
    Vec_t edgeXY(    20.0,  30., 0.0); 
    Vec_t edgemXmY( -20.0, -30., 0.0); 
    Vec_t edgeXmY(   20.0, -30., 0.0); 
    Vec_t edgemXY(  -20.0,  30., 0.0); 
    Vec_t edgeXZ(    20.0, 0.0, 40.0); 
    Vec_t edgemXmZ( -20.0, 0.0, -40.0); 
    Vec_t edgeXmZ(   20.0, 0.0, -40.0); 
    Vec_t edgemXZ(  -20.0, 0.0, 40.0); 
    Vec_t edgeYZ(    0.0,  30.0,  40.0); 
    Vec_t edgemYmZ(  0.0, -30.0, -40.0); 
    Vec_t edgeYmZ(   0.0,  30.0, -40.0); 
    Vec_t edgemYZ(   0.0, -30.0,  40.0); 

    double invSqrt2 = 1.0 / std::sqrt( 2.0); 
    double invSqrt3 = 1.0 / std::sqrt( 3.0); 

    valid= trap1.Normal( edgeXY,normal ); 
    assert(ApproxEqual( normal, Vec_t( invSqrt2, invSqrt2, 0.0) )); 

    // std::cout << " Normal at " << edgeXY << " is " << normal 
    //    << " Expected is " << Vec_t( invSqrt2, invSqrt2, 0.0) << std::endl;     

    valid= trap1.Normal( edgemXmY ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt2, -invSqrt2, 0.0) )&&valid); 
    valid= trap1.Normal( edgeXmY ,normal); 
    assert(ApproxEqual( normal, Vec_t( invSqrt2, -invSqrt2, 0.0) )); 
    valid= trap1.Normal( edgemXY ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt2, invSqrt2, 0.0) )); 

    valid= trap1.Normal( edgeXZ ,normal); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt2, 0.0, invSqrt2) )); 
    valid= trap1.Normal( edgemXmZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt2, 0.0, -invSqrt2) )); 
    valid= trap1.Normal( edgeXmZ ,normal); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt2, 0.0, -invSqrt2) )); 
    valid= trap1.Normal( edgemXZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt2, 0.0, invSqrt2) )); 

    valid= trap1.Normal( edgeYZ,normal ); 
    assert(ApproxEqual( normal, Vec_t( 0.0,  invSqrt2,  invSqrt2) )); 
    valid= trap1.Normal( edgemYmZ,normal ); 
    assert(ApproxEqual( normal, Vec_t( 0.0, -invSqrt2, -invSqrt2) )); 
    valid= trap1.Normal( edgeYmZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( 0.0,  invSqrt2, -invSqrt2) )); 
    valid= trap1.Normal( edgemYZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( 0.0, -invSqrt2,  invSqrt2) )); 
*/
    // Normals on corners
/*
    Vec_t cornerXYZ(    20.0,  30., 40.0); 
    Vec_t cornermXYZ(  -20.0,  30., 40.0); 
    Vec_t cornerXmYZ(   20.0, -30., 40.0); 
    Vec_t cornermXmYZ( -20.0, -30., 40.0); 
    Vec_t cornerXYmZ(    20.0,  30., -40.0); 
    Vec_t cornermXYmZ(  -20.0,  30., -40.0); 
    Vec_t cornerXmYmZ(   20.0, -30., -40.0); 
    Vec_t cornermXmYmZ( -20.0, -30., -40.0); 
 
    valid= trap1.Normal( cornerXYZ ,normal); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt3,  invSqrt3, invSqrt3) )); 
    valid= trap1.Normal( cornermXYZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt3,  invSqrt3, invSqrt3) )); 
    valid= trap1.Normal( cornerXmYZ ,normal); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt3, -invSqrt3, invSqrt3) )); 
    valid= trap1.Normal( cornermXmYZ,normal ); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt3, -invSqrt3, invSqrt3) )); 
    valid= trap1.Normal( cornerXYmZ,normal ); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt3,  invSqrt3, -invSqrt3) )); 
    valid= trap1.Normal( cornermXYmZ,normal ); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt3,  invSqrt3, -invSqrt3) )); 
    valid= trap1.Normal( cornerXmYmZ ,normal); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt3, -invSqrt3, -invSqrt3) )); 
    valid= trap1.Normal( cornermXmYmZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt3, -invSqrt3, -invSqrt3) )); 

     
    valid=trap2.Normal(ponxside,normal);
    assert(ApproxEqual(normal,Vec_t(cosa,0,-sina)));
    valid=trap2.Normal(ponmxside,normal);
    assert(ApproxEqual(normal,Vec_t(-cosa,0,-sina)));
    valid=trap2.Normal(ponyside,normal);
    assert(ApproxEqual(normal,Vec_t(0,cosa,-sina)));
    valid=trap2.Normal(ponmyside,normal);
    assert(ApproxEqual(normal,Vec_t(0,-cosa,-sina)));
    valid=trap2.Normal(ponzside,normal);
    assert(ApproxEqual(normal,Vec_t(0,0,1)));
    valid=trap2.Normal(ponmzside,normal);
    assert(ApproxEqual(normal,Vec_t(0,0,-1)));
    valid=trap2.Normal(ponzsidey,normal);
    assert(ApproxEqual(normal,Vec_t(0,0,1)));
    valid=trap2.Normal(ponmzsidey,normal);
    assert(ApproxEqual(normal,Vec_t(0,0,-1))); // (0,cosa,-sina) ?
*/

// SafetyFromInside(P)

/*
    Dist=trap1.SafetyFromInside(pzero);
    assert(ApproxEqual(Dist,20));
    Dist=trap1.SafetyFromInside(vx);
    assert(ApproxEqual(Dist,19));
    Dist=trap1.SafetyFromInside(vy);
    assert(ApproxEqual(Dist,20));
    Dist=trap1.SafetyFromInside(vz);
    assert(ApproxEqual(Dist,20));

    Dist=trap2.SafetyFromInside(pzero);
    assert(ApproxEqual(Dist,20*cosa));
    Dist=trap2.SafetyFromInside(vx);
    assert(ApproxEqual(Dist,19*cosa));
    Dist=trap2.SafetyFromInside(vy);
    assert(ApproxEqual(Dist,20*cosa));
    Dist=trap2.SafetyFromInside(vz);
    assert(ApproxEqual(Dist,20*cosa+sina));
*/

// DistanceToOut(P,V)

/*
    Dist=trap1.DistanceToOut(pzero,vx,norm,convex);
    assert(ApproxEqual(Dist,20)&&ApproxEqual(norm,vx));
    if(!testvecgeom) assert(convex);
    Dist=trap1.DistanceToOut(pzero,vmx,norm,convex);
    assert(ApproxEqual(Dist,20)&&ApproxEqual(norm,vmx));
    if(!testvecgeom) assert(convex);
    Dist=trap1.DistanceToOut(pzero,vy,norm,convex);
    assert(ApproxEqual(Dist,30)&&ApproxEqual(norm,vy));
    if(!testvecgeom) assert(convex);
    Dist=trap1.DistanceToOut(pzero,vmy,norm,convex);
    assert(ApproxEqual(Dist,30)&&ApproxEqual(norm,vmy));
    if(!testvecgeom) assert(convex);
    Dist=trap1.DistanceToOut(pzero,vz,norm,convex);
    assert(ApproxEqual(Dist,40)&&ApproxEqual(norm,vz));
    if(!testvecgeom) assert(convex);
    Dist=trap1.DistanceToOut(pzero,vmz,norm,convex);
    assert(ApproxEqual(Dist,40)&&ApproxEqual(norm,vmz));
    if(!testvecgeom) assert(convex);
    Dist=trap1.DistanceToOut(pzero,vxy,norm,convex);
    assert(ApproxEqual(Dist,std::sqrt(800.))&&ApproxEqual(norm,vx));
    if(!testvecgeom) assert(convex);

    Dist=trap1.DistanceToOut(ponxside,vx,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vx));
    if(!testvecgeom) assert(convex);
    Dist=trap1.DistanceToOut(ponmxside,vmx,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vmx));
    if(!testvecgeom) assert(convex);
    Dist=trap1.DistanceToOut(ponyside,vy,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vy));
    if(!testvecgeom) assert(convex);
    Dist=trap1.DistanceToOut(ponmyside,vmy,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vmy));
    if(!testvecgeom) assert(convex);
    Dist=trap1.DistanceToOut(ponzside,vz,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vz));
    if(!testvecgeom) assert(convex);
    Dist=trap1.DistanceToOut(ponmzside,vmz,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vmz));
    if(!testvecgeom) assert(convex);

    Dist=trap1.DistanceToOut(ponxside,vmx,norm,convex);
    assert(ApproxEqual(Dist,40)&&ApproxEqual(norm,vmx));
    if(!testvecgeom) assert(convex);
    Dist=trap1.DistanceToOut(ponmxside,vx,norm,convex);
    assert(ApproxEqual(Dist,40)&&ApproxEqual(norm,vx));
    if(!testvecgeom) assert(convex);
    Dist=trap1.DistanceToOut(ponyside,vmy,norm,convex);
    assert(ApproxEqual(Dist,60)&&ApproxEqual(norm,vmy));
    if(!testvecgeom) assert(convex);
    Dist=trap1.DistanceToOut(ponmyside,vy,norm,convex);
    assert(ApproxEqual(Dist,60)&&ApproxEqual(norm,vy));
    if(!testvecgeom) assert(convex);
    Dist=trap1.DistanceToOut(ponzside,vmz,norm,convex);
    assert(ApproxEqual(Dist,80)&&ApproxEqual(norm,vmz));
    if(!testvecgeom) assert(convex);
    Dist=trap1.DistanceToOut(ponmzside,vz,norm,convex);
    assert(ApproxEqual(Dist,80)&&ApproxEqual(norm,vz));
    if(!testvecgeom) assert(convex);

    Dist=trap2.DistanceToOut(pzero,vx,norm,convex);
    assert(ApproxEqual(Dist,20)&&ApproxEqual(norm,Vec_t(cosa,0,-sina)));
    if(!testvecgeom) assert(convex);
    Dist=trap2.DistanceToOut(pzero,vmx,norm,convex);
    assert(ApproxEqual(Dist,20)&&ApproxEqual(norm,Vec_t(-cosa,0,-sina)));
    if(!testvecgeom) assert(convex);
    Dist=trap2.DistanceToOut(pzero,vy,norm,convex);
    assert(ApproxEqual(Dist,30)&&ApproxEqual(norm,Vec_t(0,cosa,-sina)));
    if(!testvecgeom) assert(convex);
    Dist=trap2.DistanceToOut(pzero,vmy,norm,convex);
    assert(ApproxEqual(Dist,30)&&ApproxEqual(norm,Vec_t(0,-cosa,-sina)));
    if(!testvecgeom) assert(convex);
    Dist=trap2.DistanceToOut(pzero,vz,norm,convex);
    assert(ApproxEqual(Dist,40)&&ApproxEqual(norm,vz));
    if(!testvecgeom) assert(convex);
    Dist=trap2.DistanceToOut(pzero,vmz,norm,convex);
    assert(ApproxEqual(Dist,40)&&ApproxEqual(norm,vmz));
    if(!testvecgeom) assert(convex);
    Dist=trap2.DistanceToOut(pzero,vxy,norm,convex);
    assert(ApproxEqual(Dist,std::sqrt(800.)));
    if(!testvecgeom) assert(convex);

    Dist=trap2.DistanceToOut(ponxside,vx,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,Vec_t(cosa,0,-sina)));
    if(!testvecgeom) assert(convex);
    Dist=trap2.DistanceToOut(ponmxside,vmx,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,Vec_t(-cosa,0,-sina)));
    if(!testvecgeom) assert(convex);
    Dist=trap2.DistanceToOut(ponyside,vy,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,Vec_t(0,cosa,-sina)));
    if(!testvecgeom) assert(convex);
    Dist=trap2.DistanceToOut(ponmyside,vmy,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,Vec_t(0,-cosa,-sina)));
    if(!testvecgeom) assert(convex);
    Dist=trap2.DistanceToOut(ponzside,vz,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vz));
    if(!testvecgeom) assert(convex);
    Dist=trap2.DistanceToOut(ponmzside,vmz,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vmz));
    if(!testvecgeom) assert(convex);
*/

//SafetyFromOutside(P)
    
/*
    Dist=trap1.SafetyFromOutside(pbig);
    //  std::cout<<"trap1.SafetyFromOutside(pbig) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,80));

    Dist=trap1.SafetyFromOutside(pbigx);
    assert(ApproxEqual(Dist,80));

    Dist=trap1.SafetyFromOutside(pbigmx);
    assert(ApproxEqual(Dist,80));

    Dist=trap1.SafetyFromOutside(pbigy);
    assert(ApproxEqual(Dist,70));

    Dist=trap1.SafetyFromOutside(pbigmy);
    assert(ApproxEqual(Dist,70));

    Dist=trap1.SafetyFromOutside(pbigz);
    assert(ApproxEqual(Dist,60));

    Dist=trap1.SafetyFromOutside(pbigmz);
    assert(ApproxEqual(Dist,60));

    Dist=trap2.SafetyFromOutside(pbigx);
    assert(ApproxEqual(Dist,80*cosa));
    Dist=trap2.SafetyFromOutside(pbigmx);
    assert(ApproxEqual(Dist,80*cosa));
    Dist=trap2.SafetyFromOutside(pbigy);
    assert(ApproxEqual(Dist,70*cosa));
    Dist=trap2.SafetyFromOutside(pbigmy);
    assert(ApproxEqual(Dist,70*cosa));
    Dist=trap2.SafetyFromOutside(pbigz);
    assert(ApproxEqual(Dist,60));
    Dist=trap2.SafetyFromOutside(pbigmz);
    assert(ApproxEqual(Dist,60));

    //=== add test cases to reproduce a crash in Geant4: negative SafetyFromInside() is not acceptable
    //std::cout <<"trap1.S2O(): Line "<< __LINE__ <<", p="<< testp <<", saf2out=" << Dist <<"\n";

    Vec_t testp;
    double testValue = 0.11;
    testp = ponxside + testValue*vx;
    Dist = trap1.SafetyFromOutside(testp);
    assert(ApproxEqual(Dist,testValue));
    Dist = trap1.SafetyFromInside(testp);
    assert(Dist>=0);

    testp = ponxside - testValue*vx;
    Dist=trap1.SafetyFromOutside(testp);
    std::cout <<"trap1.S2I(): Line "<< __LINE__ <<", p="<< testp <<", saf2in=" << Dist <<"\n";
    Dist=trap1.SafetyFromInside(testp);
    assert(Dist>=0);

    testp = ponmxside + 0.11*vx;
    Dist=trap1.SafetyFromOutside(testp);
    std::cout <<"trap1.S2I(): Line "<< __LINE__ <<", p="<< testp <<", saf2in=" << Dist <<"\n";
    Dist=trap1.SafetyFromInside(testp);
    assert(Dist>=0);

    testp = ponmxside - 0.11*vx;
    Dist=trap1.SafetyFromOutside(testp);
    assert(ApproxEqual(Dist,testValue));
    Dist=trap1.SafetyFromInside(testp);
    assert(Dist>=0);

    testp = ponyside + 0.11*vy;
    Dist=trap1.SafetyFromOutside(testp);
    assert(ApproxEqual(Dist,testValue));
    Dist=trap1.SafetyFromInside(testp);
    assert(Dist>=0);

    testp = ponyside - 0.11*vy;
    Dist=trap1.SafetyFromOutside(testp);
    std::cout <<"trap1.S2I(): Line "<< __LINE__ <<", p="<< testp <<", saf2in=" << Dist <<"\n";
    Dist=trap1.SafetyFromInside(testp);
    assert(Dist>=0);

    testp = ponmyside + 0.11*vy;
    Dist=trap1.SafetyFromOutside(testp);
    std::cout <<"trap1.S2I(): Line "<< __LINE__ <<", p="<< testp <<", saf2in=" << Dist <<"\n";
    Dist=trap1.SafetyFromInside(testp);
    assert(Dist>=0);

    testp = ponmyside - 0.11*vy;
    Dist=trap1.SafetyFromOutside(testp);
    assert(ApproxEqual(Dist,testValue));
    Dist=trap1.SafetyFromInside(testp);
    assert(Dist>=0);

    testp = ponzside + 0.11*vz;
    Dist=trap1.SafetyFromOutside(testp);
    assert(ApproxEqual(Dist,testValue));
    Dist=trap1.SafetyFromInside(testp);
    assert(Dist>=0);

    testp = ponzside - 0.11*vz;
    Dist=trap1.SafetyFromOutside(testp);
    std::cout <<"trap1.S2I(): Line "<< __LINE__ <<", p="<< testp <<", saf2in=" << Dist <<"\n";
    Dist=trap1.SafetyFromInside(testp);
    assert(Dist>=0);

    testp = ponmzside + 0.11*vz;
    Dist=trap1.SafetyFromOutside(testp);
    std::cout <<"trap1.S2I(): Line "<< __LINE__ <<", p="<< testp <<", saf2in=" << Dist <<"\n";
    Dist=trap1.SafetyFromInside(testp);
    assert(Dist>=0);

    testp = ponmzside - 0.11*vz;
    Dist=trap1.SafetyFromOutside(testp);
    assert(ApproxEqual(Dist,testValue));
    Dist=trap1.SafetyFromInside(testp);
    assert(Dist>=0);
*/

// DistanceToIn(P,V)
/*
    Dist=trap1.DistanceToIn(pbigx,vmx);
    assert(ApproxEqual(Dist,80));
    Dist=trap1.DistanceToIn(pbigmx,vx);
    assert(ApproxEqual(Dist,80));
    Dist=trap1.DistanceToIn(pbigy,vmy);
    assert(ApproxEqual(Dist,70));
    Dist=trap1.DistanceToIn(pbigmy,vy);
    assert(ApproxEqual(Dist,70));
    Dist=trap1.DistanceToIn(pbigz,vmz);
    assert(ApproxEqual(Dist,60));
    Dist=trap1.DistanceToIn(pbigmz,vz);
    assert(ApproxEqual(Dist,60));
    Dist=trap1.DistanceToIn(pbigx,vxy);
    assert(ApproxEqual(Dist,Constants::kInfinity));
    Dist=trap1.DistanceToIn(pbigmx,vxy);
    assert(ApproxEqual(Dist,Constants::kInfinity));

    Dist=trap2.DistanceToIn(pbigx,vmx);
    assert(ApproxEqual(Dist,80));
    Dist=trap2.DistanceToIn(pbigmx,vx);
    assert(ApproxEqual(Dist,80));
    Dist=trap2.DistanceToIn(pbigy,vmy);
    assert(ApproxEqual(Dist,70));
    Dist=trap2.DistanceToIn(pbigmy,vy);
    assert(ApproxEqual(Dist,70));
    Dist=trap2.DistanceToIn(pbigz,vmz);
    assert(ApproxEqual(Dist,60));
    Dist=trap2.DistanceToIn(pbigmz,vz);
    assert(ApproxEqual(Dist,60));
    Dist=trap2.DistanceToIn(pbigx,vxy);
    assert(ApproxEqual(Dist,Constants::kInfinity));
    Dist=trap2.DistanceToIn(pbigmx,vxy);
    assert(ApproxEqual(Dist,Constants::kInfinity));

    dist=trap3.DistanceToIn(Vec_t(50,-50,0),vy);
    assert(ApproxEqual(dist,50));

    dist=trap3.DistanceToIn(Vec_t(50,-50,0),vmy);
    assert(ApproxEqual(dist,Constants::kInfinity));

    dist=trap4.DistanceToIn(Vec_t(50,50,0),vy);
    assert(ApproxEqual(dist,Constants::kInfinity));

    dist=trap4.DistanceToIn(Vec_t(50,50,0),vmy);
    assert(ApproxEqual(dist,50));

    dist=trap1.DistanceToIn(Vec_t(0,60,0),vxmy);
    assert(ApproxEqual(dist,Constants::kInfinity));

    dist=trap1.DistanceToIn(Vec_t(0,50,0),vxmy);
    std::cout<<"trap1.DistanceToIn(Vec_t(0,50,0),vxmy) = "<<dist<<" and vxmy="<< vxmy << std::endl ;
    // assert(ApproxEqual(dist,sqrt(800.)));  // A bug in UTrap!!!  Just keep printout above as a reminder

    dist=trap1.DistanceToIn(Vec_t(0,40,0),vxmy);
    assert(ApproxEqual(dist,10.0*std::sqrt(2.0)));

    dist=trap1.DistanceToIn(Vec_t(0,40,50),vxmy);
    assert(ApproxEqual(dist,Constants::kInfinity));

    // Parallel to side planes

    dist=trap1.DistanceToIn(Vec_t(40,60,0),vmx);
    assert(ApproxEqual(dist,Constants::kInfinity));

    dist=trap1.DistanceToIn(Vec_t(40,60,0),vmy);
    assert(ApproxEqual(dist,Constants::kInfinity));

    dist=trap1.DistanceToIn(Vec_t(40,60,50),vmz);
    assert(ApproxEqual(dist,Constants::kInfinity));

    dist=trap1.DistanceToIn(Vec_t(0,0,50),vymz);
    assert(ApproxEqual(dist,10.0*std::sqrt(2.0)));

    dist=trap1.DistanceToIn(Vec_t(0,0,80),vymz);
    assert(ApproxEqual(dist,Constants::kInfinity));

    dist=trap1.DistanceToIn(Vec_t(0,0,70),vymz);
    std::cout<<"trap1.DistanceToIn(Vec_t(0,0,70),vymz) = "<<dist<<", vymz="<< vymz << std::endl ;
    //assert(ApproxEqual(dist,30.0*sqrt(2.0)));  // A bug in UTrap!!!  Just keep printout above as a reminder
*/
// CalculateExtent
/*     
   Vec_t minExtent,maxExtent;
   trap1.Extent(minExtent,maxExtent);
   assert(ApproxEqual(minExtent,Vec_t(-20,-30,-40)));
   assert(ApproxEqual(maxExtent,Vec_t( 20, 30, 40)));
   trap2.Extent(minExtent,maxExtent);
   assert(ApproxEqual(minExtent,Vec_t(-30,-40,-40)));
   assert(ApproxEqual(maxExtent,Vec_t( 30, 40, 40)));
*/   
   return true;
}

int main(int argc, char *argv[]) {
  TestGenTrap();
  std::cout << "UTrap --> VecGeom trap passed\n";
  return 0;
}
