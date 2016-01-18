/*
 * NavigationSpecializerTest.cpp
 *
 *  Created on: 11.09.2015
 *      Author: swenzel
 */
#include "base/Global.h"
#include "base/Vector3D.h"
using namespace vecgeom;
void foo( double r1, double r2, Vector3D<Precision> & __restrict__ localdir, Vector3D<Precision> & __restrict__ globaldir )
{
  localdir=Vector3D<Precision>(0.);
  localdir[0]=globaldir[0];
  localdir[1]+=globaldir[1]*r1;
  localdir[2]+=globaldir[2]*r2;
}

void bar( double r1, double r2, Vector3D<Precision> & __restrict__ localdir, Vector3D<Precision> & __restrict__ globaldir )
{
  localdir[0]=globaldir[0];
  localdir[1]=globaldir[1]*r1;
  localdir[2]=globaldir[2]*r2;
}


#include "services/NavigationSpecializer.h"
#include "management/RootGeoManager.h"
#include <iostream>
#include <fstream>
int main(int argc, char * argv[]){
    if(argc < 3){
    	std::cerr << "usage: " << argv[0] << " geometryfile.root volumename [--loopunroll]\n";
    	return 1;
    }

	vecgeom::NavigationSpecializer specializer;
    vecgeom::RootGeoManager::Instance().LoadRootGeometry(argv[1]);

    for (auto i = 1; i < argc; i++) {
      if (strcmp(argv[i], "--loopunroll") == 0)
        specializer.EnableLoopUnrolling();
    }


    for (auto i = 1; i < argc; i++) {
      if (strcmp(argv[i], "--basenav") == 0)
        specializer.SetBaseNavigator(argv[i+1]);
    }

    std::ofstream outputfile;
    outputfile.open("GeneratedNavigator.h");
    specializer.ProduceSpecializedNavigator( vecgeom::GeoManager::Instance().FindLogicalVolume(argv[2]), outputfile );
    outputfile.close();
    return 0;
}


