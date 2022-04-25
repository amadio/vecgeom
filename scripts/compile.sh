#!/bin/bash
# store build in some random place
rm -rf build
mkdir build
cd build
cmake ../../ -DVECGEOM_GEANT4=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release -DVECGEOM_BACKEND=Vc -DVECGEOM_TEST_BENCHMARK=ON -DVECGEOM_ROOT=ON -DVECGEOM_VECTOR=avx -DVECTOR=avx -DVc_DIR=/home/swenzel/local/vc0.8/lib/cmake/Vc
make -j 8
cd ../../
