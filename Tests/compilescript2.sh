g++ -O2 -fabi-version=6 -std=c++11 -I /home/swenzel/local/root_geantv/include -I ../ -I /home/swenzel/local/vc/include/ ../PhysicalBox.cpp ../GeoManager.cpp ../TransformationMatrix.cpp TransformOfVectorsTest.cpp -o foo.x -L /home/swenzel/local/vc/lib/ -lVc  -L /home/swenzel/local/root_geantv/lib/ -lGeom
