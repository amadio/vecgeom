# a shell script which steers the process to produces a shared lib of specialized navigators
# basic usage (in build directory) run script with a ROOT geometry file and a list of logical volume names
# author: Sandro Wenzel; 26.2.2015

# the script assumes an environment variable VECGEOM_INSTALL_DIR 
# note that this script currently only serves demonstrating purposes; any sort of error handling is missing

geomfile=$1;
shift;

assemblydir="navigatorAssemblyArea";
mkdir ${assemblydir};

# rest of arguments are interpreted as list of volumes
for i in $@;do

# for each volume make an tmp dir
mkdir $i

cd $i
echo "executing ../NavigationKernelBenchmarker ${geomfile} ${i}"
cp ../NavigationKernelBenchmarker .
cp ../NavigationSpecializerTest .
cp ../${geomfile} .

./NavigationKernelBenchmarker ${geomfile} $i
./NavigationSpecializerTest ${geomfile} $i --loopunroll
cp ${i}Navigator.h ../${assemblydir}
cd ..

done

cd ${assemblydir}
../LibraryGenerator $@
mkdir build
cd build
cmake -DVecGeom_DIR=${VECGEOM_INSTALL_DIR}/lib/CMake/VecGeom ../
make
