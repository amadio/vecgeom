#!/bin/bash -x

export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8

THIS=$(dirname ${BASH_SOURCE[0]})

# first arguments is the source directory
if [ $# -ge 5 ]; then
  LABEL=$1 ; shift
  COMPILER=$1 ; shift
  BUILDTYPE=$1 ; shift
  EXTERNALS=$1 ; shift
  BACKEND=$1 ; shift
else
  echo "$0: expecting 4 arguments [LABEL]  [COMPILER] [BUILDTYPE] [EXTERNALS]"
  return
fi

if [ $LABEL == slc6 ] || [ $LABEL == cc7 ] || [ $LABEL == cuda7 ] || [ $LABEL == slc6-physical ] || [  $LABEL == continuous-sl6 ] || [  $LABEL == continuous-cuda7 ] || [ $LABEL == continuous-xeonphi ]
then
  export PATH=/afs/cern.ch/sw/lcg/contrib/CMake/3.3.2/Linux-x86_64/bin/:${PATH}
  kinit sftnight@CERN.CH -5 -V -k -t /ec/conf/sftnight.keytab
elif [ $LABEL == xeonphi ]
then
  export PATH=/afs/cern.ch/sw/lcg/contrib/CMake/3.3.2/Linux-x86_64/bin:${PATH}
  kinit sftnight@CERN.CH -5 -V -k -t /data/sftnight/ec/conf/sftnight.keytab
else
  export EXTERNALDIR=$HOME/ROOT-externals/
fi

if [[ $COMPILER == *gcc* ]]; then
  gcc47version=4.7
  gcc48version=4.8
  gcc49version=4.9
  COMPILERversion=${COMPILER}version
  ARCH=$(uname -m)
  if [ $LABEL == cuda7 ] || [ $LABEL == slc6-physical ] || [  $LABEL == continuous-sl6 ] || [  $LABEL == continuous-cuda7 ]; then
    . /afs/cern.ch/sw/lcg/contrib/gcc/${!COMPILERversion}/${ARCH}-slc6/setup.sh
  else
    . /afs/cern.ch/sw/lcg/contrib/gcc/${!COMPILERversion}/${ARCH}-${LABEL}/setup.sh
  fi
  export FC=gfortran
  export CXX=`which g++`
  export CC=`which gcc`
  export CMAKE_SOURCE_DIR=$WORKSPACE/VecGeom
  export CMAKE_BINARY_DIR=$WORKSPACE/VecGeom/builds
  export CMAKE_BUILD_TYPE=$BUILDTYPE
  export CMAKE_INSTALL_PREFIX=$WORKSPACE/VecGeom/installation
  export BACKEND=$BACKEND
  export CTEST_BUILD_OPTIONS="-DROOT=ON -DCTEST=ON -DBENCHMARK=ON ${ExtraCMakeOptions}"
#  export BACKEND=Vc
#  export CTEST_BUILD_OPTIONS="-DROOT=ON -DVc=ON -DCTEST=ON -DBENCHMARK=ON -DUSOLIDS=OFF ${ExtraCMakeOptions}"
elif [[ $COMPILER == *native* && $PLATFORM == *mac* ]]; then
  export LD_LIBRARY_PATH=/usr/local/gfortran/lib
  export PATH=${PATH}:/usr/local/bin:/opt/X11/bin
  export CC=`which clang`
  export CXX=`which clang++`
  export FC=`which gfortran`
elif [[ $PLATFORM == *native* ]]; then
  export CC=`which gcc`
  export CXX=`which g++`
  export FC=`which gfortran`
elif [[ $COMPILER == *icc* ]]; then
  iccyear=2013
  icc14year=2013
  icc15year=2015
  icc16year=2016
  COMPILERyear=${COMPILER}year
  iccgcc=4.9
  icc14gcc=4.9
  icc15gcc=4.9
  icc16gcc=4.9
  GCCversion=${COMPILER}gcc
  ARCH=$(uname -m)
  . /afs/cern.ch/sw/lcg/contrib/gcc/${!GCCversion}/${ARCH}-slc6/setup.sh
  . /afs/cern.ch/sw/IntelSoftware/linux/setup.sh
  . /afs/cern.ch/sw/IntelSoftware/linux/${ARCH}/xe${!COMPILERyear}/bin/ifortvars.sh intel64
  . /afs/cern.ch/sw/IntelSoftware/linux/${ARCH}/xe${!COMPILERyear}/bin/iccvars.sh intel64
  export CC=icc
  export CXX=icc
  export FC=ifort
  export CMAKE_SOURCE_DIR=$WORKSPACE/VecGeom
  export CMAKE_BINARY_DIR=$WORKSPACE/VecGeom/builds
  export CMAKE_BUILD_TYPE=$BUILDTYPE
  export CMAKE_INSTALL_PREFIX=$WORKSPACE/VecGeom/installation
  export BACKEND=$BACKEND
  export CTEST_BUILD_OPTIONS="-DROOT=ON -DCTEST=ON -DBENCHMARK=ON ${ExtraCMakeOptions}"
elif [[ $COMPILER == *clang* ]]; then
  clang34version=3.4
  clang35version=3.5
  clang36version=3.6
  clang37version=3.7
  clang38version=3.8
  COMPILERversion=${COMPILER}version
  clang34gcc=48
  clang35gcc=49
  clang37gcc=49
  clang38gcc=49
  GCCversion=${COMPILER}gcc
  . /afs/cern.ch/sw/lcg/external/llvm/${!COMPILERversion}/${ARCH}-${LABEL_COMPILER}/setup.sh
# . /cvmfs/sft.cern.ch/lcg/contrib/llvm/${!COMPILERversion}/${ARCH}-${LABEL_COMPILER}/setup.sh
  export CC=`which clang`
  export CXX=`which clang++`
  export FC=`which gfortran`
fi



echo ${THIS}/setup.py -o ${LABEL} -c ${COMPILER} -b ${BUILDTYPE} -v ${EXTERNALS}
eval `${THIS}/setup.py -o ${LABEL} -c ${COMPILER} -b ${BUILDTYPE} -v ${EXTERNALS}`
