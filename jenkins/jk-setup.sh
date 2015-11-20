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

if [ $LABEL == slc6 ] || [ $LABEL == cc7 ] || [ $LABEL == cuda7 ] || [ $LABEL == xeonphi ] || [ $LABEL == slc6-physical ]
then
  kinit sftnight@CERN.CH -5 -V -k -t /ec/conf/sftnight.keytab
  export PATH=/afs/cern.ch/sw/lcg/contrib/CMake/3.0.0/Linux-i386/bin:${PATH}
else
  export EXTERNALDIR=$HOME/ROOT-externals/
fi

if [[ $COMPILER == *gcc* ]]
then
  gcc47version=4.7
  gcc48version=4.8
  gcc49version=4.9
  COMPILERversion=${COMPILER}version

  ARCH=$(uname -m)

  if [ $LABEL == cuda7 ] || [ $LABEL == slc6-physical ]
  then
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
fi

if [[ $COMPILER == *icc* ]]; then

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
fi



echo ${THIS}/setup.py -o ${LABEL} -c ${COMPILER} -b ${BUILDTYPE} -v ${EXTERNALS}
eval `${THIS}/setup.py -o ${LABEL} -c ${COMPILER} -b ${BUILDTYPE} -v ${EXTERNALS}`
