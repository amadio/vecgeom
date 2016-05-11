#!/bin/bash -xv
#
# File: perfHistory.sh Purpose: store performance data for long-term
# comparisons, referenced by commit SHA-1.
#
# Notes:
# * always check for new versions of this script on HEAD of master!
# * all *Benchmark executables available in the local directory
#   will be executed multiple times for performance measurements.
#
# Usage:
#   cd <path-to-benchmark-executables>
#   source ../scripts/perfHistory.sh
#

#.. Safety check -- make sure we are in right place
CACHECHECK=`ls ./CMakeCache.txt`
if [ ${#CACHECHECK} == 0 ]; then echo Error: wrong directory.  Please run from build area.; return 0; fi

BINARIES=`ls ./*Benchmark`
if [ ${#BINARIES} == 0 ]; then echo Error: no benchmarking binaries found.; return 0; fi

#.. get commit ID to be used as reference string
CURPWD=`pwd`
cd `grep VecGeom_SOURCE CMakeCache.txt | cut -d'=' -f2`
SRCDIR=`pwd`
COMMIT=`git log | head -1 | cut -c8-15`
cd ${CURPWD}

#.. list of binaries to run
ls *Benchmark | sed 's#Benchmark##' > shapes.lis

#.. Run jobs to generate perf data
bash ${SRCDIR}/scripts/perfDataGenerate.sh `pwd` $COMMIT

#.. Split perf data into separate files per shape/algo/implementation,
#   calculate averages per shape/algo/implementation
#   and save results into a single file named after this commit
ln -sf ${SRCDIR}/scripts/average.py .
bash ${SRCDIR}/scripts/perfDataSplit.sh `pwd`/$COMMIT $COMMIT

