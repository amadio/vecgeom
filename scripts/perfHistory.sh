#!/bin/bash -xv
#
# File: perfHistory.sh Purpose: store performance data for long-term
# comparisons, referenced by commit SHA-1.
#
# Note: all *Benchmark executables available in the local directory
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
SRCPWD=`pwd`
COMMIT=`git log | head -1 | cut -c8-15`
echo \$COMMIT=$COMMIT
cd ${CURPWD}

#.. Prepare output directories
rm -rf logs dats pngs
mkdir -p logs dats pngs

#.. Run benchmark jobs -- final version: 21 jobs per shape
echo Running performance jobs for commit ${COMMIT}...
rm -f shapes.lis
touch shapes.lis
for j in *Benchmark
do jshort=`echo $j | sed 's#Benchmark##'`
   echo ${jshort} >> shapes.lis
   for i in `seq 0 20`
   do echo Running $j-job$i
      ./$j -npoints 131072 -nrep 1  &> logs/$jshort-job${i}.log
   done
done

#.. savings: remove unneeded printouts of mismatches from logfiles
for i in logs/*-job*.log; do grep -v Point $i > x.x; /bin/mv x.x $i; done

#.. extract performance data from log files
mkdir dats
for shape in `cat shapes.lis`
do echo Extracting perf data for $shape
   rm -rf dats/$shape-*.dat
   for file in logs/${shape}-job*.log
   do grep "Inside:"   $file | cut -c-15,24-33 | sed 's#-.------#0.000000#' >> dats/$shape-inside.dat
      grep "Contains:" $file | cut -c-16,58-66 | sed 's#-.------#0.000000#' >> dats/$shape-contains.dat
      grep "nceToIn:"  $file | cut -c-16,31-40 >> dats/$shape-distToIn.dat
      grep "nceToIn:"  $file | cut -c-16,66-74 >> dats/$shape-safeToIn.dat
      grep "nceToOut:" $file | cut -c-15,31-40 >> dats/$shape-distToOut.dat
      grep "nceToOut:" $file | cut -c-16,68-76 >> dats/$shape-safeToOut.dat
   done

   #.. classify by algorith + implementation
   for algo in inside contains distToIn distToOut safeToIn safeToOut
   do grep Vector dats/$shape-$algo.dat | cut -c17-24 | sed 's#-#0#g' >> dats/$shape-$algo-vect.dat
      grep Specia dats/$shape-$algo.dat | cut -c17-24 | sed 's#-#0#g' >> dats/$shape-$algo-spec.dat
      grep Unspec dats/$shape-$algo.dat | cut -c17-24 | sed 's#-#0#g' >> dats/$shape-$algo-unspec.dat
      grep ROOT   dats/$shape-$algo.dat | cut -c17-24 | sed 's#-#0#g' >> dats/$shape-$algo-root.dat
      grep USolid dats/$shape-$algo.dat | cut -c17-24 | sed 's#-#0#g' >> dats/$shape-$algo-usolids.dat
      grep Geant4 dats/$shape-$algo.dat | cut -c17-24 | sed 's#-#0#g' >> dats/$shape-$algo-geant4.dat
      grep CUDA   dats/$shape-$algo.dat | cut -c17-24 | sed 's#-#0#g' >> dats/$shape-$algo-cuda.dat
   done
done

#.. calculate average performances and save results on a single file
cd dats
ln -sf $SRCPWD/scripts/average.py .

rm -rf $COMMIT-perfs.dat
for shape in `cat ../shapes.lis`
do echo === averaging perf data for $shape
   for algo in inside contains distToIn distToOut safeToIn safeToOut
   do for impl in vect spec unspec root usolids geant4
      do fname=$shape-$algo-$impl
	 echo `./average.py -p -f $fname.dat` ${fname} >> $COMMIT-perfs.dat
	 test -e last.png && mv last.png $fname.png
      done
   done
done

mv *.png ../pngs/
mv ${COMMIT}-perfs.dat ..

#.. cleanup
cd ..
rm -rf ${COMMIT}
mkdir ${COMMIT}
mv logs/ dats/ pngs/ ${COMMIT}

rm ${COMMIT}/logs/*.log
#rm ${COMMIT}/dats/*.dat
