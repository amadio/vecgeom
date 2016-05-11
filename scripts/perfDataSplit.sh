#!/bin/sh

LOGDIR=$1/logs
DATDIR=$1/dats
PNGDIR=$1/pngs
COMMIT=$2

#.. extract performance data from log files
mkdir -p ${DATDIR}
for shape in `cat shapes.lis`
do echo Extracting perf data for $shape
   rm -rf ${DATDIR}/$shape-*.dat
   for file in ${LOGDIR}/${shape}-job*.log
   do grep "Inside:"   $file | cut -c-15,24-33 | sed 's#-.------#0.000000#' >> ${DATDIR}/$shape-inside.dat
      grep "Contains:" $file | cut -c-16,58-66 | sed 's#-.------#0.000000#' >> ${DATDIR}/$shape-contains.dat
      grep "nceToIn:"  $file | cut -c-16,31-40 >> ${DATDIR}/$shape-distToIn.dat
      grep "nceToIn:"  $file | cut -c-16,66-74 >> ${DATDIR}/$shape-safeToIn.dat
      grep "nceToOut:" $file | cut -c-15,31-40 >> ${DATDIR}/$shape-distToOut.dat
      grep "nceToOut:" $file | cut -c-16,68-76 >> ${DATDIR}/$shape-safeToOut.dat
   done

   #.. classify by algorith + implementation
   for algo in inside contains distToIn distToOut safeToIn safeToOut
   do fname=${shape}-${algo}
      grep Vector ${DATDIR}/${fname}.dat | cut -c17-24 | sed 's#-#0#g' >> ${DATDIR}/${fname}-vect.dat
      grep Specia ${DATDIR}/${fname}.dat | cut -c17-24 | sed 's#-#0#g' >> ${DATDIR}/${fname}-spec.dat
      grep Unspec ${DATDIR}/${fname}.dat | cut -c17-24 | sed 's#-#0#g' >> ${DATDIR}/${fname}-unspec.dat
      grep ROOT   ${DATDIR}/${fname}.dat | cut -c17-24 | sed 's#-#0#g' >> ${DATDIR}/${fname}-root.dat
      grep USolid ${DATDIR}/${fname}.dat | cut -c17-24 | sed 's#-#0#g' >> ${DATDIR}/${fname}-usolids.dat
      grep Geant4 ${DATDIR}/${fname}.dat | cut -c17-24 | sed 's#-#0#g' >> ${DATDIR}/${fname}-geant4.dat
      grep CUDA   ${DATDIR}/${fname}.dat | cut -c17-24 | sed 's#-#0#g' >> ${DATDIR}/${fname}-cuda.dat
   done
done

#.. write perf machine info on top of output file
outfile=$1/${COMMIT}-perf.dat
echo \# `date` > ${outfile}
echo \# `uname -a` >> ${outfile}
echo \# `cat /proc/cpuinfo | grep "model name" | uniq` >> ${outfile}
echo \# `grep repetitions $LOGDIR/*.log | cut -d' ' -f5- | uniq` >> ${outfile}

#.. save averages into shape's performance history
mkdir -p ${PNGDIR}
for shape in `cat shapes.lis`
do echo === averaging perf data for $shape
   for algo in inside contains distToIn distToOut safeToIn safeToOut
   do for impl in vect spec unspec root usolids geant4
      do dataTag=${shape}-${algo}-${impl}
	 printf " %-28s %9.6f %9.6f  %3i  %6.3f\n" \
		$(echo ${dataTag} `python ./average.py -p -f ${DATDIR}/${dataTag}.dat`) >> ${outfile}
	 test -e last.png && mv last.png ${PNGDIR}/${dataTag}.png
      done
   done
done
