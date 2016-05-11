#!/bin/bash

BINDIR=$1
COMMIT=$2
LOGDIR=$BINDIR/$COMMIT/logs
echo Logfiles saved in ${LOGDIR}

mkdir -p ${LOGDIR}

#.. Run benchmark jobs -- for now: 21 jobs per shape
for i in `seq 0 20`
do for shape in `cat shapes.lis`
   do echo Running ${shape}-job${i}
      ${shape}Benchmark -npoints 131172 -nrep 1  &> ${LOGDIR}/${shape}-job${i}.log
   done
done

#.. savings: remove unneeded printouts of mismatches from logfiles
for i in ${LOGDIR}/*-job*.log; do grep -v Point $i > x.x; mv x.x $i; done
