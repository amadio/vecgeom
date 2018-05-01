
Performance history
===================

Performance scripts can be used to document the evolution of performance of shape algorithms.

Instructions for the impatient
______________________________

Maybe you want to try the script/perfHistory.sh script and provide feedback?

First run it from the area where your Benchmark binaries are build. Remove all *Benchmark binaries you don't want to
run (e.g. leave the BoxBenchmark and TrapBenchmark only, if you will), then run the script (assuming that ${TOPDIR}
points to the top of your local source code/git repository):

  export TOPDIR=/path/to/VecGeom
  cd ${TOPDIR}/build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DGEANT4=ON -DROOT=ON -DBACKEND=VC -DBENCHMARK=ON -D<etc...>  #.. see instructions

  make -j8                      # all benchmarks will be built, or
  make -j8 TrapezoidBenchmark   # for specific benchmarks 

  #.. You'll probably need this symlink
  ln -sf ${TOPDIR}/scripts/average.py
  source ${TOPDIR}/scripts/perfHistory.sh

This will create a subdirectory named with the first 8 characters from your "git status" commit (note that any local
source code changes will affect performance, but will not be associated with the commit name of the directory. So I
suggest to create one commit per measurement conditions, even if most commits will never make into the official
repository.

Then add the commits you want to include in your comparison plots, in the file scripts/plotEvolution.py, variable
commits (a list), and run it:

  python scripts/plotEvolution.py


Let us know if you have any problems, by filing a bug report on JIRA (https://its.cern.ch/jira/projects/VECGEOM/issues/)


For the patient
---------------

Just wait some more...  LOL :-)
