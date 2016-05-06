

       Developer notes to monitor shape performance history
       ====================================================


* Shape performance data generation

A script has been developed to automate the generation of shape
performance data.  The script is available at script/perfHistory.sh,
and it has been tested on Ubuntu Linux.

The script must be run from a build area, and it collects performance
data for each benchmark executable *Benchmark previously built in that
area.

The script checks it is a valid build area, by extracting the source
directory used, and takes the git commit SHA1 for reference.  Please
note that the script does not verify that the executables are up to
date, nor tries to rebuild them.  This is on purpose, since the user
might want to collect performance data from a subset of shapes,
therefore the user must build only those benchmark binaries intended
to be evaluated.

** Instructions

Since the performance data is associated with a particular git commit,
user must make sure the intended commit is checked out and benchmark
binaries are properly built, before executing the script.

For the impatient, here is a basic list of suggested instructions:

  cd vecgeom
  git checkout master

  mkdir build
  cd build
  Vc_DIR=/work/local/vc/1.2.0 \
    Geant4_DIR=/work/local/geant4/10.02.p01-install-geant4 \
    ROOT_DIR=/work/local/root/v6.06.04 \
    cmake  -DCMAKE_BUILD_TYPE=Release \
    -DBACKEND=Vc -DVc=ON -DVECGEOM_VECTOR=avx \
    -DUSOLIDS=ON -DROOT=ON -DGEANT4=ON -DBENCHMARK=ON \
    -DCTEST=ON -DVALIDATION=OFF -DNO_SPECIALIZATION=ON

  make -j8 [myShapeBenchmark]

  ln -sf ../scripts/perfHistory.sh
  source ./perfHistory.sh

Important notes: for performance evaluation, build type Release must
be used.  Comparisons with with vectorization, Root, Geant4 and
USolids require those modes to be enabled, and also BENCHMARK=ON.

For the sake of historical performances, it is strongly suggested that
all available shape benchmarks are built and executed.

* More details

The script uses CMake configuration files to extract the git commit
SHA, which is used to name the performance data files produced, hence
keeping the association between performance data and the specific
version used.

For each shape tested, a total of 21 jobs are executed and their log
files saved.  Once all jobs are completed, the logfiles are processed
to collect, classify and summarize performance data.  As a result, a
single ASCII file is produced, with contents similar to the lines
shown below:

0.000507 0.000019 20 3.7 Box-inside-vect
0.001735 0.000043 20 2.5 Box-inside-spec
0.001729 0.000037 20 2.2 Box-inside-unspec
...
0.000958 0.000021 20 2.2 Box-safeToOut-usolids
0.000887 0.000013 20 1.4 Box-safeToOut-geant4
0.002754 0.000055 20 2.0 BoxScaled-inside-vect
...
0.001093 0.000028 20 2.5 Tube-safeToOut-root
0.001101 0.000040 20 3.7 Tube-safeToOut-usolids
0.001092 0.000033 20 3.0 Tube-safeToOut-geant4


The last string clearly identifies a specific combination of
shape+algorithm+implementation.  The four first fields in each line
represent processing time, standard deviation (both in seconds),
number of measurements used for the averaging, and statistical
quality, estimated as the ratio st.dev/mean (the smaller, the
better).

Please note that the lines above show that 20 data points were used to
calculate the average. The average is calculated in script/average.py,
and the single largest measurement is always discarded, to improve
the quality of processing time averages.

* Some free advice...

The user is recommended to close as many applications as possible, to
minimize the interference with the benchmark jobs.  In my own studies,
I observed a noticeable spread in processing times when a browser is
used, or too many files are open in my emacs session.  I also turn off
the mail client and disable all networking before running performance
jobs in my laptop.

Please send any comments to lima@fnalSPAMNOT.gov.
