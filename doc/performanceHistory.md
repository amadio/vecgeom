

       Developer notes to monitor shape performance history
       ====================================================


* Shape performance data generation

Scripts have been developed to automate the generation of shape
performance data.  The script is available at script/perfHistory.sh,
and it has been tested on Ubuntu Linux.

The script must be run from a build area, and it collects performance
data for each benchmark executable *Benchmark previously built in that
area.

The script checks it is a valid build area, by extracting the source
directory used, and takes the git commit SHA1 for reference.  Please
note that the script does not verify whether the executables are up to
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
    -DROOT=ON -DGEANT4=ON -DBENCHMARK=ON \
    -DBUILD_TESTING=ON -DVALIDATION=OFF -DNO_SPECIALIZATION=ON

  make -j8 [myShapeBenchmark]

  ln -sf ../scripts/perfHistory.sh
  source ./perfHistory.sh

Important notes: for performance evaluation, build type Release must
always be used.  Comparisons with vectorization, Root and Geant4
require those modes to be enabled, and also BENCHMARK=ON.

For the sake of historical performances, it is strongly suggested that
all available shape benchmarks are built and executed regularly, and
at specific commits when any shape algorithms have been modified.

* More details

The script uses CMake configuration files to extract the git commit
SHA, which is used to name the performance data files produced, hence
keeping the association between performance data and the specific
source versions used.

For each shape tested, a total of 21 jobs are executed and their log
files saved.  Once all jobs are completed, the logfiles are processed
to collect, classify and summarize performance data.  As a result, a
single ASCII file is produced, with contents similar to the lines
shown below:

======================================
# Wed May 11 13:48:34 CDT 2016
# Linux lima-mbp 3.13.0-86-generic #130-Ubuntu SMP Mon Apr 18 18:27:15 UTC 2016 x86_64 x86_64 x86_64 GNU/Linux
# model name : Intel(R) Core(TM) i7-3615QM CPU @ 2.30GHz
# benchmark for 131172 points for 1 repetitions.
Box-inside-vect               0.000508  0.000005    2   1.000
Box-inside-spec               0.001722  0.000067    2   3.900
Box-inside-unspec             0.001762  0.000003    2   0.200
...
Box-safeToOut-geant4          0.000887  0.000054    2   6.100
Trapezoid-inside-vect         0.000815  0.000039    2   4.800
...
Trd-safeToOut-root            0.005385  0.000033    2   0.600
Trd-safeToOut-geant4          0.003807  0.000077    2   2.000
======================================

Note that the first few lines on top of the file are "commented out"
from downstream processing, but keep performance-relevant information
from the machine used for the measurements.

The first string on the line clearly identifies a specific combination
of shape+algorithm+implementation.  The remaining four fields in each
line represent processing time, standard deviation (both in seconds),
number of measurements used for the averaging, and statistical
quality, estimated as the ratio st.dev/mean (the smaller, the better).

Please note that the lines above show that 20 data points were used to
calculate the average. The average is calculated in scripts/average.py,
and the single largest measurement is always discarded, to improve
the quality of processing time averages.

* Some free advice...

The user is recommended to close as many applications as possible, to
minimize the interference with the benchmark jobs.  In my own studies,
I observed a noticeable spread in processing times when a browser is
kept open, or too many files are open in my emacs session.  I also
turn off the mail client and disable all networking before running
performance jobs in my laptop.

* Making plots

A few plotting scripts are provided as examples.  They have also been
tested in Ubuntu Linux, using ROOT v6.06.04 built with python 2.7.

The plotting scripts expect a few commits to be provided in the script
source code, at the array /commits/:

  commits   = ["c4154901", "1f673e0d", "458e6f08"]

and the performance data should be available in the build area:

  > ls -la */*-perf.dat
  -rw-rw-r-- 1 lima lima 6944 May 11 16:33 1f673e0d/1f673e0d-perf.dat
  -rw-rw-r-- 1 lima lima 6944 May 11 16:33 458e6f08/458e6f08-perf.dat
  -rw-rw-r-- 1 lima lima 6944 May 12 14:51 c4154901/c4154901-perf.dat

Scripts can then be executed from the build area:

  > python ../scripts/plotNormalizedEvolution.py
  Commit=<c415490> - 99 datasets read.
  Commit=<1f673e0> - 99 datasets read.
  Commit=<458e6f0> - 99 datasets read.
  Info in <TCanvas::Print>: file normTimes-Box.png has been created
  Info in <TCanvas::Print>: file normTimes-Trapezoid.png has been created
  Info in <TCanvas::Print>: file normTimes-Trd.png has been created


Please send any comments or suggestions to lima@fnalSPAMNOT.gov.
