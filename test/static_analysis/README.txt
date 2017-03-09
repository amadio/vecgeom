this directory contains static analysis checks
specific to VecGeom ... implemented as a clang-tidy
module

This module can be hooked into standard clang-tidy via a
LD_PRELOAD/DYLD_INSERT_LIBRARIES TRICK

Current limitations/drawback
 * need to copy some clang-tidy headers here
   (because they are not offered by the llvm installation)
 * need a dynamic library build of llvm + clang + clang-extra-tools


USAGE:
  cmake -DSTATIC_ANALYSIS=ON -DLLVM_DIR=PATH_TO_LLVM/lib/cmake/llvm -DClang_DIR=PATH_TO_LLVM/lib/cmake/clang
