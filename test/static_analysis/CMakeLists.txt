# find clang + llvm
find_package(Clang REQUIRED 3.9)

if( LLVM_FOUND )
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(AddLLVM)

  # set the compiler flags to match llvm
  include(HandleLLVMOptions)
endif()

# Make sure that our source directory is on the current cmake module path so that
# we can include cmake files from this directory.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

# include Clang macros (unfortunately they are not part of the cmake installation)
# taken from llvm/clang 3.9.1
include(AddClang)

# add include directories
include_directories(${LLVM_INCLUDE_DIRS})

# verify that we have a dynamic library libClangTidy.[so|dylib]
# (otherwise we cannot hook our plugin)
find_file(CLANG_TIDY_LIB libClangTidy.so libClangTidy.dylib HINTS ${LLVM_CMAKE_DIR}/../../../lib)
if (!CLANG_TIDY_LIB)
  message (FATAL "no dynamic libClangTidy found in the llvm installation")
endif()

# Adding clang-tidy target if executable is found
find_program(CLANG_TIDY "clang-tidy" HINTS ${LLVM_CMAKE_DIR}/../../../bin)
find_program(CLANG_TIDY_SCRIPT "run-clang-tidy.py" HINTS ${CMAKE_CURRENT_SOURCE_DIR})

# add a unit test which will check clang-tidy conformance
add_test(static_code_test ${CMAKE_CURRENT_SOURCE_DIR}/RunChecks.sh
    ${PROJECT_BINARY_DIR}/test/static_analysis/libclangTidyVecGeomModule.so
    ${CLANG_TIDY}
    ${CLANG_TIDY_SCRIPT}
    ${PROJECT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}
    )

#
# plugin specific stuff starts here
#
set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyVecGeomModule
  SHARED

  MaskedAssignCheck.cpp
  VecGeomTidyModule.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyUtils
  clangTooling
  )
