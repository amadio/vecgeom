# find clang + llvm
find_package(Clang REQUIRED) # unfortunately we cannot ask for version 14 directly :/

if (LLVM_VERSION VERSION_LESS 14.0.0)
  message (FATAL_ERROR "Need at least clang/LLVM version 14.0.0")
endif()


list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

# include Clang macros from current source directory
# unfortunately they are not part of the cmake installation
# taken from llvm/clang 14.0.0
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
include(AddClang)

find_program(CLANG_TIDY "clang-tidy" HINTS ${LLVM_CMAKE_DIR}/../../../bin)

# add a unit test which will check clang-tidy conformance
add_test(static_code_test
    ${CMAKE_CURRENT_SOURCE_DIR}/run-clang-tidy.py
    -clang-tidy-binary ${CLANG_TIDY}
    -load=${PROJECT_BINARY_DIR}/test/static_analysis/libclangTidyVecGeomModule.so
    -checks=-*,vecgeom*
    -p ${PROJECT_BINARY_DIR}
    )

#
# plugin specific stuff starts here
#
set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyVecGeomModule
  SHARED
  MaskedAssignCheck.cpp
  VecGeomTidyModule.cpp
  )
target_include_directories(clangTidyVecGeomModule PRIVATE ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
target_compile_features(clangTidyVecGeomModule PUBLIC cxx_std_14)
