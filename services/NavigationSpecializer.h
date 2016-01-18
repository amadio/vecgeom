/*
 * NavigationSpecializer.h
 *
 *  Created on: 11.09.2015
 *      Author: swenzel
 */

#ifndef VECGEOM_SERVICES_NAVIGATIONSPECIALIZER_H_
#define VECGEOM_SERVICES_NAVIGATIONSPECIALIZER_H_

#include "navigation/NavigationState.h"
#include <string>
#include <iosfwd>
#include <map>
#include <sstream>
#include <list>
#include <vector>
#include "base/Global.h"

namespace vecgeom {
inline namespace cxx {
class LogicalVolume;
class NavStatePool;
/* A class which can produce (per logical volume) specialized C++ code
 * for navigation routines.
 *
 * A service which can transform generic navigation algorithms into specialized kernels
 * based on static analysis of the logical volume and the geometry hierarchy.
 * Example: It can convert a SimpleSafetyEstimator to FOOSafetyEstimator for a logical volume called
 *          FOO.
 * The output of the service are C++ files which can be (re- or just-in-time) compiled.
 *
 * Things which this service can (might) do are:
 *
 * a) avoid virtual functions in navigation methods
 * b) precompute all possible global transformations of a logical volume and cache them in the specialized kernels
 * c) provide a fast hard-coded lookup table to map NavigationStates objects to cached global transformations
 * d) vectorize loops over daughter volumes of the same type
 * e) loop splitting of daughter lists
 * f) realizing that daughtervolumes use the same rotations and do this transformation only once
 * g) ....
 *
 * The present class is an R&D prototype. Nothing is fix here!!
 *
 * 09/2015: idea and first implementation ( sandro.wenzel@cern.ch )
 */
class NavigationSpecializer {

public:
    // is this class a singleton ??
  NavigationSpecializer()
      : fLogicalVolumeName(), fClassName(), fLogicalVolume(nullptr), fGeometryDepth(0), fIndexMap(),
        fStaticArraysInitStream(), fStaticArraysDefinitions(), fTransformationCode(), fVectorTransformVariables(),
        fVectorTransformationCode(), // to collect the many-path/SIMD transformation statements
        fTransformVariables(),       // stores the list of relevant transformation variables
        fUnrollLoops(false),         // whether to manually unroll all loops
        fUseBaseNavigator(false),    // whether to use the DaughterDetection from another navigator ( makes sense when
                                     // combined with voxel techniques )
        fBaseNavigator(){};

    // produce a specialized SafetyEstimator class for a given logical volume
    // currently this is only done using the SimpleEstimator base algorithm
    // TODO: we could template here on some base algorithm in general and we could
    // specialize voxel algorithms and the like
    void ProduceSpecializedNavigator( LogicalVolume const *, std::ostream & );


private :
    typedef std::map< size_t, std::map< size_t, size_t >> PathLevelIndexMap_t;

    // analysis functions
    void AnalyseLogicalVolume();
    void AnalysePaths( std::list<NavigationState *> const & /* inpaths */ );
    void AnalyseTargetPaths( NavStatePool const &, NavStatePool const &);
   // void GeneratePathClassifierCode(std::list<std::pair<int, std::set<NavigationState::Value_t>>> const &pathclassification,
   //                                 PathLevelIndexMap_t &map);


    void AddToIndexMap( size_t, size_t );
    size_t PathToIndex( NavigationState const * );
    void AnalyseIndexCorrelations( std::list<NavigationState *> const & );

    // writer functions
    void DumpDisclaimer(std::ostream &);
    void DumpIncludeFiles( std::ostream & );
    void DumpNamespaceOpening( std::ostream & );
    void DumpNamespaceClosing( std::ostream & );
    void DumpStaticConstExprData( std::ostream & );
    void DumpStaticConstExprVariableDefinitions( std::ostream & );
    void DumpStaticInstanceFunction( std::ostream & );
    void DumpConstructor( std::ostream & ) const;
    void DumpClassOpening( std::ostream & );
    void DumpClassDefinitions( std::ostream & );
    void DumpClassDeclarations( std::ostream & );
    void DumpClassClosing( std::ostream & );
    void DumpPathToIndexFunction( std::ostream & );
    void DumpVectorTransformationFunction( std::ostream & );
    void DumpLocalSafetyFunction( std::ostream & );
    void DumpPrivateClassDefinitions( std::ostream & );
    void DumpPublicClassDefinitions( std::ostream & );
    void DumpLocalSafetyFunctionDeclaration( std::ostream & );

    void DumpRelocateMethod(std::ostream &) const;

    void DumpLocalVectorSafetyFunctionDeclaration( std::ostream & );

    void DumpLocalVectorSafetyFunctionDeclarationPerSIMDVector( std::ostream & );

    void DumpSafetyFunctionDeclaration( std::ostream & );
    void DumpVectorSafetyFunctionDeclaration( std::ostream & );
    void DumpTransformationAsserts( std::ostream & );

    void DumpLocalHitDetectionFunction(std::ostream &) const;

    // they produce static component functions which are plugged into the generic implementation of VNavigatorHelper
    void DumpFoo(std::ostream &) const;
    void DumpStaticTreatGlobalToLocalTransformationFunction( std::ostream & ) const;
    void DumpStaticTreatDistanceToMotherFunction( std::ostream & ) const;
    void DumpStaticPrepareOutstateFunction( std::ostream & ) const;

public:
  void EnableLoopUnrolling() { fUnrollLoops = true; }
  void DisableLoopUnrolling() { fUnrollLoops = false; }

  void SetBaseNavigator(std::string const & nav) {
    fUseBaseNavigator = true;
    fBaseNavigator = nav;
  }

  typedef std::pair<int, std::string> FinalDepthShapeType_t;
private:
    // private state
    std::string fLogicalVolumeName;
    std::string fClassName;
    LogicalVolume const * fLogicalVolume;
    unsigned int fGeometryDepth; // the depth of instances of fLogicalVolumes in the geometry hierarchy ( must be unique )
    unsigned int fNumberOfPossiblePaths;
    PathLevelIndexMap_t fIndexMap; // in memory structure; used to map a NavigationState object to an index
    std::stringstream fStaticArraysInitStream;    // stream to collect code for the static arrays
    std::stringstream fStaticArraysDefinitions;   // stream to collect code for constexpr static array definitions
    std::stringstream fTransformationCode;        // to collect the specialized transformation statements for points
    std::stringstream fTransformationCodeDir;     // to collect the specialized transformation statements for dirs
    std::stringstream fVectorTransformVariables;  // to collect relevant vector variables for transformation
    std::stringstream fVectorTransformationCode;  // to collect the many-path/SIMD transformation statements
    std::vector<std::string> fTransformVariables; // stores the list of relevant transformation variables
    bool fUnrollLoops; // whether to manually unroll all loops
    bool fUseBaseNavigator; // whether to use the DaughterDetection from another navigator ( makes sense when combined with voxel techniques )
    std::string fBaseNavigator;

    std::stringstream fDeltaTransformationCode;
    std::vector<std::string> fTransitionTransformVariables; // stores the vector of relevant variables for the relocation transformations

    //
    std::vector<std::string> fTransitionStrings;
    std::vector<FinalDepthShapeType_t> fTransitionTargetTypes; // the vector of possible relocation target types
    std::vector<size_t> fTransitionOrder; // the vector keeping the order of indices of relocation transitions (
                                                 // as stored in fTransitionTargetsTypes )
    std::vector<NavigationState::Value_t> fTargetVolIds; // the ids of the target volumes ( to quickly fetch a representative volume pointer )

    std::vector<std::vector<int>> fPathxTargetToMatrixTable; // an in - memory table to fetch the correct transition matrix index
    std::stringstream fPathxTargetToMatrixTableStringStream; // string represenation of the above
}; // end class


}} // end namespace

#endif /* VECGEOM_SERVICES_NAVIGATIONSPECIALIZER_H_ */
