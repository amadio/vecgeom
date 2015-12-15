//
// Definition of the batch solid test
//

#ifndef ShapeTester_hh
#define ShapeTester_hh

#include <iostream>
#include <sstream>

#include "VUSolid.hh"
#include "UUtils.hh"

class ShapeTester{

public:
	ShapeTester();
	~ShapeTester();

	int Run(VUSolid *testVolume);//std::ofstream &fLogger);
	int RunMethod(VUSolid *testVolume, std::string fMethod1 );//std::ofstream &fLogger);

	inline void SetFilename( const std::string &newFilename ) { fFilename = newFilename; }
	inline void SetMaxPoints( const int newMaxPoints ) { fMaxPoints = newMaxPoints; }
	inline void SetRepeat( const int newRepeat ) { fRepeat = newRepeat; }
	inline void SetMethod( const std::string &newMethod ) { fMethod = newMethod; }
	inline void SetInsidePercent( const double percent ) { fInsidePercent = percent; }
	inline void SetOutsidePercent( const double percent ) { fOutsidePercent = percent; }
        inline void SetEdgePercent( const double percent ) { fEdgePercent = percent; }

	inline void SetOutsideMaxRadiusMultiple( const double percent ) { fOutsideMaxRadiusMultiple = percent; }
	inline void SetOutsideRandomDirectionPercent( const double percent ) { fOutsideRandomDirectionPercent = percent; }
	inline void SetDifferenceTolerance( const double tolerance ) { fDifferenceTolerance = tolerance; }
        inline void SetNewSaveValue( const double tolerance ) { fMinDifference = tolerance; }
        inline void SetSaveAllData( const bool safe ) { fIfSaveAllData = safe; }
        inline void SetRunAllTests( const bool safe ) { fIfMoreTests = safe; }
	void SetFolder( const std::string &newFolder );
        void SetVerbose(int verbose){ fVerbose = verbose; }
        inline int GetMaxPoints() const { return fMaxPoints; }
        inline int GetRepeat() const { return fRepeat; }
        inline UVector3 GetPoint(int index){ return fPoints[index];}
        inline void SetNumberOfScans(int num){ fGNumberOfScans = num; } 
    
  	std::vector<UVector3> fPoints, fDirections;
private:
	void SetDefaults();

	int SaveVectorToExternalFile(const std::vector<double> &vector, const std::string &fFilename);
	int SaveVectorToExternalFile(const std::vector<UVector3> &vector, const std::string &fFilename);
	int SaveLegend(const std::string &fFilename);
        int SaveDifLegend(const std::string &fFilename);
	int SaveDoubleResults(const std::string &fFilename);
        int SaveDifDoubleResults(const std::string &fFilename);
	int SaveVectorResults(const std::string &fFilename);
        int SaveDifVectorResults(const std::string &fFilename);
         int SaveDifVectorResults1(const std::string &fFilename);

	std::string PrintCoordinates (const UVector3 &vec, const std::string &delimiter, int precision=4);
	std::string PrintCoordinates (const UVector3 &vec, const char *delimiter, int precision=4);
	void PrintCoordinates (std::stringstream &ss, const UVector3 &vec, const std::string &delimiter, int precision=4);
	void PrintCoordinates (std::stringstream &ss, const UVector3 &vec, const char *delimiter, int precision=4);

	template <class T> void VectorDifference(const std::vector<T> &first, const std::vector<T> &second, std::vector<T> &result);
	
	void VectorToDouble(const std::vector<UVector3> &vectorUVector, std::vector<double> &vectorDouble);

  void BoolToDouble(const std::vector<bool> &vectorBool, std::vector<double> &vectorDouble);
	
	int CountDoubleDifferences(const std::vector<double> &differences);
	int CountDoubleDifferences(const std::vector<double> &differences, const std::vector<double> &values1, const std::vector<double> &values2);

 //	int CompareVectorDifference(std::string fFilename);

protected:
	UVector3	GetRandomPoint() const;
	double	GaussianRandom(const double cutoff) const;

	void	ReportError( int *nError, UVector3 &p, 
		UVector3 &v, double distance,
			     std::string comment);//, std::ostream &fLogger );
	void 	ClearErrors();		
	int 	CountErrors() const;

        
protected:

	int fMaxPoints, fRepeat;
        int fVerbose;   
        double	fInsidePercent, fOutsidePercent,fEdgePercent, fOutsideMaxRadiusMultiple, fOutsideRandomDirectionPercent, fDifferenceTolerance;
        // XRay profile statistics
        int fGNumberOfScans ;
        double fGCapacitySampled,fGCapacityError ,fGCapacityAnalytical ;
	std::string fMethod;


	typedef struct sShapeTesterErrorList {
	  std::string	sMessage;
	  int		sNUsed;
		struct sShapeTesterErrorList *sNext;
	} ShapeTesterErrorList;

	ShapeTesterErrorList *fErrorList;

private:
	int fNumCheckPoints;

	int fCompositeCounter;

	void FlushSS(std::stringstream &ss);
	void Flush(const std::string &s);
       
	VUSolid *fVolumeUSolids;
        std::stringstream fVolumeSS;
	std::string fVolumeString;
   

   
 	//std::vector<UVector3> fPoints, fDirections;
	std::vector<UVector3> fResultVectorGeant4;
	std::vector<UVector3> fResultVectorRoot;
        std::vector<UVector3> fResultVectorUSolids,fResultVectorDifference;
        std::vector<double> fResultDoubleGeant4, fResultDoubleRoot, fResultDoubleUSolids, fResultDoubleDifference;
        std::vector<bool> fResultBoolGeant4, fResultBoolUSolids, fResultBoolDifference;
       
  int fOffsetSurface, fOffsetInside, fOffsetOutside, fOffsetEdge;
  int fMaxPointsInside, fMaxPointsOutside, fMaxPointsSurface,fMaxPointsEdge;
	std::ostream *fLog, *fPerftab, *fPerflabels;
	std::string fFolder;
	std::string fFilename;
        //Save only differences
        bool fIfSaveAllData;//save alldata, big files
        bool fIfMoreTests;//do all additional tests,
                         //take more time, but not affect performance measures
        bool fIfDifUSolids;//save differences of Geant4 with Usolids or with ROOT
        double fMinDifference;//save data, when difference is bigger that min
        bool fDefinedNormal, fIfException;
        std::vector<UVector3> fDifPoints;
        std::vector<UVector3> fDifDirections;
        std::vector<UVector3> fDifVectorGeant4,fDifVectorRoot,fDifVectorUSolids;
        std::vector<double> fDifGeant4,fDifRoot,fDifUSolids;
        int fDifPointsInside,fDifPointsSurface,fDifPointsOutside;
        int fMaxErrorBreak;

	UVector3 GetPointOnOrb(double r);
	UVector3 GetRandomDirection();


	int TestConsistencySolids();
        int TestInsidePoint();
        int TestOutsidePoint();
        int TestSurfacePoint();

	int TestNormalSolids();


	int TestSafetyFromInsideSolids();
        int TestSafetyFromOutsideSolids();
        int ShapeSafetyFromInside(int max);
	int ShapeSafetyFromOutside(int max);

	void PropagatedNormal(const UVector3 &point, const UVector3 &direction, double distance, UVector3 &normal);
        void PropagatedNormalU(const UVector3 &point, const UVector3 &direction, double distance, UVector3 &normal);

	int TestDistanceToInSolids();
        int TestAccuracyDistanceToIn(double dist);
        int ShapeDistances();
        int TestFarAwayPoint();

	int TestDistanceToOutSolids();             
        int ShapeNormal();
        int TestXRayProfile();
        int XRayProfile(double theta=45, int nphi=15, int ngrid=1000, bool useeps=true);
	int Integration(double theta=45, double phi=45, int ngrid=1000, bool useeps=true, int npercell=1, bool graphics=true);
	double CrossedLength(const UVector3 &point, const UVector3 &dir, bool useeps);

	void CreatePointsAndDirections();
	void CreatePointsAndDirectionsSurface();
        void CreatePointsAndDirectionsEdge();
	void CreatePointsAndDirectionsInside();
	void CreatePointsAndDirectionsOutside();

	void CompareAndSaveResults(const std::string &fMethod, double resG, double resR, double resU);

	int SaveResultsToFile(const std::string &fMethod);

	void SavePolyhedra(const std::string &fMethod);

	double MeasureTest (int (ShapeTester::*funcPtr)(int), const std::string &fMethod);

	double NormalizeToNanoseconds(double time);

	int TestMethod(int (ShapeTester::*funcPtr)());
	int TestMethodAll();

     inline double RandomRange(double min, double max)
  {
    double rand = min + (max - min) * UUtils::Random();
    return rand;
  }
    inline void GetVectorUSolids(UVector3 &point, const std::vector<UVector3> &afPoints, int index)
  {
    const UVector3 &p = afPoints[index];
    point.Set(p.x(), p.y(), p.z());
  }
 
  inline void SetVectorUSolids(const UVector3 &point, std::vector<UVector3> &afPoints, int index)
  {
    UVector3 &p = afPoints[index];
    p.Set(point.x(), point.y(), point.z());
  }
  
  inline double RandomIncrease()
  {
    double tolerance = VUSolid::Tolerance();
    double rand = -1 + 2 * UUtils::Random();
    double dif = tolerance * 0.1 * rand;
    return dif;
  }
  

};

#endif
