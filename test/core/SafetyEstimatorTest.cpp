/*
 * SafetyEstimatorTest.cpp
 *
 *  Created on: Aug 28, 2015
 *      Author: swenzel
 */

#include "VecGeom/navigation/SimpleSafetyEstimator.h"
#include "VecGeom/navigation/SimpleABBoxSafetyEstimator.h"
#include <iostream>

using namespace vecgeom;

int main()
{
  // used to test compilation
  VSafetyEstimator *estimator = SimpleSafetyEstimator::Instance();
  std::cerr << estimator->GetName() << "\n";

  estimator = SimpleABBoxSafetyEstimator::Instance();
  std::cerr << estimator->GetName() << "\n";
  return 0;
}
