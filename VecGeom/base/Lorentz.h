/// \file LorentzVector.h
/// \author Andrei Gheata
#include "LorentzRotation.h"

template <typename T>
vecgeom::LorentzVector<T> &vecgeom::LorentzVector<T>::operator*=(const vecgeom::LorentzRotation<T> &m1)
{
  return *this = m1.vectorMultiplication(*this);
}

template <typename T>
vecgeom::LorentzVector<T> &vecgeom::LorentzVector<T>::transform(const vecgeom::LorentzRotation<T> &m1)
{
  return *this = m1.vectorMultiplication(*this);
}
