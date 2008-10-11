#ifndef NTKSIMILARITYMEASURE_H
#define NTKSIMILARITYMEASURE_H

#include "ntk3DData.h"

class ntkSimilarityMeasure{
 public:
  ntkSimilarityMeasure();
  virtual ~ntkSimilarityMeasure(){};
  
  virtual void loadData(ntk3DData* reference)=0;
  virtual double doSimilarityMeasure(ntk3DData* temp)=0;

 protected:
  ntk3DData* m_reference;
  ntk3DData* m_temp;
  
};

#endif
