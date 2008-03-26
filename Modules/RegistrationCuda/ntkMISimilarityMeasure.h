#ifndef NTKMISIMILARITYMEASURE_H
#define NTKMISIMILARITYMEASURE_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ntkSimilarityMeasure.h"

class ntkMISimilarityMeasure : public ntkSimilarityMeasure{
 public:
  ntkMISimilarityMeasure();
  ~ntkMISimilarityMeasure();
  void loadData(ntk3DData* reference);
  double doSimilarityMeasure(ntk3DData* temp);
  void setAlignTemplateToReference();
  void setAlignReferenceToTemplate();
  void setHistogramBinSize(int histFactor);
 protected:
  double* m_histogram;
  double* m_histogramA;
  int* m_histogramB;
  int m_temflag;
  int m_refflag;
  int m_histFactor;
  ntkFloatDimension m_temthick;
  double getInterpolationValue(double x, double y, double z);
  void distributeInterpolationValue(double x, double y, double z, unsigned char tempValue);
};

#endif
