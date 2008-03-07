#ifndef NTKDEFORMATIONSPLINE_H
#define NTKDEFORMATIONSPLINE_H

#include "ntkProperties.h"
#include "ntk3DData.h"
#include <stdlib.h>
#include <math.h>

class ntkDeformationSpline{
 public:
  ntkDeformationSpline(ntkIntDimension splineSize);
  ~ntkDeformationSpline();

  float *getBuffer();
  ntkIntDimension getSize();
  void readFile(char* filename);
  void writeFile(char* filename);
  ntk3DData* getDeformationSplineImage(float weight);
  void multiplyFactor(float factor);
 protected:
  ntkIntDimension m_splineSize;
  float *m_splineParam;
};

#endif
