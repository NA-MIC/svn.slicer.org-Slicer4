#ifndef NTKBSPLINE3FUNCTIONH
#define NTKBSPLINE3FUNCTIONH

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

class ntkBSpline3Function{
 public:
  ntkBSpline3Function();
  ~ntkBSpline3Function();
  double getValue(double x);
  void createPreCalculation(int knotNum); //not implemented yet
  double getInterpolationValue(double x); //not implemented yet
  double getDifferentialValue(double x);
 private:
  int  m_knotNum;
  double m_step;
  double *m_val;
};

#endif
