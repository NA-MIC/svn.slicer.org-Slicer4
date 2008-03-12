#ifndef NTK2DELASTICTRANSFORMATION_H
#define NTK2DELASTICTRANSFORMATION_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "ntkTransformation.h"
#include "ntkBSpline3Function.h"
#include "ntk2DBSpline3Transform.h"
#include "ntkDeformationSpline.h"

class ntk2DElasticTransformation : public ntkTransformation{
 public:
  ntk2DElasticTransformation();
  ~ntk2DElasticTransformation();
  
  void loadData(ntk2DData* input);
  void loadData(ntk3DData* input){};
  ntk2DData* applyTransformation(ntkDeformationSpline* splineParam, int splineSizeLevel);
  
  ntkFloatDimension getNewPositionFromOld(ntkDeformationSpline* splineParam, int splineSizeLevel, ntkFloatDimension oldPosition);
  
  //ntkFloatDimension getDeformationFieldVector(ntkDeformationSpline *splineParam, int splineSizeLevel, ntkFloatDimension position);
  
  ntk2DData *getDeformationFieldImage(ntkDeformationSpline* splineParam, int splineSizeLevel, float weight);

  ntkDeformationSpline* getReverseTransformationSpline(ntkDeformationSpline* inputSpline);

  /**
   * dummy function
   */
  
  void setTransformationMatrix(vtkMatrix4x4* tMatrix){};
  
  /**
   * dummy function
   */
  
  ntk3DData *applyTransformation(ntkIntDimension outputSize, ntkFloatDimension outputThickness){return NULL;};
 protected:
  ntk2DData* m_input;
  ntkIntDimension m_inputSize;
  /**
   * Spline parameters (splineParam+N*knotNum.x*knotNum.y*knotNum.z+k*knotNum.x*knotNum.y+j*knotNum.x+i)
   * N 0:X, 1:Y, 2:Z
   */

  ntkDeformationSpline *m_splineParam;

  /**
   * Number of thread
   */

  int m_MTlevel;
};

#endif
