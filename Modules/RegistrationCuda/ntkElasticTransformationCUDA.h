#ifndef NTKELASTICTRANSFORMATIONCUDA_H
#define NTKELASTICTRANSFORMATIONCUDA_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "ntkTransformation.h"
#include "ntkBSpline3Function.h"
#include "ntkBSpline3Transform.h"
#include "ntkDeformationSpline.h"
#include "ntkTimer.h"
#include "ntkCudaDeviceMemory.h"

#include "CUDA_elasticTransformation.h"

class ntkElasticTransformationCUDA : public ntkTransformation{
 public:
  ntkElasticTransformationCUDA();
  ~ntkElasticTransformationCUDA();
  void loadData(ntk2DData* input){};
  void loadData(ntk3DData* input);
  ntk3DData* applyTransformation(ntkDeformationSpline* splineParam, int splineSizeLevel);
  
  ntkFloatDimension getNewPositionFromOld(ntkDeformationSpline* splineParam, int splineSizeLevel, ntkFloatDimension oldPosition, float weight);
  
  //ntkFloatDimension getDeformationFieldVector(ntkDeformationSpline *splineParam, int splineSizeLevel, ntkFloatDimension position);
  
  ntk3DData *getDeformationFieldImage(ntkDeformationSpline* splineParam, int splineSizeLevel, float weight);

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
  ntk3DData* m_input;
  ntkIntDimension m_inputSize;
  /**
   * Spline parameters (splineParam+N*knotNum.x*knotNum.y*knotNum.z+k*knotNum.x*knotNum.y+j*knotNum.x+i)
   * N 0:X, 1:Y, 2:Z
   */

  ntkDeformationSpline *m_splineParam;

  ntkCudaDeviceMemory *m_dInput;

  /**
   * Number of thread
   */

  int m_MTlevel;
};

#endif
