#ifndef NTKLINEARTRANSFORMATIONCUDA_H
#define NTKLINEARTRANSFORMATIONCUDA_H

#include <math.h>
#include "ntkTransformation.h"
#include "ntkBSpline3Transform.h"
#include "CUDA_linearTransformation.h"
#include "ntkCudaDeviceMemory.h"

class ntkLinearTransformationCUDA : public ntkTransformation{
 public:
  ntkLinearTransformationCUDA();
  ~ntkLinearTransformationCUDA();
  void loadData(ntk2DData* input){};
  void loadData(ntk3DData* input);
  void setTransformationMatrix(vtkMatrix4x4* tMatrix);
  ntk3DData *applyTransformation(ntkIntDimension outputSize, ntkFloatDimension outputThickness);
ntk3DData *applyTransformation();
  void doTranslation(double x, double y, double z);
  void doRotationX(double xAngle);
  void doRotationY(double yAngle);
  void doRotationZ(double zAngle);
  void doScaling(double scale);
  vtkMatrix4x4* getMatrix(void);
  
  ntkFloatDimension getNewPositionFromOld(ntkIntDimension outputSize, ntkFloatDimension outputThickness, ntkFloatDimension oldPosition);
  ntkFloatDimension getOldPositionFromNew(ntkIntDimension outputSize, ntkFloatDimension outputThickness, ntkFloatDimension newPosition);
    
 protected:
  double getInterpolationValue(double x, double y, double z);
};

#endif
