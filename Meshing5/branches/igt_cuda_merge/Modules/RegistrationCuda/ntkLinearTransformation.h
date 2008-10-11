#ifndef NTKLINEARTRANSFORMATION_H
#define NTKLINEARTRANSFORMATION_H

#include <math.h>
#include "ntkTransformation.h"
#include "ntkBSpline3Transform.h"

class ntkLinearTransformation : public ntkTransformation{
 public:
  ntkLinearTransformation();
  ~ntkLinearTransformation();
  void loadData(ntk2DData* input){};
  void loadData(ntk3DData* input);
  void setTransformationMatrix(vtkMatrix4x4* tMatrix);
  ntk3DData *applyTransformation(ntkIntDimension outputSize, ntkFloatDimension outputThickness);
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
