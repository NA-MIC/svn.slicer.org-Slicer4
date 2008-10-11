#ifndef NTKTRANSFORMATION_H
#define NTKTRANSFORMATION_H

#include <stdio.h>
#include <stdlib.h>
#include "vtkMatrix4x4.h"
#include "ntk3DData.h"

class ntkTransformation{
 public:
  ntkTransformation();
  virtual ~ntkTransformation(){};
  
  virtual void loadData(ntk3DData* input)=0;
  virtual void loadData(ntk2DData* input)=0;
  virtual void setTransformationMatrix(vtkMatrix4x4* tMatrix)=0;
  virtual ntk3DData *applyTransformation(ntkIntDimension outputSize, ntkFloatDimension outputThickness)=0;
  ntk3DData *applyTransformation(){}
  
  void printMatrix();
 protected:
  ntk3DData* m_input;
  ntk3DData* m_output;
  vtkMatrix4x4* m_tMatrix;
};

#endif
