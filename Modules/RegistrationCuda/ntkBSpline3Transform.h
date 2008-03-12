#ifndef NTKBSPLINE3TRANSFORM
#define NTKBSPLINE3TRANSFORM

#include <math.h>
#include "ntkTensor.h"
#include "ntk3DData.h"
#include "ntkBSpline3Function.h"

typedef enum{
  CURRENT_RES,
  LOWER_RES
}ntkBSpline3TransformMode;

class ntkBSpline3Transform{
 public:
  ntkBSpline3Transform();
  ~ntkBSpline3Transform();
  
  /**
   * do transformation from 3D image data to B-Spline representation
   */

  ntkTensor* doForwardTransform(ntk3DData* inputImage, ntkBSpline3TransformMode mode);

  /**
   * do reverse transformation from B-Spline representation to 3D image data
   */

  ntk3DData* doReverseTransform(ntkTensor* inputSpline);

 /**
   * do reverse transformation from B-Spline representation to 3D image data
   */

  ntkTensor* doSplineToSplineHigherResTransform(ntkTensor* inputSpline);

  /**
   * do reverse transformation from B-Spline representation to 3D image data
   */

  ntk3DData* doReverseHigherResTransform(ntkTensor* inputSpline);
  
  /**
   * get interpolation value of voxel at position dataPos using B-Spline interpolation
   */

  float getInterpolationValue(ntkTensor* inputSpline, float x, float y, float z);
  
 protected:
  ntkBSpline3Function *m_func;
  float m_factor;
};

#endif
