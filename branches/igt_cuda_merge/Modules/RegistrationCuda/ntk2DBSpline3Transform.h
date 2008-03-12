#ifndef NTK2DBSPLINE3TRANSFORM
#define NTK2DBSPLINE3TRANSFORM

#include <math.h>
#include "ntkMatrix.h"
#include "ntk2DData.h"
#include "ntkBSpline3Function.h"

typedef enum{
  CURRENT_RES,
  LOWER_RES
}ntk2DBSpline3TransformMode;

class ntk2DBSpline3Transform{
 public:
  ntk2DBSpline3Transform();
  ~ntk2DBSpline3Transform();
  
  /**
   * do transformation from 3D image data to B-Spline representation
   */

  ntkMatrix* doForwardTransform(ntk2DData* inputImage, ntk2DBSpline3TransformMode mode);

  /**
   * do reverse transformation from B-Spline representation to 3D image data
   */

  ntk2DData* doReverseTransform(ntkMatrix* inputSpline);

 /**
   * do reverse transformation from B-Spline representation to 3D image data
   */

  ntkMatrix* doSplineToSplineHigherResTransform(ntkMatrix* inputSpline);

  /**
   * do reverse transformation from B-Spline representation to 3D image data
   */

  ntk2DData* doReverseHigherResTransform(ntkMatrix* inputSpline);
  
  /**
   * get interpolation value of voxel at position dataPos using B-Spline interpolation
   */

  float getInterpolationValue(ntkMatrix* inputSpline, float x, float y);
  
 protected:
  ntkBSpline3Function *m_func;
  float m_factor;
};

#endif
