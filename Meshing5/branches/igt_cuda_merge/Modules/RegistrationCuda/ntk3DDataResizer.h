#ifndef NTK3DDATARESIZER_H
#define NTK3DDATARESIZER_H

#include <stdlib.h>
#include "ntk3DData.h"
#include "ntkBSpline3Transform.h"

class ntk3DDataResizer{
 public:
  ntk3DDataResizer();
  ~ntk3DDataResizer();

  ntk3DData *m_input;
  
  void loadData(ntk3DData* input);
  ntk3DData *resizeData(ntkIntDimension newDataSize);
  ntk3DData *resizeDataBSplineInterpolation(ntkIntDimension newDataSize);
 protected:
  double getInterpolationValue(double x, double y, double z);
};


#endif
