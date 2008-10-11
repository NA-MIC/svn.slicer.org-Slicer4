#ifndef NTK2DDATARESIZER_H
#define NTK2DDATARESIZER_H

#include <stdlib.h>
#include "ntk2DData.h"
#include "ntk2DBSpline3Transform.h"

class ntk2DDataResizer{
 public:
  ntk2DDataResizer();
  ~ntk2DDataResizer();

  ntk2DData *m_input;
  
  void loadData(ntk2DData* input);
  ntk2DData *resizeData(ntkIntDimension newDataSize);
  ntk2DData *resizeDataBSplineInterpolation(ntkIntDimension newDataSize);
 protected:
  double getInterpolationValue(double x, double y);
};


#endif
