#ifndef NTK3DDATACROPPER_H
#define NTK3DDATACROPPER_H

#include "ntk3DData.h"

class ntk3DDataCropper{
 public:
  ntk3DDataCropper();
  ~ntk3DDataCropper();
  
  ntk3DData* cropX(ntk3DData* input, int xFrom, int xTo);
  ntk3DData* cropY(ntk3DData* input, int yFrom, int yTo);
  ntk3DData* cropZ(ntk3DData* input, int zFrom, int zTo);
};

#endif
