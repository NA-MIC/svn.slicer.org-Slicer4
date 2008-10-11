#ifndef NTKDATAALIGNER_H
#define NTKDATAALIGNER_H

#include "ntk3DData.h"
#include "ntkProperties.h"

class ntkDataAligner{
 public:
  ntkDataAligner();
  ~ntkDataAligner();
  void alignBeforeToAfter(ntk3DData* before, ntk3DData* after);
};


#endif
