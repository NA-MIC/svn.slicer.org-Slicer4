#include "ntkDataAligner.h"

ntkDataAligner::ntkDataAligner(){

}

ntkDataAligner::~ntkDataAligner(){

}

void ntkDataAligner:: alignBeforeToAfter(ntk3DData* before, ntk3DData* after){
  ntkIntDimension befdim=before->getDataSize();
  ntkIntDimension afdim=after->getDataSize();
  ntkFloatDimension befthick=before->getDataThickness();
  ntkFloatDimension afthick=after->getDataThickness();
  
  befthick.x=afthick.x*afdim.x/(float)befdim.x;
  befthick.y=afthick.y*afdim.y/(float)befdim.y;
  befthick.z=afthick.z*afdim.z/(float)befdim.z;
  
  before->setDataThickness(befthick);
}
