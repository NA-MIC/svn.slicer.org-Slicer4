#include "ntk3DDataCropper.h"

ntk3DDataCropper::ntk3DDataCropper(){

}

ntk3DDataCropper::~ntk3DDataCropper(){

}

ntk3DData* ntk3DDataCropper::cropX(ntk3DData* input, int xFrom, int xTo){
  ntkIntDimension dataSize=input->getDataSize();
  ntkFloatDimension dataThickness=input->getDataThickness();
  ntk3DData *output=new ntk3DData(dataSize, dataThickness);
  int i,j,k;

  unsigned char *inputBuffer=(unsigned char*)input->getBuffer();
  unsigned char *outputBuffer=(unsigned char*)output->getBuffer();
  
  for(i=0;i<xFrom;i++){
    for(j=0;j<dataSize.y;j++){
      for(k=0;k<dataSize.z;k++){
    *(outputBuffer+k*dataSize.x*dataSize.y+j*dataSize.x+i)=0;
      }
    }
  }
  
  for(i=xFrom;i<xTo;i++){
    for(j=0;j<dataSize.y;j++){
      for(k=0;k<dataSize.z;k++){
    *(outputBuffer+k*dataSize.x*dataSize.y+j*dataSize.x+i)=*(inputBuffer+k*dataSize.x*dataSize.y+j*dataSize.x+i);
      }
    }
  }

  for(i=xTo;i<dataSize.x;i++){
    for(j=0;j<dataSize.y;j++){
      for(k=0;k<dataSize.z;k++){
    *(outputBuffer+k*dataSize.x*dataSize.y+j*dataSize.x+i)=0;
      }
    }
  }
  
  return output;
}

ntk3DData* ntk3DDataCropper::cropY(ntk3DData* input, int yFrom, int yTo){
  ntkIntDimension dataSize=input->getDataSize();
  ntkFloatDimension dataThickness=input->getDataThickness();
  ntk3DData *output=new ntk3DData(dataSize, dataThickness);
  int i,j,k;

  unsigned char *inputBuffer=(unsigned char*)input->getBuffer();
  unsigned char *outputBuffer=(unsigned char*)output->getBuffer();
  
  for(i=0;i<dataSize.x;i++){
    for(j=0;j<yFrom;j++){
      for(k=0;k<dataSize.z;k++){
    *(outputBuffer+k*dataSize.x*dataSize.y+j*dataSize.x+i)=0;
      }
    }
  }
  
  for(i=0;i<dataSize.x;i++){
    for(j=yFrom;j<yTo;j++){
      for(k=0;k<dataSize.z;k++){
    *(outputBuffer+k*dataSize.x*dataSize.y+j*dataSize.x+i)=*(inputBuffer+k*dataSize.x*dataSize.y+j*dataSize.x+i);
      }
    }
  }

  for(i=0;i<dataSize.x;i++){
    for(j=yTo;j<dataSize.y;j++){
      for(k=0;k<dataSize.z;k++){
    *(outputBuffer+k*dataSize.x*dataSize.y+j*dataSize.x+i)=0;
      }
    }
  }
  
  return output;
}

ntk3DData* ntk3DDataCropper::cropZ(ntk3DData* input, int zFrom, int zTo){
  ntkIntDimension dataSize=input->getDataSize();
  ntkFloatDimension dataThickness=input->getDataThickness();
  ntk3DData *output=new ntk3DData(dataSize, dataThickness);
  int i,j,k;

  unsigned char *inputBuffer=(unsigned char*)input->getBuffer();
  unsigned char *outputBuffer=(unsigned char*)output->getBuffer();
  
  for(i=0;i<dataSize.x;i++){
    for(j=0;j<dataSize.y;j++){
      for(k=0;k<zFrom;k++){
    *(outputBuffer+k*dataSize.x*dataSize.y+j*dataSize.x+i)=0;
      }
    }
  }
  
  for(i=0;i<dataSize.x;i++){
    for(j=0;j<dataSize.y;j++){
      for(k=zFrom;k<zTo;k++){
    *(outputBuffer+k*dataSize.x*dataSize.y+j*dataSize.x+i)=*(inputBuffer+k*dataSize.x*dataSize.y+j*dataSize.x+i);
      }
    }
  }

  for(i=0;i<dataSize.x;i++){
    for(j=0;j<dataSize.y;j++){
      for(k=zTo;k<dataSize.z;k++){
    *(outputBuffer+k*dataSize.x*dataSize.y+j*dataSize.x+i)=0;
      }
    }
  }
  
  return output;
}
