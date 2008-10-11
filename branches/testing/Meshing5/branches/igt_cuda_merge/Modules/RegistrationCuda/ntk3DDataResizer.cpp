#include "ntk3DDataResizer.h"


ntk3DDataResizer::ntk3DDataResizer(){
  m_input=NULL;
}

ntk3DDataResizer::~ntk3DDataResizer(){
  
}

void ntk3DDataResizer::loadData(ntk3DData *input){
  m_input=input;
}

ntk3DData *ntk3DDataResizer::resizeData(ntkIntDimension newDataSize){
  ntkIntDimension oldDataSize= m_input->getDataSize();
  ntkFloatDimension oldDataThickness= m_input->getDataThickness();
  ntkFloatDimension newDataThickness=ntkFloatDimension(oldDataSize.x*oldDataThickness.x/newDataSize.x, oldDataSize.y*oldDataThickness.y/newDataSize.y, oldDataSize.z*oldDataThickness.z/newDataSize.z);
  unsigned char *outputBuffer=(unsigned char*)malloc(newDataSize.x*newDataSize.y*newDataSize.z*sizeof(unsigned char));

  int i,j,k;

  for(i=0;i<newDataSize.x;i++){
    for(j=0;j<newDataSize.y;j++){
      for(k=0;k<newDataSize.z;k++){
    *(outputBuffer+k*newDataSize.x*newDataSize.y+j*newDataSize.x+i)=(unsigned char)getInterpolationValue(((i+0.5)*(newDataThickness.x)-0.5)/(oldDataThickness.x),((j+0.5)*(newDataThickness.y)-0.5)/(oldDataThickness.y),((k+0.5)*(newDataThickness.z)-0.5)/(oldDataThickness.z));
      }
    }
  }

  ntk3DData *output=new ntk3DData(newDataSize, newDataThickness);
  output->readBuffer(outputBuffer);

  return output;
}

ntk3DData *ntk3DDataResizer::resizeDataBSplineInterpolation(ntkIntDimension newDataSize){
   int i,j,k;
  double tempValue;


  ntkIntDimension oldDataSize= m_input->getDataSize();
  ntkFloatDimension oldDataThickness= m_input->getDataThickness();
  ntkFloatDimension newDataThickness=ntkFloatDimension(oldDataSize.x*oldDataThickness.x/newDataSize.x, oldDataSize.y*oldDataThickness.y/newDataSize.y, oldDataSize.z*oldDataThickness.z/newDataSize.z);

  /*bspline transformation of input*/

  ntk3DData* oldDataPlus=new ntk3DData(ntkIntDimension(oldDataSize.x+2, oldDataSize.y+2, oldDataSize.z+2));
  
  unsigned char* oldDataBuffer=(unsigned char*)m_input->getBuffer();
  unsigned char* oldDataPlusBuffer=(unsigned char*)oldDataPlus->getBuffer();

  for(k=0;k<oldDataSize.z;k++){
    for(j=0;j<oldDataSize.y;j++){
      memcpy(oldDataPlusBuffer+(k+1)*(oldDataSize.x+2)*(oldDataSize.y+2)+(j+1)*(oldDataSize.x+2)+1, oldDataBuffer+k*oldDataSize.x*oldDataSize.y+j*oldDataSize.x, oldDataSize.x);
    }
  }

  ntkBSpline3Transform *trans=new ntkBSpline3Transform();
  ntkTensor* splineInput=trans->doForwardTransform(oldDataPlus, CURRENT_RES);

  delete oldDataPlus;

  ntk3DData *output=new ntk3DData(newDataSize, newDataThickness);
  unsigned char *outputBuffer=(unsigned char*)output->getBuffer();

  for(i=0;i<newDataSize.x;i++){
    for(j=0;j<newDataSize.y;j++){
      for(k=0;k<newDataSize.z;k++){
    //*(outputBuffer+k*newDataSize.x*newDataSize.y+j*newDataSize.x+i)=(unsigned char)getInterpolationValue(i*(oldDataSize.x-1)/(newDataSize.x-1),j*(oldDataSize.y-1)/(newDataSize.y-1),k*(oldDataSize.z-1)/(newDataSize.z-1));
    tempValue=trans->getInterpolationValue(splineInput, ((i+0.5)*(newDataThickness.x)-0.5)/(oldDataThickness.x)+1.0,((j+0.5)*(newDataThickness.y)-0.5)/(oldDataThickness.y)+1.0,((k+0.5)*(newDataThickness.z)-0.5)/(oldDataThickness.z)+1.0);
    if(tempValue>255)tempValue=255.0;
    else if(tempValue<0)tempValue=0.0;
    
    *(outputBuffer+k*newDataSize.x*newDataSize.y+j*newDataSize.x+i)=(unsigned char)tempValue;
    
      }
    }
  }

  delete trans;
  delete splineInput;
  return output;
}

double ntk3DDataResizer::getInterpolationValue(double x, double y, double z){
  ntkIntDimension inputSize=m_input->getDataSize();
  unsigned char* inputBuffer=(unsigned char*)m_input->getBuffer();
  
  int xint=(int)x;
  int yint=(int)y;
  int zint=(int)z;
  float xsmall=(float)x-(float)xint;
  float ysmall=(float)y-(float)yint;
  float zsmall=(float)z-(float)zint;

  if(xint>=inputSize.x-1||yint>=inputSize.y-1||zint>=inputSize.z-1||xint<0||yint<0||zint<0){
    return 0;
  }

  
  /*
  unsigned char value1=0, value2=0, value3=0, value4=0, value5=0, value6=0, value7=0, value8=0;
  ntkIntDimension pos;

  pos.x=xint;pos.y=yint;pos.z=zint;m_input->getValue(pos,&value1);
  pos.x=xint;pos.y=yint;pos.z=zint+1;m_input->getValue(pos,&value2);
  pos.x=xint;pos.y=yint+1;pos.z=zint;m_input->getValue(pos,&value3);
  pos.x=xint;pos.y=yint+1;pos.z=zint+1;m_input->getValue(pos,&value4);
  pos.x=xint+1;pos.y=yint;pos.z=zint;m_input->getValue(pos,&value5);
  pos.x=xint+1;pos.y=yint;pos.z=zint+1;m_input->getValue(pos,&value6);
  pos.x=xint+1;pos.y=yint+1;pos.z=zint;m_input->getValue(pos,&value7);
  pos.x=xint+1;pos.y=yint+1;pos.z=zint+1;m_input->getValue(pos,&value8);
  */


  return (double)((1-xsmall)*(1-ysmall)*(1-zsmall)* *(inputBuffer+zint*inputSize.x*inputSize.y+yint*inputSize.x+xint)+
          (1-xsmall)*(1-ysmall)*zsmall* *(inputBuffer+(zint+1)*inputSize.x*inputSize.y+yint*inputSize.x+xint)+
          (1-xsmall)*ysmall*(1-zsmall)* *(inputBuffer+zint*inputSize.x*inputSize.y+(yint+1)*inputSize.x+xint)+
          (1-xsmall)*ysmall*zsmall* *(inputBuffer+(zint+1)*inputSize.x*inputSize.y+(yint+1)*inputSize.x+xint)+
          xsmall*(1-ysmall)*(1-zsmall)* *(inputBuffer+zint*inputSize.x*inputSize.y+yint*inputSize.x+xint+1)+
          xsmall*(1-ysmall)*zsmall* *(inputBuffer+(zint+1)*inputSize.x*inputSize.y+yint*inputSize.x+xint+1)+ 
          xsmall*ysmall*(1-zsmall)* *(inputBuffer+zint*inputSize.x*inputSize.y+(yint+1)*inputSize.x+xint+1)+
          xsmall*ysmall*zsmall* *(inputBuffer+(zint+1)*inputSize.x*inputSize.y+(yint+1)*inputSize.x+xint+1));
  
}
