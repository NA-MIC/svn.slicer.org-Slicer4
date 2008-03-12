#include "ntk2DDataResizer.h"


ntk2DDataResizer::ntk2DDataResizer(){
  m_input=NULL;
}

ntk2DDataResizer::~ntk2DDataResizer(){
  
}

void ntk2DDataResizer::loadData(ntk2DData *input){
  m_input=input;
}

ntk2DData *ntk2DDataResizer::resizeData(ntkIntDimension newDataSize){
  ntkIntDimension oldDataSize= m_input->getDataSize();
  ntkFloatDimension oldDataThickness= m_input->getDataThickness();
  ntkFloatDimension newDataThickness=ntkFloatDimension(oldDataSize.x*oldDataThickness.x/newDataSize.x, oldDataSize.y*oldDataThickness.y/newDataSize.y);
  unsigned char *outputBuffer=(unsigned char*)malloc(newDataSize.x*newDataSize.y*sizeof(unsigned char));

  int i,j;

  for(i=0;i<newDataSize.x;i++){
    for(j=0;j<newDataSize.y;j++){
      *(outputBuffer+j*newDataSize.x+i)=(unsigned char)getInterpolationValue(((i+0.5)*(newDataThickness.x)-0.5)/(oldDataThickness.x),((j+0.5)*(newDataThickness.y)-0.5)/(oldDataThickness.y));
    }
  }

  ntk2DData *output=new ntk2DData(newDataSize, newDataThickness);
  output->readBuffer(outputBuffer);

  return output;
}

ntk2DData *ntk2DDataResizer::resizeDataBSplineInterpolation(ntkIntDimension newDataSize){
  int i,j;
  double tempValue;


  ntkIntDimension oldDataSize= m_input->getDataSize();
  ntkFloatDimension oldDataThickness= m_input->getDataThickness();
  ntkFloatDimension newDataThickness=ntkFloatDimension(oldDataSize.x*oldDataThickness.x/newDataSize.x, oldDataSize.y*oldDataThickness.y/newDataSize.y);

  /*bspline transformation of input*/

  ntk2DData* oldDataPlus=new ntk2DData(ntkIntDimension(oldDataSize.x+2, oldDataSize.y+2));
  
  unsigned char* oldDataBuffer=(unsigned char*)m_input->getBuffer();
  unsigned char* oldDataPlusBuffer=(unsigned char*)oldDataPlus->getBuffer();

  for(j=0;j<oldDataSize.y;j++){
      memcpy(oldDataPlusBuffer+(j+1)*(oldDataSize.x+2)+1, oldDataBuffer+j*oldDataSize.x, oldDataSize.x);
  }

  ntk2DBSpline3Transform *trans=new ntk2DBSpline3Transform();
  ntkMatrix* splineInput=trans->doForwardTransform(oldDataPlus, CURRENT_RES);

  delete oldDataPlus;

  ntk2DData *output=new ntk2DData(newDataSize, newDataThickness);
  unsigned char *outputBuffer=(unsigned char*)output->getBuffer();

  for(i=0;i<newDataSize.x;i++){
    for(j=0;j<newDataSize.y;j++){
      tempValue=trans->getInterpolationValue(splineInput, ((i+0.5)*(newDataThickness.x)-0.5)/(oldDataThickness.x)+1.0,((j+0.5)*(newDataThickness.y)-0.5)/(oldDataThickness.y)+1.0);
      if(tempValue>255)tempValue=255.0;
      else if(tempValue<0)tempValue=0.0;
      
      *(outputBuffer+j*newDataSize.x+i)=(unsigned char)tempValue;
    
    }
  }

  delete trans;
  delete splineInput;
  return output;
}

double ntk2DDataResizer::getInterpolationValue(double x, double y){
  ntkIntDimension inputSize=m_input->getDataSize();
  unsigned char* inputBuffer=(unsigned char*)m_input->getBuffer();
  
  int xint=(int)x;
  int yint=(int)y;
  float xsmall=(float)x-(float)xint;
  float ysmall=(float)y-(float)yint;

  if(xint>=inputSize.x-1||yint>=inputSize.y-1||xint<0||yint<0){
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


  return (double)((1-xsmall)*(1-ysmall)* *(inputBuffer+yint*inputSize.x+xint)+
          (1-xsmall)*ysmall* *(inputBuffer+(yint+1)*inputSize.x+xint)+
          xsmall*(1-ysmall)* *(inputBuffer+yint*inputSize.x+xint+1)+
          xsmall*ysmall* *(inputBuffer+(yint+1)*inputSize.x+xint+1));
          
  
}
