#include "ntk2DElasticTransformation.h"

ntk2DElasticTransformation::ntk2DElasticTransformation(){

}

ntk2DElasticTransformation::~ntk2DElasticTransformation(){
  
}

void ntk2DElasticTransformation::loadData(ntk2DData* input){
  m_input=input;
  m_inputSize=m_input->getDataSize();
}

ntk2DData* ntk2DElasticTransformation::applyTransformation(ntkDeformationSpline* splineParam, int splineSizeLevel){
  int i,j,k,a,b,c;
  ntk2DData* output=new ntk2DData(m_inputSize);
  
  ntk2DData* inputPlus=new ntk2DData(ntkIntDimension(m_inputSize.x+2, m_inputSize.y+2));
  float newX, newY;

  float tempPow;
  int tempSize; 
  
  if(splineSizeLevel==0){
    tempPow=1;
    tempSize=1;
  }else{
    tempPow=pow(2, splineSizeLevel);
    tempSize=(int)pow(2, -splineSizeLevel);
  }
  ntkIntDimension splineSize((int)(tempPow*(m_inputSize.x+1)+1), (int)(tempPow*(m_inputSize.y+1)+1), 1);

  unsigned char* outputBuffer=(unsigned char*)output->getBuffer();
  unsigned char* inputBuffer=(unsigned char*)m_input->getBuffer();
  unsigned char* inputPlusBuffer=(unsigned char*)inputPlus->getBuffer();
 

  for(j=0;j<m_inputSize.y;j++){
      memcpy(inputPlusBuffer+(j+1)*(m_inputSize.x+2)+1, inputBuffer+j*m_inputSize.x, m_inputSize.x);
      
  }
    
  ntk2DBSpline3Transform *trans=new ntk2DBSpline3Transform();
  ntkMatrix* splineInput=trans->doForwardTransform(inputPlus, CURRENT_RES);

  //printf("inside here ok\n");fflush(stdout);
  //printf("input size: %d %d %d\n", m_inputSize.x, m_inputSize.y, m_inputSize.z);fflush(stdout);
  //printf("tempSize: %d\n", tempSize);fflush(stdout);

  float relPosX, relPosY;
  float functemp;
  ntkBSpline3Function *func = new ntkBSpline3Function();
  float tempValue;
 
  float *splineParamBuffer=splineParam->getBuffer();

  for(i=0;i<m_inputSize.x;i++){
    for(j=0;j<m_inputSize.y;j++){
      newX=i; newY=j;
  relPosX=(float)i*tempPow+tempPow;
  relPosY=(float)j*tempPow+tempPow;
  for(a=(int)relPosX-1;a<(int)relPosX+3;a++){
    if(a<0||a>=splineSize.x)continue;
    for(b=(int)relPosY-1;b<(int)relPosY+3;b++){
      if(b<0||b>=splineSize.y)continue;
        functemp=func->getValue(relPosX-(float)a)*func->getValue(relPosY-(float)b);
        newX+=*(splineParamBuffer+b*splineSize.x+a)*functemp;
        newY+=*(splineParamBuffer+splineSize.x*splineSize.y+b*splineSize.x+a)*functemp;
    
        //printf("%d %d %lf %lf\n", i,j,newX, newY);
    }
  }
  
  tempValue=trans->getInterpolationValue(splineInput, newX+1.0, newY+1.0);
  if(tempValue<0)tempValue=0;
  if(tempValue>255)tempValue=255;
  *(outputBuffer+j*m_inputSize.x+i)=(unsigned char)tempValue;

    }
  }
  
  delete func;
  delete trans;
  delete inputPlus;
  delete splineInput;
  return output;
}

ntkFloatDimension ntk2DElasticTransformation::getNewPositionFromOld(ntkDeformationSpline* splineParam, int splineSizeLevel, ntkFloatDimension oldPosition){
  int a,b,c;
  ntkFloatDimension newPosition=oldPosition;

  float tempPow;
  int tempSize; 
  
  if(splineSizeLevel==0){
    tempPow=1;
    tempSize=1;
  }else{
    tempPow=pow(2, splineSizeLevel);
    tempSize=(int)pow(2, -splineSizeLevel);
  }
  ntkIntDimension splineSize((int)(tempPow*(m_inputSize.x+1)+1), (int)(tempPow*(m_inputSize.y+1)+1), 1);

  float *splineParamBuffer=splineParam->getBuffer();
  float relPosX, relPosY;
  float functemp;
  ntkBSpline3Function *func = new ntkBSpline3Function();
  
  relPosX=(float)oldPosition.x*tempPow+tempPow;
  relPosY=(float)oldPosition.y*tempPow+tempPow;
  
  for(a=(int)relPosX-1;a<(int)relPosX+3;a++){
    if(a<0||a>=splineSize.x)continue;
    for(b=(int)relPosY-1;b<(int)relPosY+3;b++){
      if(b<0||b>=splineSize.y)continue;
    functemp=func->getValue(relPosX-(float)a)*func->getValue(relPosY-(float)b);
  newPosition.x+=*(splineParamBuffer+b*splineSize.x+a)*functemp;
  newPosition.y+=*(splineParamBuffer+splineSize.x*splineSize.y+b*splineSize.x+a)*functemp;
  
    }
  }

  //printf("%lf %lf %lf %lf %lf %lf\n", oldPosition.x, oldPosition.y, oldPosition.z, newPosition.x, newPosition.y, newPosition.z);

  delete func;
  return newPosition;
}

ntk2DData* ntk2DElasticTransformation::getDeformationFieldImage(ntkDeformationSpline* splineParam, int splineSizeLevel, float weight){
  int i,j,k,a,b,c;
  ntk2DData* output=new ntk2DData(m_inputSize);
  ntk2DData* inputPlus=new ntk2DData(ntkIntDimension(m_inputSize.x+2, m_inputSize.y+2));
  float newX, newY;

  float tempPow;
  int tempSize; 
  
  if(splineSizeLevel==0){
    tempPow=1;
    tempSize=1;
  }else{
    tempPow=pow(2, splineSizeLevel);
    tempSize=(int)pow(2, -splineSizeLevel);
  }
  ntkIntDimension splineSize((int)(tempPow*(m_inputSize.x+1)+1), (int)(tempPow*(m_inputSize.y+1)+1), 1);

  unsigned char* outputBuffer=(unsigned char*)output->getBuffer();
  unsigned char* inputBuffer=(unsigned char*)m_input->getBuffer();
  unsigned char* inputPlusBuffer=(unsigned char*)inputPlus->getBuffer();
  
  for(j=0;j<m_inputSize.y;j++){
    memcpy(inputPlusBuffer+(j+1)*(m_inputSize.x+2)+1, inputBuffer+j*m_inputSize.x, m_inputSize.x);
  }

  ntk2DBSpline3Transform *trans=new ntk2DBSpline3Transform();
  ntkMatrix* splineInput=trans->doForwardTransform(inputPlus, CURRENT_RES);

  //printf("inside here ok\n");fflush(stdout);
  //printf("input size: %d %d %d\n", m_inputSize.x, m_inputSize.y, m_inputSize.z);fflush(stdout);
  //printf("tempSize: %d\n", tempSize);fflush(stdout);

  float relPosX, relPosY;
  float functemp;
  ntkBSpline3Function *func = new ntkBSpline3Function();
  float tempValue;
 
  float *splineParamBuffer=splineParam->getBuffer();

  for(i=0;i<m_inputSize.x;i++){
    for(j=0;j<m_inputSize.y;j++){
        newX=0; newY=0;
  relPosX=(float)i*tempPow+tempPow;
  relPosY=(float)j*tempPow+tempPow;
  for(a=(int)relPosX-1;a<(int)relPosX+3;a++){
    if(a<0||a>=splineSize.x)continue;
    for(b=(int)relPosY-1;b<(int)relPosY+3;b++){
      if(b<0||b>=splineSize.y)continue;
      functemp=func->getValue(relPosX-(float)a)*func->getValue(relPosY-(float)b);
      newX+=*(splineParamBuffer+b*splineSize.x+a)*functemp;
      newY+=*(splineParamBuffer+splineSize.x*splineSize.y+b*splineSize.x+a)*functemp;
    }
  }
  tempValue=sqrt(newX*newX+newY*newY)*weight;
  if(tempValue<0)tempValue=0;
  if(tempValue>255)tempValue=255;
  *(outputBuffer+j*m_inputSize.x+i)=(unsigned char)(tempValue);
    }
  }
  
  delete func;
  delete trans;
  delete inputPlus;
  delete splineInput;
  return output;
}

ntkDeformationSpline *ntk2DElasticTransformation::getReverseTransformationSpline(ntkDeformationSpline *inputSpline){
  int i,j,k,a,b,c;
  float relPosX, relPosY;
  float functemp;
  ntkBSpline3Function *func = new ntkBSpline3Function();
  float tempValue;

  float *inputSplineBuffer=inputSpline->getBuffer();
  ntkIntDimension splineSize=inputSpline->getSize();

  float *newX=(float*)malloc(splineSize.x*splineSize.y*sizeof(float));
  float *newY=(float*)malloc(splineSize.x*splineSize.y*sizeof(float));
  
  for(i=0;i<splineSize.x;i++){
    for(j=0;j<splineSize.y;j++){
      *(newX+j*splineSize.x+i)=(float)i; 
      *(newY+j*splineSize.x+i)=(float)j; 
  relPosX=(float)i;
  relPosY=(float)j;
  for(a=(int)relPosX-1;a<(int)relPosX+3;a++){
    if(a<0||a>=splineSize.x)continue;
    for(b=(int)relPosY-1;b<(int)relPosY+3;b++){
      if(b<0||b>=splineSize.y)continue;
      functemp=func->getValue(relPosX-(float)a)*func->getValue(relPosY-(float)b);
      *(newX+j*splineSize.x+i)+=*(inputSplineBuffer+b*splineSize.x+a)*functemp;
      *(newY+j*splineSize.x+i)+=*(inputSplineBuffer+splineSize.x*splineSize.y+b*splineSize.x+a)*functemp;
      
    }
  }
  //printf("%d %d %d %lf %lf %lf\n", i,j,k, *(newX+k*splineSize.x*splineSize.y+j*splineSize.x+i), *(newY+k*splineSize.x*splineSize.y+j*splineSize.x+i), *(newZ+k*splineSize.x*splineSize.y+j*splineSize.x+i));
    
    }
  }
  
  ntkMatrix* newXMatrix=new ntkMatrix(newX, splineSize);
  free(newX);
  ntkMatrix* newYMatrix=new ntkMatrix(newY, splineSize);
  free(newY);
   
  Matrix matTx(splineSize.x,splineSize.x,0.0);
  Matrix matTxI;
  Matrix matTy(splineSize.y,splineSize.y,0.0);
  Matrix matTyI;
 
  for(i=0;i<splineSize.x;i++){
    for(j=0;j<splineSize.y;j++){
      matTx(i,j)=func->getValue(0);
    }
  }

  delete func;
  delete newXMatrix;
  delete newYMatrix;
    
  return 0;
}

