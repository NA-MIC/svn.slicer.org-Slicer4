#include "ntkElasticTransformation.h"

ntkElasticTransformation::ntkElasticTransformation(){

}

ntkElasticTransformation::~ntkElasticTransformation(){
  
}

void ntkElasticTransformation::loadData(ntk3DData* input){
  m_input=input;
  m_inputSize=m_input->getDataSize();
}

ntk3DData* ntkElasticTransformation::applyTransformation(ntkDeformationSpline* splineParam, int splineSizeLevel){
  int i,j,k,a,b,c;
  ntk3DData* output=new ntk3DData(m_inputSize);
  ntk3DData* inputPlus=new ntk3DData(ntkIntDimension(m_inputSize.x+2, m_inputSize.y+2, m_inputSize.z+2));
  float newX, newY, newZ;

  float tempPow;
  int tempSize; 
  
  if(splineSizeLevel==0){
    tempPow=1;
    tempSize=1;
  }else{
    tempPow=pow(2, splineSizeLevel);
    tempSize=(int)pow(2, -splineSizeLevel);
  }
  ntkIntDimension splineSize((int)(tempPow*(m_inputSize.x+1)+1), (int)(tempPow*(m_inputSize.y+1)+1), (int)(tempPow*(m_inputSize.z+1)+1));

  unsigned char* outputBuffer=(unsigned char*)output->getBuffer();
  unsigned char* inputBuffer=(unsigned char*)m_input->getBuffer();
  unsigned char* inputPlusBuffer=(unsigned char*)inputPlus->getBuffer();
  
  for(k=0;k<m_inputSize.z;k++){
    for(j=0;j<m_inputSize.y;j++){
      memcpy(inputPlusBuffer+(k+1)*(m_inputSize.x+2)*(m_inputSize.y+2)+(j+1)*(m_inputSize.x+2)+1, inputBuffer+k*m_inputSize.x*m_inputSize.y+j*m_inputSize.x, m_inputSize.x);
    }
  }
  ntkBSpline3Transform *trans=new ntkBSpline3Transform();
  ntkTensor* splineInput=trans->doForwardTransform(inputPlus, CURRENT_RES);

  //printf("inside here ok\n");fflush(stdout);
  //printf("input size: %d %d %d\n", m_inputSize.x, m_inputSize.y, m_inputSize.z);fflush(stdout);
  //printf("tempSize: %d\n", tempSize);fflush(stdout);

  float relPosX, relPosY, relPosZ;
  float functemp;
  ntkBSpline3Function *func = new ntkBSpline3Function();
  float tempValue;
 
  float *splineParamBuffer=splineParam->getBuffer();

  for(i=0;i<m_inputSize.x;i++){
    for(j=0;j<m_inputSize.y;j++){
      for(k=0;k<m_inputSize.z;k++){
  newX=i; newY=j; newZ=k;
  relPosX=(float)i*tempPow+tempPow;
  relPosY=(float)j*tempPow+tempPow;
  relPosZ=(float)k*tempPow+tempPow;
  for(a=(int)relPosX-1;a<(int)relPosX+3;a++){
    if(a<0||a>=splineSize.x)continue;
    for(b=(int)relPosY-1;b<(int)relPosY+3;b++){
      if(b<0||b>=splineSize.y)continue;
      for(c=(int)relPosZ-1;c<(int)relPosZ+3;c++){
        if(c<0||c>=splineSize.z)continue;
        functemp=func->getValue(relPosX-(float)a)*func->getValue(relPosY-(float)b)*func->getValue(relPosZ-(float)c);
        newX+=*(splineParamBuffer+c*splineSize.x*splineSize.y+b*splineSize.x+a)*functemp;
        newY+=*(splineParamBuffer+splineSize.x*splineSize.y*splineSize.z+c*splineSize.x*splineSize.y+b*splineSize.x+a)*functemp;
        newZ+=*(splineParamBuffer+2*splineSize.x*splineSize.y*splineSize.z+c*splineSize.x*splineSize.y+b*splineSize.x+a)*functemp;
      }
    }
  }
  
  tempValue=trans->getInterpolationValue(splineInput, newX+1.0, newY+1.0, newZ+1.0);
  if(tempValue<0)tempValue=0;
  if(tempValue>255)tempValue=255;
  *(outputBuffer+k*m_inputSize.x*m_inputSize.y+j*m_inputSize.x+i)=(unsigned char)tempValue;
      }
    }
  }
  
  delete func;
  delete trans;
  delete inputPlus;
  delete splineInput;
  return output;
}

ntkFloatDimension ntkElasticTransformation::getNewPositionFromOld(ntkDeformationSpline* splineParam, int splineSizeLevel, ntkFloatDimension oldPosition, float weight=1.0){
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
  ntkIntDimension splineSize((int)(tempPow*(m_inputSize.x+1)+1), (int)(tempPow*(m_inputSize.y+1)+1), (int)(tempPow*(m_inputSize.z+1)+1));

  float *splineParamBuffer=splineParam->getBuffer();
  float relPosX, relPosY, relPosZ;
  float functemp;
  ntkBSpline3Function *func = new ntkBSpline3Function();
  
  relPosX=(float)oldPosition.x*tempPow+tempPow;
  relPosY=(float)oldPosition.y*tempPow+tempPow;
  relPosZ=(float)oldPosition.z*tempPow+tempPow;
  for(a=(int)relPosX-1;a<(int)relPosX+3;a++){
    if(a<0||a>=splineSize.x)continue;
    for(b=(int)relPosY-1;b<(int)relPosY+3;b++){
      if(b<0||b>=splineSize.y)continue;
      for(c=(int)relPosZ-1;c<(int)relPosZ+3;c++){
  if(c<0||c>=splineSize.z)continue;
  functemp=func->getValue(relPosX-(float)a)*func->getValue(relPosY-(float)b)*func->getValue(relPosZ-(float)c);
  newPosition.x+=*(splineParamBuffer+c*splineSize.x*splineSize.y+b*splineSize.x+a)*functemp*weight;
  newPosition.y+=*(splineParamBuffer+splineSize.x*splineSize.y*splineSize.z+c*splineSize.x*splineSize.y+b*splineSize.x+a)*functemp*weight;
  newPosition.z+=*(splineParamBuffer+2*splineSize.x*splineSize.y*splineSize.z+c*splineSize.x*splineSize.y+b*splineSize.x+a)*functemp*weight;
      }
    }
  }

  //printf("%lf %lf %lf %lf %lf %lf\n", oldPosition.x, oldPosition.y, oldPosition.z, newPosition.x, newPosition.y, newPosition.z);

  delete func;
  return newPosition;
}

ntk3DData* ntkElasticTransformation::getDeformationFieldImage(ntkDeformationSpline* splineParam, int splineSizeLevel, float weight){
  int i,j,k,a,b,c;
  ntk3DData* output=new ntk3DData(m_inputSize);
  ntk3DData* inputPlus=new ntk3DData(ntkIntDimension(m_inputSize.x+2, m_inputSize.y+2, m_inputSize.z+2));
  float newX, newY, newZ;

  float tempPow;
  int tempSize; 
  
  if(splineSizeLevel==0){
    tempPow=1;
    tempSize=1;
  }else{
    tempPow=pow(2, splineSizeLevel);
    tempSize=(int)pow(2, -splineSizeLevel);
  }
  ntkIntDimension splineSize((int)(tempPow*(m_inputSize.x+1)+1), (int)(tempPow*(m_inputSize.y+1)+1), (int)(tempPow*(m_inputSize.z+1)+1));

  unsigned char* outputBuffer=(unsigned char*)output->getBuffer();
  unsigned char* inputBuffer=(unsigned char*)m_input->getBuffer();
  unsigned char* inputPlusBuffer=(unsigned char*)inputPlus->getBuffer();
  
  for(k=0;k<m_inputSize.z;k++){
    for(j=0;j<m_inputSize.y;j++){
      memcpy(inputPlusBuffer+(k+1)*(m_inputSize.x+2)*(m_inputSize.y+2)+(j+1)*(m_inputSize.x+2)+1, inputBuffer+k*m_inputSize.x*m_inputSize.y+j*m_inputSize.x, m_inputSize.x);
    }
  }
  ntkBSpline3Transform *trans=new ntkBSpline3Transform();
  ntkTensor* splineInput=trans->doForwardTransform(inputPlus, CURRENT_RES);

  //printf("inside here ok\n");fflush(stdout);
  //printf("input size: %d %d %d\n", m_inputSize.x, m_inputSize.y, m_inputSize.z);fflush(stdout);
  //printf("tempSize: %d\n", tempSize);fflush(stdout);

  float relPosX, relPosY, relPosZ;
  float functemp;
  ntkBSpline3Function *func = new ntkBSpline3Function();
  float tempValue;
 
  float *splineParamBuffer=splineParam->getBuffer();

  for(i=0;i<m_inputSize.x;i++){
    for(j=0;j<m_inputSize.y;j++){
      for(k=0;k<m_inputSize.z;k++){
  newX=0; newY=0; newZ=0;
  relPosX=(float)i*tempPow+tempPow;
  relPosY=(float)j*tempPow+tempPow;
  relPosZ=(float)k*tempPow+tempPow;
  for(a=(int)relPosX-1;a<(int)relPosX+3;a++){
    if(a<0||a>=splineSize.x)continue;
    for(b=(int)relPosY-1;b<(int)relPosY+3;b++){
      if(b<0||b>=splineSize.y)continue;
      for(c=(int)relPosZ-1;c<(int)relPosZ+3;c++){
        if(c<0||c>=splineSize.z)continue;
        functemp=func->getValue(relPosX-(float)a)*func->getValue(relPosY-(float)b)*func->getValue(relPosZ-(float)c);
        newX+=*(splineParamBuffer+c*splineSize.x*splineSize.y+b*splineSize.x+a)*functemp;
        newY+=*(splineParamBuffer+splineSize.x*splineSize.y*splineSize.z+c*splineSize.x*splineSize.y+b*splineSize.x+a)*functemp;
        newZ+=*(splineParamBuffer+2*splineSize.x*splineSize.y*splineSize.z+c*splineSize.x*splineSize.y+b*splineSize.x+a)*functemp;
      }
    }
  }
  tempValue=sqrt(newX*newX+newY*newY+newZ*newZ)*weight;
  if(tempValue<0)tempValue=0;
  if(tempValue>255)tempValue=255;
  *(outputBuffer+k*m_inputSize.x*m_inputSize.y+j*m_inputSize.x+i)=(unsigned char)(tempValue);
      }
    }
  }
  
  delete func;
  delete trans;
  delete inputPlus;
  delete splineInput;
  return output;
}

ntkDeformationSpline *ntkElasticTransformation::getReverseTransformationSpline(ntkDeformationSpline *inputSpline){
  int i,j,k,a,b,c;
  float relPosX, relPosY, relPosZ;
  float functemp;
  ntkBSpline3Function *func = new ntkBSpline3Function();
  float tempValue;

  float *inputSplineBuffer=inputSpline->getBuffer();
  ntkIntDimension splineSize=inputSpline->getSize();

  float *newX=(float*)malloc(splineSize.x*splineSize.y*splineSize.z*sizeof(float));
  float *newY=(float*)malloc(splineSize.x*splineSize.y*splineSize.z*sizeof(float));
  float *newZ=(float*)malloc(splineSize.x*splineSize.y*splineSize.z*sizeof(float));

  for(i=0;i<splineSize.x;i++){
    for(j=0;j<splineSize.y;j++){
      for(k=0;k<splineSize.z;k++){
  *(newX+k*splineSize.x*splineSize.y+j*splineSize.x+i)=(float)i; 
  *(newY+k*splineSize.x*splineSize.y+j*splineSize.x+i)=(float)j; 
  *(newZ+k*splineSize.x*splineSize.y+j*splineSize.x+i)=(float)k;
  relPosX=(float)i;
  relPosY=(float)j;
  relPosZ=(float)k;
  for(a=(int)relPosX-1;a<(int)relPosX+3;a++){
    if(a<0||a>=splineSize.x)continue;
    for(b=(int)relPosY-1;b<(int)relPosY+3;b++){
      if(b<0||b>=splineSize.y)continue;
      for(c=(int)relPosZ-1;c<(int)relPosZ+3;c++){
        if(c<0||c>=splineSize.z)continue;
        functemp=func->getValue(relPosX-(float)a)*func->getValue(relPosY-(float)b)*func->getValue(relPosZ-(float)c);
        *(newX+k*splineSize.x*splineSize.y+j*splineSize.x+i)+=*(inputSplineBuffer+c*splineSize.x*splineSize.y+b*splineSize.x+a)*functemp;
        *(newY+k*splineSize.x*splineSize.y+j*splineSize.x+i)+=*(inputSplineBuffer+splineSize.x*splineSize.y*splineSize.z+c*splineSize.x*splineSize.y+b*splineSize.x+a)*functemp;
        *(newZ+k*splineSize.x*splineSize.y+j*splineSize.x+i)+=*(inputSplineBuffer+2*splineSize.x*splineSize.y*splineSize.z+c*splineSize.x*splineSize.y+b*splineSize.x+a)*functemp;
      }
    }
  }
  //printf("%d %d %d %lf %lf %lf\n", i,j,k, *(newX+k*splineSize.x*splineSize.y+j*splineSize.x+i), *(newY+k*splineSize.x*splineSize.y+j*splineSize.x+i), *(newZ+k*splineSize.x*splineSize.y+j*splineSize.x+i));
      }
    }
  }
  
  ntkTensor* newXTensor=new ntkTensor(newX, splineSize);
  free(newX);
  ntkTensor* newYTensor=new ntkTensor(newY, splineSize);
  free(newY);
  ntkTensor* newZTensor=new ntkTensor(newZ, splineSize);
  free(newZ);
  
  Matrix matTx(splineSize.x,splineSize.x,0.0);
  Matrix matTxI;
  Matrix matTy(splineSize.y,splineSize.y,0.0);
  Matrix matTyI;
  Matrix matTz(splineSize.z,splineSize.z,0.0);
  Matrix matCzI;

  for(i=0;i<splineSize.x;i++){
    for(j=0;j<splineSize.y;j++){
      matTx(i,j)=func->getValue(0);
    }
  }

  delete func;
  delete newXTensor;
  delete newYTensor;
  delete newZTensor;
  
  return 0;
}


/*
ntkFloatDimension ntkElasticTransformation::getDeformationFieldVector(ntkDeformationSpline *splineParam, int splineSizeLevel, ntkFloatDimension position){
  ntkFloatDimension outputVector;
  ntkFloatDimension tempVector;
  
  tempVector=getNewPositionFromOld();
  
}
*/
