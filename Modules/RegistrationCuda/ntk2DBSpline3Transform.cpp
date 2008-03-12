#include "ntk2DBSpline3Transform.h"

ntk2DBSpline3Transform::ntk2DBSpline3Transform(){
  m_func=new ntkBSpline3Function();
  m_factor=1.0;
}

ntk2DBSpline3Transform::~ntk2DBSpline3Transform(){
  delete m_func;
}

ntkMatrix* ntk2DBSpline3Transform::doForwardTransform(ntk2DData* inputImage, ntk2DBSpline3TransformMode mode=CURRENT_RES){
  int i;
  ntkIntDimension inputDim=inputImage->getDataSize();
  if(mode==CURRENT_RES){
    ntkMatrix *tenImage= new ntkMatrix(inputImage);
    Matrix matCx(inputDim.x,inputDim.x,0.0);
    Matrix matCxI;
    Matrix matCy(inputDim.y,inputDim.y,0.0);
    Matrix matCyI;
        
    for(i=0;i<inputDim.x;i++){
      matCx(i,i)=1;
    }
    
    for(i=0;i<inputDim.x-1;i++){
      matCx(i,i+1)=0.25;
    }
    
    for(i=0;i<inputDim.x-1;i++){
      matCx(i+1,i)=0.25;
    }

    for(i=0;i<inputDim.y;i++){
      matCy(i,i)=1;
    }
    
    for(i=0;i<inputDim.y-1;i++){
      matCy(i,i+1)=0.25;
    }
    
    for(i=0;i<inputDim.y-1;i++){
      matCy(i+1,i)=0.25;
    }

    matCxI=matCx.inverse();
    matCyI=matCy.inverse();
    
    
    ntkMatrix *tenTemp1=tenImage->multiplyMatrixLeft(matCxI);
    tenTemp1->switchAxisForward();
    
    ntkMatrix *tenResult=tenTemp1->multiplyMatrixLeft(matCyI);
    delete tenTemp1;
    tenResult->switchAxisForward();
        
    delete tenImage;

    return tenResult;
    
  }else if(mode==LOWER_RES){
    
  }
  return 0;
}

ntk2DData* ntk2DBSpline3Transform::doReverseTransform(ntkMatrix* inputSpline){
  int i,j;
  ntkIntDimension inputDim=inputSpline->getMatrixSize();
  Matrix matCx(inputDim.x,inputDim.x,0.0);
  Matrix matCy(inputDim.y,inputDim.y,0.0);
    
  for(i=0;i<inputDim.x;i++){
    matCx(i,i)=1;
  }
  
  for(i=0;i<inputDim.x-1;i++){
    matCx(i,i+1)=0.25;
  }
    
  for(i=0;i<inputDim.x-1;i++){
    matCx(i+1,i)=0.25;
  }

  for(i=0;i<inputDim.x;i++){
    matCx(i,i)=1;
  }
  
  for(i=0;i<inputDim.y-1;i++){
    matCy(i,i+1)=0.25;
  }
    
  for(i=0;i<inputDim.y-1;i++){
    matCy(i+1,i)=0.25;
  }
  
  for(i=0;i<inputDim.y;i++){
    matCy(i,i)=1;
  }
  
  for(i=0;i<inputDim.y-1;i++){
    matCy(i,i+1)=0.25;
  }
    
  ntkMatrix *tenTemp1=inputSpline->multiplyMatrixLeft(matCx);
  tenTemp1->switchAxisForward();
  ntkMatrix *tenResult=tenTemp1->multiplyMatrixLeft(matCy);
  delete tenTemp1;
  tenResult->switchAxisForward();
    
  ntkIntDimension resultDim=tenResult->getMatrixSize();
  unsigned char *resultBuffer=(unsigned char*)malloc(resultDim.x*resultDim.y*sizeof(unsigned char));
  float tempValue=0.0;
  for(i=0;i<resultDim.x;i++){
    for(j=0;j<resultDim.y;j++){
      tempValue=tenResult->getValue(i,j);
      if(tempValue<0.0)tempValue=0.0;
      if(tempValue>255.0)tempValue=255.0;
      *(resultBuffer+j*resultDim.x+i)=(unsigned char)(round(tempValue));
    }
  }
  delete tenResult;
  ntk2DData* resultData=new ntk2DData(resultDim, resultBuffer);
  
  return resultData;
}

ntkMatrix *ntk2DBSpline3Transform::doSplineToSplineHigherResTransform(ntkMatrix *inputSpline){
  int i,j,k;
  ntkIntDimension inputSize=inputSpline->getMatrixSize();
  ntkIntDimension outputSize=ntkIntDimension(inputSize.x*2-1, inputSize.y*2-1);
  
  ntkMatrix *outputSpline=new ntkMatrix(outputSize.x, outputSize.y);
  
  float *outputSplineBuffer=outputSpline->getMatrixPointer();

  Matrix matCx(outputSize.x, inputSize.y, 0.0);
  Matrix matAx(outputSize.x, outputSize.y, 0.0);
  Matrix matCy(outputSize.y, inputSize.x, 0.0);
  Matrix matAy(outputSize.y, outputSize.x, 0.0);
    
  for(i=0;i<outputSize.x;i++){
    matAx(i,i)=1;
  }
  
  for(i=0;i<outputSize.x-1;i++){
    matAx(i,i+1)=0.25;
  }
  
  for(i=0;i<outputSize.x-1;i++){
    matAx(i+1,i)=0.25;
  }

  for(i=0;i<outputSize.y;i++){
    matAy(i,i)=1;
  }
  
  for(i=0;i<outputSize.y-1;i++){
    matAy(i,i+1)=0.25;
  }
  
  for(i=0;i<outputSize.y-1;i++){
    matAy(i+1,i)=0.25;
  }


  for(i=0;i<outputSize.x/2;i++){
      matCx(i*2,i)=1;
      matCx(i*2,i+1)=0.25;
      matCx(i*2+1,i)=0.71875;
      matCx(i*2+1,i+1)=0.71875;
  }

  for(i=1;i<outputSize.x/2+1;i++){
      matCx(i*2,i-1)=0.25;
  }

  for(i=0;i<outputSize.x/2-1;i++){
    matCx(i*2+1,i+2)=0.03125;
  }

  for(i=1;i<outputSize.x/2;i++){
    matCx(i*2+1,i-1)=0.03125;
  }

  matCx(outputSize.x-1,outputSize.x/2-1)=0.25;
  matCx(outputSize.x-1,outputSize.x/2)=1;
  

  for(i=0;i<outputSize.y/2;i++){
      matCy(i*2,i)=1;
      matCy(i*2,i+1)=0.25;
      matCy(i*2+1,i)=0.71875;
      matCy(i*2+1,i+1)=0.71875;
  }

  for(i=1;i<outputSize.y/2+1;i++){
      matCy(i*2,i-1)=0.25;
  }

  for(i=0;i<outputSize.y/2-1;i++){
    matCy(i*2+1,i+2)=0.03125;
  }

  for(i=1;i<outputSize.y/2;i++){
    matCy(i*2+1,i-1)=0.03125;
  }

  matCy(outputSize.y-1,outputSize.y/2-1)=0.25;
  matCy(outputSize.y-1,outputSize.y/2)=1;


  Matrix matAxinv=matAx.pseudo_inverse();
  Matrix matAyinv=matAy.pseudo_inverse();
  Matrix matTx=matAxinv*matCx;
  Matrix matTy=matAyinv*matCy;
    
  ntkMatrix *tenTemp1=inputSpline->multiplyMatrixLeft(matTy);
  tenTemp1->switchAxisForward();
  ntkMatrix *tenResult=tenTemp1->multiplyMatrixLeft(matTy);
  delete tenTemp1;
  tenResult->switchAxisForward();
  
  for(i=0;i<outputSize.x;i++){
    for(j=0;j<outputSize.y;j++){
      *(outputSplineBuffer+j*outputSize.x+i)=tenResult->getValue(i,j);
    }
  }
  delete tenResult;
  
  return outputSpline;
}

ntk2DData *ntk2DBSpline3Transform::doReverseHigherResTransform(ntkMatrix *inputSpline){
  int i,j,k;
  ntkIntDimension inputSize=inputSpline->getMatrixSize();
  ntkIntDimension outputSize=ntkIntDimension(inputSize.x*2-1, inputSize.y*2-1);

  Matrix matCx(outputSize.x, inputSize.y, 0.0);
  Matrix matCy(outputSize.y, inputSize.z, 0.0);
 
  for(i=0;i<outputSize.x/2;i++){
      matCx(i*2,i)=1;
      matCx(i*2,i+1)=0.25;
      matCx(i*2+1,i)=0.71875;
      matCx(i*2+1,i+1)=0.71875;
  }

  for(i=1;i<outputSize.x/2+1;i++){
      matCx(i*2,i-1)=0.25;
  }

  for(i=0;i<outputSize.x/2-1;i++){
    matCx(i*2+1,i+2)=0.03125;
  }

  for(i=1;i<outputSize.x/2;i++){
    matCx(i*2+1,i-1)=0.03125;
  }
  
  matCx(outputSize.x-1,outputSize.x/2-1)=0.25;
  matCx(outputSize.x-1,outputSize.x/2)=1;
  
  for(i=0;i<outputSize.y/2;i++){
      matCy(i*2,i)=1;
      matCy(i*2,i+1)=0.25;
      matCy(i*2+1,i)=0.71875;
      matCy(i*2+1,i+1)=0.71875;
  }

  for(i=1;i<outputSize.y/2+1;i++){
      matCy(i*2,i-1)=0.25;
  }

  for(i=0;i<outputSize.y/2-1;i++){
    matCy(i*2+1,i+2)=0.03125;
  }

  for(i=1;i<outputSize.y/2;i++){
    matCy(i*2+1,i-1)=0.03125;
  }

  matCy(outputSize.y-1,outputSize.y/2-1)=0.25;
  matCy(outputSize.y-1,outputSize.y/2)=1;


  ntkMatrix *tenTemp1=inputSpline->multiplyMatrixLeft(matCx);
  tenTemp1->switchAxisForward();
  ntkMatrix *tenResult=tenTemp1->multiplyMatrixLeft(matCy);
  delete tenTemp1;
  tenResult->switchAxisForward();
  
  ntkIntDimension resultDim=tenResult->getMatrixSize();
  unsigned char *resultBuffer=(unsigned char*)malloc(resultDim.x*resultDim.y*sizeof(unsigned char));
  float tempValue=0.0;
  for(i=0;i<resultDim.x;i++){
    for(j=0;j<resultDim.y;j++){
      tempValue=tenResult->getValue(i,j);
      if(tempValue<0.0)tempValue=0.0;
      if(tempValue>255.0)tempValue=255.0;
      *(resultBuffer+j*resultDim.x+i)=(unsigned char)(round(tempValue));
    }
  }
  delete tenResult;
  ntk2DData* resultData=new ntk2DData(resultDim, resultBuffer);
  
  return resultData;
}

float ntk2DBSpline3Transform::getInterpolationValue(ntkMatrix* inputSpline, float x, float y){
  int i,j,k;
  float result=0.0;
  ntkIntDimension splineDim=inputSpline->getMatrixSize();
  for(i=(int)x-1;i<(int)x+3;i++){
    if(i<0||i>=splineDim.x)continue;
    for(j=(int)y-1;j<(int)y+3;j++){
      if(j<0||j>=splineDim.y)continue;
      result+=(inputSpline->getValue(i,j)*m_func->getValue(x-(float)i)*m_func->getValue(y-(float)j));
      /*
      if(x==21 && y==40){
    printf("%lf ", result);
      }
      */
    }
  }
  return result;
}
