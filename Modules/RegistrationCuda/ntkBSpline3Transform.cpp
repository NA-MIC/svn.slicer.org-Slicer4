#include "ntkBSpline3Transform.h"

ntkBSpline3Transform::ntkBSpline3Transform(){
  m_func=new ntkBSpline3Function();
  m_factor=1.0;
}

ntkBSpline3Transform::~ntkBSpline3Transform(){
  delete m_func;
}

ntkTensor* ntkBSpline3Transform::doForwardTransform(ntk3DData* inputImage, ntkBSpline3TransformMode mode=CURRENT_RES){
  int i;
  ntkIntDimension inputDim=inputImage->getDataSize();
  if(mode==CURRENT_RES){
    ntkTensor *tenImage= new ntkTensor(inputImage);
    Matrix matCx(inputDim.x,inputDim.x,0.0);
    Matrix matCxI;
    Matrix matCy(inputDim.y,inputDim.y,0.0);
    Matrix matCyI;
    Matrix matCz(inputDim.z,inputDim.z,0.0);
    Matrix matCzI;
    
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

    for(i=0;i<inputDim.z;i++){
      matCz(i,i)=1;
    }
    
    for(i=0;i<inputDim.z-1;i++){
      matCz(i,i+1)=0.25;
    }
    
    for(i=0;i<inputDim.z-1;i++){
      matCz(i+1,i)=0.25;
    }

    matCxI=matCx.inverse();
    matCyI=matCy.inverse();
    matCzI=matCz.inverse();
    
    ntkTensor *tenTemp1=tenImage->multiplyMatrixLeft(matCxI);
    tenTemp1->switchAxisForward();
    
    ntkTensor *tenTemp2=tenTemp1->multiplyMatrixLeft(matCzI);
    delete tenTemp1;
    tenTemp2->switchAxisForward();
    ntkTensor *tenResult=tenTemp2->multiplyMatrixLeft(matCyI);
    delete tenTemp2;
    tenResult->switchAxisForward();
    
    delete tenImage;

    return tenResult;
    
  }else if(mode==LOWER_RES){
    
  }
  return 0;
}

ntk3DData* ntkBSpline3Transform::doReverseTransform(ntkTensor* inputSpline){
  int i,j,k;
  ntkIntDimension inputDim=inputSpline->getTensorSize();
  Matrix matCx(inputDim.x,inputDim.x,0.0);
  Matrix matCy(inputDim.y,inputDim.y,0.0);
  Matrix matCz(inputDim.z,inputDim.z,0.0);
  
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
    
  for(i=0;i<inputDim.z-1;i++){
    matCz(i+1,i)=0.25;
  }

  for(i=0;i<inputDim.z;i++){
    matCz(i,i)=1;
  }
  
  for(i=0;i<inputDim.z-1;i++){
    matCz(i,i+1)=0.25;
  }
    
  for(i=0;i<inputDim.z-1;i++){
    matCz(i+1,i)=0.25;
  }

  //float det=-(matCx.determinant()).value();
  //m_factor=1.0/(pow(det,1.0/inputDim.x));
  //matCx=matCx*m_factor;
  //matCy=matCy*m_factor;
  //matCz=matCz*m_factor;
    
  ntkTensor *tenTemp1=inputSpline->multiplyMatrixLeft(matCx);
  tenTemp1->switchAxisForward();
  ntkTensor *tenTemp2=tenTemp1->multiplyMatrixLeft(matCz);
  delete tenTemp1;
  tenTemp2->switchAxisForward();
  ntkTensor *tenResult=tenTemp2->multiplyMatrixLeft(matCy);
  delete tenTemp2;
  tenResult->switchAxisForward();
  
  ntkIntDimension resultDim=tenResult->getTensorSize();
  unsigned char *resultBuffer=(unsigned char*)malloc(resultDim.x*resultDim.y*resultDim.z*sizeof(unsigned char));
  float tempValue=0.0;
  for(i=0;i<resultDim.x;i++){
    for(j=0;j<resultDim.y;j++){
      for(k=0;k<resultDim.z;k++){
    tempValue=tenResult->getValue(i,j,k);
    if(tempValue<0.0)tempValue=0.0;
    if(tempValue>255.0)tempValue=255.0;
    *(resultBuffer+k*resultDim.x*resultDim.y+j*resultDim.x+i)=(unsigned char)(round(tempValue));
      }
    }
  }
  delete tenResult;
  ntk3DData* resultData=new ntk3DData(resultDim, resultBuffer);
  
  return resultData;
}

ntkTensor *ntkBSpline3Transform::doSplineToSplineHigherResTransform(ntkTensor *inputSpline){
  int i,j,k;
  ntkIntDimension inputSize=inputSpline->getTensorSize();
  ntkIntDimension outputSize=ntkIntDimension(inputSize.x*2-1, inputSize.y*2-1, inputSize.z*2-1);
  
  ntkTensor *outputSpline=new ntkTensor(outputSize.x, outputSize.y, outputSize.z);
  
  float *outputSplineBuffer=outputSpline->getTensorPointer();

  Matrix matCx(outputSize.x, inputSize.y, 0.0);
  Matrix matAx(outputSize.x, outputSize.y, 0.0);
  Matrix matCy(outputSize.z, inputSize.x, 0.0);
  Matrix matAy(outputSize.z, outputSize.x, 0.0);
  Matrix matCz(outputSize.y, inputSize.z, 0.0);
  Matrix matAz(outputSize.y, outputSize.z, 0.0);
  
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

  for(i=0;i<outputSize.z;i++){
    matAz(i,i)=1;
  }
  
  for(i=0;i<outputSize.z-1;i++){
    matAz(i,i+1)=0.25;
  }
  
  for(i=0;i<outputSize.z-1;i++){
    matAz(i+1,i)=0.25;
  }

  //float det=-(matA.determinant()).value();
  //m_factor=1.0/(pow(det,1.0/outputSize.x));
  //matA=matA*m_factor;

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


  for(i=0;i<outputSize.z/2;i++){
      matCz(i*2,i)=1;
      matCz(i*2,i+1)=0.25;
      matCz(i*2+1,i)=0.71875;
      matCz(i*2+1,i+1)=0.71875;
  }

  for(i=1;i<outputSize.z/2+1;i++){
      matCz(i*2,i-1)=0.25;
  }

  for(i=0;i<outputSize.z/2-1;i++){
    matCz(i*2+1,i+2)=0.03125;
  }

  for(i=1;i<outputSize.z/2;i++){
    matCz(i*2+1,i-1)=0.03125;
  }
  
  matCz(outputSize.z-1,outputSize.z/2-1)=0.25;
  matCz(outputSize.z-1,outputSize.z/2)=1;
  
  

  //matCx=matC*m_factor;

  Matrix matAxinv=matAx.pseudo_inverse();
  Matrix matAyinv=matAy.pseudo_inverse();
  Matrix matAzinv=matAz.pseudo_inverse();
  Matrix matTx=matAxinv*matCx;
  Matrix matTy=matAyinv*matCy;
  Matrix matTz=matAzinv*matCz;
  
  ntkTensor *tenTemp1=inputSpline->multiplyMatrixLeft(matTy);
  tenTemp1->switchAxisForward();
  ntkTensor *tenTemp2=tenTemp1->multiplyMatrixLeft(matTz);
  delete tenTemp1;
  tenTemp2->switchAxisForward();
  ntkTensor *tenResult=tenTemp2->multiplyMatrixLeft(matTx);
  delete tenTemp2;
  tenResult->switchAxisForward();

  for(i=0;i<outputSize.x;i++){
    for(j=0;j<outputSize.y;j++){
      for(k=0;k<outputSize.z;k++){
    *(outputSplineBuffer+k*outputSize.x*outputSize.y+j*outputSize.x+i)=tenResult->getValue(i,j,k);
      }
    }
  }
  delete tenResult;
  
  return outputSpline;
}

ntk3DData *ntkBSpline3Transform::doReverseHigherResTransform(ntkTensor *inputSpline){
  int i,j,k;
  ntkIntDimension inputSize=inputSpline->getTensorSize();
  ntkIntDimension outputSize=ntkIntDimension(inputSize.x*2-1, inputSize.y*2-1, inputSize.z*2-1);

  Matrix matCx(outputSize.x, inputSize.y, 0.0);
  Matrix matCy(outputSize.y, inputSize.z, 0.0);
  Matrix matCz(outputSize.z, inputSize.x, 0.0);

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


  for(i=0;i<outputSize.z/2;i++){
      matCz(i*2,i)=1;
      matCz(i*2,i+1)=0.25;
      matCz(i*2+1,i)=0.71875;
      matCz(i*2+1,i+1)=0.71875;
  }

  for(i=1;i<outputSize.z/2+1;i++){
      matCz(i*2,i-1)=0.25;
  }

  for(i=0;i<outputSize.z/2-1;i++){
    matCz(i*2+1,i+2)=0.03125;
  }

  for(i=1;i<outputSize.z/2;i++){
    matCz(i*2+1,i-1)=0.03125;
  }
  
  matCz(outputSize.z-1,outputSize.z/2-1)=0.25;
  matCz(outputSize.z-1,outputSize.z/2)=1;

  ntkTensor *tenTemp1=inputSpline->multiplyMatrixLeft(matCx);
  tenTemp1->switchAxisForward();
  ntkTensor *tenTemp2=tenTemp1->multiplyMatrixLeft(matCz);
  delete tenTemp1;
  tenTemp2->switchAxisForward();
  ntkTensor *tenResult=tenTemp2->multiplyMatrixLeft(matCy);
  delete tenTemp2;
  tenResult->switchAxisForward();

  ntkIntDimension resultDim=tenResult->getTensorSize();
  unsigned char *resultBuffer=(unsigned char*)malloc(resultDim.x*resultDim.y*resultDim.z*sizeof(unsigned char));
  float tempValue=0.0;
  for(i=0;i<resultDim.x;i++){
    for(j=0;j<resultDim.y;j++){
      for(k=0;k<resultDim.z;k++){
    tempValue=tenResult->getValue(i,j,k);
    if(tempValue<0.0)tempValue=0.0;
    if(tempValue>255.0)tempValue=255.0;
    *(resultBuffer+k*resultDim.x*resultDim.y+j*resultDim.x+i)=(unsigned char)(round(tempValue));
      }
    }
  }
  delete tenResult;
  ntk3DData* resultData=new ntk3DData(resultDim, resultBuffer);
  
  return resultData;
}

float ntkBSpline3Transform::getInterpolationValue(ntkTensor* inputSpline, float x, float y, float z){
  int i,j,k;
  float result=0.0;
  ntkIntDimension splineDim=inputSpline->getTensorSize();
  for(i=(int)x-1;i<(int)x+3;i++){
    if(i<0||i>=splineDim.x)continue;
    for(j=(int)y-1;j<(int)y+3;j++){
      if(j<0||j>=splineDim.y)continue;
      for(k=(int)z-1;k<(int)z+3;k++){
    if(k<0||k>=splineDim.z)continue;
    result+=(inputSpline->getValue(i,j,k)*m_func->getValue(x-(float)i)*m_func->getValue(y-(float)j)*m_func->getValue(z-(float)k));
      }
    }
  }
  return result;
}
