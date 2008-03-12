#include "ntkTensor.h"

#define TENSOR_ACC(X,Y,Z) (*(m_tensor+ (Z) *m_tensorSize.x*m_tensorSize.y + (Y)* m_tensorSize.x + (X)))

ntkTensor::ntkTensor(int x, int y, int z){
  m_tensorSize=ntkIntDimension(x,y,z);
  m_tensor=(float*)malloc(x*y*z*sizeof(float));
  initialize();
  m_tenTemp=NULL;
}

ntkTensor::ntkTensor(ntk3DData* inputData){
  int i,j,k;
  m_tensorSize=inputData->getDataSize();
  m_tensor=(float*)malloc(m_tensorSize.x*m_tensorSize.y*m_tensorSize.z*sizeof(float));
  m_tenTemp=NULL;
  unsigned char* inputBuffer=(unsigned char*)(inputData->getBuffer());
  for(i=0;i<m_tensorSize.x;i++){
    for(j=0;j<m_tensorSize.y;j++){
      for(k=0;k<m_tensorSize.z;k++){
    TENSOR_ACC(i,j,k)=(float)(*(inputBuffer+k*m_tensorSize.x*m_tensorSize.y+j*m_tensorSize.x+i));
      }
    }
  }
}

ntkTensor::ntkTensor(float* inputBuffer, ntkIntDimension inputSize){
  m_tensorSize=inputSize;
  m_tensor=(float*)malloc(m_tensorSize.x*m_tensorSize.y*m_tensorSize.z*sizeof(float));
  m_tenTemp=NULL;
  memcpy(m_tensor, inputBuffer, m_tensorSize.x*m_tensorSize.y*m_tensorSize.z*sizeof(float));
}

ntkTensor::~ntkTensor(){
  if(m_tensor!=NULL)free(m_tensor);
}

void ntkTensor::initialize(){
  int i,j,k;
  for(i=0;i<m_tensorSize.x;i++){
    for(j=0;j<m_tensorSize.y;j++){
      for(k=0;k<m_tensorSize.z;k++){
    TENSOR_ACC(i,j,k)=0.0;
      }
    }
  }
}

ntkTensor* ntkTensor::multiplyMatrixLeft(Matrix matA){
  int i,j,k;
  ntkIntDimension matSize(matA.rows(),matA.cols());
  ntkTensor *dummy=NULL;

  if(matSize.y!=m_tensorSize.x){
    printf("Size does not match. matrix-tensor multiplication failed\n");
    return dummy;
  }
  
  ntkTensor *tenResult=new ntkTensor(matSize.x,m_tensorSize.y, m_tensorSize.z);
    
  Matrix matTemp(m_tensorSize.x, m_tensorSize.y);
  Matrix matResTemp;
  
  //float* tenResPointer;
  
  for(k=0;k<m_tensorSize.z;k++){
    for(i=0;i<m_tensorSize.x;i++){
      for(j=0;j<m_tensorSize.y;j++){
    matTemp(i,j)=TENSOR_ACC(i,j,k);
      }
    }
    matResTemp=matA*matTemp;
    //tenResPointer=tenResult->getTensorPointer();
    for(i=0;i<matSize.x;i++){
      for(j=0;j<m_tensorSize.y;j++){
    //*(tenResPointer+k*matSize.x*m_tensorSize.y+j*matSize.x+i)=matResTemp(i,j);
    tenResult->setValue(i,j,k,matResTemp(i,j));
      }
    }
  }

  return tenResult;
}

void ntkTensor::switchAxisForward(){
  int i,j,k;
  
  ntkIntDimension orgDim=m_tensorSize;
  m_tensorSize=ntkIntDimension(orgDim.z, orgDim.x, orgDim.y);

  m_tenTemp=(float*)malloc(orgDim.x*orgDim.y*orgDim.z*sizeof(float));
  
  memcpy(m_tenTemp, m_tensor, orgDim.x*orgDim.y*orgDim.z*sizeof(float));

  for(i=0;i<m_tensorSize.x;i++){
    for(j=0;j<m_tensorSize.y;j++){
      for(k=0;k<m_tensorSize.z;k++){
    TENSOR_ACC(i,j,k)=*(m_tenTemp+i*orgDim.x*orgDim.y+k*orgDim.x+j);
      }
    }
  }

  free(m_tenTemp);
}

void ntkTensor::switchAxisBackward(){
  int i,j,k;
  
  ntkIntDimension orgDim=m_tensorSize;
  m_tensorSize=ntkIntDimension(orgDim.y, orgDim.z, orgDim.x);

  m_tenTemp=(float*)malloc(orgDim.x*orgDim.y*orgDim.z*sizeof(float));
  
  memcpy(m_tenTemp, m_tensor, orgDim.x*orgDim.y*orgDim.z*sizeof(float));

  for(i=0;i<m_tensorSize.x;i++){
    for(j=0;j<m_tensorSize.y;j++){
      for(k=0;k<m_tensorSize.z;k++){
    TENSOR_ACC(i,j,k)=*(m_tenTemp+j*orgDim.x*orgDim.y+i*orgDim.x+k);
      }
    }
  }

  free(m_tenTemp);
}

void ntkTensor::printTensor(){
  int i,j,k;
  for(k=0;k<m_tensorSize.z;k++){
    for(i=0;i<m_tensorSize.x;i++){
      for(j=0;j<m_tensorSize.y;j++){
    printf("%f ", TENSOR_ACC(i,j,k));fflush(stdout);
      }
      printf("\n");
    }
    printf("\n");
  }
}

void ntkTensor::setValue(int x, int y, int z, float value){
  TENSOR_ACC(x,y,z)=value;
}

float ntkTensor::getValue(int x, int y, int z){
  return TENSOR_ACC(x,y,z);
}

ntkIntDimension ntkTensor::getTensorSize(){
  return m_tensorSize;
}

float* ntkTensor::getTensorPointer(){
  return m_tensor;
}
