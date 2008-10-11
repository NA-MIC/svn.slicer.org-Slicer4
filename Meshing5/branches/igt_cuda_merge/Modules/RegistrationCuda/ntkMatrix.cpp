#include "ntkMatrix.h"

#define MATRIX_ACC(X,Y) (*(m_matrix+ (Y)* m_matrixSize.x + (X)))

ntkMatrix::ntkMatrix(int x, int y){
  m_matrixSize=ntkIntDimension(x,y);
  m_matrix=(float*)malloc(x*y*sizeof(float));
  initialize();
  m_matTemp=NULL;
}

ntkMatrix::ntkMatrix(ntk2DData* inputData){
  int i,j;
  m_matrixSize=inputData->getDataSize();
  m_matrix=(float*)malloc(m_matrixSize.x*m_matrixSize.y*sizeof(float));
  m_matTemp=NULL;
  unsigned char* inputBuffer=(unsigned char*)(inputData->getBuffer());
  for(i=0;i<m_matrixSize.x;i++){
    for(j=0;j<m_matrixSize.y;j++){
      MATRIX_ACC(i,j)=(float)(*(inputBuffer+j*m_matrixSize.x+i));
    }
  }
}

ntkMatrix::ntkMatrix(float* inputBuffer, ntkIntDimension inputSize){
  m_matrixSize=inputSize;
  m_matrix=(float*)malloc(m_matrixSize.x*m_matrixSize.y*sizeof(float));
  m_matTemp=NULL;
  memcpy(m_matrix, inputBuffer, m_matrixSize.x*m_matrixSize.y*sizeof(float));
}

ntkMatrix::~ntkMatrix(){
  if(m_matrix!=NULL)free(m_matrix);
}

void ntkMatrix::initialize(){
  int i,j,k;
  for(i=0;i<m_matrixSize.x;i++){
    for(j=0;j<m_matrixSize.y;j++){
    MATRIX_ACC(i,j)=0.0;
    }
  }
}

ntkMatrix* ntkMatrix::multiplyMatrixLeft(Matrix matA){
  int i,j,k;
  ntkIntDimension matSize(matA.rows(),matA.cols());
  ntkMatrix *dummy=NULL;

  if(matSize.y!=m_matrixSize.x){
    printf("Size does not match. matrix-tensor multiplication failed\n");
    return dummy;
  }
  
  ntkMatrix *matResult=new ntkMatrix(matSize.x,m_matrixSize.y);
    
  Matrix matTemp(m_matrixSize.x, m_matrixSize.y);
  Matrix matResTemp;
  
  //float* matResPointer;
  
  for(i=0;i<m_matrixSize.x;i++){
    for(j=0;j<m_matrixSize.y;j++){
      matTemp(i,j)=MATRIX_ACC(i,j);
    }
  }
  
  matResTemp=matA*matTemp;
  //matResPointer=matResult->getMatrixPointer();
  for(i=0;i<matSize.x;i++){
    for(j=0;j<m_matrixSize.y;j++){
      //*(matResPointer+k*matSize.x*m_matrixSize.y+j*matSize.x+i)=matResTemp(i,j);
      matResult->setValue(i,j,matResTemp(i,j));
      
    }
  }

  return matResult;
}

void ntkMatrix::switchAxisForward(){
  int i,j;
  
  ntkIntDimension orgDim=m_matrixSize;
  m_matrixSize=ntkIntDimension(orgDim.y, orgDim.x);

  m_matTemp=(float*)malloc(orgDim.x*orgDim.y*sizeof(float));
  
  memcpy(m_matTemp, m_matrix, orgDim.x*orgDim.y*sizeof(float));

  for(i=0;i<m_matrixSize.x;i++){
    for(j=0;j<m_matrixSize.y;j++){
      MATRIX_ACC(i,j)=*(m_matTemp+i*orgDim.x+j);
    }
  }

  free(m_matTemp);
}

void ntkMatrix::switchAxisBackward(){
  int i,j;
  
  ntkIntDimension orgDim=m_matrixSize;
  m_matrixSize=ntkIntDimension(orgDim.y, orgDim.x);

  m_matTemp=(float*)malloc(orgDim.x*orgDim.y*sizeof(float));
  
  memcpy(m_matTemp, m_matrix, orgDim.x*orgDim.y*sizeof(float));

  for(i=0;i<m_matrixSize.x;i++){
    for(j=0;j<m_matrixSize.y;j++){
      MATRIX_ACC(i,j)=*(m_matTemp+i*orgDim.x+j);
    }
  }

  free(m_matTemp);
}

void ntkMatrix::printMatrix(){
  int i,j;
  for(i=0;i<m_matrixSize.x;i++){
    for(j=0;j<m_matrixSize.y;j++){
      printf("%f ", MATRIX_ACC(i,j));fflush(stdout);
    }
    printf("\n");
  }
}

void ntkMatrix::setValue(int x, int y, float value){
  MATRIX_ACC(x,y)=value;
}

float ntkMatrix::getValue(int x, int y){
  return MATRIX_ACC(x,y);
}

ntkIntDimension ntkMatrix::getMatrixSize(){
  return m_matrixSize;
}

float* ntkMatrix::getMatrixPointer(){
  return m_matrix;
}
