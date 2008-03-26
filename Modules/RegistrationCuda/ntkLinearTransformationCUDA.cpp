#include "ntkLinearTransformationCUDA.h"

ntkLinearTransformationCUDA::ntkLinearTransformationCUDA(){
  m_tMatrix=vtkMatrix4x4::New();
  m_tMatrix->Identity();
}

ntkLinearTransformationCUDA::~ntkLinearTransformationCUDA(){

}

void ntkLinearTransformationCUDA::loadData(ntk3DData* input){
  m_input=input;
  ntkIntDimension dataSize=m_input->getDataSize();
}

void ntkLinearTransformationCUDA::setTransformationMatrix(vtkMatrix4x4* tMatrix){
  m_tMatrix=tMatrix;
}

ntk3DData *ntkLinearTransformationCUDA::applyTransformation(ntkIntDimension outputSize, ntkFloatDimension outputThickness){
  int i,j,k;
  ntkIntDimension inputSize=m_input->getDataSize();
  ntkFloatDimension inputThickness=m_input->getDataThickness();
  
  /*bspline transformation of input*/

  ntk3DData* inputPlus=new ntk3DData(ntkIntDimension(inputSize.x+2, inputSize.y+2, inputSize.z+2));
  
  unsigned char* inputBuffer=(unsigned char*)m_input->getBuffer();
  unsigned char* inputPlusBuffer=(unsigned char*)inputPlus->getBuffer();

  for(k=0;k<inputSize.z;k++){
    for(j=0;j<inputSize.y;j++){
      memcpy(inputPlusBuffer+(k+1)*(inputSize.x+2)*(inputSize.y+2)+(j+1)*(inputSize.x+2)+1, inputBuffer+k*inputSize.x*inputSize.y+j*inputSize.x, inputSize.x);
    }
  }

  ntkBSpline3Transform *trans=new ntkBSpline3Transform();
  ntkTensor* splineInput=trans->doForwardTransform(inputPlus, CURRENT_RES);

  delete inputPlus;

  ntk3DData *output=new ntk3DData(outputSize, outputThickness);
  unsigned char *outputBuffer=(unsigned char*)output->getBuffer();

  vtkMatrix4x4 *inverseMatrix=vtkMatrix4x4::New();
  vtkMatrix4x4::Invert(m_tMatrix, inverseMatrix);

  ntkFloatDimension out_origin;
  out_origin.x=outputSize.x*outputThickness.x/2.0;
  out_origin.y=outputSize.y*outputThickness.y/2.0;
  out_origin.z=outputSize.z*outputThickness.z/2.0;
  ntkFloatDimension in_origin;
  in_origin.x=inputSize.x*inputThickness.x/2.0;
  in_origin.y=inputSize.y*inputThickness.y/2.0;
  in_origin.z=inputSize.z*inputThickness.z/2.0;
  double vect_in[4],vect_out[4];
  double pos_in[3];
  double tempValue=0.0;
 
  for(i=0;i<outputSize.x;i++){
    for(j=0;j<outputSize.y;j++){
      for(k=0;k<outputSize.z;k++){
    vect_out[0]=((i+0.5)*outputThickness.x)-out_origin.x;
    vect_out[1]=((j+0.5)*outputThickness.y)-out_origin.y;
    vect_out[2]=((k+0.5)*outputThickness.z)-out_origin.z;
    vect_out[3]=1.0;
    
    inverseMatrix->MultiplyPoint(vect_out,vect_in);
    pos_in[0]=(vect_in[0]+in_origin.x)/inputThickness.x-0.5;
    pos_in[1]=(vect_in[1]+in_origin.y)/inputThickness.y-0.5;
    pos_in[2]=(vect_in[2]+in_origin.z)/inputThickness.z-0.5;
    if(pos_in[0]>=0 && pos_in[0]<inputSize.x-1 && pos_in[1]>=0 && pos_in[1]<inputSize.y-1 && pos_in[2]>=0 && pos_in[2]<inputSize.z-1){
      *(outputBuffer+k*outputSize.x*outputSize.y+j*outputSize.x+i)=(unsigned char)getInterpolationValue(pos_in[0],pos_in[1],pos_in[2]);
      
    }else{
      *(outputBuffer+k*outputSize.x*outputSize.y+j*outputSize.x+i)=0;
    }
      } 
    }
  }
  delete trans;
  delete splineInput;
  return output;
}

ntk3DData *ntkLinearTransformationCUDA::applyTransformation(){
  ntkIntDimension outDataSize=m_input->getDataSize();
  ntkFloatDimension outDataThickness=m_input->getDataThickness();
  unsigned char* inBuffer=(unsigned char*)m_input->getBuffer();
  unsigned char* outBuffer=(unsigned char*)malloc(outDataSize.x*outDataSize.y*outDataSize.z*sizeof(unsigned char));
;
 unsigned char* bufferTemp;
 float matTemp[16];

 vtkMatrix4x4 *inverseMatrix=vtkMatrix4x4::New();
 vtkMatrix4x4::Invert(m_tMatrix, inverseMatrix);

 matTemp[0]=inverseMatrix->GetElement(0,0);
 matTemp[1]=inverseMatrix->GetElement(0,1);
 matTemp[2]=inverseMatrix->GetElement(0,2);
 matTemp[3]=inverseMatrix->GetElement(0,3);
 matTemp[4]=inverseMatrix->GetElement(1,0);
 matTemp[5]=inverseMatrix->GetElement(1,1);
 matTemp[6]=inverseMatrix->GetElement(1,2);
 matTemp[7]=inverseMatrix->GetElement(1,3);
 matTemp[8]=inverseMatrix->GetElement(2,0);
 matTemp[9]=inverseMatrix->GetElement(2,1);
 matTemp[10]=inverseMatrix->GetElement(2,2);
 matTemp[11]=inverseMatrix->GetElement(2,3);
 matTemp[12]=inverseMatrix->GetElement(3,0);
 matTemp[13]=inverseMatrix->GetElement(3,1);
 matTemp[14]=inverseMatrix->GetElement(3,2);
 matTemp[15]=inverseMatrix->GetElement(3,3);

 ntkCudaDeviceMemory *dInputData = new ntkCudaDeviceMemory();
 dInputData->Allocate<unsigned char>(outDataSize.x*outDataSize.y*outDataSize.z);
 
 dInputData->copyFromHost(m_input->getBuffer());
 
 ntkCudaDeviceMemory *dOutputData = new ntkCudaDeviceMemory();
 dOutputData->Allocate<unsigned char>(outDataSize.x*outDataSize.y*outDataSize.z);

 CUDAlinearTransformation_doTransformation((unsigned char*)dInputData->getDeviceBuffer(), (unsigned char*)dOutputData->getDeviceBuffer(), matTemp, outDataSize.x, outDataSize.y, outDataSize.z, outDataSize.x, outDataSize.y, outDataSize.z, (float)outDataThickness.x, (float)outDataThickness.y, (float)outDataThickness.z, (float)outDataThickness.x, (float)outDataThickness.y,(float)outDataThickness.z );

 ntk3DData * output = new ntk3DData(outDataSize, outDataThickness, outBuffer);
 dOutputData->copyToHost(output->getBuffer());

 delete dInputData;
 delete dOutputData;
 
 return output;
}

void ntkLinearTransformationCUDA::doTranslation(double x, double y, double z){
  vtkMatrix4x4* transMatrix=vtkMatrix4x4::New();
  transMatrix->Identity();
  transMatrix->SetElement(0,3,x);
  transMatrix->SetElement(1,3,y);
  transMatrix->SetElement(2,3,z);
  vtkMatrix4x4* tempMatrix=vtkMatrix4x4::New();
  vtkMatrix4x4::Multiply4x4(transMatrix,m_tMatrix, tempMatrix);
  m_tMatrix->Delete();
  m_tMatrix=tempMatrix;
}

void ntkLinearTransformationCUDA::doRotationX(double xAngle){
  vtkMatrix4x4* transMatrix=vtkMatrix4x4::New();
  xAngle*=2*M_PI/360.0;
  transMatrix->Identity();
  transMatrix->SetElement(1,1,cos(xAngle));
  transMatrix->SetElement(1,2,-sin(xAngle));
  transMatrix->SetElement(2,1,sin(xAngle));
  transMatrix->SetElement(2,2,cos(xAngle));
  vtkMatrix4x4* tempMatrix=vtkMatrix4x4::New();
  vtkMatrix4x4::Multiply4x4(transMatrix,m_tMatrix, tempMatrix);
  m_tMatrix->Delete();
  m_tMatrix=tempMatrix;
}

void ntkLinearTransformationCUDA::doRotationY(double yAngle){
  vtkMatrix4x4* transMatrix=vtkMatrix4x4::New();
  yAngle*=2*M_PI/360.0;
  transMatrix->Identity();
  transMatrix->SetElement(2,2,cos(yAngle));
  transMatrix->SetElement(2,0,-sin(yAngle));
  transMatrix->SetElement(0,2,sin(yAngle));
  transMatrix->SetElement(0,0,cos(yAngle));
  vtkMatrix4x4* tempMatrix=vtkMatrix4x4::New();
  vtkMatrix4x4::Multiply4x4(transMatrix,m_tMatrix, tempMatrix);
  m_tMatrix->Delete();
  m_tMatrix=tempMatrix;
}

void ntkLinearTransformationCUDA::doRotationZ(double zAngle){
  vtkMatrix4x4* transMatrix=vtkMatrix4x4::New();
  zAngle*=2*M_PI/360.0;
  transMatrix->Identity();
  transMatrix->SetElement(0,0,cos(zAngle));
  transMatrix->SetElement(0,1,-sin(zAngle));
  transMatrix->SetElement(1,0,sin(zAngle));
  transMatrix->SetElement(1,1,cos(zAngle));
  vtkMatrix4x4* tempMatrix=vtkMatrix4x4::New();
  vtkMatrix4x4::Multiply4x4(transMatrix,m_tMatrix, tempMatrix);
  m_tMatrix->Delete();
  m_tMatrix=tempMatrix;
}

void ntkLinearTransformationCUDA::doScaling(double scale){
  vtkMatrix4x4* transMatrix=vtkMatrix4x4::New();
  transMatrix->Identity();
  transMatrix->SetElement(0,0,scale);
  transMatrix->SetElement(1,1,scale);
  transMatrix->SetElement(2,2,scale);
  vtkMatrix4x4* tempMatrix=vtkMatrix4x4::New();
  vtkMatrix4x4::Multiply4x4(transMatrix,m_tMatrix, tempMatrix);
  m_tMatrix->Delete();
  m_tMatrix=tempMatrix;
}

vtkMatrix4x4* ntkLinearTransformationCUDA::getMatrix(){
  return m_tMatrix;
}

double ntkLinearTransformationCUDA::getInterpolationValue(double x, double y, double z){
  int xint=(int)x;
  int yint=(int)y;
  int zint=(int)z;
  float xsmall=(float)x-(float)xint;
  float ysmall=(float)y-(float)yint;
  float zsmall=(float)z-(float)zint;

  unsigned char value1=0, value2=0, value3=0, value4=0, value5=0, value6=0, value7=0, value8=0;
  ntkIntDimension pos;

  pos.x=xint;pos.y=yint;pos.z=zint;value1=m_input->getValue(pos);
  pos.x=xint;pos.y=yint;pos.z=zint+1;value2=m_input->getValue(pos);
  pos.x=xint;pos.y=yint+1;pos.z=zint;value3=m_input->getValue(pos);
  pos.x=xint;pos.y=yint+1;pos.z=zint+1;value4=m_input->getValue(pos);
  pos.x=xint+1;pos.y=yint;pos.z=zint;value5=m_input->getValue(pos);
  pos.x=xint+1;pos.y=yint;pos.z=zint+1;value6=m_input->getValue(pos);
  pos.x=xint+1;pos.y=yint+1;pos.z=zint;value7=m_input->getValue(pos);
  pos.x=xint+1;pos.y=yint+1;pos.z=zint+1;value8=m_input->getValue(pos);

  return (double)((1-xsmall)*(1-ysmall)*(1-zsmall)*value1+
          (1-xsmall)*(1-ysmall)*zsmall*value2+
          (1-xsmall)*ysmall*(1-zsmall)*value3+
          (1-xsmall)*ysmall*zsmall*value4+
          xsmall*(1-ysmall)*(1-zsmall)*value5+
          xsmall*(1-ysmall)*zsmall*value6+
          xsmall*ysmall*(1-zsmall)*value7+
          xsmall*ysmall*zsmall*value8);
  
}

ntkFloatDimension ntkLinearTransformationCUDA::getNewPositionFromOld(ntkIntDimension outputSize, ntkFloatDimension outputThickness, ntkFloatDimension oldPosition){
  ntkFloatDimension newPosition;
  ntkIntDimension inputSize=m_input->getDataSize();
  ntkFloatDimension inputThickness=m_input->getDataThickness();

  ntkFloatDimension out_origin;
  out_origin.x=outputSize.x*outputThickness.x/2.0;
  out_origin.y=outputSize.y*outputThickness.y/2.0;
  out_origin.z=outputSize.z*outputThickness.z/2.0;
  ntkFloatDimension in_origin;
  in_origin.x=inputSize.x*inputThickness.x/2.0;
  in_origin.y=inputSize.y*inputThickness.y/2.0;
  in_origin.z=inputSize.z*inputThickness.z/2.0;
  double vect_in[4],vect_out[4];
  double pos_out[3];
  double tempValue=0.0;

  vect_in[0]=((oldPosition.x+0.5)*inputThickness.x)-in_origin.x;
  vect_in[1]=((oldPosition.y+0.5)*inputThickness.y)-in_origin.y;
  vect_in[2]=((oldPosition.z+0.5)*inputThickness.z)-in_origin.z;
  vect_in[3]=1.0;
    
  m_tMatrix->MultiplyPoint(vect_in,vect_out);

  pos_out[0]=(vect_out[0]+out_origin.x)/outputThickness.x-0.5;
  pos_out[1]=(vect_out[1]+out_origin.y)/outputThickness.y-0.5;
  pos_out[2]=(vect_out[2]+out_origin.z)/outputThickness.z-0.5;

  newPosition.x=pos_out[0];
  newPosition.y=pos_out[1];
  newPosition.z=pos_out[2];

  return newPosition;
}

ntkFloatDimension ntkLinearTransformationCUDA::getOldPositionFromNew(ntkIntDimension outputSize, ntkFloatDimension outputThickness, ntkFloatDimension newPosition){
  ntkFloatDimension oldPosition;
  ntkIntDimension inputSize=m_input->getDataSize();
  ntkFloatDimension inputThickness=m_input->getDataThickness();  
  
  vtkMatrix4x4 *inverseMatrix=vtkMatrix4x4::New();
  vtkMatrix4x4::Invert(m_tMatrix, inverseMatrix);
  
  ntkFloatDimension out_origin;
  out_origin.x=outputSize.x*outputThickness.x/2.0;
  out_origin.y=outputSize.y*outputThickness.y/2.0;
  out_origin.z=outputSize.z*outputThickness.z/2.0;
  ntkFloatDimension in_origin;
  in_origin.x=inputSize.x*inputThickness.x/2.0;
  in_origin.y=inputSize.y*inputThickness.y/2.0;
  in_origin.z=inputSize.z*inputThickness.z/2.0;
  double vect_in[4],vect_out[4];
  double pos_in[3];
  double tempValue=0.0;

  vect_out[0]=((newPosition.x+0.5)*outputThickness.x)-out_origin.x;
  vect_out[1]=((newPosition.y+0.5)*outputThickness.y)-out_origin.y;
  vect_out[2]=((newPosition.z+0.5)*outputThickness.z)-out_origin.z;
  vect_out[3]=1.0;
  
  inverseMatrix->MultiplyPoint(vect_out,vect_in);
 

  pos_in[0]=(vect_in[0]+in_origin.x)/inputThickness.x-0.5;
  pos_in[1]=(vect_in[1]+in_origin.y)/inputThickness.y-0.5;
  pos_in[2]=(vect_in[2]+in_origin.z)/inputThickness.z-0.5;

  oldPosition.x=pos_in[0];
  oldPosition.y=pos_in[1];
  oldPosition.z=pos_in[2];

  return oldPosition;
}

