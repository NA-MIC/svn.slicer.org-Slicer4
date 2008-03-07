#include "ntkMISimilarityMeasureCUDA.h"

ntkMISimilarityMeasureCUDA::ntkMISimilarityMeasureCUDA(){
  m_histogram=(double*)malloc(256*256*sizeof(double));
  m_histogramA=(double*)malloc(256*sizeof(double));
  m_histogramB=(int*)malloc(256*sizeof(int));
  m_histFactor=1;
  m_temflag=0;
  m_refflag=0;
}

ntkMISimilarityMeasureCUDA::~ntkMISimilarityMeasureCUDA(){
  free(m_histogram);
  free(m_histogramA);
  free(m_histogramB);

}

void ntkMISimilarityMeasureCUDA::loadData(ntk3DData* reference){
  m_reference=reference;
  ntkIntDimension dataSize=m_reference->getDataSize();
}

double ntkMISimilarityMeasureCUDA::doSimilarityMeasure(ntk3DData* temp){
  int i,j,k;
  double value=0.0;
  int count;

  ntkIntDimension refdim=m_reference->getDataSize();
  ntkIntDimension temdim=temp->getDataSize();
  ntkFloatDimension refthick=m_reference->getDataThickness();
  ntkFloatDimension temthick=temp->getDataThickness();

  ntkCudaDeviceMemory *dRef=new ntkCudaDeviceMemory();
  dRef->Allocate<unsigned char>(refdim.x*refdim.y*refdim.z);
  ntkCudaDeviceMemory *dTemp=new ntkCudaDeviceMemory();
  dTemp->Allocate<unsigned char>(temdim.x*temdim.y*temdim.z);

  dRef->copyFromHost(m_reference->getBuffer());
  dTemp->copyFromHost(temp->getBuffer());

  value = CUDAcalculateMI_doCalculation((unsigned char*)dRef->getDeviceBuffer(), (unsigned char*)dTemp->getDeviceBuffer(), refdim.x, refdim.y, refdim.z, temdim.x, temdim.y, temdim.z, refthick.x, refthick.y, refthick.z, temthick.x, temthick.y, temthick.z );   
 
  delete dRef;
  delete dTemp;

  return (double)value;
}

void ntkMISimilarityMeasureCUDA::setAlignTemplateToReference(){
  m_temflag=1;
  m_refflag=0;
}

void ntkMISimilarityMeasureCUDA::setAlignReferenceToTemplate(){
  m_refflag=1;
  m_temflag=0;
}

double ntkMISimilarityMeasureCUDA::getInterpolationValue(double x, double y, double z){
  int xint=(int)x;
  int yint=(int)y;
  int zint=(int)z;
  float xsmall=(float)x-(float)xint;
  float ysmall=(float)y-(float)yint;
  float zsmall=(float)z-(float)zint;
  
  unsigned char value1=0, value2=0, value3=0, value4=0, value5=0, value6=0, value7=0, value8=0;
  ntkIntDimension pos;

  pos.x=xint;pos.y=yint;pos.z=zint;value1=m_reference->getValue(pos);
  pos.x=xint;pos.y=yint;pos.z=zint+1;value2=m_reference->getValue(pos);
  pos.x=xint;pos.y=yint+1;pos.z=zint;value3=m_reference->getValue(pos);
  pos.x=xint;pos.y=yint+1;pos.z=zint+1;value4=m_reference->getValue(pos);
  pos.x=xint+1;pos.y=yint;pos.z=zint;value5=m_reference->getValue(pos);
  pos.x=xint+1;pos.y=yint;pos.z=zint+1;value6=m_reference->getValue(pos);
  pos.x=xint+1;pos.y=yint+1;pos.z=zint;value7=m_reference->getValue(pos);
  pos.x=xint+1;pos.y=yint+1;pos.z=zint+1;value8=m_reference->getValue(pos);

  return (double)((1-xsmall)*(1-ysmall)*(1-zsmall)*value1+
      (1-xsmall)*(1-ysmall)*zsmall*value2+
      (1-xsmall)*ysmall*(1-zsmall)*value3+
      (1-xsmall)*ysmall*zsmall*value4+
      xsmall*(1-ysmall)*(1-zsmall)*value5+
      xsmall*(1-ysmall)*zsmall*value6+
      xsmall*ysmall*(1-zsmall)*value7+
      xsmall*ysmall*zsmall*value8);
  
}

void ntkMISimilarityMeasureCUDA::distributeInterpolationValue(double x, double y, double z, unsigned char tempValue){
  int xint=(int)x;
  int yint=(int)y;
  int zint=(int)z;
  float xsmall=(float)x-(float)xint;
  float ysmall=(float)y-(float)yint;
  float zsmall=(float)z-(float)zint;
  
  unsigned char value1=0, value2=0, value3=0, value4=0, value5=0, value6=0, value7=0, value8=0;
  ntkIntDimension pos;

  pos.x=xint;pos.y=yint;pos.z=zint;value1=m_reference->getValue(pos);
  pos.x=xint;pos.y=yint;pos.z=zint+1;value2=m_reference->getValue(pos);
  pos.x=xint;pos.y=yint+1;pos.z=zint;value3=m_reference->getValue(pos);
  pos.x=xint;pos.y=yint+1;pos.z=zint+1;value4=m_reference->getValue(pos);
  pos.x=xint+1;pos.y=yint;pos.z=zint;value5=m_reference->getValue(pos);
  pos.x=xint+1;pos.y=yint;pos.z=zint+1;value6=m_reference->getValue(pos);
  pos.x=xint+1;pos.y=yint+1;pos.z=zint;value7=m_reference->getValue(pos);
  pos.x=xint+1;pos.y=yint+1;pos.z=zint+1;value8=m_reference->getValue(pos);

  tempValue/=m_histFactor;
  value1/=m_histFactor;
  value2/=m_histFactor;
  value3/=m_histFactor;
  value4/=m_histFactor;
  value5/=m_histFactor;
  value6/=m_histFactor;
  value7/=m_histFactor;
  value8/=m_histFactor;

  (*(m_histogram+value1*256+(int)tempValue))+=((1-xsmall)*(1-ysmall)*(1-zsmall));
  (*(m_histogram+value2*256+(int)tempValue))+=((1-xsmall)*(1-ysmall)*(zsmall));
  (*(m_histogram+value3*256+(int)tempValue))+=((1-xsmall)*(ysmall)*(1-zsmall));
  (*(m_histogram+value4*256+(int)tempValue))+=((1-xsmall)*(ysmall)*(zsmall));
  (*(m_histogram+value5*256+(int)tempValue))+=((xsmall)*(1-ysmall)*(1-zsmall));
  (*(m_histogram+value6*256+(int)tempValue))+=((xsmall)*(1-ysmall)*(zsmall));
  (*(m_histogram+value7*256+(int)tempValue))+=((xsmall)*(ysmall)*(1-zsmall));
  (*(m_histogram+value8*256+(int)tempValue))+=((xsmall)*(ysmall)*(zsmall));

  (*(m_histogramA+value1))+=((1-xsmall)*(1-ysmall)*(1-zsmall));
  (*(m_histogramA+value2))+=((1-xsmall)*(1-ysmall)*(zsmall));
  (*(m_histogramA+value3))+=((1-xsmall)*(ysmall)*(1-zsmall));
  (*(m_histogramA+value4))+=((1-xsmall)*(ysmall)*(zsmall));
  (*(m_histogramA+value5))+=((xsmall)*(1-ysmall)*(1-zsmall));
  (*(m_histogramA+value6))+=((xsmall)*(1-ysmall)*(zsmall));
  (*(m_histogramA+value7))+=((xsmall)*(ysmall)*(1-zsmall));
  (*(m_histogramA+value8))+=((xsmall)*(ysmall)*(zsmall));

  (*(m_histogramB+tempValue))++;  
}

void ntkMISimilarityMeasureCUDA::setHistogramBinSize(int histFactor){
  m_histFactor=histFactor;
}
