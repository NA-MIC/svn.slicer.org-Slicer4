#include "ntkMISimilarityMeasure.h"

ntkMISimilarityMeasure::ntkMISimilarityMeasure(){
  m_histogram=(double*)malloc(256*256*sizeof(double));
  m_histogramA=(double*)malloc(256*sizeof(double));
  m_histogramB=(int*)malloc(256*sizeof(int));
  m_histFactor=1;
  m_temflag=0;
  m_refflag=0;
}

ntkMISimilarityMeasure::~ntkMISimilarityMeasure(){
  free(m_histogram);
  free(m_histogramA);
  free(m_histogramB);
}

void ntkMISimilarityMeasure::loadData(ntk3DData* reference){
  m_reference=reference;
}

double ntkMISimilarityMeasure::doSimilarityMeasure(ntk3DData* temp){
  int i,j,k;
  double value=0.0;
  int count;

  ntkIntDimension refdim=m_reference->getDataSize();
  ntkIntDimension temdim=temp->getDataSize();
  ntkFloatDimension refthick=m_reference->getDataThickness();
  ntkFloatDimension temthick=temp->getDataThickness();
   
  if(m_temflag==1){
    temthick.x=refdim.x*refthick.x/(float)temdim.x;
    temthick.y=refdim.y*refthick.y/(float)temdim.y;
    temthick.z=refdim.z*refthick.z/(float)temdim.z;
  }else if(m_refflag==1){
    refthick.x=temdim.x*temthick.x/(float)refdim.x;
    refthick.y=temdim.y*temthick.y/(float)refdim.y;
    refthick.z=temdim.z*temthick.z/(float)refdim.z;
  }

  for(i=0;i<256;i++){
    for(j=0;j<256;j++){
      *(m_histogram+i*256+j)=0;
    }
    *(m_histogramA+i)=0;
    *(m_histogramB+i)=0;
  }

  count=0;
  double xdif, ydif, zdif;
  xdif=((refdim.x*refthick.x)-(temdim.x*temthick.x))/2.0;
  ydif=((refdim.y*refthick.y)-(temdim.y*temthick.y))/2.0;
  zdif=((refdim.z*refthick.z)-(temdim.z*temthick.z))/2.0;
  double xpos, ypos,zpos;
  ntkIntDimension pos;
  unsigned char tempValue;
  int tempValue2;
  for(i=1;i<temdim.x-1;i++){
    for(j=1;j<temdim.y-1;j++){
      for(k=1;k<temdim.z-1;k++){
    //if(temp->getValue(i,j,k)!=0){
      xpos=((i+0.5)*temthick.x+xdif)/refthick.x-0.5;
      ypos=((j+0.5)*temthick.y+ydif)/refthick.y-0.5;
      zpos=((k+0.5)*temthick.z+zdif)/refthick.z-0.5;
      if(xpos>=0 && xpos <= refdim.x*refthick.x-1 && ypos>=0 && ypos <= refdim.y*refthick.y-1 && zpos>=0 && zpos <= refdim.z*refthick.z-1){
        count++;
        pos.x=i;pos.y=j;pos.z=k;tempValue=temp->getValue(pos);
        distributeInterpolationValue(xpos, ypos, zpos, tempValue);
        
       }
      //}
      }
    }
  }
  

  double tempval;
  
  for(i=0;i<256;i++){
    if(*(m_histogramA+i)!=0){
      tempval=*(m_histogramA+i)/(double)(count);
      value-=tempval*log(tempval)/log((double)10);
    }
    if(*(m_histogramB+i)!=0){
      tempval=*(m_histogramB+i)/(double)(count);
      value-=tempval*log((double)tempval)/log((double)10);
    }
 
    for(j=0;j<256;j++){
      /*
    if(*(m_histogram+i*256+j)!=1){
    printf("%d %d %d:",i,j,*(m_histogram+i*256+j));
    }
      */
      if(*(m_histogram+i*256+j)!=0){
    tempval=*(m_histogram+i*256+j)/(double)(count);
    value+=tempval*log(tempval)/log(10.0f);
      }
    }
    //printf("value:%f %d", value, i);
  }
  
  return (double)value;
}

void ntkMISimilarityMeasure::setAlignTemplateToReference(){
  m_temflag=1;
  m_refflag=0;
}

void ntkMISimilarityMeasure::setAlignReferenceToTemplate(){
  m_refflag=1;
  m_temflag=0;
}

double ntkMISimilarityMeasure::getInterpolationValue(double x, double y, double z){
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

void ntkMISimilarityMeasure::distributeInterpolationValue(double x, double y, double z, unsigned char tempValue){
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

void ntkMISimilarityMeasure::setHistogramBinSize(int histFactor){
  m_histFactor=histFactor;
}
