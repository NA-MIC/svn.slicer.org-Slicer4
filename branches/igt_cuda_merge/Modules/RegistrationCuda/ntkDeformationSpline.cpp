#include "ntkDeformationSpline.h"


ntkDeformationSpline::ntkDeformationSpline(ntkIntDimension splineSize){
  int i,j,k,n;
  m_splineParam=(float*)malloc(3*splineSize.x*splineSize.y*splineSize.z*sizeof(float));
  m_splineSize=splineSize;

  for(n=0;n<3;n++){
    for(i=0;i<splineSize.x;i++){
      for(j=0;j<splineSize.y;j++){
    for(k=0;k<splineSize.z;k++){
      *(m_splineParam+n*splineSize.x*splineSize.y*splineSize.z+k*splineSize.x*splineSize.y+j*splineSize.x+i)=0.0;
    }
      }
    }
  }
}

ntkDeformationSpline::~ntkDeformationSpline(){
  if(m_splineParam!=NULL)delete m_splineParam;
}

float *ntkDeformationSpline::getBuffer(){
  return m_splineParam;
}

ntkIntDimension ntkDeformationSpline::getSize(){
  return m_splineSize;
}

void ntkDeformationSpline::readFile(char* filename){
  fstream fp;
  fp.open(filename, ios::in);
  
  fp.read((char*)m_splineParam, sizeof(float)*m_splineSize.x*m_splineSize.y*m_splineSize.z*3);
  fp.close();
}

void ntkDeformationSpline::writeFile(char* filename){
  fstream fp;
  fp.open(filename, ios::out);
  
  fp.write((char*)m_splineParam, sizeof(float)*m_splineSize.x*m_splineSize.y*m_splineSize.z*3);
  fp.close();
}

ntk3DData* ntkDeformationSpline::getDeformationSplineImage(float weight){
  int i;
  int dataSize=m_splineSize.x*m_splineSize.y*m_splineSize.z;

  ntk3DData* output=new ntk3DData(m_splineSize);
  float temp=0.0;
  unsigned char* outputBuffer=(unsigned char*)output->getBuffer(); 
  for(i=0;i<dataSize;i++){
    temp=sqrt((*(m_splineParam+i)* *(m_splineParam+i))+(*(m_splineParam+i+dataSize)* *(m_splineParam+i+dataSize))+(*(m_splineParam+i+2*dataSize)* *(m_splineParam+i+2*dataSize)))*weight;
    
    if(temp<0.0)temp=0.0;
    if(temp>255.0)temp=255.0;
    
    *(outputBuffer+i)=(unsigned char)temp;
    
  }

  return output;
}

void ntkDeformationSpline::multiplyFactor(float factor){
  int i,j,k,n;
  
  for(n=0;n<3;n++){
    for(i=0;i<m_splineSize.x;i++){
      for(j=0;j<m_splineSize.y;j++){
    for(k=0;k<m_splineSize.z;k++){
      *(m_splineParam+n*m_splineSize.x*m_splineSize.y*m_splineSize.z+k*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+i)*=factor;
    }
      }
    }
  }
}
