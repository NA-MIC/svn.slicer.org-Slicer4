#include "ntkElasticRegistrationCUDA.h"

ntkElasticRegistrationCUDA::ntkElasticRegistrationCUDA(){
  m_reference=NULL;
  m_splineParam=NULL;
  m_transform=new ntkBSpline3Transform();
  m_function=new ntkBSpline3Function();
  m_SSDgradient=NULL;
  m_maxGradient=0.0;
  m_splineSizeLevel=0;
  m_imageSizeLevel=0;
  m_MTlevel=4;
  m_firstScheme=NULL;
  m_lastScheme=NULL;
  m_minGradientLimit=0.0001;
  m_PEweight=0.0;
  m_dRef=NULL;
  m_dTar=NULL;
}

ntkElasticRegistrationCUDA::~ntkElasticRegistrationCUDA(){
  delete m_transform;
  delete m_function;
  if(m_splineParam!=NULL)delete m_splineParam;
  if(m_SSDgradient!=NULL)free(m_SSDgradient);
  resetRegistrationScheme();
}

void ntkElasticRegistrationCUDA::loadData(ntk3DData* reference){
  m_reference=reference;
  m_referenceSize=reference->getDataSize();
  m_referenceBuffer=(unsigned char*)m_reference->getBuffer();
}

void ntkElasticRegistrationCUDA::doRegistration(ntk3DData* target){}//dummy function

ntkDeformationSpline* ntkElasticRegistrationCUDA::doRegistration(ntk3DData* reference, ntk3DData* target, ntkElasticityMap* emap, int splineSizeLevel=0, ntkDeformationSpline* splineParam=NULL, float step=1.0, float PEweight=1.0){
  int i,j,k,n;
  float maxGradient;
  m_PEweight=PEweight;
  
  m_reference=reference;
  m_referenceSize=reference->getDataSize();
  m_referenceBuffer=(unsigned char*)m_reference->getBuffer();

  m_splineSizeLevel=splineSizeLevel;
  m_target=target;
  m_targetSize=target->getDataSize();

  m_emap=emap;

  if(m_targetSize.x!=m_referenceSize.x || m_targetSize.y!=m_referenceSize.y || m_targetSize.z!=m_referenceSize.z){
    printf("Size of target and reference do not match.\n");
    return 0;
  }

  /*targetPlus is 3D data of target with one additional voxel in each direction*/
  ntk3DData *targetPlus=new ntk3DData(ntkIntDimension(m_targetSize.x+2, m_targetSize.y+2, m_targetSize.z+2));
  unsigned char *targetBuffer=(unsigned char*)target->getBuffer();
  unsigned char *targetPlusBuffer=(unsigned char*)targetPlus->getBuffer();
  
  for(k=0;k<m_targetSize.z;k++){
    for(j=0;j<m_targetSize.y;j++){
      memcpy(targetPlusBuffer+(k+1)*(m_targetSize.x+2)*(m_targetSize.y+2)+(j+1)*(m_targetSize.x+2)+1, targetBuffer+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x, m_targetSize.x);
    }
  }
  
  m_targetPlus=targetPlus;

  /*m_targetPlusSpline is B-Spline representation of targetPlus*/
  m_targetPlusSpline=m_transform->doForwardTransform(m_targetPlus, CURRENT_RES);
  float tempPow=pow(2, splineSizeLevel);
  float knotDistance=pow(2, -splineSizeLevel);
  m_knotDistance=(int)knotDistance;
  m_splineSize= ntkIntDimension((int)(tempPow*(m_targetSize.x+1)+1), (int)(tempPow*(m_targetSize.y+1)+1), (int)(tempPow*(m_targetSize.z+1)+1));
  
  ntkDeformationSpline *outputSpline=new ntkDeformationSpline(ntkIntDimension(m_splineSize.x, m_splineSize.y, m_splineSize.z));
  float *outputSplineBuffer=outputSpline->getBuffer();

  printf("m_splineSize: %d %d %d\n", m_splineSize.x, m_splineSize.y, m_splineSize.z);

  if(splineParam!=NULL){
    float *inputSplineBuffer=splineParam->getBuffer();
    memcpy(outputSplineBuffer, inputSplineBuffer, 3*m_splineSize.x*m_splineSize.y*m_splineSize.z*sizeof(float));
  }

  float SSDvalue=0.0;
  float SSDvalueMin=0.0;
  float SSDvalueTemp=0.0;
   
  ntkElasticTransformationCUDA *transform=new ntkElasticTransformationCUDA();
  transform->loadData(target);
  ntk3DData* result=transform->applyTransformation(outputSpline, m_splineSizeLevel);
  if(m_PEweight==0.0){
   
    SSDvalue=calculateSSD(m_reference, result); //outputSpline

  }else{
    SSDvalue=calculateSSDPE(m_reference, result, outputSpline); //outputSpline
  }
  printf("Initial SSDvalue=%lf\n", SSDvalue);
  
  delete result;
  delete transform;

  float stepOrg=step;

  do{
    SSDvalueMin=SSDvalue;
    SSDvalueTemp=SSDvalue;

    if(m_PEweight==0.0){
      ntkTimer *timer =new ntkTimer();
      timer->Start();
      maxGradient=calculateSSDGradient(outputSpline);
      timer->End();
      timer->printDetailedElapsedTime("SSD gradient calculation.");
      delete timer;
    }
    else{
      maxGradient=calculateSSDPEGradient(outputSpline);
    }

    do{
      if(maxGradient*stepOrg<m_minGradientLimit){
  step=m_minGradientLimit/maxGradient;
      }else{
  step=stepOrg;
      }

      SSDvalue=SSDvalueTemp;
      for(i=0;i<m_splineSize.x;i++){
  for(j=0;j<m_splineSize.y;j++){
    for(k=0;k<m_splineSize.z;k++){
      for(n=0;n<3;n++){
        *(outputSplineBuffer+n*m_splineSize.x*m_splineSize.y*m_splineSize.z+k*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+i)-=*(m_SSDgradient+n*m_splineSize.x*m_splineSize.y*m_splineSize.z+k*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+i)*step;
      }
    }
  }
      }
      
      transform=new ntkElasticTransformationCUDA();
      transform->loadData(target);
      result=transform->applyTransformation(outputSpline, m_splineSizeLevel);
      
      if(m_PEweight==0.0){
  SSDvalueTemp=calculateSSD(m_reference, result);//outputSpline
  //SSDvalueTemp=CUDAelasticRegistration_calculateSSD((unsigned char*)m_reference->getBuffer(), (unsigned char*)result->getBuffer(), m_referenceSize.x, m_referenceSize.y, m_referenceSize.z);
      }else{
  SSDvalueTemp=calculateSSDPE(m_reference, result, outputSpline);//outputSpline
      }
      printf("SSDvalueTemp=%lf\n", SSDvalueTemp);
      
      delete result;
      delete transform; 
      
      SSDvalue=SSDvalueTemp;
    }while(SSDvalueTemp<SSDvalue); 
    
  }while(SSDvalue<SSDvalueMin);
  
  for(i=0;i<m_splineSize.x;i++){
    for(j=0;j<m_splineSize.y;j++){
      for(k=0;k<m_splineSize.z;k++){
  for(n=0;n<3;n++){
    *(outputSplineBuffer+n*m_splineSize.x*m_splineSize.y*m_splineSize.z+k*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+i)+=*(m_SSDgradient+n*m_splineSize.x*m_splineSize.y*m_splineSize.z+k*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+i)*step;
  }
      }
    }
  }
  
  m_PEweight=0.0;
  delete targetPlus;
  return outputSpline;
}

float ntkElasticRegistrationCUDA::calculateSSDGradient(ntkDeformationSpline *splineParam){

  ntkIntDimension refSize=m_reference->getDataSize();
  ntkIntDimension tarSize=m_target->getDataSize();

  ntkCudaDeviceMemory *dRef = new ntkCudaDeviceMemory();
  dRef->Allocate<unsigned char>(refSize.x*refSize.y*refSize.z);
  ntkCudaDeviceMemory *dTar = new ntkCudaDeviceMemory();
  dTar->Allocate<unsigned char>(tarSize.x*tarSize.y*tarSize.z);
  ntkCudaDeviceMemory *dSpline = new ntkCudaDeviceMemory();
  dSpline->Allocate<float>(m_splineSize.x*m_splineSize.y*m_splineSize.z*3);
  ntkCudaDeviceMemory *dSSDGradient = new ntkCudaDeviceMemory();
  dSSDGradient->Allocate<float>(m_splineSize.x*m_splineSize.y*m_splineSize.z*3);
  
  dRef->copyFromHost(m_reference->getBuffer());
  dTar->copyFromHost(m_target->getBuffer());
  dSpline->copyFromHost(splineParam->getBuffer());

  float max=CUDAcalculateSSDGradient_doCalculation((unsigned char*)dRef->getDeviceBuffer(), (unsigned char*)dTar->getDeviceBuffer(), (float*)dSpline->getDeviceBuffer(), (float*)dSSDGradient->getDeviceBuffer(), refSize.x, refSize.y, refSize.z, m_splineSizeLevel);

  if(m_SSDgradient!=NULL)free( m_SSDgradient);
  m_SSDgradient=(float*)malloc(m_splineSize.x*m_splineSize.y*m_splineSize.z*3*sizeof(float));
  dSSDGradient->copyToHost(m_SSDgradient);

  delete dRef;
  delete dTar;
  delete dSpline;
  delete dSSDGradient;
  
  printf("Max value=%lf\n", max);
  
  return max;
}

float ntkElasticRegistrationCUDA::calculateSSDPEGradient(ntkDeformationSpline *splineParam){
  int i,j,k,n; 

  float *splineParamBuffer=splineParam->getBuffer();
  if(m_SSDgradient!=NULL)delete m_SSDgradient;
  m_SSDgradient=(float*)malloc(m_splineSize.x*m_splineSize.y*m_splineSize.z*3*sizeof(float));

  for(i=0;i<m_splineSize.x;i++){
    for(j=0;j<m_splineSize.y;j++){
      for(k=0;k<m_splineSize.z;k++){
  for(n=0;n<3;n++){
    *(m_SSDgradient+n*m_splineSize.x*m_splineSize.y*m_splineSize.z+k*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+i)=0.0;
  }
      }
    }
  }
  
  int d,e,f;
  float functemp;
  float tempValue;
  float tempPow=1.0/(float)m_knotDistance;
  float relPosX, relPosY, relPosZ;
  float *newX=(float*)malloc(m_targetSize.x*m_targetSize.y*m_targetSize.z*sizeof(float));
  float *newY=(float*)malloc(m_targetSize.x*m_targetSize.y*m_targetSize.z*sizeof(float));
  float *newZ=(float*)malloc(m_targetSize.x*m_targetSize.y*m_targetSize.z*sizeof(float));
  float *diff=(float*)malloc(m_targetSize.x*m_targetSize.y*m_targetSize.z*sizeof(float));

    /*calculating difference between each pixel of reference and transformed target*/
  for(i=0;i<m_targetSize.x;i++){
    for(j=0;j<m_targetSize.y;j++){
      for(k=0;k<m_targetSize.z;k++){
  *(newX+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)=i; 
  *(newY+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)=j; 
  *(newZ+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)=k; 
  relPosX=(float)(i+1)*tempPow;
  relPosY=(float)(j+1)*tempPow;
  relPosZ=(float)(k+1)*tempPow;
  
  for(d=(int)relPosX-1;d<(int)relPosX+3;d++){
    if(d<0||d>m_splineSize.x-1)continue;
    for(e=(int)relPosY-1;e<(int)relPosY+3;e++){
      if(e<0||e>m_splineSize.y-1)continue;
      for(f=(int)relPosZ-1;f<(int)relPosZ+3;f++){
        if(f<0||f>m_splineSize.z-1)continue;        
        
        functemp=m_function->getValue(relPosX-(float)d)*m_function->getValue(relPosY-(float)e)*m_function->getValue(relPosZ-(float)f);
        *(newX+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)+=*(splineParamBuffer+f*m_splineSize.x*m_splineSize.y+e*m_splineSize.x+d)*functemp;
        *(newY+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)+=*(splineParamBuffer+m_splineSize.x*m_splineSize.y*m_splineSize.z+f*m_splineSize.x*m_splineSize.y+e*m_splineSize.x+d)*functemp;
        *(newZ+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)+=*(splineParamBuffer+2*m_splineSize.x*m_splineSize.y*m_splineSize.z+f*m_splineSize.x*m_splineSize.y+e*m_splineSize.x+d)*functemp;
      }
    }
  }
  tempValue=m_transform->getInterpolationValue(m_targetPlusSpline, *(newX+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)+1.0, *(newY+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)+1.0,  *(newZ+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)+1.0);
  
  if(tempValue<0)tempValue=0;
  if(tempValue>255)tempValue=255;

  *(diff+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)=tempValue-*(m_referenceBuffer+k*m_referenceSize.x*m_referenceSize.y+j*m_referenceSize.x+i);
  if(*(diff+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)==1 || *(diff+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)==-1)*(diff+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)=0;
      }
    }
  }
  
  pthread_t tid[m_MTlevel];

  ntkCalculateSSDGradientThreadParam *threadParam = new ntkCalculateSSDGradientThreadParam[m_MTlevel];
 
  for(int i=0;i<m_MTlevel;i++){
    threadParam[i].here=this;
    threadParam[i].threadNumber=i;
    threadParam[i].newX=newX;
    threadParam[i].newY=newY;
    threadParam[i].newZ=newZ;
    threadParam[i].diff=diff;
    if(pthread_create(&tid[i], NULL, startCalculateSSDPEGradientThread, &(threadParam[i]))){
      printf("create thread %d failed\n",i);
      exit(1);
    }
  }
  for(int i=0;i<m_MTlevel;i++){
    if(pthread_join(tid[i], NULL)){
      printf("join thread %d failed\n",i);
      exit(1);
    }
  }

  float max=0.0;
    
  for(i=0;i<m_splineSize.x;i++){
    for(j=0;j<m_splineSize.y;j++){
      for(k=0;k<m_splineSize.z;k++){
  for(n=0;n<3;n++){
    *(m_SSDgradient+n*m_splineSize.x*m_splineSize.y*m_splineSize.z+k*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+i)/=m_targetSize.x*m_targetSize.y*m_targetSize.z;
  }
      }
    }
  }
  
  for(i=0;i<m_splineSize.x;i++){
    for(j=0;j<m_splineSize.y;j++){
      for(k=0;k<m_splineSize.z;k++){
  for(n=0;n<3;n++){
    if(fabs(*(m_SSDgradient+n*m_splineSize.x*m_splineSize.y*m_splineSize.z+k*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+i))>max){
      max=fabs(*(m_SSDgradient+n*m_splineSize.x*m_splineSize.y*m_splineSize.z+k*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+i));
    }
  }
      }
    }
  }

  printf("Max value=%lf\n", max);
  
  delete threadParam;
  free(diff);
  free(newX);
  free(newY);
  free(newZ);
  return max;
}

void ntkElasticRegistrationCUDA::threadCalculateSSDGradient(float *newX, float *newY, float* newZ, float *diff,  int threadNumber){
  int i,j,k;
  int a,b,c;
  int d,e,f;
  float val1, val2, val3;
  float tempPow=1.0/(float)m_knotDistance;
  float relPosX, relPosY, relPosZ;

  int startx;
  int endx;

  startx=threadNumber*m_splineSize.x/m_MTlevel;
  endx=(threadNumber+1)*m_splineSize.x/m_MTlevel;

  int indexabc=0;
  int indexijk=0;
  int splSize=m_splineSize.x*m_splineSize.y*m_splineSize.z;

  for(a=startx;a<endx;a++){
    for(b=0;b<m_splineSize.y;b++){
      for(c=0;c<m_splineSize.z;c++){
  indexabc=c*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a;
  *(m_SSDgradient+indexabc)=0;
  for(i=((a-2)*m_knotDistance-1);i<((a+2)*m_knotDistance);i++){
    if(i<0 || i>m_targetSize.x-1) continue;
    for(j=((b-2)*m_knotDistance-1);j<((b+2)*m_knotDistance);j++){
      if(j<0 || j>m_targetSize.y-1) continue;
      for(k=((c-2)*m_knotDistance-1);k<((c+2)*m_knotDistance);k++){
        if(k<0 || k>m_targetSize.z-1) continue;
        /*calculation of SSD gradient against x axis deformation parameter*/
        /*val1 calculation*/
        indexijk=k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i;

        val1=*(diff+indexijk);
        if(fabs(val1)>0.0001){

  

    /*val2 calculation*/
    val2=0;    
    relPosX=(float)*(newX+indexijk)+1.0;
    relPosY=(float)*(newY+indexijk)+1.0;
    relPosZ=(float)*(newZ+indexijk)+1.0;
    
    for(d=(int)relPosX-1;d<(int)relPosX+3;d++){
      if(d<0||d>=m_targetSize.x+2)continue;
      for(e=(int)relPosY-1;e<(int)relPosY+3;e++){
        if(e<0||e>=m_targetSize.y+2)continue;
        for(f=(int)relPosZ-1;f<(int)relPosZ+3;f++){
          if(f<0||f>=m_targetSize.z+2)continue;   
          val2+=m_targetPlusSpline->getValue(d,e,f)*m_function->getDifferentialValue(relPosX-(float)d)*m_function->getValue(relPosY-(float)e)*m_function->getValue(relPosZ-(float)f);
        }   
      }
    }
    //val2/=m_knotDistance;
    if(fabs(val2)>0.0001){
      
      /*val3 calculation*/
     
      *(m_SSDgradient+indexabc)+=val1*val2*m_function->getValue((float)i*tempPow+tempPow-(float)a)*m_function->getValue((float)j*tempPow+tempPow-(float)b)*m_function->getValue((float)k*tempPow+tempPow-(float)c)/m_knotDistance;
     
    }
        }

        
        /*calculation of SSD gradient against y axis deformation parameter*/
        /*val1 calculation*/
        val1=*(diff+indexijk);
        if(fabs(val1)>0.0001){
    /*val2 calculation*/
    val2=0;
    relPosX=(float)*(newX+indexijk)+1.0;
    relPosY=(float)*(newY+indexijk)+1.0;
    relPosZ=(float)*(newZ+indexijk)+1.0;
    
    for(d=(int)relPosX-1;d<(int)relPosX+3;d++){
      if(d<0||d>=m_targetSize.x+2)continue;
      for(e=(int)relPosY-1;e<(int)relPosY+3;e++){
        if(e<0||e>=m_targetSize.y+2)continue;
        for(f=(int)relPosZ-1;f<(int)relPosZ+3;f++){
          if(f<0||f>=m_targetSize.z+2)continue;   
          val2+=m_targetPlusSpline->getValue(d,e,f)*m_function->getValue(relPosX-(float)d)*m_function->getDifferentialValue(relPosY-(float)e)*m_function->getValue(relPosZ-(float)f);
        }   
      }
    }
    //val2/=m_knotDistance;
    if(fabs(val2)>0.0001){
      
      /*val3 calculation*/
     
      *(m_SSDgradient+splSize+indexabc)+=val1*val2*m_function->getValue((float)i*tempPow+tempPow-(float)a)*m_function->getValue((float)j*tempPow+tempPow-(float)b)*m_function->getValue((float)k*tempPow+tempPow-(float)c)/m_knotDistance;
    }
        }


        /*calculation of SSD gradient against z axis deformation parameter*/
        /*val1 calculation*/
        val1=*(diff+indexijk);
        if(fabs(val1)>0.0001){
    /*val2 calculation*/
    val2=0;
    relPosX=(float)*(newX+indexijk)+1.0;
    relPosY=(float)*(newY+indexijk)+1.0;
    relPosZ=(float)*(newZ+indexijk)+1.0;
    
    for(d=(int)relPosX-1;d<(int)relPosX+3;d++){
      if(d<0||d>=m_targetSize.x+2)continue;
      for(e=(int)relPosY-1;e<(int)relPosY+3;e++){
        if(e<0||e>=m_targetSize.y+2)continue;
        for(f=(int)relPosZ-1;f<(int)relPosZ+3;f++){
          if(f<0||f>=m_targetSize.z+2)continue;   
          val2+=m_targetPlusSpline->getValue(d,e,f)*m_function->getValue(relPosX-(float)d)*m_function->getValue(relPosY-(float)e)*m_function->getDifferentialValue(relPosZ-(float)f);
        }   
      }
    }

    //val2/=m_knotDistance;
    if(fabs(val2)>0.0001){
      
      /*val3 calculation*/
      
      *(m_SSDgradient+2*splSize+indexabc)+=val1*val2*m_function->getValue((float)i*tempPow+tempPow-(float)a)*m_function->getValue((float)j*tempPow+tempPow-(float)b)*m_function->getValue((float)k*tempPow+tempPow-(float)c)/m_knotDistance;
    }
        }
        
        
      }
    }
  }
      }
    }
  }
  
}

void ntkElasticRegistrationCUDA::threadCalculateSSDPEGradient(float *newX, float *newY, float* newZ, float *diff,  int threadNumber){
  int i,j,k;
  int a,b,c;
  int d,e,f;
  float val1, val2, val3;
  float tempPow=1.0/(float)m_knotDistance;
  float relPosX, relPosY, relPosZ;

  float *potEnergy=(float*)malloc(m_splineSize.x*m_splineSize.y*m_splineSize.z*3*sizeof(float));

  int startx;
  int endx;

  startx=threadNumber*m_splineSize.x/m_MTlevel;
  endx=(threadNumber+1)*m_splineSize.x/m_MTlevel;

  for(a=startx;a<endx;a++){
    for(b=0;b<m_splineSize.y;b++){
      for(c=0;c<m_splineSize.z;c++){
  *(m_SSDgradient+c*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)=0;
  for(i=((a-2)*m_knotDistance-1);i<((a+2)*m_knotDistance);i++){
    if(i<0 || i>m_targetSize.x-1) continue;
    for(j=((b-2)*m_knotDistance-1);j<((b+2)*m_knotDistance);j++){
      if(j<0 || j>m_targetSize.y-1) continue;
      for(k=((c-2)*m_knotDistance-1);k<((c+2)*m_knotDistance);k++){
        if(k<0 || k>m_targetSize.z-1) continue;
        /*calculation of SSD gradient against x axis deformation parameter*/
        /*val1 calculation*/
        val1=*(diff+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
        if(fabs(val1)>0.0001){

  

    /*val2 calculation*/
    val2=0;    
    relPosX=(float)*(newX+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)+1.0;
    relPosY=(float)*(newY+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)+1.0;
    relPosZ=(float)*(newZ+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)+1.0;
    
    for(d=(int)relPosX-1;d<(int)relPosX+3;d++){
      if(d<0||d>=m_targetSize.x+2)continue;
      for(e=(int)relPosY-1;e<(int)relPosY+3;e++){
        if(e<0||e>=m_targetSize.y+2)continue;
        for(f=(int)relPosZ-1;f<(int)relPosZ+3;f++){
          if(f<0||f>=m_targetSize.z+2)continue;   
          val2+=m_targetPlusSpline->getValue(d,e,f)*m_function->getDifferentialValue(relPosX-(float)d)*m_function->getValue(relPosY-(float)e)*m_function->getValue(relPosZ-(float)f);
        }   
      }
    }
    //val2/=m_knotDistance;
    if(fabs(val2)>0.0001){
      
      /*val3 calculation*/
  
      *(m_SSDgradient+c*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)+=val1*val2*m_function->getValue((float)i*tempPow+tempPow-(float)a)*m_function->getValue((float)j*tempPow+tempPow-(float)b)*m_function->getValue((float)k*tempPow+tempPow-(float)c)/m_knotDistance;
     
    }
        }

        
        /*calculation of SSD gradient against y axis deformation parameter*/
        /*val1 calculation*/
        val1=*(diff+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
        if(fabs(val1)>0.0001){
    /*val2 calculation*/
    val2=0;
    relPosX=(float)*(newX+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)+1.0;
    relPosY=(float)*(newY+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)+1.0;
    relPosZ=(float)*(newZ+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)+1.0;
    
    for(d=(int)relPosX-1;d<(int)relPosX+3;d++){
      if(d<0||d>=m_targetSize.x+2)continue;
      for(e=(int)relPosY-1;e<(int)relPosY+3;e++){
        if(e<0||e>=m_targetSize.y+2)continue;
        for(f=(int)relPosZ-1;f<(int)relPosZ+3;f++){
          if(f<0||f>=m_targetSize.z+2)continue;   
          val2+=m_targetPlusSpline->getValue(d,e,f)*m_function->getValue(relPosX-(float)d)*m_function->getDifferentialValue(relPosY-(float)e)*m_function->getValue(relPosZ-(float)f);
        }   
      }
    }
    //val2/=m_knotDistance;
    if(fabs(val2)>0.0001){
      
      /*val3 calculation*/
    

      *(m_SSDgradient+m_splineSize.x*m_splineSize.y*m_splineSize.z+c*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)+=val1*val2*m_function->getValue((float)i*tempPow+tempPow-(float)a)*m_function->getValue((float)j*tempPow+tempPow-(float)b)*m_function->getValue((float)k*tempPow+tempPow-(float)c)/m_knotDistance;
    }
        }

        /*calculation of SSD gradient against z axis deformation parameter*/
        /*val1 calculation*/
        val1=*(diff+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
        if(fabs(val1)>0.0001){
    /*val2 calculation*/
    val2=0;
    relPosX=(float)*(newX+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)+1.0;
    relPosY=(float)*(newY+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)+1.0;
    relPosZ=(float)*(newZ+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)+1.0;
    
    for(d=(int)relPosX-1;d<(int)relPosX+3;d++){
      if(d<0||d>=m_targetSize.x+2)continue;
      for(e=(int)relPosY-1;e<(int)relPosY+3;e++){
        if(e<0||e>=m_targetSize.y+2)continue;
        for(f=(int)relPosZ-1;f<(int)relPosZ+3;f++){
          if(f<0||f>=m_targetSize.z+2)continue;   
          val2+=m_targetPlusSpline->getValue(d,e,f)*m_function->getValue(relPosX-(float)d)*m_function->getValue(relPosY-(float)e)*m_function->getDifferentialValue(relPosZ-(float)f);
        }   
      }
    }

    //val2/=m_knotDistance;
    if(fabs(val2)>0.0001){
      
      /*val3 calculation*/
      
  
      *(m_SSDgradient+2*m_splineSize.x*m_splineSize.y*m_splineSize.z+c*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)+=val1*val2*m_function->getValue((float)i*tempPow+tempPow-(float)a)*m_function->getValue((float)j*tempPow+tempPow-(float)b)*m_function->getValue((float)k*tempPow+tempPow-(float)c)/m_knotDistance;
    }
        }
        
        
      }
    }
  }
      }
    }
  }

  /*calculate SSDPE gradient based on spring potential energy*/

  float distance=1.0;
  float distance2=1.0;
  float orgPosX;
  float orgPosY;
  float orgPosZ;
  float relPos2X;
  float relPos2Y;
  float relPos2Z;
  float size=m_targetSize.x*m_targetSize.y*m_targetSize.z;
  float *emapBuffer=m_emap->getBuffer();
  float elasticity;
  float elasticity2;
  
  for(a=startx;a<endx;a++){
    for(b=0;b<m_splineSize.y;b++){
      for(c=0;c<m_splineSize.z;c++){
  orgPosX=a-1;orgPosY=b-1;orgPosZ=c-1;
  *(potEnergy+0*m_splineSize.x*m_splineSize.y*m_splineSize.z+c*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)=0;
  *(potEnergy+1*m_splineSize.x*m_splineSize.y*m_splineSize.z+c*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)=0;
  *(potEnergy+2*m_splineSize.x*m_splineSize.y*m_splineSize.z+c*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)=0;
  //*(m_SSDgradient+c*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)=0;
  for(i=((a-2)*m_knotDistance-1);i<((a+2)*m_knotDistance);i++){
    if(i<0 || i>m_targetSize.x-2) continue;
    for(j=((b-2)*m_knotDistance-1);j<((b+2)*m_knotDistance);j++){
      if(j<0 || j>m_targetSize.y-1) continue;
      for(k=((c-2)*m_knotDistance-1);k<((c+2)*m_knotDistance);k++){
        if(k<0 || k>m_targetSize.z-1) continue;
        
        relPosX=(float)*(newX+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
        relPosY=(float)*(newY+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
        relPosZ=(float)*(newZ+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);

        relPos2X=(float)*(newX+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i+1);
        relPos2Y=(float)*(newY+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i+1);
        relPos2Z=(float)*(newZ+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i+1);
        
        distance=sqrt((relPos2X-relPosX)*(relPos2X-relPosX)+(relPos2Y-relPosY)*(relPos2Y-relPosY)+(relPos2Z-relPosZ)*(relPos2Z-relPosZ));
        
        elasticity=*(emapBuffer+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
        elasticity2=*(emapBuffer+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+(i+1));
        
        //printf("elasticity:%lf %lf \n", elasticity, elasticity2);

        *(potEnergy+c*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)+=((relPos2X-relPosX)/distance-(relPos2X-relPosX)/pow(distance,3))*(m_function->getValue(i-orgPosX+1)*m_function->getValue(j-orgPosY)*m_function->getValue(k-orgPosZ)-m_function->getValue(i-orgPosX)*m_function->getValue(j-orgPosY)*m_function->getValue(k-orgPosZ))*(elasticity+elasticity2)/2.0;
        
        //*(potEnergy+c*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)+=(2*(distance-1)*(relPos2X-relPosX)/distance)*(m_function->getValue(i-orgPosX+1)*m_function->getValue(j-orgPosY)*m_function->getValue(k-orgPosZ)-m_function->getValue(i-orgPosX)*m_function->getValue(j-orgPosY)*m_function->getValue(k-orgPosZ));
         
       
      }
    }
  }
  
  *(m_SSDgradient+0*m_splineSize.x*m_splineSize.y*m_splineSize.z+c*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)+=*(potEnergy+0*m_splineSize.x*m_splineSize.y*m_splineSize.z+c*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)*size*m_PEweight*pow(8,-m_imageSizeLevel);//*pow(8,m_knotDistance-1);///m_knotDistance/m_knotDistance;
  
  for(i=((a-2)*m_knotDistance-1);i<((a+2)*m_knotDistance);i++){
    if(i<0 || i>m_targetSize.x-1) continue;
    for(j=((b-2)*m_knotDistance-1);j<((b+2)*m_knotDistance);j++){
      if(j<0 || j>m_targetSize.y-2) continue;
      for(k=((c-2)*m_knotDistance-1);k<((c+2)*m_knotDistance);k++){
        if(k<0 || k>m_targetSize.z-1) continue;
        
        relPosX=(float)*(newX+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
        relPosY=(float)*(newY+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
        relPosZ=(float)*(newZ+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);

        relPos2X=(float)*(newX+k*m_targetSize.x*m_targetSize.y+(j+1)*m_targetSize.x+i);
        relPos2Y=(float)*(newY+k*m_targetSize.x*m_targetSize.y+(j+1)*m_targetSize.x+i);
        relPos2Z=(float)*(newZ+k*m_targetSize.x*m_targetSize.y+(j+1)*m_targetSize.x+i);
        
        distance=sqrt((relPos2X-relPosX)*(relPos2X-relPosX)+(relPos2Y-relPosY)*(relPos2Y-relPosY)+(relPos2Z-relPosZ)*(relPos2Z-relPosZ));
        
        elasticity=*(emapBuffer+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
        elasticity2=(*(emapBuffer+k*m_targetSize.x*m_targetSize.y+(j+1)*m_targetSize.x+i));
        
        //printf("elasticity:%lf %lf \n", elasticity, elasticity2);

        *(potEnergy+m_splineSize.x*m_splineSize.y*m_splineSize.z+c*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)+=((relPos2Y-relPosY)/distance-(relPos2Y-relPosY)/pow(distance,3))*(m_function->getValue(i-orgPosX)*m_function->getValue(j-orgPosY+1)*m_function->getValue(k-orgPosZ)-m_function->getValue(i-orgPosX)*m_function->getValue(j-orgPosY)*m_function->getValue(k-orgPosZ))*(elasticity+elasticity2)/2.0;
        
        //*(potEnergy+c*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)+=(2*(distance-1)*(relPos2Y-relPosY)/distance)*(m_function->getValue(i-orgPosX)*m_function->getValue(j-orgPosY+1)*m_function->getValue(k-orgPosZ)-m_function->getValue(i-orgPosX)*m_function->getValue(j-orgPosY)*m_function->getValue(k-orgPosZ));
      }
    }
  }

  *(m_SSDgradient+1*m_splineSize.x*m_splineSize.y*m_splineSize.z+c*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)+=*(potEnergy+1*m_splineSize.x*m_splineSize.y*m_splineSize.z+c*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)*size*m_PEweight*pow(8,-m_imageSizeLevel);//*pow(8,m_knotDistance-1);///m_knotDistance/m_knotDistance;

  for(i=((a-2)*m_knotDistance-1);i<((a+2)*m_knotDistance);i++){
    if(i<0 || i>m_targetSize.x-1) continue;
    for(j=((b-2)*m_knotDistance-1);j<((b+2)*m_knotDistance);j++){
      if(j<0 || j>m_targetSize.y-1) continue;
      for(k=((c-2)*m_knotDistance-1);k<((c+2)*m_knotDistance);k++){
        if(k<0 || k>m_targetSize.z-2) continue;
        
        relPosX=(float)*(newX+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
        relPosY=(float)*(newY+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
        relPosZ=(float)*(newZ+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);

        relPos2X=(float)*(newX+(k+1)*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
        relPos2Y=(float)*(newY+(k+1)*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
        relPos2Z=(float)*(newZ+(k+1)*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
        
        distance=sqrt((relPos2X-relPosX)*(relPos2X-relPosX)+(relPos2Y-relPosY)*(relPos2Y-relPosY)+(relPos2Z-relPosZ)*(relPos2Z-relPosZ));

        elasticity=*(emapBuffer+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
        elasticity2=(*(emapBuffer+(k+1)*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i));
        
        //printf("elasticity:%lf %lf \n", elasticity, elasticity2);

        *(potEnergy+2*m_splineSize.x*m_splineSize.y*m_splineSize.z+c*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)+=((relPos2Z-relPosZ)/distance-(relPos2Z-relPosZ)/pow(distance,3))*(m_function->getValue(i-orgPosX)*m_function->getValue(j-orgPosY)*m_function->getValue(k-orgPosZ+1)-m_function->getValue(i-orgPosX)*m_function->getValue(j-orgPosY)*m_function->getValue(k-orgPosZ))*(elasticity+elasticity2)/2.0;
        
        //*(potEnergy+c*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)+=(2*(distance-1)*(relPos2Z-relPosZ)/distance)*(m_function->getValue(i-orgPosX)*m_function->getValue(j-orgPosY)*m_function->getValue(k-orgPosZ+1)-m_function->getValue(i-orgPosX)*m_function->getValue(j-orgPosY)*m_function->getValue(k-orgPosZ));

      }
    }
  }
  
    
  
  *(m_SSDgradient+2*m_splineSize.x*m_splineSize.y*m_splineSize.z+c*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)+=*(potEnergy+2*m_splineSize.x*m_splineSize.y*m_splineSize.z+c*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)*size*m_PEweight*pow(8,-m_imageSizeLevel);//*pow(8,m_knotDistance-1);///m_knotDistance/m_knotDistance;
  
      }
    }
  }

  free(potEnergy);
}

void* ntkElasticRegistrationCUDA::startCalculateSSDGradientThread(void *arg){
  ntkCalculateSSDGradientThreadParam *temp=(ntkCalculateSSDGradientThreadParam*)arg;
  (temp->here)->threadCalculateSSDGradient(temp->newX, temp->newY, temp->newZ, temp->diff, temp->threadNumber);
  return 0;
}

void* ntkElasticRegistrationCUDA::startCalculateSSDPEGradientThread(void *arg){
  ntkCalculateSSDGradientThreadParam *temp=(ntkCalculateSSDGradientThreadParam*)arg;
  (temp->here)->threadCalculateSSDPEGradient(temp->newX, temp->newY, temp->newZ, temp->diff, temp->threadNumber);
  return 0;
}

float ntkElasticRegistrationCUDA::calculateSSDGradient(ntk3DData* target, int splineSizeLevel=0, ntkDeformationSpline *splineParam=NULL){
  int j,k;
  m_splineSizeLevel=splineSizeLevel;
  m_target=target;
  m_targetSize=target->getDataSize();
  if(m_targetSize.x!=m_referenceSize.x || m_targetSize.y!=m_referenceSize.y || m_targetSize.z!=m_referenceSize.z){
    printf("Size of target and reference do not match.\n");
    return 0;
  }

  /*targetPlus is 3D data of target with one additional voxel in each direction*/
  ntk3DData *targetPlus=new ntk3DData(ntkIntDimension(m_targetSize.x+2, m_targetSize.y+2, m_targetSize.z+2));
  unsigned char *targetBuffer=(unsigned char*)target->getBuffer();
  unsigned char *targetPlusBuffer=(unsigned char*)targetPlus->getBuffer();
  
  for(k=0;k<m_targetSize.z;k++){
    for(j=0;j<m_targetSize.y;j++){
      memcpy(targetPlusBuffer+(k+1)*(m_targetSize.x+2)*(m_targetSize.y+2)+(j+1)*(m_targetSize.x+2)+1, targetBuffer+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x, m_targetSize.x);
    }
  }
  
  m_targetPlus=targetPlus;

  /*m_targetPlusSpline is B-Spline representation of targetPlus*/
  m_targetPlusSpline=m_transform->doForwardTransform(m_targetPlus, CURRENT_RES);
  float tempPow=pow(2, splineSizeLevel);
  float knotDistance=pow(2, -splineSizeLevel);
  m_knotDistance=(int)knotDistance;
  m_splineSize= ntkIntDimension((int)(tempPow*(m_targetSize.x+1)+1), (int)(tempPow*(m_targetSize.y+1)+1), (int)(tempPow*(m_targetSize.z+1)+1));
  
  if(splineParam==NULL){
    splineParam=new ntkDeformationSpline(ntkIntDimension(m_splineSize.x,m_splineSize.y,m_splineSize.z));
  }
  m_splineParam=splineParam;
  
  float max=calculateSSDGradient(splineParam);
  return max;
}

float* ntkElasticRegistrationCUDA::getGradientBuffer(){
  return m_SSDgradient;
}

ntkIntDimension ntkElasticRegistrationCUDA::getSplineSize(){
  return m_splineSize;
}

float ntkElasticRegistrationCUDA::calculateSSD(ntk3DData* reference, ntk3DData* target){
  int i,j,k;
  ntkIntDimension refDim=reference->getDataSize();
  ntkIntDimension tarDim=target->getDataSize();

  if(refDim.x!=tarDim.x||refDim.y!=tarDim.y||refDim.z!=tarDim.z){
    printf("Reference and Target size do not match\n");
    return 0;
  }

  if(m_dRef!=NULL)delete m_dRef;
  if(m_dTar!=NULL)delete m_dTar;
  
  m_dRef = new ntkCudaDeviceMemory();
  m_dRef->Allocate<unsigned char>(refDim.x*refDim.y*refDim.z);
  m_dTar = new ntkCudaDeviceMemory();
  m_dTar->Allocate<unsigned char>(refDim.x*refDim.y*refDim.z);

  m_dRef->copyFromHost(reference->getBuffer());
  m_dTar->copyFromHost(target->getBuffer());
  
  return CUDAcalculateSSD_doCalculation((unsigned char*)m_dRef->getDeviceBuffer(), (unsigned char*)m_dTar->getDeviceBuffer(), refDim.x, refDim.y, refDim.z);

}

float ntkElasticRegistrationCUDA::calculateSSDPE(ntk3DData* reference, ntk3DData* target, ntkDeformationSpline *splineParam){
  int i,j,k;
  float potEnergy=0;
  ntkIntDimension refDim=reference->getDataSize();
  ntkIntDimension tarDim=target->getDataSize();
  
  unsigned char *refBuffer=(unsigned char*)reference->getBuffer();
  unsigned char *tarBuffer=(unsigned char*)target->getBuffer();
  
  if(refDim.x!=tarDim.x||refDim.y!=tarDim.y||refDim.z!=tarDim.z){
    printf("Reference and Target size do not match\n");
    return 0;
  }
  
  float value=0;
  
  for(i=0;i<refDim.x;i++){
    for(j=0;j<refDim.y;j++){
      for(k=0;k<refDim.z;k++){
  value+=((int)*(refBuffer+k*refDim.x*refDim.y+j*refDim.x+i)-(int)*(tarBuffer+k*refDim.x*refDim.y+j*refDim.x+i))*((int)*(refBuffer+k*refDim.x*refDim.y+j*refDim.x+i)-(int)*(tarBuffer+k*refDim.x*refDim.y+j*refDim.x+i));
      }
    }
  }
  
  /*calculate energy potential*/
  
 int n; 

  float *splineParamBuffer=splineParam->getBuffer();
  int a,b,c;
  int d,e,f;
  float functemp;
  float tempPow=1.0/(float)m_knotDistance;
  float relPosX, relPosY, relPosZ;
  float *newX=(float*)malloc(m_targetSize.x*m_targetSize.y*m_targetSize.z*sizeof(float));
  float *newY=(float*)malloc(m_targetSize.x*m_targetSize.y*m_targetSize.z*sizeof(float));
  float *newZ=(float*)malloc(m_targetSize.x*m_targetSize.y*m_targetSize.z*sizeof(float));
  float *diff=(float*)malloc(m_targetSize.x*m_targetSize.y*m_targetSize.z*sizeof(float));

    /*calculating difference between each pixel of reference and transformed target*/
  for(i=0;i<m_targetSize.x;i++){
    for(j=0;j<m_targetSize.y;j++){
      for(k=0;k<m_targetSize.z;k++){
  *(newX+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)=i; 
  *(newY+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)=j; 
  *(newZ+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)=k; 
  relPosX=(float)(i+1)*tempPow;
  relPosY=(float)(j+1)*tempPow;
  relPosZ=(float)(k+1)*tempPow;
  
  for(d=(int)relPosX-1;d<(int)relPosX+3;d++){
    if(d<0||d>m_splineSize.x-1)continue;
    for(e=(int)relPosY-1;e<(int)relPosY+3;e++){
      if(e<0||e>m_splineSize.y-1)continue;
      for(f=(int)relPosZ-1;f<(int)relPosZ+3;f++){
        if(f<0||f>m_splineSize.z-1)continue;        
        
        functemp=m_function->getValue(relPosX-(float)d)*m_function->getValue(relPosY-(float)e)*m_function->getValue(relPosZ-(float)f);
        *(newX+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)+=*(splineParamBuffer+f*m_splineSize.x*m_splineSize.y+e*m_splineSize.x+d)*functemp;
        *(newY+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)+=*(splineParamBuffer+m_splineSize.x*m_splineSize.y*m_splineSize.z+f*m_splineSize.x*m_splineSize.y+e*m_splineSize.x+d)*functemp;
        *(newZ+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i)+=*(splineParamBuffer+2*m_splineSize.x*m_splineSize.y*m_splineSize.z+f*m_splineSize.x*m_splineSize.y+e*m_splineSize.x+d)*functemp;
      }
    }
  }
  
      }
    }
  }

  /*calculate SSDPE based on spring potential energy*/

  float distance=1.0;
  float relPos2X;
  float relPos2Y;
  float relPos2Z;
  float elasticity;
  float elasticity2;
  float* emapBuffer=m_emap->getBuffer();

  for(i=0;i<m_targetSize.x-1;i++){
    for(j=0;j<m_targetSize.y;j++){
      for(k=0;k<m_targetSize.z;k++){
    relPosX=(float)*(newX+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
    relPosY=(float)*(newY+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
    relPosZ=(float)*(newZ+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
    
    relPos2X=(float)*(newX+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i+1);
    relPos2Y=(float)*(newY+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i+1);
    relPos2Z=(float)*(newZ+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i+1);
    
    distance=sqrt((relPos2X-relPosX)*(relPos2X-relPosX)+(relPos2Y-relPosY)*(relPos2Y-relPosY)+(relPos2Z-relPosZ)*(relPos2Z-relPosZ));
    
    elasticity=*(emapBuffer+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
    elasticity2=(*(emapBuffer+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i+1));
    
    potEnergy+=((distance-1)*(distance-1))/distance*(elasticity+elasticity2)/2.0;;
    //potEnergy+=((distance-1)*(distance-1));
      }
    }
  }

  for(i=0;i<m_targetSize.x;i++){
    for(j=0;j<m_targetSize.y-1;j++){
      for(k=0;k<m_targetSize.z;k++){
  relPosX=(float)*(newX+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
  relPosY=(float)*(newY+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
  relPosZ=(float)*(newZ+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
  
  relPos2X=(float)*(newX+k*m_targetSize.x*m_targetSize.y+(j+1)*m_targetSize.x+i);
  relPos2Y=(float)*(newY+k*m_targetSize.x*m_targetSize.y+(j+1)*m_targetSize.x+i);
  relPos2Z=(float)*(newZ+k*m_targetSize.x*m_targetSize.y+(j+1)*m_targetSize.x+i);
  
  distance=sqrt((relPos2X-relPosX)*(relPos2X-relPosX)+(relPos2Y-relPosY)*(relPos2Y-relPosY)+(relPos2Z-relPosZ)*(relPos2Z-relPosZ));
  
  elasticity=(*(emapBuffer+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i));
  elasticity2=(*(emapBuffer+k*m_targetSize.x*m_targetSize.y+(j+1)*m_targetSize.x+i));
    
  potEnergy+=((distance-1)*(distance-1))/distance*(elasticity+elasticity2)/2.0;;
  //potEnergy+=((distance-1)*(distance-1));
      }
    }
  }
  
  for(i=0;i<m_targetSize.x;i++){
    for(j=0;j<m_targetSize.y;j++){
      for(k=0;k<m_targetSize.z-1;k++){
  relPosX=(float)*(newX+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
  relPosY=(float)*(newY+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
  relPosZ=(float)*(newZ+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
  
  relPos2X=(float)*(newX+(k+1)*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
  relPos2Y=(float)*(newY+(k+1)*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
  relPos2Z=(float)*(newZ+(k+1)*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i);
  
  distance=sqrt((relPos2X-relPosX)*(relPos2X-relPosX)+(relPos2Y-relPosY)*(relPos2Y-relPosY)+(relPos2Z-relPosZ)*(relPos2Z-relPosZ));
  
  elasticity=(*(emapBuffer+k*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i));
  elasticity2=(*(emapBuffer+(k+1)*m_targetSize.x*m_targetSize.y+j*m_targetSize.x+i));
    
  potEnergy+=((distance-1)*(distance-1))/distance*(elasticity+elasticity2)/2.0;;
  //potEnergy+=((distance-1)*(distance-1));
      }
    }
  }
  

  printf("potential energy: %lf\n", potEnergy*m_PEweight*pow(8,-m_imageSizeLevel));

  delete newX;
  delete newY;
  delete newZ;
  delete diff;

  return value/refDim.x/refDim.y/refDim.z+potEnergy*m_PEweight*pow(8,-m_imageSizeLevel);
}

ntkDeformationSpline *ntkElasticRegistrationCUDA::doSplineParamTransform(ntkDeformationSpline* inputSpline){
  float *inputSplineBuffer=inputSpline->getBuffer();
  ntkIntDimension inputSplineSize=inputSpline->getSize();
  ntkTensor *inputTensorX=new ntkTensor(inputSplineBuffer, inputSplineSize);
  ntkTensor *inputTensorY=new ntkTensor(inputSplineBuffer+inputSplineSize.x*inputSplineSize.y*inputSplineSize.z, inputSplineSize);
  ntkTensor *inputTensorZ=new ntkTensor(inputSplineBuffer+2*inputSplineSize.x*inputSplineSize.y*inputSplineSize.z, inputSplineSize);
  ntkBSpline3Transform *transform=new ntkBSpline3Transform();
  
  ntkTensor *outputTensorX=transform->doSplineToSplineHigherResTransform(inputTensorX);
  ntkTensor *outputTensorY=transform->doSplineToSplineHigherResTransform(inputTensorY);
  ntkTensor *outputTensorZ=transform->doSplineToSplineHigherResTransform(inputTensorZ);

  ntkIntDimension outputSize=outputTensorX->getTensorSize();

  ntkDeformationSpline *outputSpline=new ntkDeformationSpline(ntkIntDimension(outputSize.x, outputSize.y, outputSize.z));
  float *outputSplineBuffer=outputSpline->getBuffer();

  float *outputTensorXBuffer=outputTensorX->getTensorPointer();
  float *outputTensorYBuffer=outputTensorY->getTensorPointer();
  float *outputTensorZBuffer=outputTensorZ->getTensorPointer();

  memcpy(outputSplineBuffer, outputTensorXBuffer, outputSize.x*outputSize.y*outputSize.z*sizeof(float));
  memcpy(outputSplineBuffer+outputSize.x*outputSize.y*outputSize.z, outputTensorYBuffer, outputSize.x*outputSize.y*outputSize.z*sizeof(float));
  memcpy(outputSplineBuffer+2*outputSize.x*outputSize.y*outputSize.z, outputTensorZBuffer, outputSize.x*outputSize.y*outputSize.z*sizeof(float));

  
  delete transform;
  delete inputTensorX;
  delete inputTensorY;
  delete inputTensorZ;
  delete outputTensorX;
  delete outputTensorY;
  delete outputTensorZ;
  
  return outputSpline;
}

void ntkElasticRegistrationCUDA::addRegistrationScheme(int imageResLevel, int splineResLevel, float step=1.0, float PEweight=0.0){
  //ntkElasticRegistrationCUDAScheme *newScheme=new ntkElasticRegistrationCUDAScheme();
  ntkElasticRegistrationCUDAScheme *newScheme=(ntkElasticRegistrationCUDAScheme*)malloc(sizeof(ntkElasticRegistrationCUDAScheme));
  newScheme->imageResLevel=imageResLevel;
  newScheme->splineResLevel=splineResLevel;
  newScheme->step=step;
  newScheme->PEweight=PEweight;
  newScheme->next=NULL;
  
  ntkElasticRegistrationCUDAScheme *tempScheme;

  if(m_firstScheme==NULL && m_lastScheme==NULL){  
    m_firstScheme=newScheme;
    m_lastScheme=newScheme;
  }else{
    tempScheme=m_lastScheme;
    m_lastScheme=newScheme;
    tempScheme->next=m_lastScheme;
  }
}

void ntkElasticRegistrationCUDA::resetRegistrationScheme(){
  ntkElasticRegistrationCUDAScheme *tempScheme;

  while(m_firstScheme!=NULL){
    tempScheme=m_firstScheme;
    m_firstScheme=tempScheme->next;
    //delete tempScheme;
    free(tempScheme);
  }
  
  m_firstScheme=NULL;
  m_lastScheme=NULL;
}

void ntkElasticRegistrationCUDA::printRegistrationScheme(){
  ntkElasticRegistrationCUDAScheme *tempScheme;

  tempScheme=m_firstScheme;

  if(tempScheme==NULL){
    printf("There is no elastic registration scheme defined\n");
    return;
  }

  int count=1;

  while(tempScheme!=NULL){
    printf("Elastic registration scheme No.%d : imageResLevel=%d, splineResLevel=%d, PEweight=%lf\n", count, tempScheme->imageResLevel, tempScheme->splineResLevel, tempScheme->PEweight);
    count++;
    tempScheme=tempScheme->next;
  }
}

bool ntkElasticRegistrationCUDA::checkRegistrationScheme(){
  if(m_firstScheme==NULL){
    return false;
  }else if(m_firstScheme->next==NULL){
    return true;
  }else{
    ntkElasticRegistrationCUDAScheme *tempScheme=m_firstScheme;
    do{
      if(tempScheme->imageResLevel+tempScheme->splineResLevel>tempScheme->next->imageResLevel+tempScheme->next->splineResLevel){
  return false;
      }
      tempScheme=tempScheme->next;
    }while(tempScheme->next!=NULL);
  }
  return true;
}

ntkDeformationSpline* ntkElasticRegistrationCUDA::executeRegistrationSchemes(ntk3DData* reference, ntk3DData *target, ntkDeformationSpline* inputSpline=NULL, ntkElasticityMap* emap=NULL){
  int i;
  
  if(!checkRegistrationScheme()){
    printf("Registration scheme check failed. Registration is terminated\n");
    return NULL;
  }
  
  ntkIntDimension targetSize=target->getDataSize();
  ntkIntDimension referenceSize=reference->getDataSize();

  if(targetSize.x!=referenceSize.x || targetSize.y!=referenceSize.y || targetSize.z!=referenceSize.z){
    printf("Size of target and reference do not match.\n");
    return 0;
  }

  ntkDeformationSpline *outputSpline;
  ntkDeformationSpline *tempSpline;
  
  ntkElasticRegistrationCUDAScheme *tempScheme=m_firstScheme;
  ntk3DData* tempReference;
  ntk3DData* tempTarget;
  ntkElasticityMap* tempMap;
  bool needRelease=false;

  int imageResLevel;
  int splineResLevel;
  int prevImageResLevel;
  int prevSplineResLevel;
  int splineTransformNum;

  imageResLevel=tempScheme->imageResLevel;
  splineResLevel=tempScheme->splineResLevel;
  
  if(imageResLevel<0){
    tempReference=resizeDataSmaller(reference, imageResLevel);
    tempTarget=resizeDataSmaller(target, imageResLevel);
    tempMap=resizeMapSmaller(emap, imageResLevel);
    needRelease=true;
  }else{
    tempReference=reference;
    tempTarget=target;
    tempMap=emap;
    needRelease=false;
  }
  
  m_imageSizeLevel=imageResLevel;
  outputSpline=doRegistration(tempReference, tempTarget, tempMap, tempScheme->splineResLevel, inputSpline, tempScheme->step, tempScheme->PEweight);
  
  if(needRelease){
    delete tempReference;
    delete tempTarget;
    delete tempMap;
  }
  
  tempScheme=tempScheme->next;

  while(tempScheme!=NULL){
    prevImageResLevel=imageResLevel;
    prevSplineResLevel=splineResLevel;
    imageResLevel=tempScheme->imageResLevel;
    splineResLevel=tempScheme->splineResLevel;

    splineTransformNum=(imageResLevel+splineResLevel)-(prevImageResLevel+prevSplineResLevel);
    
    for(i=0;i<splineTransformNum;i++){
      tempSpline=outputSpline;
      outputSpline=doSplineParamTransform(tempSpline);
      delete tempSpline;
    }
    
    int imageResLevelUp=imageResLevel-prevImageResLevel;
    
    if(imageResLevelUp!=0){
      float *splineBuffer=outputSpline->getBuffer();
      ntkIntDimension splineSize=outputSpline->getSize();
      
      for(i=0;i<splineSize.x*splineSize.y*splineSize.z*3;i++){
  *(splineBuffer+i) *= pow(2, imageResLevelUp);
      }
    }
    
    if(imageResLevel<0){
      tempReference=resizeDataSmaller(reference, imageResLevel);
      tempTarget=resizeDataSmaller(target, imageResLevel);
      tempMap=resizeMapSmaller(emap, imageResLevel);
      needRelease=true;
    }else{
      tempReference=reference;
      tempTarget=target;
      tempMap=emap;
      needRelease=false;
    }
    
    tempSpline=outputSpline;
    m_imageSizeLevel=imageResLevel;
    outputSpline=doRegistration(tempReference, tempTarget, tempMap, tempScheme->splineResLevel, tempSpline, tempScheme->step, tempScheme->PEweight);
    delete tempSpline;

    if(needRelease){
      delete tempReference;
      delete tempTarget;
      delete tempMap;
    }
    
    tempScheme=tempScheme->next;
  }
  m_imageSizeLevel=0;

  return outputSpline;
}

ntk3DData *ntkElasticRegistrationCUDA::resizeDataSmaller(ntk3DData* input, int imageResLevel){
  int i,j,k;
  ntkIntDimension inputSize=input->getDataSize();
  ntkFloatDimension inputThickness=input->getDataThickness();
  ntkIntDimension outputSize=inputSize;
  ntkFloatDimension outputThickness=inputThickness;

  for(i=0;i<-imageResLevel;i++){
    outputSize.x=(outputSize.x+1)/2-1;
    outputSize.y=(outputSize.y+1)/2-1;
    outputSize.z=(outputSize.z+1)/2-1;
  }
  ntk3DData *output=new ntk3DData(outputSize, outputThickness);
  unsigned char* inputBuffer=(unsigned char*)input->getBuffer();
  unsigned char* outputBuffer=(unsigned char*)output->getBuffer();
  
  int factor=(int)pow(2, -imageResLevel);
  
  for(i=0;i<outputSize.x;i++){
    for(j=0;j<outputSize.y;j++){
      for(k=0;k<outputSize.z;k++){
  *(outputBuffer+k*outputSize.x*outputSize.y+j*outputSize.x+i)=*(inputBuffer+(((k+1)*factor)-1)*inputSize.x*inputSize.y+(((j+1)*factor)-1)*inputSize.x+(((i+1)*factor)-1));
      }
    }
  }  
  
  return output;
}

ntkElasticityMap *ntkElasticRegistrationCUDA::resizeMapSmaller(ntkElasticityMap* input, int imageResLevel){
  int i,j,k;
  ntkIntDimension inputSize=input->getMapSize();
  ntkIntDimension outputSize=inputSize;
  
  for(i=0;i<-imageResLevel;i++){
    outputSize.x=(outputSize.x+1)/2-1;
    outputSize.y=(outputSize.y+1)/2-1;
    outputSize.z=(outputSize.z+1)/2-1;
  }
  ntkElasticityMap *output=new ntkElasticityMap(outputSize);
  float* inputBuffer=input->getBuffer();
  float* outputBuffer=output->getBuffer();
  
  int factor=(int)pow(2, -imageResLevel);
  
  for(i=0;i<outputSize.x;i++){
    for(j=0;j<outputSize.y;j++){
      for(k=0;k<outputSize.z;k++){
  *(outputBuffer+k*outputSize.x*outputSize.y+j*outputSize.x+i)=*(inputBuffer+(((k+1)*factor)-1)*inputSize.x*inputSize.y+(((j+1)*factor)-1)*inputSize.x+(((i+1)*factor)-1));
      }
    }
  }  
  
  return output;
}

void ntkElasticRegistrationCUDA::setMinGradientLimit(float value){
  m_minGradientLimit=value;
}
