#include "ntk2DElasticRegistration.h"

ntk2DElasticRegistration::ntk2DElasticRegistration(){
  m_reference=NULL;
  m_splineParam=NULL;
  m_transform=new ntk2DBSpline3Transform();
  m_function=new ntkBSpline3Function();
  m_SSDgradient=NULL;
  m_maxGradient=0.0;
  m_splineSizeLevel=0;
  m_imageSizeLevel=0;
  m_MTlevel=4;
  m_firstScheme=NULL;
  m_lastScheme=NULL;
  m_minGradientLimit=0.1;
  m_PEweight=0.0;
}

ntk2DElasticRegistration::~ntk2DElasticRegistration(){
  delete m_transform;
  delete m_function;
  if(m_splineParam!=NULL)delete m_splineParam;
  if(m_SSDgradient!=NULL)free(m_SSDgradient);
  resetRegistrationScheme();
}

void ntk2DElasticRegistration::loadData(ntk2DData* reference){
  m_reference=reference;
  m_referenceSize=reference->getDataSize();
  m_referenceBuffer=(unsigned char*)m_reference->getBuffer();
}

void ntk2DElasticRegistration::doRegistration(ntk3DData* target){}//dummy function

ntkDeformationSpline* ntk2DElasticRegistration::doRegistration(ntk2DData* reference, ntk2DData* target, ntkElasticityMap* emap, int splineSizeLevel=0, ntkDeformationSpline* splineParam=NULL, float step=1.0, float PEweight=1.0){
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

  if(m_targetSize.x!=m_referenceSize.x || m_targetSize.y!=m_referenceSize.y){
    printf("Size of target and reference do not match.\n");
    return 0;
  }

  /*targetPlus is 2D data of target with one additional voxel in each direction*/
  ntk2DData *targetPlus=new ntk2DData(ntkIntDimension(m_targetSize.x+2, m_targetSize.y+2));
  unsigned char *targetBuffer=(unsigned char*)target->getBuffer();
  unsigned char *targetPlusBuffer=(unsigned char*)targetPlus->getBuffer();
  
  for(j=0;j<m_targetSize.y;j++){
    memcpy(targetPlusBuffer+(j+1)*(m_targetSize.x+2)+1, targetBuffer+j*m_targetSize.x, m_targetSize.x);
  }
  
  
  m_targetPlus=targetPlus;

  /*m_targetPlusSpline is B-Spline representation of targetPlus*/

  m_targetPlusSpline=m_transform->doForwardTransform(m_targetPlus, CURRENT_RES);
  float tempPow=pow(2, splineSizeLevel);
  float knotDistance=pow(2, -splineSizeLevel);
  m_knotDistance=(int)knotDistance;
  m_splineSize= ntkIntDimension((int)(tempPow*(m_targetSize.x+1)+1), (int)(tempPow*(m_targetSize.y+1)+1), 1);
  
  ntkDeformationSpline *outputSpline=new ntkDeformationSpline(ntkIntDimension(m_splineSize.x, m_splineSize.y, 1));
  float *outputSplineBuffer=outputSpline->getBuffer();

  printf("m_splineSize: %d %d %d\n", m_splineSize.x, m_splineSize.y, 1);

  if(splineParam!=NULL){
    float *inputSplineBuffer=splineParam->getBuffer();
    memcpy(outputSplineBuffer, inputSplineBuffer, 3*m_splineSize.x*m_splineSize.y*sizeof(float));
  }

  float SSDvalue=0.0;
  float SSDvalueMin=0.0;
  float SSDvalueTemp=0.0;
   
  ntk2DElasticTransformation *transform=new ntk2DElasticTransformation();
  transform->loadData(target);
  ntk2DData* result=transform->applyTransformation(outputSpline, m_splineSizeLevel);
  
  if(m_PEweight==0.0){
    SSDvalue=calculateSSD(m_reference, result);//outputSpline
  }else{
    SSDvalue=calculateSSDPE(m_reference, result, outputSpline);//outputSpline
  }
  printf("Initial SSDvalue=%lf\n", SSDvalue);
  
  delete result;
  delete transform; 
  
  float stepOrg=step;
  
  do{
    SSDvalueMin=SSDvalue;
    SSDvalueTemp=SSDvalue;

    if(m_PEweight==0.0){
      maxGradient=calculateSSDGradient(outputSpline);
    }
    else{
      maxGradient=calculateSSDPEGradient(outputSpline);
    }

    


    do{
      if(maxGradient==0){
  step=1;
      }else if(maxGradient*stepOrg<m_minGradientLimit){
  step=m_minGradientLimit/maxGradient;
      }else{
  step=stepOrg;
      }

      SSDvalue=SSDvalueTemp;
      for(i=0;i<m_splineSize.x;i++){
  for(j=0;j<m_splineSize.y;j++){
    for(n=0;n<2;n++){
        *(outputSplineBuffer+n*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+i)-=*(m_SSDgradient+n*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+i)*step;
        //printf("%lf %lf\n", *(outputSplineBuffer+n*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+i),*(outputSplineBuffer+n*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+(m_splineSize.x-i-1)));
    }
  }
      }
            
      transform=new ntk2DElasticTransformation();
      transform->loadData(target);
      
      result=transform->applyTransformation(outputSpline, m_splineSizeLevel);
      
      float *splineBuffer=outputSpline->getBuffer();
      
      
      if(m_PEweight==0.0){
  SSDvalueTemp=calculateSSD(m_reference, result);//outputSpline
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
      for(n=0;n<2;n++){
    *(outputSplineBuffer+n*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+i)+=*(m_SSDgradient+n*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+i)*step;
      }
    }
  }
    
  m_PEweight=0.0;
  delete targetPlus;
  return outputSpline;
}

float ntk2DElasticRegistration::calculateSSDGradient(ntkDeformationSpline *splineParam){
  int i,j,k,n; 

  float *splineParamBuffer=splineParam->getBuffer();
  if(m_SSDgradient!=NULL)delete m_SSDgradient;
  m_SSDgradient=(float*)malloc(m_splineSize.x*m_splineSize.y*2*sizeof(float));

  for(i=0;i<m_splineSize.x;i++){
    for(j=0;j<m_splineSize.y;j++){
      for(n=0;n<2;n++){
  *(m_SSDgradient+n*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+i)=0.0;
      }
    }
  }
  
  int d,e,f;
  float functemp;
  float tempValue;
  float tempPow=1.0/(float)m_knotDistance;
  float relPosX, relPosY;
  float *newX=(float*)malloc(m_targetSize.x*m_targetSize.y*sizeof(float));
  float *newY=(float*)malloc(m_targetSize.x*m_targetSize.y*sizeof(float));
  float *diff=(float*)malloc(m_targetSize.x*m_targetSize.y*sizeof(float));

    /*calculating difference between each pixel of reference and transformed target*/
  for(i=0;i<m_targetSize.x;i++){
    for(j=0;j<m_targetSize.y;j++){
        *(newX+j*m_targetSize.x+i)=i; 
  *(newY+j*m_targetSize.x+i)=j; 
  relPosX=(float)(i+1)*tempPow;
  relPosY=(float)(j+1)*tempPow;
    
  for(d=(int)relPosX-1;d<(int)relPosX+3;d++){
    if(d<0||d>m_splineSize.x-1)continue;
    for(e=(int)relPosY-1;e<(int)relPosY+3;e++){
      if(e<0||e>m_splineSize.y-1)continue;
      functemp=m_function->getValue(relPosX-(float)d)*m_function->getValue(relPosY-(float)e);
        *(newX+j*m_targetSize.x+i)+=*(splineParamBuffer+e*m_splineSize.x+d)*functemp;
        *(newY+j*m_targetSize.x+i)+=*(splineParamBuffer+e*m_splineSize.x+d)*functemp;
      
  
    }
  }
  tempValue=m_transform->getInterpolationValue(m_targetPlusSpline, *(newX+j*m_targetSize.x+i)+1.0, *(newY+j*m_targetSize.x+i)+1.0);
  
  if(tempValue<0)tempValue=0;
  if(tempValue>255)tempValue=255;
  
  *(diff+j*m_targetSize.x+i)=tempValue-*(m_referenceBuffer+j*m_referenceSize.x+i);
  //if(*(diff+j*m_targetSize.x+i)==1 || *(diff+j*m_targetSize.x+i)==-1)*(diff+j*m_targetSize.x+i)=0;
  
  

    }
  }
  
  pthread_t tid[m_MTlevel];

  ntkCalculateSSDGradientThreadParam *threadParam = new ntkCalculateSSDGradientThreadParam[m_MTlevel];
 
  for(int i=0;i<m_MTlevel;i++){
    threadParam[i].here=this;
    threadParam[i].threadNumber=i;
    threadParam[i].newX=newX;
    threadParam[i].newY=newY;
    threadParam[i].diff=diff;
    if(pthread_create(&tid[i], NULL, startCalculateSSDGradientThread, &(threadParam[i]))){
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
      for(n=0;n<2;n++){
  *(m_SSDgradient+n*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+i)/=m_targetSize.x*m_targetSize.y;
  
      }
    }
  }
  
  for(i=0;i<m_splineSize.x;i++){
    for(j=0;j<m_splineSize.y;j++){
      for(n=0;n<2;n++){
  if(fabs(*(m_SSDgradient+n*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+i))>max){
    max=fabs(*(m_SSDgradient+n*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+i));
  }
      }
    }
  }

  printf("Max value=%lf\n", max);
  
  delete threadParam;
  free(diff);
  free(newX);
  free(newY);
return max;
}

float ntk2DElasticRegistration::calculateSSDPEGradient(ntkDeformationSpline *splineParam){
  int i,j,n; 

  float *splineParamBuffer=splineParam->getBuffer();
  if(m_SSDgradient!=NULL)delete m_SSDgradient;
  m_SSDgradient=(float*)malloc(m_splineSize.x*m_splineSize.y*3*sizeof(float));

  for(i=0;i<m_splineSize.x;i++){
    for(j=0;j<m_splineSize.y;j++){
      for(n=0;n<2;n++){
  *(m_SSDgradient+n*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+i)=0.0;
      }
    }
  }
  
  int d,e,f;
  float functemp;
  float tempValue;
  float tempPow=1.0/(float)m_knotDistance;
  float relPosX, relPosY;
  float *newX=(float*)malloc(m_targetSize.x*m_targetSize.y*sizeof(float));
  float *newY=(float*)malloc(m_targetSize.x*m_targetSize.y*sizeof(float));
  float *diff=(float*)malloc(m_targetSize.x*m_targetSize.y*sizeof(float));

    /*calculating difference between each pixel of reference and transformed target*/
  for(i=0;i<m_targetSize.x;i++){
    for(j=0;j<m_targetSize.y;j++){
      *(newX+j*m_targetSize.x+i)=i; 
      *(newY+j*m_targetSize.x+i)=j; 
  
  relPosX=(float)(i+1)*tempPow;
  relPosY=(float)(j+1)*tempPow;
  
  for(d=(int)relPosX-1;d<(int)relPosX+3;d++){
    if(d<0||d>m_splineSize.x-1)continue;
    for(e=(int)relPosY-1;e<(int)relPosY+3;e++){
      if(e<0||e>m_splineSize.y-1)continue;
        
        functemp=m_function->getValue(relPosX-(float)d)*m_function->getValue(relPosY-(float)e);
        *(newX+j*m_targetSize.x+i)+=*(splineParamBuffer+e*m_splineSize.x+d)*functemp;
        *(newY+j*m_targetSize.x+i)+=*(splineParamBuffer+e*m_splineSize.x+d)*functemp;
  
    }
  }
  tempValue=m_transform->getInterpolationValue(m_targetPlusSpline, *(newX+j*m_targetSize.x+i)+1.0, *(newY+j*m_targetSize.x+i)+1.0);
  
  if(tempValue<0)tempValue=0;
  if(tempValue>255)tempValue=255;

  *(diff+j*m_targetSize.x+i)=tempValue-*(m_referenceBuffer+j*m_referenceSize.x+i);
  if(*(diff+j*m_targetSize.x+i)==1 || *(diff+j*m_targetSize.x+i)==-1)*(diff+j*m_targetSize.x+i)=0;
    
    }
  }
  
  pthread_t tid[m_MTlevel];

  ntkCalculateSSDGradientThreadParam *threadParam = new ntkCalculateSSDGradientThreadParam[m_MTlevel];
 
  for(int i=0;i<m_MTlevel;i++){
    threadParam[i].here=this;
    threadParam[i].threadNumber=i;
    threadParam[i].newX=newX;
    threadParam[i].newY=newY;
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
      for(n=0;n<2;n++){
  *(m_SSDgradient+n*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+i)/=m_targetSize.x*m_targetSize.y*1;
      }
    }
  }
  
  for(i=0;i<m_splineSize.x;i++){
    for(j=0;j<m_splineSize.y;j++){
      for(n=0;n<2;n++){
  if(fabs(*(m_SSDgradient+n*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+i))>max){
    max=fabs(*(m_SSDgradient+n*m_splineSize.x*m_splineSize.y+j*m_splineSize.x+i));
  }
      }
    }
  }

  printf("Max value=%lf\n", max);
  
  delete threadParam;
  free(diff);
  free(newX);
  free(newY);
  return max;
}

void ntk2DElasticRegistration::threadCalculateSSDGradient(float *newX, float *newY, float *diff,  int threadNumber){
  int i,j,k;
  int a,b,c;
  int d,e,f;
  float val1, val2, val3;
  float tempPow=1.0/(float)m_knotDistance;
  float relPosX, relPosY;
  
  int startx;
  int endx;
  
  startx=threadNumber*m_splineSize.x/m_MTlevel;
  endx=(threadNumber+1)*m_splineSize.x/m_MTlevel;
  
  for(a=startx;a<endx;a++){
    for(b=0;b<m_splineSize.y;b++){
      *(m_SSDgradient+b*m_splineSize.x+a)=0;
      for(i=((a-2)*m_knotDistance-1);i<((a+2)*m_knotDistance);i++){
  if(i<0 || i>m_targetSize.x-1) continue;
  for(j=((b-2)*m_knotDistance-1);j<((b+2)*m_knotDistance);j++){
    if(j<0 || j>m_targetSize.y-1) continue;
    
    /*calculation of SSD gradient against x axis deformation parameter*/
    /*val1 calculation*/
    val1=*(diff+j*m_targetSize.x+i);
    if(fabs(val1)>0){
      
      
      
      /*val2 calculation*/
      val2=0;    
      relPosX=(float)*(newX+j*m_targetSize.x+i)+1.0;
      relPosY=(float)*(newY+j*m_targetSize.x+i)+1.0;
      
      for(d=(int)relPosX-1;d<(int)relPosX+3;d++){
        if(d<0||d>=m_targetSize.x+2)continue;
        for(e=(int)relPosY-1;e<(int)relPosY+3;e++){
    if(e<0||e>=m_targetSize.y+2)continue;
    
    val2+=m_targetPlusSpline->getValue(d,e)*m_function->getDifferentialValue(relPosX-(float)d)*m_function->getValue(relPosY-(float)e);
        }
      }
            
      if(fabs(val2)>0){
        
        /*val3 calculation*/
        
    *(m_SSDgradient+b*m_splineSize.x+a)+=val1*val2*m_function->getValue((float)i*tempPow+tempPow-(float)a)*m_function->getValue((float)j*tempPow+tempPow-(float)b)/m_knotDistance;
    
    //if((b==0 || b ==4) && a==1 )printf("inside %d, %d, %d, %d %lf\n",a, b, i, j, val1*val2*m_function->getValue((float)i*tempPow+tempPow-(float)a)*m_function->getValue((float)j*tempPow+tempPow-(float)b)/m_knotDistance);
      }
    }
    
    
    
    /*calculation of SSD gradient against y axis deformation parameter*/
    /*val1 calculation*/
    val1=*(diff+j*m_targetSize.x+i);
    if(fabs(val1)>0){
      /*val2 calculation*/
      val2=0;
      relPosX=(float)*(newX+j*m_targetSize.x+i)+1.0;
      relPosY=(float)*(newY+j*m_targetSize.x+i)+1.0;
      
      for(d=(int)relPosX-1;d<(int)relPosX+3;d++){
        if(d<0||d>=m_targetSize.x+2)continue;
      for(e=(int)relPosY-1;e<(int)relPosY+3;e++){
        if(e<0||e>=m_targetSize.y+2)continue;
        
        val2+=m_targetPlusSpline->getValue(d,e)*m_function->getValue(relPosX-(float)d)*m_function->getDifferentialValue(relPosY-(float)e);
        
      }
      }
      //val2/=m_knotDistance;
      if(fabs(val2)>0){
      
        /*val3 calculation*/
        
        *(m_SSDgradient+m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)+=val1*val2*m_function->getValue((float)i*tempPow+tempPow-(float)a)*m_function->getValue((float)j*tempPow+tempPow-(float)b)/m_knotDistance;
      }
    }
    
  }
      }
    }
  }
  
  for(a=startx;a<endx;a++){
    for(b=0;b<m_splineSize.y;b++){
      printf("%d %d %lf %lf \n",a, b, *(m_SSDgradient+b*m_splineSize.x+a), *(m_SSDgradient+m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a));
    }
  }
  
}


void ntk2DElasticRegistration::threadCalculateSSDPEGradient(float *newX, float *newY, float *diff,  int threadNumber){
  int i,j,k;
  int a,b,c;
  int d,e,f;
  float val1, val2, val3;
  float tempPow=1.0/(float)m_knotDistance;
  float relPosX, relPosY;
  
  float *potEnergy=(float*)malloc(m_splineSize.x*m_splineSize.y*3*sizeof(float));
  
  int startx;
  int endx;
  
  startx=threadNumber*m_splineSize.x/m_MTlevel;
  endx=(threadNumber+1)*m_splineSize.x/m_MTlevel;
  
  for(a=startx;a<endx;a++){
    for(b=0;b<m_splineSize.y;b++){
      *(m_SSDgradient+b*m_splineSize.x+a)=0;
      for(i=((a-2)*m_knotDistance-1);i<((a+2)*m_knotDistance);i++){
  if(i<0 || i>m_targetSize.x-1) continue;
  for(j=((b-2)*m_knotDistance-1);j<((b+2)*m_knotDistance);j++){
    if(j<0 || j>m_targetSize.y-1) continue;
    
    /*calculation of SSD gradient against x axis deformation parameter*/
    /*val1 calculation*/
    val1=*(diff+j*m_targetSize.x+i);
    if(fabs(val1)>0.0001){
      
      
      
      /*val2 calculation*/
      val2=0;    
      relPosX=(float)*(newX+j*m_targetSize.x+i)+1.0;
      relPosY=(float)*(newY+j*m_targetSize.x+i)+1.0;
      
      for(d=(int)relPosX-1;d<(int)relPosX+3;d++){
        if(d<0||d>=m_targetSize.x+2)continue;
        for(e=(int)relPosY-1;e<(int)relPosY+3;e++){
    if(e<0||e>=m_targetSize.y+2)continue;
    val2+=m_targetPlusSpline->getValue(d,e)*m_function->getDifferentialValue(relPosX-(float)d)*m_function->getValue(relPosY-(float)e);
        }   
      }
    
      //val2/=m_knotDistance;
      if(fabs(val2)>0.0001){
        
        /*val3 calculation*/
        
        *(m_SSDgradient+b*m_splineSize.x+a)+=val1*val2*m_function->getValue((float)i*tempPow+tempPow-(float)a)*m_function->getValue((float)j*tempPow+tempPow-(float)b)/m_knotDistance;
        
      }
    }
    
    
    /*calculation of SSD gradient against y axis deformation parameter*/
    /*val1 calculation*/
    val1=*(diff+j*m_targetSize.x+i);
    if(fabs(val1)>0.0001){
      /*val2 calculation*/
      val2=0;
      relPosX=(float)*(newX+j*m_targetSize.x+i)+1.0;
      relPosY=(float)*(newY+j*m_targetSize.x+i)+1.0;
      
      for(d=(int)relPosX-1;d<(int)relPosX+3;d++){
        if(d<0||d>=m_targetSize.x+2)continue;
        for(e=(int)relPosY-1;e<(int)relPosY+3;e++){
    if(e<0||e>=m_targetSize.y+2)continue;
    
    val2+=m_targetPlusSpline->getValue(d,e)*m_function->getValue(relPosX-(float)d)*m_function->getDifferentialValue(relPosY-(float)e);
    
        }
      }
      //val2/=m_knotDistance;
      if(fabs(val2)>0.0001){
        
        /*val3 calculation*/
        
        
        *(m_SSDgradient+m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)+=val1*val2*m_function->getValue((float)i*tempPow+tempPow-(float)a)*m_function->getValue((float)j*tempPow+tempPow-(float)b)/m_knotDistance;
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
  float relPos2X;
  float relPos2Y;
  
  float size=m_targetSize.x*m_targetSize.y*1;
  float *emapBuffer=m_emap->getBuffer();
  float elasticity;
  float elasticity2;
  
  for(a=startx;a<endx;a++){
    for(b=0;b<m_splineSize.y;b++){
      orgPosX=a-1;orgPosY=b-1;
  *(potEnergy+0*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)=0;
  *(potEnergy+1*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)=0;
  
  //*(m_SSDgradient+b*m_splineSize.x+a)=0;
  for(i=((a-2)*m_knotDistance-1);i<((a+2)*m_knotDistance);i++){
    if(i<0 || i>m_targetSize.x-2) continue;
    for(j=((b-2)*m_knotDistance-1);j<((b+2)*m_knotDistance);j++){
      if(j<0 || j>m_targetSize.y-1) continue;
          
        relPosX=(float)*(newX+j*m_targetSize.x+i);
        relPosY=(float)*(newY+j*m_targetSize.x+i);
  
        relPos2X=(float)*(newX+j*m_targetSize.x+i+1);
        relPos2Y=(float)*(newY+j*m_targetSize.x+i+1);
          
        distance=sqrt((relPos2X-relPosX)*(relPos2X-relPosX)+(relPos2Y-relPosY)*(relPos2Y-relPosY));
        
        elasticity=*(emapBuffer+j*m_targetSize.x+i);
        elasticity2=*(emapBuffer+j*m_targetSize.x+(i+1));
        
        //printf("elasticity:%lf %lf \n", elasticity, elasticity2);

        *(potEnergy+b*m_splineSize.x+a)+=((relPos2X-relPosX)/distance-(relPos2X-relPosX)/pow(distance,3))*(m_function->getValue(i-orgPosX+1)*m_function->getValue(j-orgPosY)-m_function->getValue(i-orgPosX)*m_function->getValue(j-orgPosY))*(elasticity+elasticity2)/2.0;
        
        //*(potEnergy+b*m_splineSize.x+a)+=(2*(distance-1)*(relPos2X-relPosX)/distance)*(m_function->getValue(i-orgPosX+1)*m_function->getValue(j-orgPosY)*m_function->getValue(k-orgPosZ)-m_function->getValue(i-orgPosX)*m_function->getValue(j-orgPosY)*m_function->getValue(k-orgPosZ));
         
       
    }
  }
  
  *(m_SSDgradient+0*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)+=*(potEnergy+0*m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)*size*m_PEweight*pow(8,-m_imageSizeLevel);//*pow(8,m_knotDistance-1);///m_knotDistance/m_knotDistance;
  
  for(i=((a-2)*m_knotDistance-1);i<((a+2)*m_knotDistance);i++){
    if(i<0 || i>m_targetSize.x-1) continue;
    for(j=((b-2)*m_knotDistance-1);j<((b+2)*m_knotDistance);j++){
      if(j<0 || j>m_targetSize.y-2) continue;
        
        relPosX=(float)*(newX+j*m_targetSize.x+i);
        relPosY=(float)*(newY+j*m_targetSize.x+i);

        relPos2X=(float)*(newX+(j+1)*m_targetSize.x+i);
        relPos2Y=(float)*(newY+(j+1)*m_targetSize.x+i);
                
        distance=sqrt((relPos2X-relPosX)*(relPos2X-relPosX)+(relPos2Y-relPosY)*(relPos2Y-relPosY));
        
        elasticity=*(emapBuffer+j*m_targetSize.x+i);
        elasticity2=(*(emapBuffer+(j+1)*m_targetSize.x+i));
        
        //printf("elasticity:%lf %lf \n", elasticity, elasticity2);

        *(potEnergy+m_splineSize.x*m_splineSize.y+b*m_splineSize.x+a)+=((relPos2Y-relPosY)/distance-(relPos2Y-relPosY)/pow(distance,3))*(m_function->getValue(i-orgPosX)*m_function->getValue(j-orgPosY+1)-m_function->getValue(i-orgPosX)*m_function->getValue(j-orgPosY))*(elasticity+elasticity2)/2.0;
        
        //*(potEnergy+b*m_splineSize.x+a)+=(2*(distance-1)*(relPos2Y-relPosY)/distance)*(m_function->getValue(i-orgPosX)*m_function->getValue(j-orgPosY+1)*m_function->getValue(k-orgPosZ)-m_function->getValue(i-orgPosX)*m_function->getValue(j-orgPosY)*m_function->getValue(k-orgPosZ));

        
    }
  }
    
    }
  }

  free(potEnergy);
  }
}

void* ntk2DElasticRegistration::startCalculateSSDGradientThread(void *arg){
  ntkCalculateSSDGradientThreadParam *temp=(ntkCalculateSSDGradientThreadParam*)arg;
  (temp->here)->threadCalculateSSDGradient(temp->newX, temp->newY, temp->diff, temp->threadNumber);
  return 0;
}

 void* ntk2DElasticRegistration::startCalculateSSDPEGradientThread(void *arg){
  ntkCalculateSSDGradientThreadParam *temp=(ntkCalculateSSDGradientThreadParam*)arg;
  (temp->here)->threadCalculateSSDPEGradient(temp->newX, temp->newY, temp->diff, temp->threadNumber);
  return 0;
}

 float ntk2DElasticRegistration::calculateSSDGradient(ntk2DData* target, int splineSizeLevel=0, ntkDeformationSpline *splineParam=NULL){
  int j,k;
  m_splineSizeLevel=splineSizeLevel;
  m_target=target;
  m_targetSize=target->getDataSize();
  if(m_targetSize.x!=m_referenceSize.x || m_targetSize.y!=m_referenceSize.y){
    printf("Size of target and reference do not match.\n");
    return 0;
  }

  /*targetPlus is 2D data of target with one additional voxel in each direction*/
  ntk2DData *targetPlus=new ntk2DData(ntkIntDimension(m_targetSize.x+2, m_targetSize.y+2));
  unsigned char *targetBuffer=(unsigned char*)target->getBuffer();
  unsigned char *targetPlusBuffer=(unsigned char*)targetPlus->getBuffer();
  
  for(j=0;j<m_targetSize.y;j++){
    memcpy(targetPlusBuffer+(j+1)*(m_targetSize.x+2)+1, targetBuffer+j*m_targetSize.x, m_targetSize.x);
  }
  
  
  m_targetPlus=targetPlus;

  /*m_targetPlusSpline is B-Spline representation of targetPlus*/
  m_targetPlusSpline=m_transform->doForwardTransform(m_targetPlus, CURRENT_RES);
  float tempPow=pow(2, splineSizeLevel);
  float knotDistance=pow(2, -splineSizeLevel);
  m_knotDistance=(int)knotDistance;
  m_splineSize= ntkIntDimension((int)(tempPow*(m_targetSize.x+1)+1), (int)(tempPow*(m_targetSize.y+1)+1),1);
  
  if(splineParam==NULL){
    splineParam=new ntkDeformationSpline(ntkIntDimension(m_splineSize.x,m_splineSize.y,1));
  }
  m_splineParam=splineParam;
  
  float max=calculateSSDGradient(splineParam);
  return max;
}

 float* ntk2DElasticRegistration::getGradientBuffer(){
  return m_SSDgradient;
}

ntkIntDimension ntk2DElasticRegistration::getSplineSize(){
  return m_splineSize;
}
 
float ntk2DElasticRegistration::calculateSSD(ntk2DData* reference, ntk2DData* target){
  int i,j,k;
  ntkIntDimension refDim=reference->getDataSize();
  ntkIntDimension tarDim=target->getDataSize();
  
  unsigned char *refBuffer=(unsigned char*)reference->getBuffer();
  unsigned char *tarBuffer=(unsigned char*)target->getBuffer();

  if(refDim.x!=tarDim.x||refDim.y!=tarDim.y){
    printf("Reference and Target size do not match\n");
    return 0;
  }
  
  float value=0;
  for(i=0;i<refDim.x;i++){
    for(j=0;j<refDim.y;j++){
      value+=((int)*(refBuffer+j*refDim.x+i)-(int)*(tarBuffer+j*refDim.x+i))*((int)*(refBuffer+j*refDim.x+i)-(int)*(tarBuffer+j*refDim.x+i));
    }
  }
  return value/refDim.x/refDim.y;
}

 float ntk2DElasticRegistration::calculateSSDPE(ntk2DData* reference, ntk2DData* target, ntkDeformationSpline *splineParam){
  int i,j,k;
  float potEnergy=0.0;
  ntkIntDimension refDim=reference->getDataSize();
  ntkIntDimension tarDim=target->getDataSize();
  
  unsigned char *refBuffer=(unsigned char*)reference->getBuffer();
  unsigned char *tarBuffer=(unsigned char*)target->getBuffer();

  if(refDim.x!=tarDim.x||refDim.y!=tarDim.y){
    printf("Reference and Target size do not match\n");
    return 0;
  }
  
  float value=0;
  for(i=0;i<refDim.x;i++){
    for(j=0;j<refDim.y;j++){
      value+=((int)*(refBuffer+j*refDim.x+i)-(int)*(tarBuffer+j*refDim.x+i))*((int)*(refBuffer+j*refDim.x+i)-(int)*(tarBuffer+j*refDim.x+i));
    }
  }
  printf("SSD:%lf \n",value/refDim.x/refDim.y);

  /*calculate energy potential*/
  
  int n; 
  
  float *splineParamBuffer=splineParam->getBuffer();
  int a,b,c;
  int d,e,f;
  float functemp;
  float tempPow=1.0/(float)m_knotDistance;
  float relPosX, relPosY;
  float *newX=(float*)malloc(m_targetSize.x*m_targetSize.y*sizeof(float));
  float *newY=(float*)malloc(m_targetSize.x*m_targetSize.y*sizeof(float));
  float *diff=(float*)malloc(m_targetSize.x*m_targetSize.y*sizeof(float));

    /*calculating difference between each pixel of reference and transformed target*/
  for(i=0;i<m_targetSize.x;i++){
    for(j=0;j<m_targetSize.y;j++){
       *(newX+j*m_targetSize.x+i)=i; 
  *(newY+j*m_targetSize.x+i)=j; 
  relPosX=(float)(i+1)*tempPow;
  relPosY=(float)(j+1)*tempPow;
    
  for(d=(int)relPosX-1;d<(int)relPosX+3;d++){
    if(d<0||d>m_splineSize.x-1)continue;
    for(e=(int)relPosY-1;e<(int)relPosY+3;e++){
      if(e<0||e>m_splineSize.y-1)continue;
          
        functemp=m_function->getValue(relPosX-(float)d)*m_function->getValue(relPosY-(float)e);
        *(newX+j*m_targetSize.x+i)+=*(splineParamBuffer+e*m_splineSize.x+d)*functemp;
        *(newY+j*m_targetSize.x+i)+=*(splineParamBuffer+e*m_splineSize.x+d)*functemp;
    }
  }
  
      }
    }


  /*calculate SSDPE based on spring potential energy*/

  float distance=1.0;
  float relPos2X;
  float relPos2Y;
  float elasticity;
  float elasticity2;
  float* emapBuffer=m_emap->getBuffer();

  for(i=0;i<m_targetSize.x-1;i++){
    for(j=0;j<m_targetSize.y;j++){
      relPosX=(float)*(newX+j*m_targetSize.x+i);
      relPosY=(float)*(newY+j*m_targetSize.x+i);
      
      relPos2X=(float)*(newX+j*m_targetSize.x+i+1);
      relPos2Y=(float)*(newY+j*m_targetSize.x+i+1);
      
      distance=sqrt((relPos2X-relPosX)*(relPos2X-relPosX)+(relPos2Y-relPosY)*(relPos2Y-relPosY));
      
      elasticity=*(emapBuffer+j*m_targetSize.x+i);
      elasticity2=(*(emapBuffer+j*m_targetSize.x+i+1));
      
            
      potEnergy+=((distance-1)*(distance-1))/distance*(elasticity+elasticity2)/2.0;
      //potEnergy+=((distance-1)*(distance-1));
    }
  }
  
  for(i=0;i<m_targetSize.x;i++){
    for(j=0;j<m_targetSize.y-1;j++){
      relPosX=(float)*(newX+j*m_targetSize.x+i);
      relPosY=(float)*(newY+j*m_targetSize.x+i);
      
      relPos2X=(float)*(newX+(j+1)*m_targetSize.x+i);
      relPos2Y=(float)*(newY+(j+1)*m_targetSize.x+i);
      
      distance=sqrt((relPos2X-relPosX)*(relPos2X-relPosX)+(relPos2Y-relPosY)*(relPos2Y-relPosY));
      
      elasticity=(*(emapBuffer+j*m_targetSize.x+i));
      elasticity2=(*(emapBuffer+(j+1)*m_targetSize.x+i));
      
      
      
      potEnergy+=((distance-1)*(distance-1))/distance*(elasticity+elasticity2)/2.0;;
      
      //potEnergy+=((distance-1)*(distance-1));
    }
  }
  
  printf("potential energy: %lf\n", potEnergy*m_PEweight*pow(8,-m_imageSizeLevel));
  
  delete newX;
  delete newY;
  delete diff;
  
  return value/refDim.x/refDim.y+potEnergy*m_PEweight*pow(8,-m_imageSizeLevel);
}
 
 ntkDeformationSpline *ntk2DElasticRegistration::doSplineParamTransform(ntkDeformationSpline* inputSpline){
   printf("11");fflush(stdout);
   float *inputSplineBuffer=inputSpline->getBuffer();
  ntkIntDimension inputSplineSize=inputSpline->getSize();
  ntkMatrix *inputMatrixX=new ntkMatrix(inputSplineBuffer, inputSplineSize);
  ntkMatrix *inputMatrixY=new ntkMatrix(inputSplineBuffer+inputSplineSize.x*inputSplineSize.y, inputSplineSize);
  ntk2DBSpline3Transform *transform=new ntk2DBSpline3Transform();
  printf("12");fflush(stdout);
  ntkMatrix *outputMatrixX=transform->doSplineToSplineHigherResTransform(inputMatrixX);
  ntkMatrix *outputMatrixY=transform->doSplineToSplineHigherResTransform(inputMatrixY);
  
  ntkIntDimension outputSize=outputMatrixX->getMatrixSize();
  printf("13");fflush(stdout);
  ntkDeformationSpline *outputSpline=new ntkDeformationSpline(ntkIntDimension(outputSize.x, outputSize.y, 1));
  float *outputSplineBuffer=outputSpline->getBuffer();

  float *outputMatrixXBuffer=outputMatrixX->getMatrixPointer();
  float *outputMatrixYBuffer=outputMatrixY->getMatrixPointer();
  printf("14");fflush(stdout);
  memcpy(outputSplineBuffer, outputMatrixXBuffer, outputSize.x*outputSize.y*sizeof(float));
  memcpy(outputSplineBuffer+outputSize.x*outputSize.y, outputMatrixYBuffer, outputSize.x*outputSize.y*sizeof(float));
  
  
  delete transform;
  delete inputMatrixX;
  delete inputMatrixY;
  delete outputMatrixX;
  delete outputMatrixY;
    
  return outputSpline;
}

 void ntk2DElasticRegistration::addRegistrationScheme(int imageResLevel, int splineResLevel, float step=1.0, float PEweight=0.0){
  //ntkElasticRegistrationScheme *newScheme=new ntkElasticRegistrationScheme();
  ntkElasticRegistrationScheme *newScheme=(ntkElasticRegistrationScheme*)malloc(sizeof(ntkElasticRegistrationScheme));
  newScheme->imageResLevel=imageResLevel;
  newScheme->splineResLevel=splineResLevel;
  newScheme->step=step;
  newScheme->PEweight=PEweight;
  newScheme->next=NULL;
  
  ntkElasticRegistrationScheme *tempScheme;

  if(m_firstScheme==NULL && m_lastScheme==NULL){  
    m_firstScheme=newScheme;
    m_lastScheme=newScheme;
  }else{
    tempScheme=m_lastScheme;
    m_lastScheme=newScheme;
    tempScheme->next=m_lastScheme;
  }
}

 void ntk2DElasticRegistration::resetRegistrationScheme(){
  ntkElasticRegistrationScheme *tempScheme;

  while(m_firstScheme!=NULL){
    tempScheme=m_firstScheme;
    m_firstScheme=tempScheme->next;
    //delete tempScheme;
    free(tempScheme);
  }
  
  m_firstScheme=NULL;
  m_lastScheme=NULL;
}

 void ntk2DElasticRegistration::printRegistrationScheme(){
  ntkElasticRegistrationScheme *tempScheme;

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

bool ntk2DElasticRegistration::checkRegistrationScheme(){
  if(m_firstScheme==NULL){
    return false;
  }else if(m_firstScheme->next==NULL){
    return true;
  }else{
    ntkElasticRegistrationScheme *tempScheme=m_firstScheme;
    do{
      if(tempScheme->imageResLevel+tempScheme->splineResLevel>tempScheme->next->imageResLevel+tempScheme->next->splineResLevel){
  return false;
      }
      tempScheme=tempScheme->next;
    }while(tempScheme->next!=NULL);
  }
  return true;
}

 ntkDeformationSpline* ntk2DElasticRegistration::executeRegistrationSchemes(ntk2DData* reference, ntk2DData *target, ntkDeformationSpline* inputSpline=NULL, ntkElasticityMap* emap=NULL){
  int i;
  
  if(!checkRegistrationScheme()){
    printf("Registration scheme check failed. Registration is terminated\n");
    return NULL;
  }
  
  ntkIntDimension targetSize=target->getDataSize();
  ntkIntDimension referenceSize=reference->getDataSize();

  if(targetSize.x!=referenceSize.x || targetSize.y!=referenceSize.y){
    printf("Size of target and reference do not match.\n");
    return 0;
  }

  ntkDeformationSpline *outputSpline;
  ntkDeformationSpline *tempSpline;
  
  ntkElasticRegistrationScheme *tempScheme=m_firstScheme;
  ntk2DData* tempReference;
  ntk2DData* tempTarget;
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
  tempTarget->writeFile("target2.raw");
  outputSpline=doRegistration(tempReference, tempTarget, tempMap, tempScheme->splineResLevel, inputSpline, tempScheme->step, tempScheme->PEweight);
  
  if(needRelease){
    delete tempReference;
    delete tempTarget;
    delete tempMap;
  }
  
  tempScheme=tempScheme->next;

  while(tempScheme!=NULL){
    //printf("1");fflush(stdout);
    prevImageResLevel=imageResLevel;
    prevSplineResLevel=splineResLevel;
    imageResLevel=tempScheme->imageResLevel;
    splineResLevel=tempScheme->splineResLevel;

    splineTransformNum=(imageResLevel+splineResLevel)-(prevImageResLevel+prevSplineResLevel);
    //printf("2");fflush(stdout);
    for(i=0;i<splineTransformNum;i++){
      tempSpline=outputSpline;
      outputSpline=doSplineParamTransform(tempSpline);
      delete tempSpline;
    }
    //printf("3");fflush(stdout);
    int imageResLevelUp=imageResLevel-prevImageResLevel;
    
    if(imageResLevelUp!=0){
      float *splineBuffer=outputSpline->getBuffer();
      ntkIntDimension splineSize=outputSpline->getSize();
      
      for(i=0;i<splineSize.x*splineSize.y*3;i++){
  *(splineBuffer+i) *= pow(2, imageResLevelUp);
      }
    }
    //printf("4");fflush(stdout);
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
    //printf("5");fflush(stdout);
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
 
 ntk2DData *ntk2DElasticRegistration::resizeDataSmaller(ntk2DData* input, int imageResLevel){
  int i,j,k;
  ntkIntDimension inputSize=input->getDataSize();
  ntkFloatDimension inputThickness=input->getDataThickness();
  ntkIntDimension outputSize=inputSize;
  ntkFloatDimension outputThickness=inputThickness;

  for(i=0;i<-imageResLevel;i++){
    outputSize.x=(outputSize.x+1)/2-1;
    outputSize.y=(outputSize.y+1)/2-1;
  }
  ntk2DData *output=new ntk2DData(outputSize, outputThickness);
  unsigned char* inputBuffer=(unsigned char*)input->getBuffer();
  unsigned char* outputBuffer=(unsigned char*)output->getBuffer();
  
  int factor=(int)pow(2, -imageResLevel);
  
  for(i=0;i<outputSize.x;i++){
    for(j=0;j<outputSize.y;j++){
      *(outputBuffer+j*outputSize.x+i)=*(inputBuffer+(((j+1)*factor)-1)*inputSize.x+(((i+1)*factor)-1));
    }
  }  
  
  return output;
}

 ntkElasticityMap *ntk2DElasticRegistration::resizeMapSmaller(ntkElasticityMap* input, int imageResLevel){
  int i,j,k;
  ntkIntDimension inputSize=input->getMapSize();
  ntkIntDimension outputSize=inputSize;
  
  for(i=0;i<-imageResLevel;i++){
    outputSize.x=(outputSize.x+1)/2-1;
    outputSize.y=(outputSize.y+1)/2-1;
  }
  ntkElasticityMap *output=new ntkElasticityMap(outputSize);
  float* inputBuffer=input->getBuffer();
  float* outputBuffer=output->getBuffer();
  
  int factor=(int)pow(2, -imageResLevel);
  
  for(i=0;i<outputSize.x;i++){
    for(j=0;j<outputSize.y;j++){
      *(outputBuffer+j*outputSize.x+i)=*(inputBuffer+(((j+1)*factor)-1)*inputSize.x+(((i+1)*factor)-1));
    }
  }  
  
  return output;
 }
 
 void ntk2DElasticRegistration::setMinGradientLimit(float value){
  m_minGradientLimit=value;
}
 
