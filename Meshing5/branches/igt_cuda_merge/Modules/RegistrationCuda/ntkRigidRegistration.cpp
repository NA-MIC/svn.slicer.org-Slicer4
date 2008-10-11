#include "ntkRigidRegistration.h"

ntkRigidRegistration::ntkRigidRegistration(){
  m_xpos=0;
  m_ypos=0;
  m_zpos=0;
  m_rxpos=0;
  m_rypos=0;
  m_rzpos=0;
  m_scale=0;
  m_value=0;
  m_trans=new ntkLinearTransformation();
  m_transMatrix=m_trans->getMatrix();
  m_measure=new ntkMISimilarityMeasure();
}

ntkRigidRegistration::~ntkRigidRegistration(){

}

void ntkRigidRegistration::loadData(ntk3DData *reference){
  m_reference=reference;
  m_measure->loadData(m_reference);
}

void ntkRigidRegistration::setInitialPosition(double x, double y, double z, double rx, double ry, double rz){
  m_xpos=x;
  m_ypos=y;
  m_zpos=z;
  m_rxpos=rx;
  m_rypos=ry;
  m_rzpos=rz;
}

void ntkRigidRegistration::doRegistration(ntk3DData* target){
  m_target=target;
  double temp_value=-10000.0;
  double step=1;
  m_trans->loadData(m_target);

  m_value=m_measure->doSimilarityMeasure(m_target);

  temp_value=optimizeOneDirection(&m_rxpos, step);
  printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);
  temp_value=optimizeOneDirection(&m_rypos, step);
  printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);
  temp_value=optimizeOneDirection(&m_rzpos, step);
  printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);
  temp_value=optimizeOneDirection(&m_xpos, step);
  printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);
  temp_value=optimizeOneDirection(&m_ypos, step);
  printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);
  temp_value=optimizeOneDirection(&m_zpos, step);
  printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);
  
  while(temp_value<m_value){
    m_value=temp_value;
 
    temp_value=optimizeOneDirection(&m_rxpos, step);
    printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);
    temp_value=optimizeOneDirection(&m_rypos, step);
    printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);
    temp_value=optimizeOneDirection(&m_rzpos, step);
    printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);
    temp_value=optimizeOneDirection(&m_xpos, step);
    printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);
    temp_value=optimizeOneDirection(&m_ypos, step);
    printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);
    temp_value=optimizeOneDirection(&m_zpos, step);
    printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);
  }
}

void ntkRigidRegistration::doRegistration(ntk3DData* target, double step, int histFactor){
  m_target=target;
  double temp_value=10000.0;
  double before_value=0.0;

  m_measure->setHistogramBinSize(histFactor);
  m_trans->loadData(m_target);

  ntkIntDimension temp_dim=m_target->getDataSize();
  ntkFloatDimension temp_thickness=m_target->getDataThickness();
  
  printf("Start registration. step= %f\n", step);

  m_trans->getMatrix()->Identity();
  m_trans->doTranslation(m_xpos, m_ypos, m_zpos);
  m_trans->doRotationZ(m_rzpos);
  m_trans->doRotationY(m_rypos);
  m_trans->doRotationX(m_rxpos);
  ntk3DData *temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
  m_value=m_measure->doSimilarityMeasure(temp_data);
  before_value=m_value;
  
  temp_value=optimizeOneDirection(&m_xpos, step);
  m_value=temp_value;
  printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);

  temp_value=optimizeOneDirection(&m_ypos, step);
  m_value=temp_value;
  printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);

  temp_value=optimizeOneDirection(&m_zpos, step);
  m_value=temp_value;
  printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);

  temp_value=optimizeOneDirection(&m_rxpos, step);
  m_value=temp_value;
  printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);

  temp_value=optimizeOneDirection(&m_rypos, step);
  m_value=temp_value;
  printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);
 
  temp_value=optimizeOneDirection(&m_rzpos, step);
  m_value=temp_value;
  printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);
 
  
  while(temp_value>before_value){
    before_value=m_value;

    temp_value=optimizeOneDirection(&m_xpos, step);
    m_value=temp_value;
    printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);

    temp_value=optimizeOneDirection(&m_ypos, step);
    m_value=temp_value;
    printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);

    temp_value=optimizeOneDirection(&m_zpos, step);
    m_value=temp_value;
    printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);
    
 
    temp_value=optimizeOneDirection(&m_rxpos, step);
    m_value=temp_value;
    printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);

   

    temp_value=optimizeOneDirection(&m_rypos, step);
    m_value=temp_value;
    printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);
    
   
    temp_value=optimizeOneDirection(&m_rzpos, step);
    m_value=temp_value;
    printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);
  }
  delete temp_data;
}

void ntkRigidRegistration::doUphillRegistration(ntk3DData* target, double step, double moveStep, int histFactor){
  m_target=target;
  double temp_value=10000.0;
  double dx=0,dy=0,dz=0,drx=0,dry=0,drz=0;
  double tempMove=0;;

  m_measure->setHistogramBinSize(histFactor);
  m_trans->loadData(m_target);

  ntkIntDimension temp_dim=m_target->getDataSize();
  ntkFloatDimension temp_thickness=m_target->getDataThickness();
  //ntk3DData* temp_data=new ntk3DData(temp_dim);
  //temp_data->setDataThickness(temp_thickness);

  printf("Start uphill registration. step= %f\n", step);

  m_trans->getMatrix()->Identity();
  m_trans->doTranslation(m_xpos, m_ypos, m_zpos);
  m_trans->doRotationZ(m_rzpos);
  m_trans->doRotationY(m_rypos);
  m_trans->doRotationX(m_rxpos);
  ntk3DData *temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
  // m_trans->applyTransformation(temp_data);
  m_value=m_measure->doSimilarityMeasure(temp_data);
  
  while(temp_value>m_value){
    m_xpos+=dx;m_ypos+=dy;m_zpos+=dz;m_rxpos+=drx;m_rypos+=dry;m_rzpos+=drz;
    dx=0;dy=0;dz=0;drx=0;dry=0;drz=0;
    
    printf("%f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);
    printf("value: %f %f\n", m_value, temp_value);
    
    m_value=temp_value;
    m_trans->getMatrix()->Identity();
    m_trans->doTranslation(m_xpos+step, m_ypos, m_zpos);
    m_trans->doRotationZ(m_rzpos);
    m_trans->doRotationY(m_rypos);
    m_trans->doRotationX(m_rxpos);
    delete temp_data;
    temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
    dx+=m_measure->doSimilarityMeasure(temp_data);
    m_trans->getMatrix()->Identity();
    m_trans->doTranslation(m_xpos-step, m_ypos, m_zpos);
    m_trans->doRotationZ(m_rzpos);
    m_trans->doRotationY(m_rypos);
    m_trans->doRotationX(m_rxpos);
    delete temp_data;
    temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
    dx-=m_measure->doSimilarityMeasure(temp_data);
    dx/=(2*step);
    
    m_trans->getMatrix()->Identity();
    m_trans->doTranslation(m_xpos, m_ypos+step, m_zpos);
    m_trans->doRotationZ(m_rzpos);
    m_trans->doRotationY(m_rypos);
    m_trans->doRotationX(m_rxpos);
    delete temp_data;
    temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
    dy+=m_measure->doSimilarityMeasure(temp_data);
    m_trans->getMatrix()->Identity();
    m_trans->doTranslation(m_xpos, m_ypos-step, m_zpos);
    m_trans->doRotationZ(m_rzpos);
    m_trans->doRotationY(m_rypos);
    m_trans->doRotationX(m_rxpos);
    delete temp_data;
    temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
    dy-=m_measure->doSimilarityMeasure(temp_data);
    dy/=(2*step);

    m_trans->getMatrix()->Identity();
    m_trans->doTranslation(m_xpos, m_ypos, m_zpos+step);
    m_trans->doRotationZ(m_rzpos);
    m_trans->doRotationY(m_rypos);
    m_trans->doRotationX(m_rxpos);
    delete temp_data;
    temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
    dz+=m_measure->doSimilarityMeasure(temp_data);
    m_trans->getMatrix()->Identity();
    m_trans->doTranslation(m_xpos, m_ypos, m_zpos-step);
    m_trans->doRotationZ(m_rzpos);
    m_trans->doRotationY(m_rypos);
    m_trans->doRotationX(m_rxpos);
    delete temp_data;
    temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
    dz-=m_measure->doSimilarityMeasure(temp_data);
    dz/=(2*step);
    
    m_trans->getMatrix()->Identity();
    m_trans->doTranslation(m_xpos, m_ypos, m_zpos);
    m_trans->doRotationZ(m_rzpos+step);
    m_trans->doRotationY(m_rypos);
    m_trans->doRotationX(m_rxpos);
    delete temp_data;
    temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
    drz+=m_measure->doSimilarityMeasure(temp_data);
    m_trans->getMatrix()->Identity();
    m_trans->doTranslation(m_xpos, m_ypos, m_zpos);
    m_trans->doRotationZ(m_rzpos-step);
    m_trans->doRotationY(m_rypos);
    m_trans->doRotationX(m_rxpos);
    delete temp_data;
    temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
    drz-=m_measure->doSimilarityMeasure(temp_data);
    drz/=(2*step);

    m_trans->getMatrix()->Identity();
    m_trans->doTranslation(m_xpos, m_ypos, m_zpos);
    m_trans->doRotationZ(m_rzpos);
    m_trans->doRotationY(m_rypos+step);
    m_trans->doRotationX(m_rxpos);
    delete temp_data;
    temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
    dry+=m_measure->doSimilarityMeasure(temp_data);
    m_trans->getMatrix()->Identity();
    m_trans->doTranslation(m_xpos, m_ypos, m_zpos);
    m_trans->doRotationZ(m_rzpos);
    m_trans->doRotationY(m_rypos-step);
    m_trans->doRotationX(m_rxpos);
    delete temp_data;
    temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
    dry-=m_measure->doSimilarityMeasure(temp_data);
    dry/=(2*step);

    m_trans->getMatrix()->Identity();
    m_trans->doTranslation(m_xpos, m_ypos, m_zpos);
    m_trans->doRotationZ(m_rzpos);
    m_trans->doRotationY(m_rypos);
    m_trans->doRotationX(m_rxpos+step);
    delete temp_data;
    temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
    drx+=m_measure->doSimilarityMeasure(temp_data);
    m_trans->getMatrix()->Identity();
    m_trans->doTranslation(m_xpos, m_ypos, m_zpos);
    m_trans->doRotationZ(m_rzpos);
    m_trans->doRotationY(m_rypos);
    m_trans->doRotationX(m_rxpos-step);
    delete temp_data;
    temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
    drx-=m_measure->doSimilarityMeasure(temp_data);
    drx/=(2*step);

    tempMove=sqrt(dx*dx+dy*dy+dz*dz+drx*drx+dry*dry+drz*drz);
    dx*=moveStep/tempMove;
    dy*=moveStep/tempMove;
    dz*=moveStep/tempMove;
    drx*=moveStep/tempMove;
    dry*=moveStep/tempMove;
    drz*=moveStep/tempMove;
    printf("tempMove=%f(%f %f %f %f %f %f)\n", tempMove, dx, dy, dz, drx, dry, drz);
    

    m_trans->getMatrix()->Identity();
    m_trans->doTranslation(m_xpos+dx, m_ypos+dy, m_zpos+dz);
    m_trans->doRotationZ(m_rzpos+drx);
    m_trans->doRotationY(m_rypos+dry);
    m_trans->doRotationX(m_rxpos+drz);
    delete temp_data;
    temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
    temp_value=m_measure->doSimilarityMeasure(temp_data);
  }
}

void ntkRigidRegistration::doPowellRegistration(ntk3DData* target, double stepsize, int histFactor){
  m_target=target;
  double temp_value=10000.0;
  double before_value=0.0;
  ntkOptimizationStep step[6];
  ntkOptimizationStep stepBig;
  ntkOptimizationStep temp_mvalue;
  step[0]=ntkOptimizationStep(stepsize,0,0,0,0,0,0);
  step[1]=ntkOptimizationStep(0,stepsize,0,0,0,0,0);
  step[2]=ntkOptimizationStep(0,0,stepsize,0,0,0,0);
  step[3]=ntkOptimizationStep(0,0,0,stepsize,0,0,0);
  step[4]=ntkOptimizationStep(0,0,0,0,stepsize,0,0);
  step[5]=ntkOptimizationStep(0,0,0,0,0,stepsize,0);

  m_measure->setHistogramBinSize(histFactor);
  m_trans->loadData(m_target);
  
  ntkIntDimension temp_dim=m_target->getDataSize();
  ntkFloatDimension temp_thickness=m_target->getDataThickness();

  printf("Start registration. step= %f\n", stepsize);

  m_trans->getMatrix()->Identity();
  m_trans->doTranslation(m_xpos, m_ypos, m_zpos);
  m_trans->doRotationZ(m_rzpos);
  m_trans->doRotationY(m_rypos);
  m_trans->doRotationX(m_rxpos);
  ntk3DData *temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
  m_value=m_measure->doSimilarityMeasure(temp_data);
  before_value=m_value;

  temp_mvalue=ntkOptimizationStep(m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);

  temp_value=optimizeOneDirection(step[0]);
  m_value=temp_value;
  printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);

  temp_value=optimizeOneDirection(step[1]);
  m_value=temp_value;
  printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);

  temp_value=optimizeOneDirection(step[2]);
  m_value=temp_value;
  printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);

  temp_value=optimizeOneDirection(step[3]);
  m_value=temp_value;
  printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);

  temp_value=optimizeOneDirection(step[4]);
  m_value=temp_value;
  printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);
 
  temp_value=optimizeOneDirection(step[5]);
  m_value=temp_value;
  printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);
 
  stepBig=ntkOptimizationStep(m_xpos-temp_mvalue.x,m_ypos-temp_mvalue.y,m_zpos-temp_mvalue.z,m_rxpos-temp_mvalue.rx,m_rypos-temp_mvalue.ry,m_rzpos-temp_mvalue.rz, 0.0);

  double stepVal=sqrt(stepBig.x*stepBig.x+stepBig.y*stepBig.y+stepBig.z*stepBig.z+stepBig.rx*stepBig.rx+stepBig.ry*stepBig.ry+stepBig.rz*stepBig.rz);
    
  if(stepVal>0.0){
    stepBig=ntkOptimizationStep(stepBig.x*stepsize/stepVal,stepBig.y*stepsize/stepVal,stepBig.z*stepsize/stepVal,stepBig.rx*stepsize/stepVal,stepBig.ry*stepsize/stepVal,stepBig.rz*stepsize/stepVal, 0.0);
  }

  temp_value=optimizeOneDirection(stepBig);
  m_value=temp_value;
  printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);

  while(temp_value>before_value){
    step[0]=step[1];
    step[1]=step[2];
    step[2]=step[3];
    step[3]=step[4];
    step[4]=step[5];
    step[5]=stepBig;
    
    printf("step5 %f %f %f %f %f %f %f\n", step[5].x, step[5].y, step[5].z, step[5].rx, step[5].ry, step[5].rz, step[5].scale);
    
    before_value=m_value;

    temp_mvalue=ntkOptimizationStep(m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);

    temp_value=optimizeOneDirection(step[0]);
    m_value=temp_value;
    printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);

    temp_value=optimizeOneDirection(step[1]);
    m_value=temp_value;
    printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);

    temp_value=optimizeOneDirection(step[2]);
    m_value=temp_value;
    printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);
    
    temp_value=optimizeOneDirection(step[3]);
    m_value=temp_value;
    printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);

    temp_value=optimizeOneDirection(step[4]);
    m_value=temp_value;
    printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);
   
    temp_value=optimizeOneDirection(step[5]);
    m_value=temp_value;
    printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);

    stepBig=ntkOptimizationStep(m_xpos-temp_mvalue.x,m_ypos-temp_mvalue.y,m_zpos-temp_mvalue.z,m_rxpos-temp_mvalue.rx,m_rypos-temp_mvalue.ry,m_rzpos-temp_mvalue.rz, 0.0);
    
    double stepVal=sqrt(stepBig.x*stepBig.x+stepBig.y*stepBig.y+stepBig.z*stepBig.z+stepBig.rx*stepBig.rx+stepBig.ry*stepBig.ry+stepBig.rz*stepBig.rz);
    
    if(stepVal>0.0){
      stepBig=ntkOptimizationStep(stepBig.x*stepsize/stepVal,stepBig.y*stepsize/stepVal,stepBig.z*stepsize/stepVal,stepBig.rx*stepsize/stepVal,stepBig.ry*stepsize/stepVal,stepBig.rz*stepsize/stepVal, 0.0);
    }

    temp_value=optimizeOneDirection(stepBig);
    m_value=temp_value;
    printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);
  }
  delete temp_data;
}

/*
void ntkRigidRegistration::doPowellRegistration(ntk3DData* target, double stepsize, int histFactor){
  m_target=target;
  double temp_value=10000.0;
  double before_value=0.0;
  ntkOptimizationStep step[7];
  ntkOptimizationStep temp_mvalue;
  step[0]=ntkOptimizationStep(stepsize,0,0,0,0,0,0);
  step[1]=ntkOptimizationStep(0,stepsize,0,0,0,0,0);
  step[2]=ntkOptimizationStep(0,0,stepsize,0,0,0,0);
  step[3]=ntkOptimizationStep(0,0,0,stepsize,0,0,0);
  step[4]=ntkOptimizationStep(0,0,0,0,stepsize,0,0);
  step[5]=ntkOptimizationStep(0,0,0,0,0,stepsize,0);
  step[6]=ntkOptimizationStep(0,0,0,0,0,0,stepsize);

  m_measure->setHistogramBinSize(histFactor);
  m_trans->loadData(m_target);
  
  ntkIntDimension temp_dim=m_target->getDataSize();
  ntkFloatDimension temp_thickness=m_target->getDataThickness();
  //ntk3DData* temp_data=new ntk3DData(temp_dim);
  //temp_data->setDataThickness(temp_thickness);

  printf("Start registration. step= %f\n", stepsize);

  m_trans->getMatrix()->Identity();
  m_trans->doTranslation(m_xpos, m_ypos, m_zpos);
  m_trans->doRotationZ(m_rzpos);
  m_trans->doRotationY(m_rypos);
  m_trans->doRotationX(m_rxpos);
  m_trans->doScaling(m_scale/100.0+1);
  ntk3DData *temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
  //m_trans->applyTransformation(temp_data);
  m_value=m_measure->doSimilarityMeasure(temp_data);
  before_value=m_value;

  temp_mvalue=ntkOptimizationStep(m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);

  temp_value=optimizeOneDirection(step[0]);
  m_value=temp_value;
  printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);

  temp_value=optimizeOneDirection(step[1]);
  m_value=temp_value;
  printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);

  temp_value=optimizeOneDirection(step[2]);
  m_value=temp_value;
  printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);

  temp_value=optimizeOneDirection(step[3]);
  m_value=temp_value;
  printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);

  temp_value=optimizeOneDirection(step[4]);
  m_value=temp_value;
  printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);
 
  temp_value=optimizeOneDirection(step[5]);
  m_value=temp_value;
  printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);

  temp_value=optimizeOneDirection(step[6]);
  m_value=temp_value;
  printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);
 
  
  while(temp_value>before_value){
    step[0]=step[1];
    step[1]=step[2];
    step[2]=step[3];
    step[3]=step[4];
    step[4]=step[5];
    step[5]=step[6];
    ntkOptimizationStep val=ntkOptimizationStep(m_xpos-temp_mvalue.x,m_ypos-temp_mvalue.y,m_zpos-temp_mvalue.z,m_rxpos-temp_mvalue.rx,m_rypos-temp_mvalue.ry,m_rzpos-temp_mvalue.rz, m_scale-temp_mvalue.scale);
    double stepVal=sqrt(val.x*val.x+val.y*val.y+val.z*val.z+val.rx*val.rx+val.ry*val.ry+val.rz*val.rz+val.scale*val.scale);
    
    step[6]=ntkOptimizationStep(val.x*stepsize/stepVal,val.y*stepsize/stepVal,val.z*stepsize/stepVal,val.rx*stepsize/stepVal,val.ry*stepsize/stepVal,val.rz*stepsize/stepVal, val.scale*stepsize/stepVal);
    printf("step6 %f %f %f %f %f %f %f\n", step[6].x, step[6].y, step[6].z, step[6].rx, step[6].ry, step[6].rz, step[6].scale);
    before_value=m_value;

    temp_mvalue=ntkOptimizationStep(val.x, val.y, val.z, val.rx, val.ry, val.rz, val.scale);

    temp_value=optimizeOneDirection(step[0]);
    m_value=temp_value;
    printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);

    temp_value=optimizeOneDirection(step[1]);
    m_value=temp_value;
    printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);

    temp_value=optimizeOneDirection(step[2]);
    m_value=temp_value;
    printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);
    
    temp_value=optimizeOneDirection(step[3]);
    m_value=temp_value;
    printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);

    temp_value=optimizeOneDirection(step[4]);
    m_value=temp_value;
    printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);
   
    temp_value=optimizeOneDirection(step[5]);
    m_value=temp_value;
    printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);

    temp_value=optimizeOneDirection(step[6]);
    m_value=temp_value;
    printf("%f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);
  }
  delete temp_data;
}
*/

double ntkRigidRegistration::optimizeOneDirection(double* variable, double step){
  double big=100000.0;
  double temp_value=0.0;
  double temp_big=m_value;
  double temp_step_big=100000.0;
  double temp_variable=*variable;
  ntkIntDimension temp_dim=m_target->getDataSize();
  ntkFloatDimension temp_thickness=m_target->getDataThickness();
  int count=0;

  *variable+=step;
  m_trans->getMatrix()->Identity();
  m_trans->doTranslation(m_xpos, m_ypos, m_zpos);
  m_trans->doRotationZ(m_rzpos);
  m_trans->doRotationY(m_rypos);
  m_trans->doRotationX(m_rxpos);
  ntk3DData *temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
  temp_value=m_measure->doSimilarityMeasure(temp_data);

  m_trans->getMatrix()->Identity();
  m_trans->doTranslation(m_xpos, m_ypos, m_zpos);
  m_trans->doRotationZ(m_rzpos);
  m_trans->doRotationY(m_rypos);
  m_trans->doRotationX(m_rxpos);
  delete temp_data;
  temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
  temp_value=m_measure->doSimilarityMeasure(temp_data);
   
  count=0;
  while(temp_value>temp_big){// && count <10){
    count++;
    temp_big=temp_value;
    *variable+=step;
    m_trans->getMatrix()->Identity();
    m_trans->doTranslation(m_xpos, m_ypos, m_zpos);
    m_trans->doRotationZ(m_rzpos);
    m_trans->doRotationY(m_rypos);
    m_trans->doRotationX(m_rxpos);
    printf("inside %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);
    delete temp_data;
    temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
    temp_value=m_measure->doSimilarityMeasure(temp_data);
  }
  
  big=temp_big;
  temp_step_big=*variable-step;
  
  *variable=temp_variable;
  *variable-=step;
  temp_big=m_value;
  m_trans->getMatrix()->Identity();
  m_trans->doTranslation(m_xpos, m_ypos, m_zpos);
  m_trans->doRotationZ(m_rzpos);
  m_trans->doRotationY(m_rypos);
  m_trans->doRotationX(m_rxpos);
  delete temp_data;
  temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
  temp_value=m_measure->doSimilarityMeasure(temp_data);
  count=0;
  while(temp_value>temp_big){// && count <10){
    count++;
    temp_big=temp_value;
    *variable-=step;
    m_trans->getMatrix()->Identity();
    m_trans->doTranslation(m_xpos, m_ypos, m_zpos);
    m_trans->doRotationZ(m_rzpos);
    m_trans->doRotationY(m_rypos);
    m_trans->doRotationX(m_rxpos);
    printf("inside %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos);
    delete temp_data;
    temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
    temp_value=m_measure->doSimilarityMeasure(temp_data);
  }

  if(temp_big>big){
    big=temp_big;
    temp_step_big=*variable+=step;
  }
  
  *variable=temp_step_big;
    
  printf("value: %f %f\n", big, m_value);
  
  delete temp_data;
  return big;
}

double ntkRigidRegistration::optimizeOneDirection(ntkOptimizationStep step){
  double big=100000.0;
  double temp_value=0.0;
  double temp_big=m_value;
  ntkOptimizationStep temp_step_big(10000,10000,10000,10000,10000,10000,10000);
  ntkOptimizationStep temp_step=ntkOptimizationStep(m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);
  ntkIntDimension temp_dim=m_target->getDataSize();
  ntkFloatDimension temp_thickness=m_target->getDataThickness();
  int count=0;

  m_xpos+=step.x;m_ypos+=step.y;m_zpos+=step.z;m_rxpos+=step.rx;m_rypos+=step.ry;m_rzpos+=step.rz;m_scale+=step.scale;
  m_trans->getMatrix()->Identity();
  m_trans->doTranslation(m_xpos, m_ypos, m_zpos);
  m_trans->doRotationZ(m_rzpos);
  m_trans->doRotationY(m_rypos);
  m_trans->doRotationX(m_rxpos);
  m_trans->doScaling(m_scale/100.0+1);
  ntk3DData *temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
  //ntk3DData *temp_data=m_trans->applyTransformation();
  temp_value=m_measure->doSimilarityMeasure(temp_data);

  m_trans->getMatrix()->Identity();
  m_trans->doTranslation(m_xpos, m_ypos, m_zpos);
  m_trans->doRotationZ(m_rzpos);
  m_trans->doRotationY(m_rypos);
  m_trans->doRotationX(m_rxpos);
  m_trans->doScaling(m_scale/100.0+1);
  delete temp_data;
  temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
  //temp_data=m_trans->applyTransformation();
  temp_value=m_measure->doSimilarityMeasure(temp_data);
   
  count=0;
  while(temp_value>temp_big){// && count <10){
    count++;
    temp_big=temp_value;
    m_xpos+=step.x;m_ypos+=step.y;m_zpos+=step.z;m_rxpos+=step.rx;m_rypos+=step.ry;m_rzpos+=step.rz;m_scale+=step.scale;
    m_trans->getMatrix()->Identity();
    m_trans->doTranslation(m_xpos, m_ypos, m_zpos);
    m_trans->doRotationZ(m_rzpos);
    m_trans->doRotationY(m_rypos);
    m_trans->doRotationX(m_rxpos);
    m_trans->doScaling(m_scale/100.0+1);
    printf("inside %f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);
    delete temp_data;
    temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
    //temp_data=m_trans->applyTransformation();
    temp_value=m_measure->doSimilarityMeasure(temp_data);
  }
  
  big=temp_big;
  temp_step_big=ntkOptimizationStep(m_xpos-step.x,m_ypos-step.y,m_zpos-step.z,m_rxpos-step.rx,m_rypos-step.ry,m_rzpos-step.rz, m_scale-step.scale);
  
  m_xpos=temp_step.x-step.x;m_ypos=temp_step.y-step.y;m_zpos=temp_step.z-step.z;m_rxpos=temp_step.rx-step.rx;m_rypos=temp_step.ry-step.ry;m_rzpos=temp_step.rz-step.rz;m_scale=temp_step.scale-step.scale;
  temp_big=m_value;
  m_trans->getMatrix()->Identity();
  m_trans->doTranslation(m_xpos, m_ypos, m_zpos);
  m_trans->doRotationZ(m_rzpos);
  m_trans->doRotationY(m_rypos);
  m_trans->doRotationX(m_rxpos);
  m_trans->doScaling(m_scale/100.0+1);
  delete temp_data;
  temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
  //temp_data=m_trans->applyTransformation();
  temp_value=m_measure->doSimilarityMeasure(temp_data);
  count=0;
  while(temp_value>temp_big){// && count <10){
    count++;
    temp_big=temp_value;
    m_xpos-=step.x;m_ypos-=step.y;m_zpos-=step.z;m_rxpos-=step.rx;m_rypos-=step.ry;m_rzpos-=step.rz;m_scale-=step.scale;
    m_trans->getMatrix()->Identity();
    m_trans->doTranslation(m_xpos, m_ypos, m_zpos);
    m_trans->doRotationZ(m_rzpos);
    m_trans->doRotationY(m_rypos);
    m_trans->doRotationX(m_rxpos);
    m_trans->doScaling(m_scale/100.0+1);
    printf("inside %f %f %f %f %f %f %f\n",m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);
    delete temp_data;
    temp_data=m_trans->applyTransformation(temp_dim, temp_thickness);
    //temp_data=m_trans->applyTransformation();
    temp_value=m_measure->doSimilarityMeasure(temp_data);
  }

  if(temp_big>big){
    big=temp_big;
    m_xpos+=step.x;m_ypos+=step.y;m_zpos+=step.z;m_rxpos+=step.rx;m_rypos+=step.ry;m_rzpos+=step.rz;m_scale+=step.scale;
    temp_step_big=ntkOptimizationStep(m_xpos, m_ypos, m_zpos, m_rxpos, m_rypos, m_rzpos, m_scale);
  }
  
  m_xpos=temp_step_big.x;m_ypos=temp_step_big.y;m_zpos=temp_step_big.z;m_rxpos=temp_step_big.rx;m_rypos=temp_step_big.ry;m_rzpos=temp_step_big.rz;m_scale=temp_step_big.scale;
  
  printf("value: %f %f\n", big, m_value);
  
  delete temp_data;
  return big;
}

ntk3DData* ntkRigidRegistration::getResult(void){
  ntkIntDimension dim=m_target->getDataSize();
  ntkFloatDimension thickness=m_target->getDataThickness();

  m_trans->getMatrix()->Identity();
  m_trans->doTranslation(m_xpos, m_ypos, m_zpos);
  m_trans->doRotationZ(m_rzpos);
  m_trans->doRotationY(m_rypos);
  m_trans->doRotationX(m_rxpos);
  m_trans->doScaling(m_scale/100.0+1);
  ntk3DData *result=m_trans->applyTransformation(dim, thickness);
  
  return result;
}
