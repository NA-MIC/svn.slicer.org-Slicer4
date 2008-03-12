#include "ntkBSpline3Function.h"

ntkBSpline3Function::ntkBSpline3Function(){
  m_val=NULL;
}

ntkBSpline3Function::~ntkBSpline3Function(){
  if(m_val!=NULL) free(m_val);
}

double ntkBSpline3Function::getValue(double x){
  double xabs=fabs(x);
  if(xabs<=1.0){
    return (1.0-((1-xabs/2.0)*x*x*1.5));
  }else if(xabs<2.0){
    return ((2-xabs)*(2-xabs)*(2-xabs)/4.0);
  }else{
    return 0;
  }
  
  /*
  if(fabs(x)<=1.0){
    return (2.0/3.0-((1-fabs(x)/2.0)*x*x));
  }else if(fabs(x)<2.0){
    return ((2-fabs(x))*(2-fabs(x))*(2-fabs(x))/6.0);
  }else{
    return 0;
  }
  */
}

void ntkBSpline3Function::createPreCalculation(int knotNum){
  m_knotNum=knotNum;
  m_step=4.0/(double)(m_knotNum-1);
  m_val=(double*)malloc(m_knotNum*sizeof(double));
  int i;
  *(m_val+0)=0.0;
  for(i=1;i<m_knotNum-1;i++){
    *(m_val+i)=getValue(-2.0+i*m_step);
  }
  *(m_val+m_knotNum-1)=0.0;
}

double ntkBSpline3Function::getInterpolationValue(double x){
  if(x<=-2.0 || x >=2.0)return 0.0;
  double relpos=(x+2.0)/m_step;
  int intrelpos = (int)relpos;
  return *(m_val+intrelpos);
  //return (((double)(intrelpos+1)-relpos)* (*(m_val+intrelpos)))+((relpos-(double)intrelpos) * (*(m_val+intrelpos+1)));
}

double ntkBSpline3Function::getDifferentialValue(double x){
  double result=0.0;
  double xabs=fabs(x);
  if(xabs<=1.0){
    result=(2.25*xabs*xabs-3*xabs);
  }else if(xabs<2.0){
    result=(-0.75*(2.0-xabs)*(2.0-xabs));
  }else{
    return 0;
  }
  
  if(x<0){
    result=0.0-result;
  }
  return result;
}
