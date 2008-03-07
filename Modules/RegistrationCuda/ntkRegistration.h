#ifndef NTKREGISTRATION_H
#define NTKREGISTRATION_H
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>

#include "ntkProperties.h"
#include "ntk3DData.h"

class ntkRegistration{
 public:
  ntkRegistration(){}
  virtual ~ntkRegistration(){}
  
  virtual void loadData(ntk3DData* reference)=0;
  virtual void doRegistration(ntk3DData* target)=0;

  /**
   * virtual function for ntkElasticRegistration
   */

  void doRegistration(ntk3DData* target, int splineSizeLevel){};

  void setMTLevel(int);
  
 protected:
  ntk3DData* m_reference;
  ntk3DData* m_target;
  ntk3DData* m_temp;
  int m_MTlevel;
};
#endif
