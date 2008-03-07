#ifndef NTKRIGIDREGISTRATION_H
#define NTKRIGIDREGISTRATION_H

#include "ntkRegistration.h"
#include "ntkSADSimilarityMeasure.h"
#include "ntkMISimilarityMeasure.h"
#include "ntkLinearTransformation.h"
#include "ntkDataAligner.h"
#include <math.h>

struct ntkOptimizationStep{
  double x;
  double y;
  double z;
  double rx;
  double ry;
  double rz;
  double scale;
  public: 
  ntkOptimizationStep(){
    x=0;
    y=0;
    z=0;
    rx=0;
    ry=0;
    rz=0;
    scale=0;
  }
  ntkOptimizationStep(double xx, double yy, double zz, double rxx, double ryy, double rzz){
    x=xx;y=yy;z=zz;rx=rxx;ry=ryy;rz=rzz;scale=0;
  }
  ntkOptimizationStep(double xx, double yy, double zz, double rxx, double ryy, double rzz, double scl){
    x=xx;y=yy;z=zz;rx=rxx;ry=ryy;rz=rzz;scale=scl;
  }
};

class ntkRigidRegistration : public ntkRegistration{
 public:
  ntkRigidRegistration();
  ~ntkRigidRegistration();
  void loadData(ntk3DData* reference);
  void doRegistration(ntk3DData* target);
  void doRegistration(ntk3DData* target, double step, int histFactor);
  void doUphillRegistration(ntk3DData* target, double step, double stepMove, int histFactor);
  void doPowellRegistration(ntk3DData* target, double step, int histFactor);
  void setInitialPosition(double x, double y, double z, double rx, double ry, double rz);
  ntk3DData* getResult(void);
 protected:
  double m_xpos;
  double m_ypos;
  double m_zpos;
  double m_rxpos;
  double m_rypos;
  double m_rzpos;
  double m_scale;
  double m_value;
  double optimizeOneDirection(double* variable, double step);
  double optimizeOneDirection(ntkOptimizationStep step);
  vtkMatrix4x4* m_transMatrix;
  ntkLinearTransformation* m_trans;
  ntkMISimilarityMeasure* m_measure;
};

#endif
