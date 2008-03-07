#ifndef NTKRIGIDREGISTRATIONCUDA_H
#define NTKRIGIDREGISTRATIONCUDA_H

#include "ntkRegistration.h"
#include "ntkRigidRegistration.h"
#include "ntkSADSimilarityMeasure.h"
#include "ntkMISimilarityMeasure.h"
#include "ntkLinearTransformation.h"
#include "ntkDataAligner.h"
#include <math.h>

class ntkRigidRegistrationCUDA : public ntkRegistration{
 public:
  ntkRigidRegistrationCUDA();
  ~ntkRigidRegistrationCUDA();
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
