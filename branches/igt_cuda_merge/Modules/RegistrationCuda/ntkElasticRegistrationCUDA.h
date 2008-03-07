#ifndef NTKELASTICREGISTRATIONCUDA_H
#define NTKELASTICREGISTRATIONCUDA_H

#include "ntkRegistration.h"
#include "ntkElasticTransformation.h"
#include "ntkElasticTransformationCUDA.h"
#include "ntkBSpline3Transform.h"
#include "ntkBSpline3Function.h" 
#include "ntkDeformationSpline.h"
#include "ntkElasticityMap.h"
#include "ntk3DDataResizer.h"
#include "ntkTimer.h"
#include "ntkCudaDeviceMemory.h"
#include <math.h>
#include "CUDA_calculateSSD.h"
#include "CUDA_calculateSSDGradient.h"

class ntkElasticRegistrationCUDA : public ntkRegistration{
  typedef struct{
    ntkElasticRegistrationCUDA* here;
    int threadNumber;
    ntk3DData* target;
    int splineSizeLevel;
    ntkDeformationSpline *splineParam;
    
    float *newX;
    float *newY;
    float *newZ;
    float *diff;
  }ntkCalculateSSDGradientThreadParam;

  typedef struct scheme{
    struct scheme *next;
    int imageResLevel;
    int splineResLevel;
    float step;
    float PEweight;
  }ntkElasticRegistrationCUDAScheme;

 public:
  ntkElasticRegistrationCUDA();
  ~ntkElasticRegistrationCUDA();
  void loadData(ntk3DData* reference);

  /**
   * dummy function
   */
  
  void doRegistration(ntk3DData* target);
  ntkDeformationSpline* doRegistration(ntk3DData* reference, ntk3DData* target, ntkElasticityMap* emap,  int splineSizeLevel, ntkDeformationSpline* splineParam, float step, float PEweight);
  
  /**
   * get the size of spline parameter
   */

  ntkIntDimension getSplineSize();

  /**
   * calculate SSD of reference and transformed target.
   * target is elastically transformed by splineParam. Calculated SSD value is returned.
   */
  
  float calculateSSD(ntk3DData* reference, ntk3DData* target);

  /**
   * calculate SSD of reference and transformed target plus potential energy of deformation spline.
   * target is elastically transformed by splineParam. Calculated SSD value is returned.
   */
  
  float calculateSSDPE(ntk3DData* reference, ntk3DData* target, ntkDeformationSpline *splineParam);
  
  /**
   * get calculated gradient buffer
   * you need to execute calculateSSDGradient() befor calling this function. Returned gradient buffer does not need to be released from main program.
   */
  
  float* getGradientBuffer();

  /**
   * calculate gradient of SSD in spline parameter space
   * maximum gradient value is returned
   */

  float calculateSSDGradient(ntk3DData* target, int splineSizeLevel, ntkDeformationSpline *splineParam);
  
  /**
   * do spline param to higher resolution spline parameter transformation 
   */

  ntkDeformationSpline *doSplineParamTransform(ntkDeformationSpline* inputSpline);

  void addRegistrationScheme(int imageResLevel, int splineResLevel, float step, float PEweight);
  void resetRegistrationScheme();
  void printRegistrationScheme();
  bool checkRegistrationScheme();
  ntkDeformationSpline *executeRegistrationSchemes(ntk3DData *reference, ntk3DData *target, ntkDeformationSpline *inputSpline, ntkElasticityMap *emap);
  ntk3DData *resizeDataSmaller(ntk3DData* input, int imageResLevel);
  ntkElasticityMap *resizeMapSmaller(ntkElasticityMap* input, int imageResLevel);
  void setMinGradientLimit(float value);
  
 protected:
  ntkElasticTransformation* m_trans;
  ntk3DData* m_reference;
  ntk3DData* m_target;
  ntk3DData* m_targetPlus;
  unsigned char* m_referenceBuffer;
  ntkBSpline3Transform *m_transform;
  ntkBSpline3Function *m_function;
  ntkIntDimension m_referenceSize;
  ntkIntDimension m_targetSize;
  ntkIntDimension m_splineSize;
  ntkTensor* m_targetPlusSpline;
  ntkDeformationSpline* m_splineParam;
  ntkElasticityMap* m_emap;
  ntkElasticRegistrationCUDAScheme *m_firstScheme;
  ntkElasticRegistrationCUDAScheme *m_lastScheme;
  int m_knotDistance;
  float* m_SSDgradient;
  float m_maxGradient;
  int m_splineSizeLevel;
  int m_imageSizeLevel;
  int m_MTlevel;
  float m_minGradientLimit;
  float m_PEweight;

  ntkCudaDeviceMemory *m_dRef;
  ntkCudaDeviceMemory *m_dTar;

  /**
   * calculate gradient of SSD in spline parameter space
   * maximum gradient value is returned
   */

  float calculateSSDGradient(ntkDeformationSpline *splineParam);

  /**
   * calculate gradient of (SSD + Potential Energy) in spline parameter space
   * maximum gradient value is returned
   */

  float calculateSSDPEGradient(ntkDeformationSpline *splineParam);
  
 private:
  void threadCalculateSSDGradient(float *newX, float *newY, float* newZ, float *diff,  int threadNumber);
  
  void threadCalculateSSDPEGradient(float *newX, float *newY, float* newZ, float *diff,  int threadNumber);
  
  static void *startCalculateSSDGradientThread(void* threadParam);

  static void *startCalculateSSDPEGradientThread(void* threadParam);
  
};


#endif
