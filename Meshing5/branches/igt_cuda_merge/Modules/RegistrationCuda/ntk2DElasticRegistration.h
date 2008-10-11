#ifndef NTK2DELASTICREGISTRATION_H
#define NTK2DELASTICREGISTRATION_H

#include "ntkRegistration.h"
#include "ntk2DElasticTransformation.h"
#include "ntk2DBSpline3Transform.h"
#include "ntkBSpline3Function.h" 
#include "ntkDeformationSpline.h"
#include "ntkElasticityMap.h"
#include "ntk2DDataResizer.h"
#include <math.h>

class ntk2DElasticRegistration : public ntkRegistration{
  typedef struct{
    ntk2DElasticRegistration* here;
    int threadNumber;
    ntk2DData* target;
    int splineSizeLevel;
    ntkDeformationSpline *splineParam;
    
    float *newX;
    float *newY;
    
    float *diff;
  }ntkCalculateSSDGradientThreadParam;

  typedef struct scheme{
    struct scheme *next;
    int imageResLevel;
    int splineResLevel;
    float step;
    float PEweight;
  }ntkElasticRegistrationScheme;

 public:
  ntk2DElasticRegistration();
  ~ntk2DElasticRegistration();
  void loadData(ntk2DData* reference);
  void loadData(ntk3DData* reference){};

  /**
   * dummy function
   */
  
  void doRegistration(ntk3DData* target);
  ntkDeformationSpline* doRegistration(ntk2DData* reference, ntk2DData* target, ntkElasticityMap* emap,  int splineSizeLevel, ntkDeformationSpline* splineParam, float step, float PEweight);
  
  /**
   * get the size of spline parameter
   */

  ntkIntDimension getSplineSize();

  /**
   * calculate SSD of reference and transformed target.
   * target is elastically transformed by splineParam. Calculated SSD value is returned.
   */
  
  float calculateSSD(ntk2DData* reference, ntk2DData* target);

  /**
   * calculate SSD of reference and transformed target plus potential energy of deformation spline.
   * target is elastically transformed by splineParam. Calculated SSD value is returned.
   */
  
  float calculateSSDPE(ntk2DData* reference, ntk2DData* target, ntkDeformationSpline *splineParam);
  
  /**
   * get calculated gradient buffer
   * you need to execute calculateSSDGradient() befor calling this function. Returned gradient buffer does not need to be released from main program.
   */
  
  float* getGradientBuffer();

  /**
   * calculate gradient of SSD in spline parameter space
   * maximum gradient value is returned
   */

  float calculateSSDGradient(ntk2DData* target, int splineSizeLevel, ntkDeformationSpline *splineParam);
  
  /**
   * do spline param to higher resolution spline parameter transformation 
   */

  ntkDeformationSpline *doSplineParamTransform(ntkDeformationSpline* inputSpline);

  void addRegistrationScheme(int imageResLevel, int splineResLevel, float step, float PEweight);
  void resetRegistrationScheme();
  void printRegistrationScheme();
  bool checkRegistrationScheme();
  ntkDeformationSpline *executeRegistrationSchemes(ntk2DData *reference, ntk2DData *target, ntkDeformationSpline *inputSpline, ntkElasticityMap *emap);
  ntk2DData *resizeDataSmaller(ntk2DData* input, int imageResLevel);
  ntkElasticityMap *resizeMapSmaller(ntkElasticityMap* input, int imageResLevel);
  void setMinGradientLimit(float value);
  
 protected:
  ntk2DElasticTransformation* m_trans;
  ntk2DData* m_reference;
  ntk2DData* m_target;
  ntk2DData* m_targetPlus;
  unsigned char* m_referenceBuffer;
  ntk2DBSpline3Transform *m_transform;
  ntkBSpline3Function *m_function;
  ntkIntDimension m_referenceSize;
  ntkIntDimension m_targetSize;
  ntkIntDimension m_splineSize;
  ntkMatrix* m_targetPlusSpline;
  ntkDeformationSpline* m_splineParam;
  ntkElasticityMap* m_emap;
  ntkElasticRegistrationScheme *m_firstScheme;
  ntkElasticRegistrationScheme *m_lastScheme;
  int m_knotDistance;
  float* m_SSDgradient;
  float m_maxGradient;
  int m_splineSizeLevel;
  int m_imageSizeLevel;
  int m_MTlevel;
  float m_minGradientLimit;
  float m_PEweight;

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
  void threadCalculateSSDGradient(float *newX, float *newY, float *diff,  int threadNumber);
  
  void threadCalculateSSDPEGradient(float *newX, float *newY, float *diff,  int threadNumber);
  
};


#endif
