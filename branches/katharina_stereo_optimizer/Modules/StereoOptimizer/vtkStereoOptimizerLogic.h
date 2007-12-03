/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkGradientAnisotropicDiffusionFilterLogic.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.3 $

=========================================================================auto=*/
#ifndef __vtkStereoOptimizerLogic_h
#define __vtkStereoOptimizerLogic_h

#include "vtkSlicerModuleLogic.h"
#include "vtkMRMLScene.h"

#include "vtkStereoOptimizer.h"
#include "vtkMRMLStereoOptimizerNode.h"

class VTK_STEREOOPTIMIZER_EXPORT vtkStereoOptimizerLogic : public vtkSlicerModuleLogic
{
  public:
  
  static vtkStereoOptimizerLogic *New();
  vtkTypeMacro(vtkStereoOptimizerLogic,vtkSlicerModuleLogic);

  void PrintSelf(ostream& os, vtkIndent indent);

  // TODO: do we need to observe MRML here?
  virtual void ProcessMrmlEvents ( vtkObject *caller, unsigned long event,
                                   void *callData ){};

  // Description: Get/Set MRML node storing parameter values
  vtkGetObjectMacro (StereoOptimizerNode, vtkMRMLStereoOptimizerNode);
  
  void SetAndObserveStereoOptimizerNode(vtkMRMLStereoOptimizerNode *n) 
   {
   vtkSetAndObserveMRMLNodeMacro( this->StereoOptimizerNode, n);
   }
  
  // The method that creates and runs VTK or ITK pipeline
  //void Apply();
  
  //float GetProgress();
  //void SetProgress(float p);
  //vtkGetFloatMacro(Progress);
  //vtkSetFloatMacro(Progress);
  //vtkGetStringMacro(Res);
  //vtkSetStringMacro(Res);
  // vtkGetMacro(TestFloat);
  //vtkSetMacro(TestFloat);
 
 protected:
  vtkStereoOptimizerLogic();
  virtual ~vtkStereoOptimizerLogic();
  vtkStereoOptimizerLogic(const vtkStereoOptimizerLogic&);
  void operator=(const vtkStereoOptimizerLogic&);
  
  vtkMRMLStereoOptimizerNode* StereoOptimizerNode;
  // float Progress; //progress of labelStats processing in percent
  // char* Res;
  // float TestFloat;
 };

#endif

