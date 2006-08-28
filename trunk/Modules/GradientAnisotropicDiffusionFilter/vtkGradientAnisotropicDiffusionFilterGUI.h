/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkGradientAnisotropicDiffusionFilterGUI.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.3 $

=========================================================================auto=*/
#ifndef __vtkGradientAnisotropicDiffusionFilterGUI_h
#define __vtkGradientAnisotropicDiffusionFilterGUI_h

#include "vtkSlicerBaseGUIWin32Header.h"
#include "vtkSlicerModuleGUI.h"

#include "vtkMRMLScene.h"
#include "vtkGradientAnisotropicDiffusionFilterLogic.h"


class vtkSlicerSliceWidget;
class vtkKWFrame;
class vtkKWScaleWithEntry;
class vtkKWPushButton;
class vtkSlicerNodeSelectorWidget;

class VTK_GRADIENTANISOTROPICDIFFUSIONFILTER_EXPORT vtkGradientAnisotropicDiffusionFilterGUI : public vtkSlicerModuleGUI
{
  public:
  static vtkGradientAnisotropicDiffusionFilterGUI *New();
  vtkTypeMacro(vtkGradientAnisotropicDiffusionFilterGUI,vtkSlicerModuleGUI);
  void PrintSelf(ostream& os, vtkIndent indent);

   // Description: Get/Set MRML node
  vtkGetObjectMacro (Logic, vtkGradientAnisotropicDiffusionFilterLogic);
  vtkSetObjectMacro (Logic, vtkGradientAnisotropicDiffusionFilterLogic);
  
  // Description: Get/Set MRML node
  vtkGetObjectMacro (GradientAnisotropicDiffusionFilterNode, vtkMRMLGradientAnisotropicDiffusionFilterNode);
  vtkSetObjectMacro (GradientAnisotropicDiffusionFilterNode, vtkMRMLGradientAnisotropicDiffusionFilterNode);

  // Description:
  // Create widgets
  virtual void BuildGUI ( );

  // Description:
  // Add obsereves to GUI widgets
  virtual void AddGUIObservers ( );
  
  // Description:
  // Remove obsereves to GUI widgets
  virtual void RemoveGUIObservers ( );
  
  // Description:
  // Pprocess events generated by Logic
  virtual void ProcessLogicEvents ( vtkObject *caller, unsigned long event,
                                  void *callData ){};

  // Description:
  // Pprocess events generated by GUI widgets
  virtual void ProcessGUIEvents ( vtkObject *caller, unsigned long event,
                                  void *callData );

  // Description:
  // Pprocess events generated by MRML
  virtual void ProcessMRMLEvents ( vtkObject *caller, unsigned long event, 
                                  void *callData);
  // Description:
  // Describe behavior at module startup and exit.
  virtual void Enter ( ){};
  virtual void Exit ( ){};

protected:
  vtkGradientAnisotropicDiffusionFilterGUI();
  virtual ~vtkGradientAnisotropicDiffusionFilterGUI();
  vtkGradientAnisotropicDiffusionFilterGUI(const vtkGradientAnisotropicDiffusionFilterGUI&);
  void operator=(const vtkGradientAnisotropicDiffusionFilterGUI&);

  // Description:
  // Updates GUI widgets based on parameters values in MRML node
  void UpdateGUI();

  // Description:
  // Updates parameters values in MRML node based on GUI widgets 
  void UpdateMRML();
  
  vtkKWScaleWithEntry* ConductanceScale;
  vtkKWScaleWithEntry* TimeStepScale;
  vtkKWScaleWithEntry* NumberOfIterationsScale;
  vtkSlicerNodeSelectorWidget* VolumeSelector;
  vtkSlicerNodeSelectorWidget* OutVolumeSelector;
  vtkSlicerNodeSelectorWidget* GADNodeSelector;
  vtkKWPushButton* ApplyButton;
  
  vtkGradientAnisotropicDiffusionFilterLogic *Logic;
  vtkMRMLGradientAnisotropicDiffusionFilterNode* GradientAnisotropicDiffusionFilterNode;

};

#endif

