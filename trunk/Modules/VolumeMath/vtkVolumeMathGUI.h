/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkVolumeMathGUI.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.3 $

=========================================================================auto=*/
#ifndef __vtkVolumeMathGUI_h
#define __vtkVolumeMathGUI_h

#include <string>
#include <iostream>
#include <sstream>

#include "vtkSlicerBaseGUIWin32Header.h"
#include "vtkSlicerModuleGUI.h"

#include "vtkMRMLScene.h"
#include "vtkVolumeMathLogic.h"


class vtkSlicerSliceWidget;
class vtkKWFrame;
class vtkKWScaleWithEntry;
class vtkKWPushButton;
class vtkSlicerNodeSelectorWidget;
class vtkKWMultiColumnList;
class vtkKWLoadSaveButton;

class VTK_VOLUMEMATH_EXPORT vtkVolumeMathGUI : public vtkSlicerModuleGUI
{
  public:
  static vtkVolumeMathGUI *New();
  vtkTypeMacro(vtkVolumeMathGUI,vtkSlicerModuleGUI);
  void PrintSelf(ostream& os, vtkIndent indent);

   // Description: Get/Set MRML node
  vtkGetObjectMacro (Logic, vtkVolumeMathLogic);
  vtkSetObjectMacro (Logic, vtkVolumeMathLogic);
  
  // Description: Get/Set MRML node
  vtkGetObjectMacro (VolumeMathNode, vtkMRMLVolumeMathNode);

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
  // Process events generated by Logic
  virtual void ProcessLogicEvents ( vtkObject *caller, unsigned long event,
                                    void *callData );

  // Description:
  // Process events generated by GUI widgets
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

  // Description:
  // Set up primary selection tcl procedures 
  virtual void SetPrimarySelectionTclProcedures ( );

  //BTX
  // Description:
  // Sets text to be primary selection, for copy to clipboard functionality 
  virtual void SetPrimarySelection ( std::string text );
  //ETX
  
protected:
  vtkVolumeMathGUI();
  virtual ~vtkVolumeMathGUI();
  vtkVolumeMathGUI(const vtkVolumeMathGUI&);
  void operator=(const vtkVolumeMathGUI&);

  // Description:
  // Updates GUI widgets based on parameters values in MRML node
  void UpdateGUI();

  // Description:
  // Updates parameters values in MRML node based on GUI widgets 
  void UpdateMRML();
  
  vtkSlicerNodeSelectorWidget* GrayscaleSelector;
  vtkSlicerNodeSelectorWidget* LabelmapSelector; 
  vtkKWPushButton* ApplyButton;
 
  vtkVolumeMathLogic *Logic;
  vtkMRMLVolumeMathNode* VolumeMathNode;
  //vtkKWText* VolStatsResult;
  vtkKWMultiColumnList* ResultList;
  vtkKWLoadSaveButton* SaveToFile;
  vtkKWPushButton* SaveToClipboardButton;
  
};

#endif

