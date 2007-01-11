/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRAblationGUI.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.3 $

=========================================================================auto=*/
#ifndef __vtkMRAblationGUI_h
#define __vtkMRAblationGUI_h

#include "vtkSlicerBaseGUIWin32Header.h"
#include "vtkSlicerModuleGUI.h"

#include "vtkMRMLScene.h"
#include "vtkMRAblationLogic.h"


class vtkSlicerSliceWidget;
class vtkKWFrame;
class vtkKWScaleWithEntry;
class vtkKWPushButton;
class vtkSlicerNodeSelectorWidget;
class vtkKWLoadSaveButtonWithLabel;
class vtkKWEntryWithLabel;

class VTK_MRABLATION_EXPORT vtkMRAblationGUI : public vtkSlicerModuleGUI
{
  public:

  static vtkMRAblationGUI* New (  );
  vtkTypeRevisionMacro ( vtkMRAblationGUI, vtkSlicerModuleGUI );
  void PrintSelf (ostream& os, vtkIndent indent );

   // Description: Get/Set MRML node
  vtkGetObjectMacro (Logic, vtkMRAblationLogic);
  
  // Description: Get/Set MRML node
  vtkGetObjectMacro (MRAblationNode, vtkMRMLMRAblationNode);
  vtkSetObjectMacro (MRAblationNode, vtkMRMLMRAblationNode);

  // Description:
  // API for setting VolumeNode, VolumeLogic and
  // for both setting and observing them.
  void SetModuleLogic ( vtkMRAblationLogic *logic )
  { this->SetLogic ( vtkObjectPointer (&this->Logic), logic ); }
  void SetAndObserveModuleLogic ( vtkMRAblationLogic *logic )
  { this->SetAndObserveLogic ( vtkObjectPointer (&this->Logic), logic ); }

  // Description:
  // Create widgets
  virtual void BuildGUI ( );

  // Description:
  // Add obsereves to GUI widgets
  virtual void AddGUIObservers ( );
  
  // Description:
  // Remove obsereves to GUI widgets
  virtual void RemoveGUIObservers ( );
  virtual void RemoveMRMLNodeObservers ( );
  virtual void RemoveLogicObservers ( );
  
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
  vtkMRAblationGUI();
  ~vtkMRAblationGUI();
  vtkMRAblationGUI(const vtkMRAblationGUI&);
  void operator=(const vtkMRAblationGUI&);

  // Description:
  // Updates GUI widgets based on parameters values in MRML node
  void UpdateGUI();

  // Description:
  // Updates parameters values in MRML node based on GUI widgets 
  void UpdateMRML();
  
  vtkKWLoadSaveButtonWithLabel* ImageDirectory;
  vtkKWLoadSaveButtonWithLabel* WorkingDirectory;
  vtkKWScaleWithEntry* TimepointsScale;
  vtkKWScaleWithEntry* SlicesScale;
  vtkKWEntryWithLabel* TEEntry;
  vtkKWEntryWithLabel* w0Entry;
  vtkKWEntryWithLabel* TCEntry;
  vtkSlicerNodeSelectorWidget* OutVolumeSelector;
  vtkKWPushButton* ApplyButton;
  
  vtkMRAblationLogic *Logic;
  vtkMRMLMRAblationNode* MRAblationNode;

};

#endif

