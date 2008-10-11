/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMeshingWorkflowGUI.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.3 $

=========================================================================auto=*/
#ifndef __vtkMeshingWorkflowGUI_h
#define __vtkMeshingWorkflowGUI_h

#include "vtkSlicerBaseGUIWin32Header.h"
#include "vtkSlicerModuleGUI.h"

#include "vtkMRMLScene.h"
#include "vtkMeshingWorkflowLogic.h"




class vtkSlicerSliceWidget;
class vtkKWFrame;
class vtkKWScaleWithEntry;
class vtkKWPushButton;
class vtkSlicerNodeSelectorWidget;

class vtkKWMimxViewProperties;
class vtkLinkedListWrapperTree;
class vtkKWLoadSaveDialog;
class vtkKWMimxMainUserInterfacePanel;
class vtkKWMimxDisplayPropertiesGroup;


// added for UIowa Mimx integration.  Using MRML-based notebook instead of
// local list-based notebook. 

//class vtkKWMimxMainNotebook;
class vtkMeshingWorkflowMRMLNotebook;

class VTK_SLICERDAEMON_EXPORT vtkMeshingWorkflowGUI : public vtkSlicerModuleGUI
{
  public:
  static vtkMeshingWorkflowGUI *New();
  vtkTypeMacro(vtkMeshingWorkflowGUI,vtkSlicerModuleGUI);
  void PrintSelf(ostream& os, vtkIndent indent);

 
  
   // Description: Get/Set MRML node
  vtkGetObjectMacro (Logic, vtkMeshingWorkflowLogic);
  vtkSetObjectMacro (Logic, vtkMeshingWorkflowLogic);
  
  virtual void BuildGUI ( );

  virtual void AddGUIObservers ( );
  
  virtual void RemoveGUIObservers ( );
  
  virtual void ProcessLogicEvents ( vtkObject *caller, unsigned long event,
                                  void *callData ){};
  virtual void ProcessGUIEvents ( vtkObject *caller, unsigned long event,
                                  void *callData );
  virtual void ProcessMrmlEvents ( vtkObject *caller, unsigned long event,
                                   void *callData );
  // Description:
  // Describe behavior at module startup and exit.
  virtual void Enter ( ){};
  virtual void Exit ( ){};

  vtkGetObjectMacro(ViewProperties, vtkKWMimxViewProperties);
    vtkGetObjectMacro(
            MainUserInterfacePanel, vtkKWMimxMainUserInterfacePanel);
  
protected:
  vtkMeshingWorkflowGUI();
  ~vtkMeshingWorkflowGUI();
  vtkMeshingWorkflowGUI(const vtkMeshingWorkflowGUI&);
  void operator=(const vtkMeshingWorkflowGUI&);

  vtkMeshingWorkflowMRMLNotebook *MimxMainNotebook;
  //vtkKWMimxMainNotebook *MimxMainNotebook;
  vtkKWMimxViewProperties *ViewProperties;
  vtkLinkedListWrapperTree *DoUndoTree;
  vtkKWLoadSaveDialog *LoadSaveDialog;
  vtkKWMimxMainUserInterfacePanel *MainUserInterfacePanel;
  vtkKWMimxDisplayPropertiesGroup *DisplayPropertyDialog;
  vtkKWPushButton* ApplyButton;
  

  // callback to create the separate UI
  void BuildSeparateFEMeshGUI();
        
  vtkMeshingWorkflowLogic *Logic;

};

#endif

