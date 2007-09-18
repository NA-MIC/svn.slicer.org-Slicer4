/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkVolumeRenderingModuleGUI.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.3 $

=========================================================================auto=*/
#ifndef __vtkVolumeRenderingModuleGUI_h
#define __vtkVolumeRenderingModuleGUI_h

#include "vtkSlicerModuleGUI.h"
#include "vtkVolumeRenderingWin32Header.h"
#include "vtkVolumeRenderingModuleLogic.h"

class VTK_VRMODULE_EXPORT vtkVolumeRenderingModuleGUI :public vtkSlicerModuleGUI
{
public:

  static vtkVolumeRenderingModuleGUI *New();
 // vtkTypeMacro(vtkVolumeRenderingModuleGUI,vtkSlicerModuleGUI);

  void PrintSelf(ostream& os, vtkIndent indent);

   // Description: Get/Set module logic
  vtkGetObjectMacro (Logic, vtkVolumeRenderingModuleLogic);
  
  void SetModuleLogic ( vtkVolumeRenderingModuleLogic *logic )
  { this->SetLogic ( vtkObjectPointer (&this->Logic), logic ); }
   
  void SetAndObserveModuleLogic ( vtkVolumeRenderingModuleLogic *logic )
  { this->SetAndObserveLogic ( vtkObjectPointer (&this->Logic), logic ); }
  
  // Description:
  // Create widgets
  virtual void BuildGUI ( );

  // Description:
  // This method releases references and key-bindings,
  // and optionally removes observers.
  virtual void TearDownGUI ( );
  
  // Description:
  // Methods for adding module-specific key bindings and
  // removing them.
  virtual void CreateModuleEventBindings ( );
  virtual void ReleaseModuleEventBindings ( );

  // Description:
  // Add obsereves to GUI widgets
  virtual void AddGUIObservers ( );
  
  // Description:
  // Remove obsereves to GUI widgets
  virtual void RemoveGUIObservers ( );
  virtual void RemoveMRMLNodeObservers ( );
  virtual void RemoveLogicObservers ( );
  
  // Description:
  // Process events generated by Logic
  virtual void ProcessLogicEvents ( vtkObject *caller, unsigned long event,
                                  void *callData ){};

  // Description:
  // Process events generated by GUI widgets
  virtual void ProcessGUIEvents ( vtkObject *caller, unsigned long event,
                                  void *callData );

  // Description:
  // Process events generated by MRML
  virtual void ProcessMRMLEvents ( vtkObject *caller, unsigned long event, 
                                  void *callData);


  // Description:
  // Methods describe behavior at module enter and exit.
  virtual void Enter ( );
  virtual void Exit ( );
  

  // Description:
  // Get/Set the main slicer viewer widget, for picking
  vtkGetObjectMacro(ViewerWidget, vtkSlicerViewerWidget);
  virtual void SetViewerWidget(vtkSlicerViewerWidget *viewerWidget);

  // Description:
  // Get/Set the slicer interactorstyle, for picking
  vtkGetObjectMacro(InteractorStyle, vtkSlicerViewerInteractorStyle);
  virtual void SetInteractorStyle(vtkSlicerViewerInteractorStyle *interactorStyle);
  
protected:
  vtkVolumeRenderingModuleGUI();
  ~vtkVolumeRenderingModuleGUI();
  vtkVolumeRenderingModuleGUI(const vtkVolumeRenderingModuleGUI&);
  void operator=(const vtkVolumeRenderingModuleGUI&);

  // Description:
  // Updates GUI widgets based on parameters values in MRML node
  void UpdateGUI();

  // Description:
  // Updates parameters values in MRML node based on GUI widgets 
  void UpdateMRML();

  // Description:
  // GUI elements
  
  // Description:
  // Pointer to the module's logic class
  vtkVolumeRenderingModuleLogic *Logic;

  // Description:
  // A pointer back to the viewer widget, useful for picking
  vtkSlicerViewerWidget *ViewerWidget;

  // Description:
  // A poitner to the interactor style, useful for picking
  vtkSlicerViewerInteractorStyle *InteractorStyle;
};

#endif
