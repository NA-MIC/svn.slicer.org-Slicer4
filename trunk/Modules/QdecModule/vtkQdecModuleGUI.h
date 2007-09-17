/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkQdecModuleGUI.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.3 $

=========================================================================auto=*/
#ifndef __vtkQdecModuleGUI_h
#define __vtkQdecModuleGUI_h

#include "vtkQdecModuleWin32Header.h"
#include "vtkSlicerModuleGUI.h"

#include "vtkQdecModule.h"
#include "vtkMRMLScene.h"
#include "vtkQdecModuleLogic.h"

class vtkKWFrame;
class vtkKWScaleWithEntry;
class vtkKWPushButton;
class vtkSlicerNodeSelectorWidget;
class vtkKWLoadSaveButtonWithLabel;
class vtkKWEntryWithLabel;
class vtkKWMultiColumnListWithScrollbars;
class vtkKWListBoxWithScrollbarsWithLabel;
class vtkSlicerViewerWidget;
class vtkSlicerViewerInteractorStyle;
class VTK_QDECMODULE_EXPORT vtkQdecModuleGUI : public vtkSlicerModuleGUI
{
public:

  static vtkQdecModuleGUI *New();
 // vtkTypeMacro(vtkQdecModuleGUI,vtkSlicerModuleGUI);

  void PrintSelf(ostream& os, vtkIndent indent);

   // Description: Get/Set module logic
  vtkGetObjectMacro (Logic, vtkQdecModuleLogic);
  
  void SetModuleLogic ( vtkQdecModuleLogic *logic )
  { this->SetLogic ( vtkObjectPointer (&this->Logic), logic ); }
   
  void SetAndObserveModuleLogic ( vtkQdecModuleLogic *logic )
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
  // Get methods on class members ( no Set methods required. )
  vtkGetObjectMacro ( LoadTableButton, vtkKWLoadSaveButtonWithLabel );
  vtkGetObjectMacro ( LoadResultsButton, vtkKWLoadSaveButtonWithLabel );
  vtkGetObjectMacro ( ContinuousFactorsListBox, vtkKWListBoxWithScrollbarsWithLabel );
  vtkGetObjectMacro ( DiscreteFactorsListBox, vtkKWListBoxWithScrollbarsWithLabel );

  // Description:
  // Methods describe behavior at module enter and exit.
  virtual void Enter ( );
  virtual void Exit ( );
  
  // Description:
  // Update the elements of the multi column list box
  void UpdateElement(int row, int col, char *str);

  // Description:
  // Get/Set the main slicer viewer widget, for picking
  vtkGetObjectMacro(ViewerWidget, vtkSlicerViewerWidget);
  virtual void SetViewerWidget(vtkSlicerViewerWidget *viewerWidget);

  // Description:
  // Get/Set the slicer interactorstyle, for picking
  vtkGetObjectMacro(InteractorStyle, vtkSlicerViewerInteractorStyle);
  virtual void SetInteractorStyle(vtkSlicerViewerInteractorStyle *interactorStyle);
  
protected:
  vtkQdecModuleGUI();
  ~vtkQdecModuleGUI();
  vtkQdecModuleGUI(const vtkQdecModuleGUI&);
  void operator=(const vtkQdecModuleGUI&);

  // Description:
  // Updates GUI widgets based on parameters values in MRML node
  void UpdateGUI();

  // Description:
  // Updates parameters values in MRML node based on GUI widgets 
  void UpdateMRML();

  // Description:
  // GUI elements
  vtkKWLabel *NAMICLabel;
  vtkKWLoadSaveButtonWithLabel* SubjectsDirectoryButton;
  vtkKWLoadSaveButtonWithLabel* LoadTableButton;
  vtkKWLoadSaveButtonWithLabel* LoadResultsButton;
  vtkKWEntryWithLabel *DesignEntry;
  vtkKWListBoxWithScrollbarsWithLabel *DiscreteFactorsListBox;
  vtkKWListBoxWithScrollbarsWithLabel *ContinuousFactorsListBox;
  vtkKWPushButton* ApplyButton;
  vtkKWMultiColumnListWithScrollbars *MultiColumnList;
  
  
  vtkKWLabel *MeasureLabel;
  vtkKWMenuButton *MeasureMenu;

  vtkKWLabel *HemisphereLabel;
  vtkKWMenuButton *HemisphereMenu;

  vtkKWLabel *SmoothnessLabel;
  vtkKWMenuButton *SmoothnessMenu;
  
  // Description:
  // Pointer to the module's logic class
  vtkQdecModuleLogic *Logic;

  // Description:
  // A pointer back to the viewer widget, useful for picking
  vtkSlicerViewerWidget *ViewerWidget;

  // Description:
  // A pointer to the interactor style, useful for picking
  vtkSlicerViewerInteractorStyle *InteractorStyle;

  // Description:
  // A menu populated by display overlay options
  vtkKWLabel *QuestionLabel;
  vtkKWMenuButton *QuestionMenu;
};

#endif

