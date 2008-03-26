/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkOpenIGTLinkDaemonGUI.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.3 $

=========================================================================auto=*/
#ifndef __vtkOpenIGTLinkDaemonGUI_h
#define __vtkOpenIGTLinkDaemonGUI_h

#include "vtkSlicerBaseGUIWin32Header.h"
#include "vtkSlicerModuleGUI.h"

#include "vtkMRMLScene.h"
#include "vtkOpenIGTLinkDaemonLogic.h"


class vtkSlicerSliceWidget;
class vtkKWFrame;
class vtkKWScaleWithEntry;
class vtkKWPushButton;
class vtkSlicerNodeSelectorWidget;

class VTK_SLICERDAEMON_EXPORT vtkOpenIGTLinkDaemonGUI : public vtkSlicerModuleGUI
{
  public:
  static vtkOpenIGTLinkDaemonGUI *New();
  vtkTypeMacro(vtkOpenIGTLinkDaemonGUI,vtkSlicerModuleGUI);
  void PrintSelf(ostream& os, vtkIndent indent);

   // Description: Get/Set MRML node
  vtkGetObjectMacro (Logic, vtkOpenIGTLinkDaemonLogic);
  vtkSetObjectMacro (Logic, vtkOpenIGTLinkDaemonLogic);
  
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

protected:
  vtkOpenIGTLinkDaemonGUI();
  ~vtkOpenIGTLinkDaemonGUI();
  vtkOpenIGTLinkDaemonGUI(const vtkOpenIGTLinkDaemonGUI&);
  void operator=(const vtkOpenIGTLinkDaemonGUI&);

  vtkKWScaleWithEntry* ConductanceScale;
  vtkKWScaleWithEntry* TimeStepScale;
  vtkKWScaleWithEntry* NumberOfIterationsScale;
  vtkSlicerNodeSelectorWidget* VolumeSelector;
  vtkKWPushButton* ApplyButton;
  
  vtkOpenIGTLinkDaemonLogic *Logic;

};

#endif

