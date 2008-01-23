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
#include "vtkVolumeRenderingModule.h"
#include "vtkVolumeRenderingModuleLogic.h"

#include "vtkMRMLVolumeRenderingNode.h"
#include "vtkSlicerNodeSelectorWidget.h"
#include "vtkSlicerNodeSelectorVolumeRenderingWidget.h"
#include "vtkSlicerVolumePropertyWidget.h"
#include "vtkKWLabel.h"
#include "vtkKWHistogram.h"
#include "vtkKWEntryWithLabel.h"
#include "vtkKWTkUtilities.h"
#include "vtkSlicerLabelMapWidget.h"

#include <string>

class vtkSlicerVolumeTextureMapper3D;
class vtkFixedPointVolumeRayCastMapper;
class vtkSlicerVRHelper;

class VTK_VOLUMERENDERINGMODULE_EXPORT vtkVolumeRenderingModuleGUI :public vtkSlicerModuleGUI
{
public:

    static vtkVolumeRenderingModuleGUI *New();
    vtkTypeMacro(vtkVolumeRenderingModuleGUI,vtkSlicerModuleGUI);

    void PrintSelf(ostream& os, vtkIndent indent);

    // Description: Get/Set module logic
    vtkGetObjectMacro (Logic, vtkVolumeRenderingModuleLogic);
    virtual void SetLogic(vtkVolumeRenderingModuleLogic *log)
    {
        this->Logic=log;
    }

    // Description:
    // Set the logic pointer from parent class pointer.
    // Overloads implementation in vtkSlicerModulesGUI
    // to allow loadable modules.
    virtual void SetModuleLogic ( vtkSlicerLogic *logic )
    {
      this->SetLogic(reinterpret_cast<vtkVolumeRenderingModuleLogic*> (logic)); 
    }

    //vtkSetObjectMacro (Logic, vtkVolumeRenderingModuleLogic);
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

    vtkSetMacro(PipelineInitialized,int);
    vtkGetMacro(PipelineInitialized,int);
    vtkBooleanMacro(PipelineInitialized,int);

    // Description:
    // Get methods on class members ( no Set methods required. )
    vtkGetObjectMacro (PB_Testing,vtkKWPushButton);
    vtkGetObjectMacro (PB_CreateNewVolumeRenderingNode,vtkKWPushButton);
    vtkGetObjectMacro (NS_ImageData,vtkSlicerNodeSelectorWidget);
    vtkGetObjectMacro (NS_VolumeRenderingDataSlicer,vtkSlicerNodeSelectorVolumeRenderingWidget);
    vtkGetObjectMacro (NS_VolumeRenderingDataScene,vtkSlicerNodeSelectorVolumeRenderingWidget);
    vtkGetObjectMacro (EWL_CreateNewVolumeRenderingNode,vtkKWEntryWithLabel);
    vtkGetObjectMacro (detailsFrame,vtkSlicerModuleCollapsibleFrame);
    vtkGetObjectMacro (currentNode,vtkMRMLVolumeRenderingNode);
    vtkGetObjectMacro (presets, vtkMRMLScene);





protected:
    vtkVolumeRenderingModuleGUI();
    ~vtkVolumeRenderingModuleGUI();
    vtkVolumeRenderingModuleGUI(const vtkVolumeRenderingModuleGUI&);//not implemented
    void operator=(const vtkVolumeRenderingModuleGUI&);//not implemented

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

    int PipelineInitialized;//0=no,1=Yes
    void InitializePipelineNewCurrentNode();
    void InitializePipelineFromMRMLScene();
    void InitializePipelineFromSlicer();
    void InitializePipelineFromImageData();
    void LabelMapInitializePipelineNewCurrentNode();
    void LabelMapInitializePipelineFromMRMLScene();
    void LabelMapInitializePipelineFromSlicer();
    void LabelMapInitializePipelineFromImageData();

    //OWN GUI Elements

    //Frame Save/Load
    vtkKWPushButton *PB_Testing;
    vtkKWPushButton *PB_CreateNewVolumeRenderingNode;
    vtkSlicerNodeSelectorWidget *NS_ImageData;
    //BTX
    std::string PreviousNS_ImageData;
    std::string PreviousNS_VolumeRenderingSlicer;
    std::string PreviousNS_VolumeRenderingDataScene;
    //ETX
    vtkSlicerNodeSelectorVolumeRenderingWidget *NS_VolumeRenderingDataSlicer;
    vtkSlicerNodeSelectorVolumeRenderingWidget *NS_VolumeRenderingDataScene;
    vtkKWEntryWithLabel *EWL_CreateNewVolumeRenderingNode;

    //Frame Details
    vtkSlicerModuleCollapsibleFrame *detailsFrame;


    //Other members
    vtkMRMLVolumeRenderingNode  *currentNode;
    vtkMRMLScene *presets;





   
    void PackLabelMapGUI(void);
    void UnpackLabelMapGUI(void);

    void PackSvpGUI(void);
    void UnpackSvpGUI(void);
    vtkSlicerVRHelper *Helper;
    //0 means grayscale, 1 means LabelMap
    int HelperNumber;
};

#endif
