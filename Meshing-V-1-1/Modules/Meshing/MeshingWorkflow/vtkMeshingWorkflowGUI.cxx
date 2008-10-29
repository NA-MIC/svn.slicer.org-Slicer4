/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMeshingWorkflowGUI.cxx,v $
Date:      $Date: 2006/03/17 15:10:10 $
Version:   $Revision: 1.2 $

=========================================================================auto=*/

#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"

#include "vtkMeshingWorkflowGUI.h"

#include "vtkCommand.h"
#include "vtkKWApplication.h"
#include "vtkKWWidget.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerApplicationLogic.h"
#include "vtkSlicerNodeSelectorWidget.h"
#include "vtkKWScaleWithEntry.h"
#include "vtkKWEntryWithLabel.h"
#include "vtkKWMenuButtonWithLabel.h"
#include "vtkKWMenuButton.h"
#include "vtkKWScale.h"
#include "vtkKWMenu.h"
#include "vtkKWEntry.h"
#include "vtkKWFrame.h"
#include "vtkSlicerApplication.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWPushButton.h"

#include "vtkSlicerModuleCollapsibleFrame.h"
#include "vtkRenderWindowInteractor.h"

// *** Declarations added for Univ. of Iowa Meshing Integration into Slicer3

// include declarations from Univ. of Iowa standalone meshing workflow GUI class hierarchy.  The
// original notebook uses locally-maintained linked lists.  The MRML notebook moves the storage into
// the MRML tree and keeps the same API for the client module. Change of code is minimized betweeen
// the standalone application and the slicer module.

//#include "vtkKWMenuButtonWithLabel.h"
#include "vtkKWMimxMainWindow.h"
#include "vtkKWMimxViewWindow.h"
#include "vtkKWRenderWidget.h"
#include "vtkMRMLScene.h"
#include "vtkMRMLViewNode.h"

//#include "vtkKWMimxMainNotebook.h"
//#include "vtkMeshingWorkflowMRMLNotebook.h"
#include "vtkKWMimxViewProperties.h"
#include "vtkKWMimxMainUserInterfacePanel.h"
#include "vtkLinkedListWrapperTree.h"

//------------------------------------------------------------------------------
vtkMeshingWorkflowGUI* vtkMeshingWorkflowGUI::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMeshingWorkflowGUI");
  if(ret)
    {
      return (vtkMeshingWorkflowGUI*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMeshingWorkflowGUI;
}


//----------------------------------------------------------------------------
vtkMeshingWorkflowGUI::vtkMeshingWorkflowGUI()
{

    this->Logic = NULL;
//    this->MimxMainNotebook = NULL;
//    this->ViewProperties = NULL;
//    this->DoUndoTree = NULL;
//    this->MainUserInterfacePanel = NULL;
//    this->DisplayPropertyDialog = NULL;

    // try to load supporting libraries dynamically.  This is needed
    // since the toplevel is a loadable module but the other libraries
    // didn't get loaded
    Tcl_Interp* interp = this->GetApplication()->GetMainInterp();
    Mimxcommon_Init(interp);
    Buildingblock_Init(interp);
}

//----------------------------------------------------------------------------
vtkMeshingWorkflowGUI::~vtkMeshingWorkflowGUI()
{

  if (this->Logic != NULL)
    {
    this->Logic->Delete();

    }

}

//----------------------------------------------------------------------------
void vtkMeshingWorkflowGUI::PrintSelf(ostream& os, vtkIndent indent)
{

}

//---------------------------------------------------------------------------
void vtkMeshingWorkflowGUI::AddGUIObservers ( )
{


    // look in the menu and add callbacks
    // these observers don't have to be added anymore because the existing BoundingBox GUI management code handles callbacks directly. Callbacks
    // are wrapped with the libmimxBoundingBox library and are invoked automatically

//    this->SavedMimxFEMenuGroup->ObjectMenuButton->GetWidget()->GetMenu()->AddObserver (vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
//    this->SavedMimxFEMenuGroup->OperationMenuButton->GetWidget()->GetMenu()->AddObserver (vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
//    this->SavedMimxFEMenuGroup->TypeMenuButton->GetWidget()->GetMenu()->AddObserver (vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
//    this->SavedMimxFEMenuGroup->ObjectMenuButton->GetWidget()->GetMenu()->AddObserver (vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
//    this->SavedMimxFEMenuGroup->OperationMenuButton->GetWidget()->GetMenu()->AddObserver (vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
//    this->SavedMimxFEMenuGroup->TypeMenuButton->GetWidget()->GetMenu()->AddObserver (vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    //this->ApplyButton->AddObserver (vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
}



//---------------------------------------------------------------------------
void vtkMeshingWorkflowGUI::RemoveGUIObservers ( )
{
    // Fill in
    //this->ApplyButton->RemoveObservers ( vtkCommand::ModifiedEvent,  (vtkCommand *)this->GUICallbackCommand );
}

//---------------------------------------------------------------------------
void vtkMeshingWorkflowGUI::ProcessGUIEvents ( vtkObject *caller,
                                           unsigned long event,
                                           void *callData )
{
  char tempstr[256], commandStr[128];
  vtkKWMenu *m = vtkKWMenu::SafeDownCast(caller);
  vtkKWPushButton *b = vtkKWPushButton::SafeDownCast(caller);

  cout << "FE callback received!" << endl;
//
  // process events on the object menu
//  if (b == this->ApplyButton && event == vtkKWPushButton::InvokedEvent )
//      {
//      this->BuildSeparateFEMeshGUI();
//      }

}


//---------------------------------------------------------------------------
void vtkMeshingWorkflowGUI::ProcessMrmlEvents ( vtkObject *caller,
                                            unsigned long event,
                                            void *callData )
{
  /**
  vtkMRMLMeshingWorkflowNode* node = dynamic_cast<vtkMRMLMeshingWorkflowNode *> (this->ApplicationLogic->GetMRMLScene()->GetNextNodeByClass("vtkMRMLMeshingWorkflowNode"));

  if (node) {
    this->SetMeshingWorkflowNode(node);
  }
  **/
}




//---------------------------------------------------------------------------
void vtkMeshingWorkflowGUI::BuildGUI ( )
{
  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();

  this->UIPanel->AddPage ( "MeshingWorkflow", "MeshingWorkflow", NULL );
  // ---
  // MODULE GUI FRAME
  // configure a page for a volume loading UI for now.
  // later, switch on the modulesButton in the SlicerControlGUI
  // ---

  // HELP FRAME
  vtkKWFrameWithLabel *helpFrame = vtkKWFrameWithLabel::New ( );
  helpFrame->SetParent ( this->UIPanel->GetPageWidget ( "MeshingWorkflow" ) );
  helpFrame->Create ( );
  helpFrame->CollapseFrame ( );
  helpFrame->SetLabelText ("Help");
  app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                helpFrame->GetWidgetName(), this->UIPanel->GetPageWidget("MeshingWorkflow")->GetWidgetName());

  vtkSlicerModuleCollapsibleFrame *moduleFrame = vtkSlicerModuleCollapsibleFrame::New ( );
  //vtkKWFrameWithLabel *moduleFrame = vtkKWFrameWithLabel::New ( );
  moduleFrame->SetParent ( this->UIPanel->GetPageWidget ( "MeshingWorkflow" ) );
  moduleFrame->Create ( );
  moduleFrame->SetLabelText ("MeshingWorkflow");
  moduleFrame->ExpandFrame ( );
  app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s -fill both",
                moduleFrame->GetWidgetName(), this->UIPanel->GetPageWidget("MeshingWorkflow")->GetWidgetName());
   
   // Create the MIMX Main Window.  This is a composite widget which serves as the top of the independent
   // meshing application.  This widget is created and the slicer render window and KWWindow are passed so that
   // widgets created below here are controllable through slicer's interface.
  
   this->MeshingUI = vtkKWMimxMainWindow::New();
   this->MeshingUI->SetRenderWidget( this->GetApplicationGUI()->GetViewerWidget()->GetMainViewer() );
   this->MeshingUI->SetMainWindow( this->GetApplicationGUI()->GetMainSlicerWindow() );
   this->MeshingUI->SetParent( moduleFrame );
   this->MeshingUI->Create();
   app->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s -fill both",
           this->MeshingUI->GetWidgetName(), moduleFrame->GetFrame()->GetWidgetName());

 

}



// Description:
// Describe behavior at module startup and exit.
 void vtkMeshingWorkflowGUI::Enter ( )
 {
//     // get pointers to the current scene.  Make a temporary scene to hold some
//     // state that we will restore when we exit the module
//    vtkMRMLScene *SlicerScene = vtkMRMLScene::GetActiveScene();
//    this->StoredMRMLState =  vtkMRMLScene::New();
//    //this->StoredMRMLState
//    
//    // save the 3D window selections (is the box on, the background color, etc.)
//    this->StoredMRMLState->CopyNode(SlicerScene->GetNthNodeByClass(0,"vtkMRMLView"));
//    // save the arrangement of slice windows and 3D window 
//    this->StoredMRMLState->CopyNode(SlicerScene->GetNthNodeByClass(0,"vtkMRMLLayout"));
//     
//    // turn off other MRML objects to have a clean interface for meshing
//    ((vtkMRMLViewNode*)(SlicerScene->GetNthNodeByClass(1,"vtkMRMLView")))->SetBoxVisible(0);
//    ((vtkMRMLViewNode*)(SlicerScene->GetNthNodeByClass(1,"vtkMRMLView")))->SetAxisLabelsVisible(0);
//    //((vtkMRMLViewNode*)(SlicerScene->GetNthNodeByClass(1,"vtkMRMLView")))->SetBackgroundColor(0.0,0.0,0.0);
//     
    // add the specific application settings for this module here
     this->MeshingUI->AddOrientationAxis();
       
 }
 
 
 
 void vtkMeshingWorkflowGUI::Exit ( )
 {
     // restore the MRML Scene state
//     vtkMRMLScene *SlicerScene = vtkMRMLScene::GetActiveScene();
//     SlicerScene->RemoveNodeNoNotify(SlicerScene->GetNthNodeByClass(1,"vtkMRMLView"));
//     SlicerScene->CopyNode(this->StoredMRMLState->GetNthNodeByClass(1,"vtkMRMLView"));
//  
        
    // remove the specific application settings for this module here
     this->MeshingUI->RemoveOrientationAxis();
         
     
 }
