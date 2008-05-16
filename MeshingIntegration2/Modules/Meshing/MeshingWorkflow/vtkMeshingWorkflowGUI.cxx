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

// *** Declarations added for Univ. of Iowa Meshing Integration into Slicer3

// include declarations from Univ. of Iowa standalone meshing workflow GUI class hierarchy.  The
// original notebook uses locally-maintained linked lists.  The MRML notebook moves the storage into
// the MRML tree and keeps the same API for the client module. Change of code is minimized betweeen
// the standalone application and the slicer module. 

//#include "vtkKWMenuButtonWithLabel.h"
#include "vtkKWMimxMainWindow.h"
#include "vtkKWMimxViewWindow.h"
#include "vtkKWRenderWidget.h"

//#include "vtkKWMimxMainNotebook.h"
#include "vtkMeshingWorkflowMRMLNotebook.h"
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
    this->MimxMainNotebook = NULL;
    this->ViewProperties = NULL;
    this->DoUndoTree = NULL;
  this->MainUserInterfacePanel = NULL;
    this->DisplayPropertyDialog = NULL;
    
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
//  char tempstr[256], commandStr[128];
//  vtkKWMenu *m = vtkKWMenu::SafeDownCast(caller);
//  cout << "FE callback received!" << endl;
//    
//  // process events on the object menu
//  if ( m == this->SavedMimxFEMenuGroup->ObjectMenuButton->GetWidget()->GetMenu() ) 
//  {
//      cout << "callback on FE ObjectMenuButton item " << endl;
//      for (int i=0; i<m->GetNumberOfItems(); i++)
//      {
//          cout << "item " << i << "has value: " << m->GetItemSelectedState(i) << "command: " << m->GetItemCommand(i) << endl;
//          strncpy(tempstr,m->GetItemCommand(i),255); 
//          strtok(tempstr," "); 
//          strncpy(commandStr,strtok(NULL,"\0"),127);
//          cout << "command isolated was:" << commandStr << endl;
//          if(!(strcmp(commandStr,"BBMenuCallback")) && m->GetItemSelectedState(i)) {this->SavedMimxFEMenuGroup->BBMenuCallback(); break;}
//          if(!(strcmp(commandStr,"BBMeshSeedMenuCallback")) && m->GetItemSelectedState(i)) {this->SavedMimxFEMenuGroup->BBMeshSeedMenuCallback(); break;}
//          if(!(strcmp(commandStr,"FEMeshMenuCallback")) && m->GetItemSelectedState(i)) {this->SavedMimxFEMenuGroup->FEMeshMenuCallback(); break;}
//      }
//    
//  // process events on the operation menu       
//  } else if (m == this->SavedMimxFEMenuGroup->OperationMenuButton->GetWidget()->GetMenu() )
//  {
//     
//      cout << "callback on FE OperationMenuButton item " << endl;
//      for (int i=0; i<m->GetNumberOfItems(); i++)
//      {
//          strncpy(tempstr,m->GetItemCommand(i),255); 
//          strtok(tempstr," "); 
//          strncpy(commandStr,strtok(NULL,"\0"),127);
//          if(!(strcmp(commandStr,"LoadBBCallback")) && m->GetItemSelectedState(i)) {this->SavedMimxFEMenuGroup->LoadBBCallback(); break;}
//          if(!(strcmp(commandStr,"SaveBBCallback")) && m->GetItemSelectedState(i)) {this->SavedMimxFEMenuGroup->SaveBBCallback(); break;}
//          if(!(strcmp(commandStr,"CreateBBCallback")) && m->GetItemSelectedState(i)) {this->SavedMimxFEMenuGroup->CreateBBCallback(); break;}
//          if(!(strcmp(commandStr,"DeleteBBCallback")) && m->GetItemSelectedState(i)) {this->SavedMimxFEMenuGroup->DeleteBBCallback(); break;}
//          if(!(strcmp(commandStr,"EditBBCallback")) && m->GetItemSelectedState(i)) {this->SavedMimxFEMenuGroup->EditBBCallback(); break;}
//      }
//
//  // process operations on the type menu
//  } else if (m == this->SavedMimxFEMenuGroup->TypeMenuButton->GetWidget()->GetMenu() )
//  {
//      cout << "callback on FE TypeMenuButton item " << endl;
//      for (int i=0; i<m->GetNumberOfItems(); i++)
//      {
//          strncpy(tempstr,m->GetItemCommand(i),255); 
//          strtok(tempstr," "); 
//          strncpy(commandStr,strtok(NULL,"\0"),127);
//          if(!(strcmp(commandStr,"LoadVTKBBCallback")) && m->GetItemSelectedState(i)) {this->SavedMimxFEMenuGroup->LoadVTKBBCallback(); break;}
//          if(!(strcmp(commandStr,"CreateBBFromBoundsCallback")) && m->GetItemSelectedState(i)) {this->SavedMimxFEMenuGroup->CreateBBFromBoundsCallback(); break;}
//          if(!(strcmp(commandStr,"CreateFEMeshFromBBCallback")) && m->GetItemSelectedState(i)) {this->SavedMimxFEMenuGroup->CreateFEMeshFromBBCallback(); break;}
//          if(!(strcmp(commandStr,"SaveVTKBBCallback")) && m->GetItemSelectedState(i)) {this->SavedMimxFEMenuGroup->SaveVTKBBCallback(); break;}
//          if(!(strcmp(commandStr,"SmoothLaplacianFEMeshCallback")) && m->GetItemSelectedState(i)) {this->SavedMimxFEMenuGroup->SmoothLaplacianFEMeshCallback(); break;}
//      }
//  }

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

  vtkKWFrameWithLabel *moduleFrame = vtkKWFrameWithLabel::New ( );
  moduleFrame->SetParent ( this->UIPanel->GetPageWidget ( "MeshingWorkflow" ) );
  moduleFrame->Create ( );
  moduleFrame->SetLabelText ("MeshingWorkflow");
  moduleFrame->ExpandFrame ( );
  app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                moduleFrame->GetWidgetName(), this->UIPanel->GetPageWidget("MeshingWorkflow")->GetWidgetName());
  
  
 
  // create a shell of the MimxViewWindow for class compability.  This class only
  // points to an instance of the slicer viewer, but encapsulates the integration with 
  // the slicer viewer.  The renderWindow is initialized to be the main 3D window from slicer. 

  //vtkKWMimxViewWindow *viewwin = vtkKWMimxViewWindow::New();
  vtkKWMimxMainWindow *mainwin = vtkKWMimxMainWindow::New();
  mainwin->SetRenderWidget(this->GetApplicationGUI()->GetViewerWidget()->GetMainViewer());
  
    // create the undo tree 
    if(!this->DoUndoTree)
                this->DoUndoTree = vtkLinkedListWrapperTree::New();
  
  // create the view properties 
  
        this->ViewProperties = vtkKWMimxViewProperties::New();
        this->ViewProperties->SetParent(this->UIPanel->GetPageWidget ( "MeshingWorkflow" ));
        this->ViewProperties->SetMimxMainWindow(mainwin);
        this->ViewProperties->Create();
        this->ViewProperties->SetBorderWidth(3);
        this->ViewProperties->SetReliefToGroove();
        this->ViewProperties->SetDoUndoTree(this->DoUndoTree);
        this->GetApplication()->Script("pack %s -side top -anchor nw -expand n -padx 2 -pady 5 -fill x", 
             this->ViewProperties->GetMainFrame()->GetWidgetName());
        
        
//            if(!this->MainUserInterfacePanel)
//            {
//                    this->MainUserInterfacePanel = vtkKWMimxMainUserInterfacePanel::New();
//                    this->MainUserInterfacePanel->SetMimxMainWindow(mainwin);
//                    this->MainUserInterfacePanel->SetDoUndoTree(this->DoUndoTree);
//                    this->MainUserInterfacePanel->SetMultiColumnList(
//                            this->ViewProperties->GetMultiColumnList());
//            }
//            this->MainUserInterfacePanel->SetParent(this->UIPanel->GetPageWidget ( "MeshingWorkflow" ));
//            this->MainUserInterfacePanel->SetApplication(this->GetApplication());
//            this->MainUserInterfacePanel->Create();
//            this->MainUserInterfacePanel->SetBorderWidth(3);
//            this->MainUserInterfacePanel->SetReliefToGroove();
//            this->MainUserInterfacePanel->GetMainFrame()->ExpandFrame();
//            //      this->MainNotebook->SetWidth(200);
//            this->GetApplication()->Script(
//                    "pack %s -side top -anchor nw -expand n -fill y -pady 2 -fill x", 
//                    this->MainUserInterfacePanel->GetWidgetName());
//            this->ViewProperties->SetViewPropertiesGroup(
//                    this->MainUserInterfacePanel->GetViewPropertiesGroup());

        
        
  
  // create the notebook which is the root of the pre-developed meshing workflow
  //this->MimxMainNotebook = vtkKWMimxMainNotebook::New();
  this->MimxMainNotebook = vtkMeshingWorkflowMRMLNotebook::New();
  // pass in the current application MRML scene, so the nodes for storage will be stored in the scene
  //`this->MimxMainNotebook->SetMRMLSceneForStorage(this->ApplicationLogic->GetMRMLScene());
  this->MimxMainNotebook->SetParent ( this->UIPanel->GetPageWidget ( "MeshingWorkflow" ) );
  this->MimxMainNotebook->SetApplication(this->GetApplication());
  this->MimxMainNotebook->SetMimxMainWindow(mainwin);
  //this->MimxMainNotebook->SetMimxViewWindow(viewwin);
  this->MimxMainNotebook->Create ( );
  this->MimxMainNotebook->SetWidth(200);
  app->Script (
       "pack %s -side top -anchor nw  -expand y -padx 0 -pady 1", 
       this->MimxMainNotebook->GetWidgetName());
  


}
