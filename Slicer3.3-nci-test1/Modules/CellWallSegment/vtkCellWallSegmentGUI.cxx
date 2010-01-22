/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkCellWallSegmentGUI.cxx,v $
Date:      $Date: 2006/03/17 15:10:10 $
Version:   $Revision: 1.2 $

=========================================================================auto=*/

#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"

#include "vtkCellWallSegmentGUI.h"
#include "vtkCellWallVisSeg.h"
#include "vtkMRMLCellWallSegmentNode.h"
#include "vtkMRMLFiducialListNode.h"

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
#include "vtkKWFrameWithLabel.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerModuleCollapsibleFrame.h"
#include "vtkKWPushButton.h"
#include "vtkKWPushButtonSet.h"
#include "vtkKWFileBrowserDialog.h"
#include "vtkSlicerNodeSelectorWidget.h"

//------------------------------------------------------------------------------
vtkCellWallSegmentGUI* vtkCellWallSegmentGUI::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkCellWallSegmentGUI");
  if(ret)
    {
      return (vtkCellWallSegmentGUI*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkCellWallSegmentGUI;
}


//----------------------------------------------------------------------------
vtkCellWallSegmentGUI::vtkCellWallSegmentGUI()
{
  this->CellWallNodeSelector  = vtkSlicerNodeSelectorWidget::New();
  this->VolumeSelector = vtkSlicerNodeSelectorWidget::New();
  this->OutVolumeSelector = vtkSlicerNodeSelectorWidget::New();
  this->SegmentedVolumeSelector = vtkSlicerNodeSelectorWidget::New();
  this->TwoDButton = vtkKWPushButton::New();
  this->ThreeDButton = vtkKWPushButton::New();
  this->OpenFileButton = vtkKWPushButton::New();
  this->Logic = NULL;
  this->CellWallSegmentNode = NULL;
  this->FileBrowserDialog = NULL;
}

//----------------------------------------------------------------------------
vtkCellWallSegmentGUI::~vtkCellWallSegmentGUI()
{
 
    if ( this->VolumeSelector ) {
        this->VolumeSelector->SetParent(NULL);
        this->VolumeSelector->Delete();
        this->VolumeSelector = NULL;
    }
    if ( this->OutVolumeSelector ) {
        this->OutVolumeSelector->SetParent(NULL);
        this->OutVolumeSelector->Delete();
        this->OutVolumeSelector = NULL;
    }
    if ( this->SegmentedVolumeSelector ) {
         this->SegmentedVolumeSelector->SetParent(NULL);
         this->SegmentedVolumeSelector->Delete();
         this->SegmentedVolumeSelector = NULL;
     }
   if ( this->TwoDButton ) {
        this->TwoDButton->SetParent(NULL);
        this->TwoDButton->Delete();
        this->TwoDButton = NULL;
    }
   if ( this->ThreeDButton ) {
         this->ThreeDButton->SetParent(NULL);
         this->ThreeDButton->Delete();
         this->ThreeDButton = NULL;
     }
  this->SetLogic (NULL);
  vtkSetMRMLNodeMacro(this->CellWallSegmentNode, NULL);

}

//----------------------------------------------------------------------------
void vtkCellWallSegmentGUI::PrintSelf(ostream& os, vtkIndent indent)
{
  
}

//---------------------------------------------------------------------------
void vtkCellWallSegmentGUI::AddGUIObservers ( ) 
{
  this->VolumeSelector->AddObserver (vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->OutVolumeSelector->AddObserver (vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->SegmentedVolumeSelector->AddObserver (vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->FiducialListSelectorWidget->AddObserver (vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->TwoDButton->AddObserver (vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->ThreeDButton->AddObserver (vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->OpenFileButton->AddObserver (vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand ); 
}


//---------------------------------------------------------------------------
void vtkCellWallSegmentGUI::RemoveGUIObservers ( )
{
  this->VolumeSelector->RemoveObservers (vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->OutVolumeSelector->RemoveObservers (vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->SegmentedVolumeSelector->RemoveObservers (vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->TwoDButton->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
  this->ThreeDButton->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
}



//---------------------------------------------------------------------------
void vtkCellWallSegmentGUI::ProcessGUIEvents ( vtkObject *caller,
                                           unsigned long event,
                                           void *callData ) 
{
  
 
  vtkKWPushButton *b = vtkKWPushButton::SafeDownCast(caller);
  vtkSlicerNodeSelectorWidget *selector = vtkSlicerNodeSelectorWidget::SafeDownCast(caller);
 
  // check for events that specify the output volume to create
  if (selector == this->OutVolumeSelector && event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent  &&
    this->OutVolumeSelector->GetSelected() != NULL) 
    { 
    this->UpdateMRML();
    }
  
  // check for events that specify the segmented volume to create
   if (selector == this->SegmentedVolumeSelector && event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent  &&
     this->SegmentedVolumeSelector->GetSelected() != NULL) 
     { 
     this->UpdateMRML();
     }

  // check if the fiducial list selection has changed and poke the new ID into the MRML node for reference during the 
  // logic's execution method
    
//   if (selector == this->FiducialListSelectorWidget &&  event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent &&
//       this->FiducialListSelectorWidget->GetSelected() != NULL)
//     {
//       //vtkDebugMacro("vtkCellWallSegmentGUI: ProcessGUIEvent Node Selector Event: " << event << ".\n");
//       cout << "vtkCellWallSegmentGUI: ProcessGUIEvent Node Selector Event: " << event << endl;
//      vtkMRMLFiducialListNode *fidList = vtkMRMLFiducialListNode::SafeDownCast(this->FiducialListSelectorWidget->GetSelected());
//       this->CellWallSegmentNode->SetFiducialListRef(fidList->GetID());
//     }

   if (selector == this->FiducialListSelectorWidget &&
        event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent )
      {
       vtkDebugMacro("vtkCellWallSegmentGUI: ProcessGUIEvent Node Selector Event: " << event << ".\n");
       cout << "vtkCellWallSegmentGUI: ProcessGUIEvent Node Selector Event: " << event << endl;
       vtkMRMLFiducialListNode *fidList =
       vtkMRMLFiducialListNode::SafeDownCast(this->FiducialListSelectorWidget->GetSelected());
       if (fidList != NULL)
        {
          cout << "about to set MRML node fidlist" << endl;
          this->CellWallSegmentNode->SetFiducialListRef(fidList->GetID());
          cout << "completed set MRML node fidlist" << fidList->GetID() << "on MRML node: " << this->CellWallSegmentNode->GetID() << endl;
        }
       else
        {
        vtkDebugMacro("vtkCellWallSegmentGUI: ProcessGUIEvent: the selected node is null!");
        cout << "vtkCellWallSegmentGUI: ProcessGUIEvent: the selected node is null!" << endl;
        }
      return;
    }

   
  
  if (b == this->OpenFileButton && event == vtkKWPushButton::InvokedEvent ) 
     {
     this->FileSelectionCallback();
     }
  if (b == this->TwoDButton && event == vtkKWPushButton::InvokedEvent ) 
    {
      this->Logic->Perform2DSegmentation();
      this->UpdateMRML();
    }
  if (b == this->ThreeDButton && event == vtkKWPushButton::InvokedEvent ) 
     {
      this->Logic->Perform3DSegmentation();
      this->UpdateMRML();
     }
}


//---------------------------------------------------------------------------
void vtkCellWallSegmentGUI::UpdateMRML ()
{
    cout << "entering UpdateMRML" << endl;
    
  vtkMRMLCellWallSegmentNode* n = this->GetCellWallSegmentNode();
  if (n == NULL)
    {
    // set an observe new node in Logic
    this->Logic->SetAndObserveCellWallSegmentNode(n);
    vtkSetAndObserveMRMLNodeMacro(this->CellWallSegmentNode, n);
   }

  // save node parameters for Undo
  //this->GetLogic()->GetMRMLScene()->SaveStateForUndo(n);

  // set node parameters from GUI widgets
//  if (this->VolumeSelector->GetSelected() != NULL)
//    {
//    //n->SetInputVolumeRef(this->VolumeSelector->GetSelected()->GetID());
//    }

  if (this->OutVolumeSelector->GetSelected() != NULL)
    {
      n->SetOutputVolumeRef(this->OutVolumeSelector->GetSelected()->GetID());
    }

  if (this->SegmentedVolumeSelector->GetSelected() != NULL)
    {
      n->SetSegmentationVolumeRef(this->SegmentedVolumeSelector->GetSelected()->GetID());
    }
  
  
}

//---------------------------------------------------------------------------
void vtkCellWallSegmentGUI::UpdateGUI ()
{
  vtkMRMLCellWallSegmentNode* n = this->GetCellWallSegmentNode();
  if (n != NULL)
    {
  
    }
}

//---------------------------------------------------------------------------
void vtkCellWallSegmentGUI::ProcessMRMLEvents ( vtkObject *caller,
                                            unsigned long event,
                                            void *callData ) 
{
  // if parameter node has been changed externally, update GUI widgets with new values
  vtkMRMLCellWallSegmentNode* node = vtkMRMLCellWallSegmentNode::SafeDownCast(caller);
  if (node != NULL && this->GetCellWallSegmentNode() == node) 
    {
    this->UpdateGUI();
    }
}




//---------------------------------------------------------------------------
void vtkCellWallSegmentGUI::BuildGUI ( ) 
{
  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();

  vtkMRMLCellWallSegmentNode* gadNode = vtkMRMLCellWallSegmentNode::New();
  this->Logic->GetMRMLScene()->RegisterNodeClass(gadNode);
  this->Logic->GetMRMLScene()->AddNode(gadNode);
    this->Logic->SetAndObserveCellWallSegmentNode(gadNode);
    vtkSetAndObserveMRMLNodeMacro( this->CellWallSegmentNode, gadNode);
    gadNode->Delete();

  this->UIPanel->AddPage ( "CellWallSegment", "CellWallSegment", NULL );
  // ---
  // MODULE GUI FRAME 
  // ---
   // Define your help text and build the help frame here.
    const char *help = "The CellWallSegment module....";
    const char *about = "This work was supported by NCi, NA-MIC, NAC, BIRN, NCIGT, and the Slicer Community. See <a>http://www.slicer.org</a> for details. ";
    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "CellWallSegment" );
    this->BuildHelpAndAboutFrame ( page, help, about );
    
  vtkSlicerModuleCollapsibleFrame *moduleFrame = vtkSlicerModuleCollapsibleFrame::New ( );
  moduleFrame->SetParent ( this->UIPanel->GetPageWidget ( "CellWallSegment" ) );
  moduleFrame->Create ( );
  moduleFrame->SetLabelText ("Cell Wall Segmentation");
  moduleFrame->ExpandFrame ( );
  app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                moduleFrame->GetWidgetName(), this->UIPanel->GetPageWidget("CellWallSegment")->GetWidgetName());
  
  
    this->CellWallNodeSelector->SetNodeClass("vtkMRMLCellWallSegmentNode", NULL, NULL, "CellWallSegmentationParameters"); 
    this->CellWallNodeSelector->SetNewNodeEnabled(1);
    this->CellWallNodeSelector->NoneEnabledOn();
    this->CellWallNodeSelector->SetShowHidden(1);
    this->CellWallNodeSelector->SetParent( moduleFrame->GetFrame() );
    this->CellWallNodeSelector->Create();
    this->CellWallNodeSelector->SetMRMLScene(this->Logic->GetMRMLScene());
    this->CellWallNodeSelector->UpdateMenu();

    this->CellWallNodeSelector->SetBorderWidth(2);
    this->CellWallNodeSelector->SetLabelText( "Cell Wall Parameters");
    this->CellWallNodeSelector->SetBalloonHelpString("select a CellWallSegmentation node from the current mrml scene.");
    app->Script("pack %s -side top -anchor e -padx 20 -pady 4", 
                  this->CellWallNodeSelector->GetWidgetName());
  
  
  this->VolumeSelector->SetNodeClass("vtkMRMLScalarVolumeNode", NULL, NULL, NULL);
  this->VolumeSelector->SetParent( moduleFrame->GetFrame() );
  this->VolumeSelector->Create();
  this->VolumeSelector->SetMRMLScene(this->Logic->GetMRMLScene());
  this->VolumeSelector->UpdateMenu();

    //---------------------------------------------------
    //  create an instance of the Cell Wall Segmentation Algorithm
    //---------------------------------------------------
  
  
  this->OutVolumeSelector->SetNodeClass("vtkMRMLScalarVolumeNode", NULL, NULL, "CellWallSegmentInputVolume");
  this->OutVolumeSelector->SetNewNodeEnabled(1);
  this->OutVolumeSelector->SetParent( moduleFrame->GetFrame() );
  this->OutVolumeSelector->Create();
  this->OutVolumeSelector->SetMRMLScene(this->Logic->GetMRMLScene());
  this->OutVolumeSelector->UpdateMenu();

  this->OutVolumeSelector->SetBorderWidth(2);
  this->OutVolumeSelector->SetLabelText( "Output Volume: ");
  this->OutVolumeSelector->SetBalloonHelpString("select an output volume from the current mrml scene.");
  app->Script("pack %s -side top -anchor e -padx 20 -pady 4", 
                this->OutVolumeSelector->GetWidgetName());
  
  this->SegmentedVolumeSelector->SetNodeClass("vtkMRMLScalarVolumeNode", NULL, NULL, "CellWallSegmentOutputVolume");
   this->SegmentedVolumeSelector->SetNewNodeEnabled(1);
   this->SegmentedVolumeSelector->SetParent( moduleFrame->GetFrame() );
   this->SegmentedVolumeSelector->Create();
   this->SegmentedVolumeSelector->SetMRMLScene(this->Logic->GetMRMLScene());
   this->SegmentedVolumeSelector->UpdateMenu();

   this->SegmentedVolumeSelector->SetBorderWidth(2);
   this->SegmentedVolumeSelector->SetLabelText( "Segmented Volume: ");
   this->SegmentedVolumeSelector->SetBalloonHelpString("select an segmentation volume from the current mrml scene.");
   app->Script("pack %s -side top -anchor e -padx 20 -pady 4", 
                 this->SegmentedVolumeSelector->GetWidgetName());

  this->OpenFileButton->SetParent( moduleFrame->GetFrame() );
  this->OpenFileButton->Create();
  this->OpenFileButton->SetText("Open ICS File");
  this->OpenFileButton->SetWidth ( 24 );
  app->Script("pack %s -side top -anchor e -padx 20 -pady 10", 
                 this->OpenFileButton->GetWidgetName());
  
  
  // node selector
    this->FiducialListSelectorWidget = vtkSlicerNodeSelectorWidget::New();
    this->FiducialListSelectorWidget->SetParent(moduleFrame->GetFrame());
    this->FiducialListSelectorWidget->Create();
    this->FiducialListSelectorWidget->SetNodeClass("vtkMRMLFiducialListNode", NULL, NULL, NULL);
    this->FiducialListSelectorWidget->NewNodeEnabledOn();
    this->FiducialListSelectorWidget->SetMRMLScene(this->GetMRMLScene());
    this->FiducialListSelectorWidget->SetBorderWidth(2);
    this->FiducialListSelectorWidget->SetPadX(2);
    this->FiducialListSelectorWidget->SetPadY(2);
    //this->FiducialListSelectorWidget->GetWidget()->IndicatorVisibilityOff();
    this->FiducialListSelectorWidget->GetWidget()->SetWidth(24);
    this->FiducialListSelectorWidget->SetLabelText( "Fiducial List: ");
    this->FiducialListSelectorWidget->SetBalloonHelpString("Select a fiducial list from the current mrml scene.");
    this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                  this->FiducialListSelectorWidget->GetWidgetName());
  
  this->TwoDButton->SetParent( moduleFrame->GetFrame() );
  this->TwoDButton->Create();
  this->TwoDButton->SetText("2D Segmentation from Selected List");
  this->TwoDButton->SetWidth ( 38 );
  app->Script("pack %s -side top -anchor e -padx 20 -pady 10", 
                this->TwoDButton->GetWidgetName());
  

  this->ThreeDButton->SetParent( moduleFrame->GetFrame() );
  this->ThreeDButton->Create();
  this->ThreeDButton->SetText("3D Segmentation of 2D Boundary");
  this->ThreeDButton->SetWidth ( 38 );
  app->Script("pack %s -side top -anchor e -padx 20 -pady 10", 
                this->ThreeDButton->GetWidgetName());

  moduleFrame->Delete();

  
}

//---------------------------------------------------------------------------
void vtkCellWallSegmentGUI::FileSelectionCallback ( ) 
{
   if(!this->FileBrowserDialog)
   {
        this->FileBrowserDialog = vtkKWFileBrowserDialog::New();
        this->FileBrowserDialog->SetApplication(this->GetApplication());
        this->FileBrowserDialog->Create();
    }
    this->FileBrowserDialog->SetDefaultExtension(".ics");
    this->FileBrowserDialog->SetFileTypes("{{ICS files} {.ics}}");
    this->FileBrowserDialog->RetrieveLastPathFromRegistry("LastPath");
    this->FileBrowserDialog->Invoke();
    if(this->FileBrowserDialog->GetStatus() == vtkKWDialog::StatusOK)
    {
        //vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
        //callback->SetState(0);

        if(!this->FileBrowserDialog->GetFileName())
        {
                vtkErrorMacro("File name not chosen");
                return;
        }

        char *filename = FileBrowserDialog->GetFileName();
        
        this->Logic->GetCellWallVisSeg()->readImage(filename);
        this->Logic->GetCellWallVisSeg()->afterLoadingInit();
        this->Logic->InitializeMRMLVolume(filename);
 
        
//        this->GetApplication()->SetRegistryValue(
//                1, "RunTime", "LastPath", vtksys::SystemTools::GetFilenamePath( filename ).c_str());
//        this->FileBrowserDialog->SaveLastPathToRegistry("LastPath");
    }
}
