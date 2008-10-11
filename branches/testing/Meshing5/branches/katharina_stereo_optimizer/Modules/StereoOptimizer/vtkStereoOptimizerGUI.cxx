/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkStereoOptimizerGUI.cxx,v $
Date:      $Date: 2006/03/17 15:10:10 $
Version:   $Revision: 1.2 $

=========================================================================auto=*/

#include "vtkObjectFactory.h"
#include "vtkStereoOptimizerGUI.h"
#include "vtkCommand.h"

#include "vtkKWApplication.h"
#include "vtkKWWidget.h"
#include "vtkKWLabel.h"
#include "vtkSlicerModuleCollapsibleFrame.h"
#include "vtkKWScale.h"
#include "vtkKWScaleWithEntry.h"
#include "vtkKWThumbWheel.h"
#include "vtkKWNotebook.h" 
#include "vtkKWRenderWidget.h"

#include "vtkSlicerTheme.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerApplicationLogic.h"
#include "vtkSlicerNodeSelectorWidget.h"

#include "vtkRenderer.h"
//#include "vtkKWScaleWithEntry.h"
//#include "vtkKWEntryWithLabel.h"
//#include "vtkKWMenuButtonWithLabel.h"
//#include "vtkKWMenuButton.h"
//#include "vtkKWScale.h"
//#include "vtkKWMenu.h"
//#include "vtkKWEntry.h"
//#include "vtkKWFrame.h"

//#include "vtkKWPushButton.h"
//#include "vtkKWMultiColumnList.h"
//#include "vtkMRMLStereoOptimizerNode.h"
//#include "vtkKWLoadSaveButton.h"
//#include "vtkKWProgressGauge.h"
//#include "vtkSlicerWindow.h" 

//------------------------------------------------------------------------------
vtkStereoOptimizerGUI* vtkStereoOptimizerGUI::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkStereoOptimizerGUI");
  if(ret)
    {
      return (vtkStereoOptimizerGUI*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkStereoOptimizerGUI;
}


//----------------------------------------------------------------------------
vtkStereoOptimizerGUI::vtkStereoOptimizerGUI()
{
  this->ViewAngleScale = vtkKWScaleWithEntry::New();
  this->EyeAngleScale = vtkKWScaleWithEntry::New();
  this->PlaneOfProjectionScale = vtkKWScaleWithEntry::New();
  this->MaxParallaxScale = vtkKWScaleWithEntry::New();
  this->UserEyeDistanceScale = vtkKWScaleWithEntry::New();
  this->PushBehindScreenButton = vtkKWPushButton::New();
  this->OptimizeDepthButton = vtkKWPushButton::New();
  this->ResetButton = vtkKWPushButton::New();
  this->ViewDistanceWheel = vtkKWThumbWheel::New();
  this->ScreenWidthWheel = vtkKWThumbWheel::New();
  this->Logic = NULL;
  this->StereoOptimizerNode = NULL;
  this->SideViewsRenderPending = 0;
  this->SideViewWidget = NULL;
  this->BirdsViewWidget = NULL;
}

//----------------------------------------------------------------------------
vtkStereoOptimizerGUI::~vtkStereoOptimizerGUI()
{
  this->SideViewsRenderPending = 0;
  if ( this->ViewAngleScale ) 
    {
    this->ViewAngleScale->SetParent(NULL);
    this->ViewAngleScale->Delete();
    this->ViewAngleScale = NULL;
    }
  if ( this->EyeAngleScale ) 
    {
    this->EyeAngleScale->SetParent(NULL);
    this->EyeAngleScale->Delete();
    this->EyeAngleScale = NULL;
    }
  if ( this->PlaneOfProjectionScale ) 
    {
    this->PlaneOfProjectionScale->SetParent(NULL);
    this->PlaneOfProjectionScale->Delete();
    this->PlaneOfProjectionScale = NULL;
    }
 if ( this->MaxParallaxScale ) 
    {
    this->MaxParallaxScale->SetParent(NULL);
    this->MaxParallaxScale->Delete();
    this->MaxParallaxScale = NULL;
    }
  if ( this->UserEyeDistanceScale ) 
    {
    this->UserEyeDistanceScale->SetParent(NULL);
    this->UserEyeDistanceScale->Delete();
    this->UserEyeDistanceScale = NULL;
    }
  if ( this->OptimizeDepthButton ) 
    {
    this->OptimizeDepthButton->SetParent(NULL);
    this->OptimizeDepthButton->Delete();
    this->OptimizeDepthButton = NULL;
    }
  if ( this->PushBehindScreenButton ) 
    {
    this->PushBehindScreenButton->SetParent(NULL);
    this->PushBehindScreenButton->Delete();
    this->PushBehindScreenButton = NULL;
    }
  if ( this->ResetButton ) 
    {
    this->ResetButton->SetParent(NULL);
    this->ResetButton->Delete();
    this->ResetButton = NULL;
    }
  if ( this->ScreenWidthWheel ) 
    {
    this->ScreenWidthWheel->SetParent(NULL);
    this->ScreenWidthWheel->Delete();
    this->ScreenWidthWheel = NULL;
    }
  if ( this->ViewDistanceWheel ) 
    {
    this->ViewDistanceWheel->SetParent(NULL);
    this->ViewDistanceWheel->Delete();
    this->ViewDistanceWheel = NULL;
    }
  this->SetLogic (NULL);
  vtkSetMRMLNodeMacro(this->StereoOptimizerNode, NULL);
}

//----------------------------------------------------------------------------
void vtkStereoOptimizerGUI::PrintSelf(ostream& os, vtkIndent indent)
{ 
  //TODO: need to print everything
  this->vtkObject::PrintSelf ( os, indent );
  
  // eventuall these get moved into the view node...
  os << indent << "StereoOptimizerGUI: " << this->GetClassName ( ) << "\n";
  
  // class widgets
  os << indent << "SideViewsRenderPending: " << this->SideViewsRenderPending << "\n"; 
}

//---------------------------------------------------------------------------
void vtkStereoOptimizerGUI::AddGUIObservers ( ) 
{
  this->ViewAngleScale->AddObserver (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->EyeAngleScale->AddObserver (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->PlaneOfProjectionScale->AddObserver (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );

  this->PushBehindScreenButton->AddObserver (vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->OptimizeDepthButton->AddObserver (vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->ResetButton->AddObserver (vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );

  //TODO: add and remove observers for ScrrenWidthWheel and ViewDistanceWheel MaxParallaxScale  UserEyeDistanceScale
  //  this->Logic->AddObserver (vtkStereoOptimizerLogic::LabelStatsOuterLoop, (vtkCommand *)this->LogicCallbackCommand );
  //this->Logic->AddObserver (vtkStereoOptimizerLogic::LabelStatsInnerLoop, (vtkCommand *)this->LogicCallbackCommand );
  //this->Logic->AddObserver (vtkStereoOptimizerLogic::StartLabelStats, (vtkCommand *)this->LogicCallbackCommand );
  //this->Logic->AddObserver (vtkStereoOptimizerLogic::EndLabelStats, (vtkCommand *)this->LogicCallbackCommand );
}

//---------------------------------------------------------------------------
void vtkStereoOptimizerGUI::RemoveGUIObservers ( )
{
  this->ViewAngleScale->RemoveObservers (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->EyeAngleScale->RemoveObservers (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );  
  this->PlaneOfProjectionScale->RemoveObservers (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );

  this->PushBehindScreenButton->RemoveObservers (vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->OptimizeDepthButton->RemoveObservers (vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->ResetButton->RemoveObservers (vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );

 //  this->GrayscaleSelector->RemoveObservers (vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );  

//   this->LabelmapSelector->RemoveObservers (vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );  

//   this->PlaneOfProjectionScale->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );

  // this->SaveToClipboardButton->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );

  //this->SaveToFile->RemoveObservers ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );

//   this->Logic->RemoveObservers (vtkStereoOptimizerLogic::LabelStatsOuterLoop, (vtkCommand *)this->LogicCallbackCommand );

//   this->Logic->RemoveObservers (vtkStereoOptimizerLogic::LabelStatsInnerLoop, (vtkCommand *)this->LogicCallbackCommand );

//   this->Logic->RemoveObservers (vtkStereoOptimizerLogic::StartLabelStats, (vtkCommand *)this->LogicCallbackCommand );

//   this->Logic->RemoveObservers (vtkStereoOptimizerLogic::EndLabelStats, (vtkCommand *)this->LogicCallbackCommand );

}


//---------------------------------------------------------------------------
void vtkStereoOptimizerGUI::ProcessGUIEvents ( vtkObject *caller,
                                           unsigned long event,
                                           void *callData ) 
{
  vtkKWScaleWithEntry *s = vtkKWScaleWithEntry::SafeDownCast(caller);
  vtkKWPushButton *b = vtkKWPushButton::SafeDownCast(caller);
  //vtkSlicerNodeSelectorWidget *selector = vtkSlicerNodeSelectorWidget::SafeDownCast(caller);
  
 //  if (selector == this->GrayscaleSelector && event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent &&
//     this->GrayscaleSelector->GetSelected() != NULL) 
//     { 
//     this->UpdateMRML();
//     }
//   else if (selector == this->LabelmapSelector && event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent  &&
//     this->LabelmapSelector->GetSelected() != NULL) 
//     { 
//     this->UpdateMRML();
//     }
 if (s == this->ViewAngleScale && event == vtkKWScale::ScaleValueChangedEvent ) 
    {
       std::cout << "ViewAngleScale is changed to: "<<  this->ViewAngleScale->GetValue() << "\n";
    // this->ResultList->DeleteAllRows();
//     this->UpdateMRML();
//     this->Logic->Apply();
    }
 if (s == this->EyeAngleScale && event == vtkKWScale::ScaleValueChangedEvent ) 
    {
       std::cout << "EyeAngleScale is changed to: "<<  this->EyeAngleScale->GetValue() << "\n";
    }
 if (s == this->PlaneOfProjectionScale && event == vtkKWScale::ScaleValueChangedEvent ) 
    {
       std::cout << "PlaneOfProjectionScale is changed to: "<<  this->PlaneOfProjectionScale->GetValue() << "\n";
    }
 if (b == this->PushBehindScreenButton && event == vtkKWPushButton::InvokedEvent ) 
    {
      std::cout << "Push Behind the screen!\n";
    }
 if (b == this->OptimizeDepthButton && event == vtkKWPushButton::InvokedEvent ) 
    {
      std::cout << "Optimize depth!\n";
    }
 if (b == this->ResetButton && event == vtkKWPushButton::InvokedEvent ) 
   { 
      std::cout << "Reset view!\n";
   }
//  if (b == this->SaveToFile && event == vtkKWPushButton::InvokedEvent ) 
//    {
//      vtkKWLoadSaveButton *saveLoadButton = vtkKWLoadSaveButton::SafeDownCast(caller);
//      const char *fileName = saveLoadButton->GetFileName();
//      if ( fileName ) 
//        {
//          std::cout << "This is the filename: "<<  this->SaveToFile->GetFileName() << "\n";
//          vtkMRMLStereoOptimizerNode* n = this->GetStereoOptimizerNode();
//          n->SaveResultToTextFile(fileName);
//        }
//    }
}

//---------------------------------------------------------------------------
void vtkStereoOptimizerGUI::UpdateMRML ()
{
  std::cout <<"UpdateMRML gets called!" << "\n";
  // vtkMRMLStereoOptimizerNode* n = this->GetStereoOptimizerNode();
 //  if (n == NULL)
//     {
//     //no parameter node selected yet, create new
//     vtkMRMLStereoOptimizerNode* volumeMathNode = vtkMRMLStereoOptimizerNode::New();
//     n = volumeMathNode;
//     //set an observe new node in Logic
//     this->Logic->SetAndObserveStereoOptimizerNode(volumeMathNode);
//     vtkSetAndObserveMRMLNodeMacro(this->StereoOptimizerNode, volumeMathNode);
//     }
  
  // save node parameters for Undo
  //  this->GetLogic()->GetMRMLScene()->SaveStateForUndo(n);
  // set node parameters from GUI widgets
 //  if (this->GrayscaleSelector->GetSelected() != NULL)
//     {
//     n->SetInputGrayscaleRef(this->GrayscaleSelector->GetSelected()->GetID());
//     }

//   if (this->LabelmapSelector->GetSelected() != NULL)
//     {
//     n->SetInputLabelmapRef(this->LabelmapSelector->GetSelected()->GetID());
//     }
}

//---------------------------------------------------------------------------
void vtkStereoOptimizerGUI::UpdateGUI ()
{ 
  std::cout <<"UpdateGUI gets called!" << "\n";
  // vtkMRMLStereoOptimizerNode* n = this->GetStereoOptimizerNode();
  // if (n != NULL)
//     {
//       // this->VolStatsResult->SetText(n->GetResultText());
//     if(!n->LabelStats.empty()) 
//       { 
//       typedef std::list<vtkMRMLStereoOptimizerNode::LabelStatsEntry>::const_iterator LI;
//         int i = 0;
//         for (LI li = n->LabelStats.begin(); li != n->LabelStats.end(); ++li)
//           {
//            const vtkMRMLStereoOptimizerNode::LabelStatsEntry& label = *li;  
//            //  std::cout << "This is i: " << i <<std::endl;
//            // std::cout << "Label: " << label.Label << "\tMin: " << label.Min ;
//            // std::cout << "\tMax: " << label.Max << "\tMean: " << label.Mean << std::endl ;
           
//            this->ResultList->InsertCellTextAsInt(i, 0, label.Label);
//            this->ResultList->InsertCellTextAsInt(i, 1, label.Count);
           
//            this->ResultList->InsertCellTextAsInt(i, 2, label.Min);
//            this->ResultList->InsertCellTextAsInt(i, 3, label.Max);
//            this->ResultList->InsertCellTextAsDouble(i, 4, label.Mean);
//            this->ResultList->InsertCellTextAsDouble(i, 5, label.StdDev);
//            i++;
//           }
//       }
//     }
}

//---------------------------------------------------------------------------
void vtkStereoOptimizerGUI::ProcessMRMLEvents ( vtkObject *caller,
                                            unsigned long event,
                                            void *callData ) 
{
std::cout <<"ProcessMRMLEvents gets called!" << "\n";
// if parameter node has been changed externally, update GUI widgets with new values
 // vtkMRMLStereoOptimizerNode* node = vtkMRMLStereoOptimizerNode::SafeDownCast(caller);
//  if (node != NULL && this->GetStereoOptimizerNode() == node) 
//    {
//    this->UpdateGUI();
//    }
}

//---------------------------------------------------------------------------
void vtkStereoOptimizerGUI::BuildGUI ( ) 
{
  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
  // vtkMRMLStereoOptimizerNode* n = vtkMRMLStereoOptimizerNode::New();
  //this->Logic->GetMRMLScene()->RegisterNodeClass(n);
  //n->Delete();

  this->UIPanel->AddPage ( "StereoOptimizer", "StereoOptimizer", NULL );
  // ---
  // MODULE GUI FRAME 
  // ---
   // Define your help text and build the help frame here.
  const char *help = "The StereoOptimizer module....";
  const char *about = "This work was supported by NA-MIC, NAC, BIRN, NCIGT, and the Slicer Community. See http://www.slicer.org for details. ";
  vtkKWWidget *page = this->UIPanel->GetPageWidget ( "StereoOptimizer" );
  this->BuildHelpAndAboutFrame ( page, help, about );
 
  vtkSlicerModuleCollapsibleFrame *moduleSettingsFrame = vtkSlicerModuleCollapsibleFrame::New ( );
  moduleSettingsFrame->SetParent ( this->UIPanel->GetPageWidget ( "StereoOptimizer" ) );
  moduleSettingsFrame->Create ( );
  moduleSettingsFrame->CollapseFrame ( );
  moduleSettingsFrame->SetLabelText ("Settings");
  // moduleSettingsFrame->ExpandFrame ( );
  app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                moduleSettingsFrame->GetWidgetName(), this->UIPanel->GetPageWidget("StereoOptimizer")->GetWidgetName());

  // create the notebook
  vtkKWNotebook *notebook = vtkKWNotebook::New();
  notebook->SetParent ( moduleSettingsFrame->GetFrame() );
  notebook->Create();
  this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                 notebook->GetWidgetName());

  // create a tab for general and advanced
  notebook->AddPage ( "General", "Adapt module to specific user and to specific stereo system used.", NULL );
  notebook->AddPage ( "Advanced", "Change module logic.", NULL );
  
  ViewDistanceWheel->SetParent(notebook->GetFrame("General"));
  ViewDistanceWheel->PopupModeOn();
  ViewDistanceWheel->Create();
  ViewDistanceWheel->SetLength(150);
  ViewDistanceWheel->DisplayEntryOn();
  ViewDistanceWheel->DisplayLabelOn();
  ViewDistanceWheel->GetLabel()->SetText("View Distance to the Screen in cm:");
  app->Script("pack %s -side top -anchor nw -expand n -padx 2 -pady 2", 
              ViewDistanceWheel->GetWidgetName());
  
  ScreenWidthWheel->SetParent(notebook->GetFrame("General"));
  ScreenWidthWheel->PopupModeOn();  
  ScreenWidthWheel->Create();
  ScreenWidthWheel->SetLength(150);
  ScreenWidthWheel->DisplayEntryOn();
  ScreenWidthWheel->DisplayLabelOn();
  ScreenWidthWheel->GetLabel()->SetText("Screen Width in cm:");
  app->Script("pack %s -side top -anchor nw -expand n -padx 2 -pady 2", 
              ScreenWidthWheel->GetWidgetName());
  
  MaxParallaxScale->SetParent( notebook->GetFrame("Advanced") );
  MaxParallaxScale->Create();
  MaxParallaxScale->SetRange(1.0, 6.0);
  MaxParallaxScale->SetResolution(0.1);
  MaxParallaxScale->SetValue(1.5);
  //  scale2->GetScale()->SetLength(350);
  MaxParallaxScale->RangeVisibilityOn();
  MaxParallaxScale->SetLabelText("Maximum Disparity in Degrees:");
  MaxParallaxScale->SetBalloonHelpString(
    "This changes the maximum disparity that the tool will believe is comfortable to view."
    "For a rule of thumb the literature suggests a value of 1.5 degrees");

  app->Script(
    "pack %s -side top -anchor nw -expand n -fill none -padx 20 -pady 10", 
    MaxParallaxScale->GetWidgetName());

  UserEyeDistanceScale->SetParent( notebook->GetFrame("Advanced") );
  UserEyeDistanceScale->Create();
  UserEyeDistanceScale->SetRange(5.0, 8.0);
  UserEyeDistanceScale->SetResolution(0.1);
  UserEyeDistanceScale->SetValue(6.0);
  //  scale2->GetScale()->SetLength(350);
  UserEyeDistanceScale->RangeVisibilityOn();
  UserEyeDistanceScale->SetLabelText("User Eye Distance in cm:");
  UserEyeDistanceScale->SetBalloonHelpString(
    "Adapt module to physiology of specific user."
    "The Eye Distance would be measured with a ruler and is defined by the distance of the user's pupils.");

  app->Script(
    "pack %s -side top -anchor nw -expand n -fill none -padx 20 -pady 10", 
    UserEyeDistanceScale->GetWidgetName());

  vtkSlicerModuleCollapsibleFrame *moduleFrame = vtkSlicerModuleCollapsibleFrame::New ( );
  moduleFrame->SetParent ( this->UIPanel->GetPageWidget ( "StereoOptimizer" ) );
  moduleFrame->Create ( );
  moduleFrame->SetLabelText ("Stereo Camera Control");
  // moduleFrame->ExpandFrame ( );
  app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                moduleFrame->GetWidgetName(), this->UIPanel->GetPageWidget("StereoOptimizer")->GetWidgetName());
  
  ViewAngleScale->SetParent( moduleFrame->GetFrame() );
  ViewAngleScale->Create();
  ViewAngleScale->SetRange(-1.0, 1.0);
  ViewAngleScale->SetResolution(0.1);
  //  scale2->GetScale()->SetLength(350);
  ViewAngleScale->RangeVisibilityOn();
  ViewAngleScale->SetLabelText("Perspective scale:");
  ViewAngleScale->SetBalloonHelpString(
    "Changes the view angle "
    "-1: very much telephoto"
    "1: very wide angle");

  app->Script(
    "pack %s -side top -anchor nw -expand n -fill none -padx 20 -pady 10", 
    ViewAngleScale->GetWidgetName());

  EyeAngleScale->SetParent( moduleFrame->GetFrame() );
  EyeAngleScale->Create();
  EyeAngleScale->SetRange(-2.0, 2.0);
  EyeAngleScale->SetResolution(0.1);
  //  scale2->GetScale()->SetLength(350);
  EyeAngleScale->RangeVisibilityOn();
  EyeAngleScale->SetLabelText("Eye angle scale:");
  EyeAngleScale->SetBalloonHelpString(
    "Explain what this scale does "
    "Changes the eye angle "
    "-2: (Hypostereoscopic view) stereobase very small: less stereoscopic depth, shallower"
    "2: (Hyperstereoscopic view) stereobase quite large: more stereoscopic depth, deeper");

  app->Script(
    "pack %s -side top -anchor nw -expand n -fill none -padx 20 -pady 10", 
    EyeAngleScale->GetWidgetName());


  PlaneOfProjectionScale->SetParent( moduleFrame->GetFrame() );
  PlaneOfProjectionScale->Create();
  PlaneOfProjectionScale->SetRange(-1.0, 1.0);
  PlaneOfProjectionScale->SetResolution(0.1);
  //  scale2->GetScale()->SetLength(350);
  PlaneOfProjectionScale->RangeVisibilityOn();
  PlaneOfProjectionScale->SetLabelText("Plane of Projection scale:");
  PlaneOfProjectionScale->SetBalloonHelpString(
    "Explain what this sclae does "
    "Changes location of plane of projection "
    "-1: object in front of display, very close"
    "1: opbject behind the display, farther away");

  app->Script(
    "pack %s -side top -anchor nw -expand n -fill none -padx 20 -pady 10", 
    PlaneOfProjectionScale->GetWidgetName());


  vtkSlicerModuleCollapsibleFrame *moduleOptimizerFrame = vtkSlicerModuleCollapsibleFrame::New ( );
  moduleOptimizerFrame->SetParent ( this->UIPanel->GetPageWidget ( "StereoOptimizer" ) );
  moduleOptimizerFrame->Create ( );
  moduleOptimizerFrame->SetLabelText ("Optimize Stereo View");
  moduleOptimizerFrame->ExpandFrame ( );
  app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                moduleOptimizerFrame->GetWidgetName(), this->UIPanel->GetPageWidget("StereoOptimizer")->GetWidgetName());
   
  this->OptimizeDepthButton->SetParent( moduleOptimizerFrame->GetFrame() );
  this->OptimizeDepthButton->Create();
  this->OptimizeDepthButton->SetText("Optimize Depth");
  this->OptimizeDepthButton->SetWidth ( 16 );
  app->Script("pack %s -side top -anchor e -padx 20 -pady 10", 
              this->OptimizeDepthButton->GetWidgetName());

  this->PushBehindScreenButton->SetParent( moduleOptimizerFrame->GetFrame() );
  this->PushBehindScreenButton->Create();
  this->PushBehindScreenButton->SetText("Push Scene behind screen");
  this->PushBehindScreenButton->SetWidth ( 26 );
  app->Script("pack %s -side top -anchor e -padx 20 -pady 10", 
              this->PushBehindScreenButton->GetWidgetName());
 
  this->ResetButton->SetParent( moduleOptimizerFrame->GetFrame() );
  this->ResetButton->Create();
  this->ResetButton->SetText("Reset");
  this->ResetButton->SetWidth ( 7 );
  app->Script("pack %s -side top -anchor e -padx 20 -pady 10", 
              this->ResetButton->GetWidgetName());

  moduleFrame->Delete();
  moduleSettingsFrame->Delete();
  moduleOptimizerFrame->Delete();


  //Side View Frame and Widget
  vtkSlicerModuleCollapsibleFrame *moduleSideViewFrame = vtkSlicerModuleCollapsibleFrame::New ( );
  moduleSideViewFrame->SetParent ( this->UIPanel->GetPageWidget ( "StereoOptimizer" ) );
  moduleSideViewFrame->Create ( );
  moduleSideViewFrame->SetLabelText ("Side View");
  moduleSideViewFrame->ExpandFrame ( );
  app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                 moduleSideViewFrame->GetWidgetName(), this->UIPanel->GetPageWidget("StereoOptimizer")->GetWidgetName());

  // Create a render widget
 this->SideViewWidget = vtkKWRenderWidget::New( );

 // vtkKWRenderWidget *ideViewWidget = vtkKWRenderWidget::New();
 this->SideViewWidget->SetParent(moduleSideViewFrame->GetFrame());
 this->SideViewWidget->Create();
 this->SideViewWidget->SetRendererBackgroundColor ( app->GetSlicerTheme()->GetSlicerColors()->ViewerBlue );
 //this->SideViewWidget->SetWidth ( this->NavigationZoomWidgetWid );
 //this->SideViewWidget->SetHeight ( this->NavigationZoomWidgetHit );
 this->SideViewWidget->GetRenderWindow()->AddRenderer(this->SideViewWidget->GetRenderer() );
 this->SideViewWidget->GetRenderWindow()->DoubleBufferOn();
 this->SideViewWidget->GetRenderer()->GetRenderWindow()->GetInteractor()->Disable();

  app->Script("pack %s -side top -fill both -expand y -padx 0 -pady 0", 
              this->SideViewWidget->GetWidgetName());
 
  //Bird's View Frame and Widget
  vtkSlicerModuleCollapsibleFrame *moduleBirdsViewFrame = vtkSlicerModuleCollapsibleFrame::New ( );
  moduleBirdsViewFrame->SetParent ( this->UIPanel->GetPageWidget ( "StereoOptimizer" ) );
  moduleBirdsViewFrame->Create ( );
  moduleBirdsViewFrame->SetLabelText ("Bird's View");
  moduleBirdsViewFrame->ExpandFrame ( );
  app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                 moduleBirdsViewFrame->GetWidgetName(), this->UIPanel->GetPageWidget("StereoOptimizer")->GetWidgetName());

  // Create a render widget
 this->BirdsViewWidget = vtkKWRenderWidget::New( );
 this->BirdsViewWidget->SetParent(moduleBirdsViewFrame->GetFrame());
 this->BirdsViewWidget->Create();
 this->BirdsViewWidget->SetRendererBackgroundColor ( app->GetSlicerTheme()->GetSlicerColors()->ViewerBlue );
 //this->BirdsViewWidget->SetWidth ( this->NavigationZoomWidgetWid );
 //this->BirdsViewWidget->SetHeight ( this->NavigationZoomWidgetHit );
 this->BirdsViewWidget->GetRenderWindow()->AddRenderer(this->BirdsViewWidget->GetRenderer() );
 this->BirdsViewWidget->GetRenderWindow()->DoubleBufferOn();
 this->BirdsViewWidget->GetRenderer()->GetRenderWindow()->GetInteractor()->Disable();

  app->Script("pack %s -side top -fill both -expand y -padx 0 -pady 0", 
               this->BirdsViewWidget->GetWidgetName());

// app->Script(
//     "pack %s -side top -anchor nw -expand n -fill none -padx 2 -pady 6", 
//     ViewAngleScale->GetWidgetName());

//   this->GrayscaleSelector->SetNodeClass("vtkMRMLScalarVolumeNode", NULL, NULL, NULL);
//   this->GrayscaleSelector->SetParent( moduleFrame->GetFrame() );
//   this->GrayscaleSelector->Create();
//   this->GrayscaleSelector->SetMRMLScene(this->Logic->GetMRMLScene());
//   this->GrayscaleSelector->UpdateMenu();

//   this->GrayscaleSelector->SetBorderWidth(2);
//   this->GrayscaleSelector->SetLabelText( "Input Grayscale Volume: ");
//   this->GrayscaleSelector->SetBalloonHelpString("Select an input grayscale volume from the current mrml scene.");
//   app->Script("pack %s -side top -anchor e -padx 20 -pady 4", 
//                 this->GrayscaleSelector->GetWidgetName());
  
//   this->LabelmapSelector->SetNodeClass("vtkMRMLScalarVolumeNode", NULL, NULL, NULL);
//   this->LabelmapSelector->SetParent( moduleFrame->GetFrame() );
//   this->LabelmapSelector->Create();
//   this->LabelmapSelector->SetMRMLScene(this->Logic->GetMRMLScene());
//   this->LabelmapSelector->UpdateMenu();

//   this->LabelmapSelector->SetBorderWidth(2);
//   this->LabelmapSelector->SetLabelText( "Input Labelmap: ");
//   this->LabelmapSelector->SetBalloonHelpString("Select an input labelmap from the current mrml scene.");
//   app->Script("pack %s -side top -anchor e -padx 20 -pady 4", 
//                 this->LabelmapSelector->GetWidgetName());


//   this->ApplyButton->SetParent( moduleFrame->GetFrame() );
//   this->ApplyButton->Create();
//   this->ApplyButton->SetText("Apply");
//   this->ApplyButton->SetWidth ( 8 );
//   app->Script("pack %s -side top -anchor e -padx 20 -pady 10", 
//                 this->ApplyButton->GetWidgetName());

//   this->ResultList->SetParent( moduleFrame->GetFrame());
//   this->ResultList->Create();

//   this->ResultList->SetWidth(0);
//   this->ResultList->SetHeight(7);
  
//   int col_index;

//   // Add the columns (make some of them editable)

//   col_index = this->ResultList->AddColumn("Label");
//   this->ResultList->ColumnEditableOn(col_index);

//   col_index = this->ResultList->AddColumn("Count");
//   this->ResultList->ColumnEditableOn(col_index);

//   col_index = this->ResultList->AddColumn("Min");
//   this->ResultList->ColumnEditableOn(col_index);
  
//   col_index = this->ResultList->AddColumn("Max");
//   this->ResultList->ColumnEditableOn(col_index);
  
//   col_index = this->ResultList->AddColumn("Mean");
//   this->ResultList->ColumnEditableOn(col_index);
  
//   col_index = this->ResultList->AddColumn("StdDev");
//   this->ResultList->ColumnEditableOn(col_index);

//   app->Script(
//     "pack %s -side top -anchor e  -padx 20 -pady 10", 
//     this->ResultList->GetWidgetName());
 
//   // Create the button to copy result to clipboard
//   this->SaveToClipboardButton->SetParent( moduleFrame->GetFrame() );
//   this->SaveToClipboardButton->Create();
//   this->SaveToClipboardButton->SetText("Copy result to clipboard");
//   this->SaveToClipboardButton->SetWidth ( 28 );

//   this->SaveToFile->SetParent( moduleFrame->GetFrame() );
//   this->SaveToFile->Create();
//   this->SaveToFile->SetText("Save to file");
//   this->SaveToFile->GetLoadSaveDialog()->SaveDialogOn(); // load mode
  
//   this->SaveToFile->GetLoadSaveDialog()->SetFileTypes("{{Text Document} {.txt}}");
//   this->SaveToFile->GetLoadSaveDialog()->SetInitialFileName("LabelStatistics.txt"); 
//   this->SaveToFile->GetLoadSaveDialog()->SetDefaultExtension("txt");

//   app->Script(
//     "pack %s %s -side right -anchor w  -padx 20 -pady 10", 
//     this->SaveToClipboardButton->GetWidgetName(),
//     this->SaveToFile->GetWidgetName());

//   this->SaveToFile->Delete();

  ///--------

}

void vtkStereoOptimizerGUI::ProcessLogicEvents ( vtkObject *caller,
                                                  unsigned long event,
                                                  void *callData)
{
 //  vtkStereoOptimizerLogic* logic =  vtkStereoOptimizerLogic::SafeDownCast(caller);
//   const char * callDataStr = (const char *)callData;
  
//   std::string innerLoopMsg = "Computing Stats for ";

//   vtkSlicerWindow* mainWindow = this->ApplicationGUI->GetMainSlicerWindow();
//   vtkKWProgressGauge* progressGauge =  mainWindow->GetProgressGauge(); 
 
//   if (event == vtkStereoOptimizerLogic::StartLabelStats)
//     {
//       std::cout << "StartLabelStats\n"<< "\n";
//       progressGauge->SetValue(0);
//       progressGauge->SetNthValue(1, 0);

//       mainWindow->SetStatusText("Start calculating ...");
//     } 
//   else if (event == vtkStereoOptimizerLogic::EndLabelStats)
//     {
//       std::cout << "EndLabelStats\n"<< "\n";
//       mainWindow->SetStatusText("Done");
//     }
//   else if (event == vtkStereoOptimizerLogic::LabelStatsOuterLoop) 
//     {
//       std::cout << "LabelStatsOuterLoop\n"<< "\n";
//       std::cout << "This is the progress in GUI: "<< logic->GetProgress() << " .\n";
//       progressGauge->SetValue(logic->GetProgress()*100);
//       mainWindow->SetStatusText(innerLoopMsg.append( callDataStr ).c_str() );

//     } 
//   else if (event == vtkStereoOptimizerLogic::LabelStatsInnerLoop)  
//     {
//       std::cout << "LabelStatsInnerLoop\n"<< "\n";
//       std::stringstream ss ( callDataStr );
//       double innerProg = 0;
//       ss >> innerProg;
//       progressGauge->SetNthValue(1,innerProg*100);
//     }
//   else 
//     {
//       std::cout << "Event: "<< event << " is not handled here.\n";
//     }
  

}
