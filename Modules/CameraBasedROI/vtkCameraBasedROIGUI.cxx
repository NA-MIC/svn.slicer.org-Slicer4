
#include "vtkSlicerApplication.h"

#include "vtkCameraBasedROIGUI.h"

// vtkSlicer includes
#include "vtkSlicerModuleCollapsibleFrame.h"
#include "vtkSlicerNodeSelectorWidget.h"
#include "vtkSlicerApplicationGUI.h"

// KWWidgets includes
#include "vtkKWApplication.h"
#include "vtkKWWidget.h"
#include "vtkKWFrame.h"
#include "vtkKWScaleWithEntry.h"
#include "vtkKWScale.h"

// STL includes
#include <string>
#include <iostream>
#include <sstream>
#include <map>
#include <vector>
#include <iterator>

// VTK includes
#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkCommand.h"


//------------------------------------------------------------------------------
vtkCameraBasedROIGUI* vtkCameraBasedROIGUI::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkCameraBasedROIGUI");
  if(ret)
    {
      return (vtkCameraBasedROIGUI*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkCameraBasedROIGUI;
}


//----------------------------------------------------------------------------
vtkCameraBasedROIGUI::vtkCameraBasedROIGUI()
{
  this->UpdatingMRML = 0;
  this->UpdatingGUI = 0;

  this->Logic = NULL;
    
  this->CameraBasedROINode = NULL;
  this->CameraNode = NULL;

  this->SpecificationFrame = NULL;
  this->ParameterSelector = NULL;
  this->CameraSelector = NULL;
  this->ROISelector = NULL;
  this->DistanceScale = NULL;
  this->SizeScale = NULL;

  // Try to load supporting libraries dynamically.  This is needed
  // since the toplevel is a loadable module but the other libraries
  // didn't get loaded
  
  Tcl_Interp* interp = this->GetApplication()->GetMainInterp();
  if (interp)
    {
    Vtkslicercamerabasedroimodulelogic_Init(interp);
    //VtkslicerCameraBasedROImodulelogic_Init(interp);
    }
  else
    {
    vtkErrorMacro("Failed to obtain reference to application TCL interpreter");
    }
    
}

//----------------------------------------------------------------------------
vtkCameraBasedROIGUI::~vtkCameraBasedROIGUI()
{
    this->RemoveGUIObservers ( );
    this->RemoveMRMLObservers ( );

    this->UpdatingMRML = 0;
    this->UpdatingGUI = 0;

    if ( this->ParameterSelector )
      {
      this->ParameterSelector->SetParent (NULL );
      this->ParameterSelector->Delete();
      this->ParameterSelector = NULL;
      }
    if ( this->CameraSelector )
      {
      this->CameraSelector->SetParent (NULL );
      this->CameraSelector->Delete();
      this->CameraSelector = NULL;
      }
    if ( this->ROISelector )
      {
      this->ROISelector->SetParent (NULL );
      this->ROISelector->Delete();
      this->ROISelector = NULL;
      }
    if ( this->DistanceScale )
      {
      this->DistanceScale->SetParent (NULL );
      this->DistanceScale->Delete();
      this->DistanceScale = NULL;
      }
    if ( this->SizeScale )
      {
      this->SizeScale->SetParent (NULL );
      this->SizeScale->Delete();
      this->SizeScale = NULL;
      }
    if ( this->SpecificationFrame )
      {
      this->SpecificationFrame->SetParent (NULL );
      this->SpecificationFrame->Delete();
      this->SpecificationFrame = NULL;
      }

    this->Raised = false;

    if ( this->Logic )
      {
      this->Logic->Delete();
      this->Logic = NULL;
      }

//    this->SetAndObserveMRMLScene ( NULL );    
}



//----------------------------------------------------------------------------
void vtkCameraBasedROIGUI::Enter()
{
  
  
  //--- mark as currently being visited.
  this->Raised = true;

  //--- mark as visited at least once.
  this->Visited = true;

  //--- only build when first visited.
  if ( this->Built == false )
    {
    this->BuildGUI();
    this->AddObserver ( vtkSlicerModuleGUI::ModuleSelectedEvent, (vtkCommand *)this->ApplicationGUI->GetGUICallbackCommand() );
    this->AddMRMLObservers();    
    }

  this->AddGUIObservers();    
  this->CreateModuleEventBindings();

  //--- make GUI reflect current MRML state
  this->UpdateGUI();
}


//----------------------------------------------------------------------------
void vtkCameraBasedROIGUI::Exit ( )
{

  //--- mark as no longer selected.
  this->Raised = false;

  this->RemoveGUIObservers();
  this->ReleaseModuleEventBindings();
//  this->SetAndObserveMRMLScene ( NULL );
  
}


//----------------------------------------------------------------------------
void vtkCameraBasedROIGUI::AddMRMLObservers()
{
  if ( !this->Visited )
    {
    return;
    }
  
  vtkIntArray *events = vtkIntArray::New();
  events->InsertNextValue(vtkMRMLScene::NodeAddedEvent);
  events->InsertNextValue(vtkMRMLScene::NodeRemovedEvent);
  //events->InsertNextValue(vtkMRMLScene::SceneCloseEvent);
  // Slicer3.cxx calls delete on events
  this->SetAndObserveMRMLSceneEvents ( this->MRMLScene, events );
  events->Delete();
}


//----------------------------------------------------------------------------
void vtkCameraBasedROIGUI::TearDownGUI ( )
{
  if ( !this->Built )
    {
    return;
    }

  this->RemoveObservers ( vtkSlicerModuleGUI::ModuleSelectedEvent, (vtkCommand *)this->ApplicationGUI->GetGUICallbackCommand() );
  this->RemoveGUIObservers ( );
  this->ParameterSelector->SetMRMLScene ( NULL );
  this->CameraSelector->SetMRMLScene ( NULL );
  this->ROISelector->SetMRMLScene ( NULL );
  this->ReleaseModuleEventBindings();
  this->RemoveMRMLObservers ( );
}


//----------------------------------------------------------------------------
void vtkCameraBasedROIGUI::PrintSelf(ostream& os, vtkIndent indent)
{
  Superclass::PrintSelf(os, indent);
}


//---------------------------------------------------------------------------
void vtkCameraBasedROIGUI::AddGUIObservers ( ) 
{
  if ( !this->Built )
    {
    return;
    }

  //--- include this to enable lazy building
  if ( !this->Visited )
    {
    return;
    }

  if ( this->ParameterSelector )
    {
    if  (this->MRMLScene != NULL )
      {
      this->ParameterSelector->SetMRMLScene ( this->MRMLScene );
      }
    this->ParameterSelector->AddObserver ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent,
                                             ( vtkCommand *) this->GUICallbackCommand );
    this->ParameterSelector->UpdateMenu();
    }

  if ( this->CameraSelector )
    {
    if  (this->MRMLScene != NULL )
      {
      this->CameraSelector->SetMRMLScene ( this->MRMLScene );
      }
    this->CameraSelector->AddObserver ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent,
                                             ( vtkCommand *) this->GUICallbackCommand );
    this->CameraSelector->UpdateMenu();
    }


  if ( this->ROISelector )
    {
    if  (this->MRMLScene != NULL )
      {
      this->ROISelector->SetMRMLScene ( this->MRMLScene );
      }
    this->ROISelector->AddObserver ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent,
                                             ( vtkCommand *) this->GUICallbackCommand );
    this->ROISelector->UpdateMenu();
    }

  if (this->DistanceScale)
    {
    this->DistanceScale->AddObserver (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    }

  if (this->SizeScale)
    {
    this->SizeScale->AddObserver (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    }
}



//---------------------------------------------------------------------------
void vtkCameraBasedROIGUI::RemoveGUIObservers ( )
{
  if ( !this->Built )
    {
    return;
    }

  //--- include this to enable lazy building
  if ( !this->Visited )
    {
    return;
    }
  if ( this->ParameterSelector  )
    {
    this->ParameterSelector->SetMRMLScene ( NULL );
    this->ParameterSelector ->RemoveObservers ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent,
                                            ( vtkCommand *)this->GUICallbackCommand );
    }
  if ( this->CameraSelector  )
    {
    this->CameraSelector->SetMRMLScene ( NULL );
    this->CameraSelector ->RemoveObservers ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent,
                                            ( vtkCommand *)this->GUICallbackCommand );
    }
  if ( this->ROISelector  )
    {
    this->ROISelector->SetMRMLScene ( NULL );
    this->ROISelector ->RemoveObservers ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent,
                                            ( vtkCommand *)this->GUICallbackCommand );
    }
  if (this->DistanceScale)
    {
    this->DistanceScale->RemoveObservers (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    }

  if (this->SizeScale)
    {
    this->SizeScale->RemoveObservers (vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    }

}




//---------------------------------------------------------------------------
void vtkCameraBasedROIGUI::RemoveMRMLObservers ( ) {
    vtkSetAndObserveMRMLNodeMacro(this->CameraBasedROINode, NULL);
    this->SetAndObserveMRMLScene( NULL );
}


//---------------------------------------------------------------------------
void vtkCameraBasedROIGUI::ProcessGUIEvents ( vtkObject *caller,
                                           unsigned long event,
                                           void *vtkNotUsed(callData))
{

  if ( !this->Built )
    {
    return;
    }
  if (this->CameraBasedROINode == NULL)
    {
    this->CreateParameterNode();
    this->UpdateParameterNode();
    }
  if ( caller == this->ParameterSelector && event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent )
    {
    vtkMRMLCameraBasedROINode *param = vtkMRMLCameraBasedROINode::SafeDownCast ( this->ParameterSelector->GetSelected() );
    vtkSetAndObserveMRMLNodeMacro(this->CameraBasedROINode, param);
    }
  else if ( caller == this->CameraSelector && event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent )
    {
    vtkMRMLCameraNode *camera = vtkMRMLCameraNode::SafeDownCast ( this->CameraSelector->GetSelected() );
    vtkSetAndObserveMRMLNodeMacro(this->CameraNode, camera);
    if (this->CameraBasedROINode) 
      {
      this->CameraBasedROINode->SetCameraNodeID(camera->GetID());
      }
    }
  else if ( caller == this->ROISelector && event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent )
    {
    vtkMRMLROINode *roi = vtkMRMLROINode::SafeDownCast ( this->ROISelector->GetSelected() );
    if (this->CameraBasedROINode) 
      {
      this->CameraBasedROINode->SetROINodeID(roi->GetID());
      }
    }

  if ( this->DistanceScale && caller == this->DistanceScale && event == vtkKWScale::ScaleValueChangedEvent ) 
    {
    this->CameraBasedROINode->SetROIDistanceToCamera(this->DistanceScale->GetWidget()->GetValue());
    }

  if ( this->SizeScale && caller == this->SizeScale && event == vtkKWScale::ScaleValueChangedEvent ) 
    {
    this->CameraBasedROINode->SetROISize(this->SizeScale->GetWidget()->GetValue());
    }

}





//---------------------------------------------------------------------------
void vtkCameraBasedROIGUI::ProcessMRMLEvents(vtkObject *caller,
                                          unsigned long event,
                                          void *callData)
{
  if ( !this->Raised )
    {
    return;
    }
  if ( !this->Visited )
    {
    return;
    }

  if ( this->ApplicationGUI == NULL)
    {
    vtkErrorMacro ("ProcessMRMLEvents: ApplicationGUI is null");
    return;    
    }
  if (event == vtkMRMLScene::NodeAddedEvent)
    {
    vtkMRMLNode *node = (vtkMRMLNode*) (callData);
    if (node != NULL && node->IsA("vtkMRMLCameraBasedROINode") )
      {
      vtkMRMLCameraBasedROINode *pnode = vtkMRMLCameraBasedROINode::SafeDownCast(node);
      if (this->CameraBasedROINode == NULL)
        {
        this->CreateParameterNode();
        this->UpdateParameterNode();
        }
      }
    }

  if (event == vtkMRMLScene::NodeRemovedEvent)
    {
    vtkMRMLNode *node = (vtkMRMLNode*) (callData);
    if (node != NULL && node->IsA("vtkMRMLCameraBasedROINode") )
      {
      vtkMRMLCameraBasedROINode *pnode = vtkMRMLCameraBasedROINode::SafeDownCast(node);
      if (this->CameraBasedROINode == pnode)
        {
        vtkSetAndObserveMRMLNodeMacro(this->CameraNode, NULL);
        vtkSetAndObserveMRMLNodeMacro(this->CameraBasedROINode, NULL);
        }
      }
    }

  if (this->CameraBasedROINode == NULL)
    {
    this->CreateParameterNode();
    this->UpdateParameterNode();
    }

  this->Logic->UpdateROI(this->CameraBasedROINode );
  this->UpdateGUI();
  //vtkMRMLScene *scene = vtkMRMLScene::SafeDownCast ( caller );
}


//---------------------------------------------------------------------------
void vtkCameraBasedROIGUI::UpdateGUI ()
{
  if ( !this->Built && !this->MRMLScene )
    {
    return;
    }
  
  // update from MRML
  if ( this->UpdatingMRML )
    {
    return;
    }
  if ( this->UpdatingGUI )
    {
    return;
    }
  
  this->UpdatingGUI = 1;

  if (this->ParameterSelector && this->CameraBasedROINode)
    {
    this->ParameterSelector->SetSelected(this->CameraBasedROINode);
    }
  if (this->CameraSelector && this->CameraBasedROINode)
    {
    this->CameraSelector->SetSelected(this->MRMLScene->GetNodeByID(
        this->CameraBasedROINode->GetCameraNodeID()));
    }
  if (this->ROISelector && this->CameraBasedROINode)
    {
    this->ROISelector->SetSelected(this->MRMLScene->GetNodeByID(
        this->CameraBasedROINode->GetROINodeID()));
    }
  if (this->DistanceScale && this->CameraBasedROINode)
    {
    this->DistanceScale->GetWidget()->SetValue(this->CameraBasedROINode->GetROIDistanceToCamera());
    }
  if (this->SizeScale && this->CameraBasedROINode)
    {
    this->SizeScale->GetWidget()->SetValue(this->CameraBasedROINode->GetROISize());
    }

  this->UpdatingGUI = 0;
}






//---------------------------------------------------------------------------
void vtkCameraBasedROIGUI::BuildGUI ( ) 
{
  //--- include this to enable lazy building
  if ( !this->Visited )
    {
    return;
    }
  
  vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast (this->GetApplication() );
  if ( !app )
    {
    vtkErrorMacro ( "BuildGUI: got Null SlicerApplication" );
    return;
    }
  vtkSlicerApplicationGUI *appGUI = app->GetApplicationGUI();
  if ( !appGUI )
    {
    vtkErrorMacro ( "BuildGUI: got Null SlicerApplicationGUI" );
    return;
    }
  vtkSlicerWindow *win = appGUI->GetMainSlicerWindow ();
  if ( win == NULL )
    {
    vtkErrorMacro ( "BuildGUI: got NULL MainSlicerWindow");
    return;
    }
  win->SetStatusText ( "Building Interface for CameraBasedROI Module...." );
  app->Script ( "update idletasks" );


  if ( this->MRMLScene != NULL )
    {
    vtkMRMLCameraBasedROINode* m = vtkMRMLCameraBasedROINode::New();
    this->MRMLScene->RegisterNodeClass(m);
    m->Delete();
    }
  else
    {
    vtkErrorMacro("GUI is being built before MRML Scene is set");
    return;
    }
      
  this->UIPanel->AddPage ( "CameraBasedROI", "CameraBasedROI", NULL );

  // HELP FRAME
  const char* about = "CameraBasedROI was developed by Alex Yarmarkovich. This work was supported by NA-MIC, NAC, BIRN, NCIGT, Harvard CTSC, and the Slicer Community. See <a>http://www.slicer.org</a> for details.\n";
  
  const char *help = "**CameraBasedROI** is a module for placing an ROI based on the camera which can be transformed.  \n\n **Usage:** To use this module, select an existing camera from the scene, select a ROI node.\n\n";

  vtkKWWidget *page = this->UIPanel->GetPageWidget ( "CameraBasedROI" );
  this->BuildHelpAndAboutFrame ( page, help, about );
  vtkKWLabel *NACLabel = vtkKWLabel::New();
  NACLabel->SetParent ( this->GetLogoFrame() );
  NACLabel->Create();
  NACLabel->SetImageToIcon ( this->GetAcknowledgementIcons()->GetNACLogo() );

  vtkKWLabel *NAMICLabel = vtkKWLabel::New();
  NAMICLabel->SetParent ( this->GetLogoFrame() );
  NAMICLabel->Create();
  NAMICLabel->SetImageToIcon ( this->GetAcknowledgementIcons()->GetNAMICLogo() );    

  vtkKWLabel *NCIGTLabel = vtkKWLabel::New();
  NCIGTLabel->SetParent ( this->GetLogoFrame() );
  NCIGTLabel->Create();
  NCIGTLabel->SetImageToIcon ( this->GetAcknowledgementIcons()->GetNCIGTLogo() );
    
  vtkKWLabel *BIRNLabel = vtkKWLabel::New();
  BIRNLabel->SetParent ( this->GetLogoFrame() );
  BIRNLabel->Create();
  BIRNLabel->SetImageToIcon ( this->GetAcknowledgementIcons()->GetBIRNLogo() );

  vtkKWLabel *CTSCLabel = vtkKWLabel::New();
  CTSCLabel->SetParent ( this->GetLogoFrame() );
  CTSCLabel->Create();
  CTSCLabel->SetImageToIcon (this->GetAcknowledgementIcons()->GetCTSCLogo() );


  app->Script ( "grid %s -row 0 -column 0 -padx 2 -pady 2 -sticky e", NAMICLabel->GetWidgetName());
  app->Script ("grid %s -row 0 -column 1 -padx 2 -pady 2 -sticky e",  NACLabel->GetWidgetName());
  app->Script ( "grid %s -row 1 -column 0 -padx 2 -pady 2 -sticky e",  BIRNLabel->GetWidgetName());
  app->Script ( "grid %s -row 1 -column 1 -padx 2 -pady 2 -sticky e",  NCIGTLabel->GetWidgetName());                  
  app->Script ( "grid %s -row 1 -column 2 -padx 2 -pady 2 -sticky w",  CTSCLabel->GetWidgetName());                  
  app->Script ( "grid columnconfigure %s 0 -weight 0", this->GetLogoFrame()->GetWidgetName() );
  app->Script ( "grid columnconfigure %s 1 -weight 0", this->GetLogoFrame()->GetWidgetName() );
  app->Script ( "grid columnconfigure %s 2 -weight 1", this->GetLogoFrame()->GetWidgetName() );

  NACLabel->Delete();
  NAMICLabel->Delete();
  NCIGTLabel->Delete();
  BIRNLabel->Delete();
  CTSCLabel->Delete();  

  // MAIN UI FRAME
  this->SpecificationFrame = vtkSlicerModuleCollapsibleFrame::New ( );
  SpecificationFrame->SetParent ( this->UIPanel->GetPageWidget ( "CameraBasedROI" ) );
  SpecificationFrame->Create ( );
  SpecificationFrame->ExpandFrame ( );
  SpecificationFrame->SetLabelText ("Place ROI based on Camera");
  app->Script ( "pack %s -side top -anchor nw -fill x -expand y -padx 2 -pady 2 -in %s",
                  SpecificationFrame->GetWidgetName(), this->UIPanel->GetPageWidget("CameraBasedROI")->GetWidgetName());

  vtkKWFrame *f = vtkKWFrame::New();
  f->SetParent ( this->SpecificationFrame->GetFrame() );
  f->Create();
  app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",  f->GetWidgetName() );

  this->ParameterSelector = vtkSlicerNodeSelectorWidget::New();
  this->ParameterSelector->SetParent( f );
  this->ParameterSelector->Create();
  this->ParameterSelector->AddNodeClass("vtkMRMLCameraBasedROINode", NULL, NULL, NULL);
  this->ParameterSelector->SetChildClassesEnabled(0);
  this->ParameterSelector->SetNewNodeEnabled(1);
  this->ParameterSelector->SetShowHidden (1);
  this->ParameterSelector->SetMRMLScene(this->GetMRMLScene());
  this->ParameterSelector->GetWidget()->GetWidget()->SetWidth (24 );
  this->ParameterSelector->GetWidget()->GetWidget()->IndicatorVisibilityOff();
  this->ParameterSelector->SetBorderWidth(2);
  this->ParameterSelector->SetPadX(2);
  this->ParameterSelector->SetPadY(2);
  this->ParameterSelector->SetLabelText( "Parameters ");
  this->ParameterSelector->SetBalloonHelpString("Select or create module parameters.");
  this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                 this->ParameterSelector->GetWidgetName());

  this->CameraSelector = vtkSlicerNodeSelectorWidget::New();
  this->CameraSelector->SetParent( f );
  this->CameraSelector->Create();
  this->CameraSelector->AddNodeClass("vtkMRMLCameraNode", NULL, NULL, NULL);
  this->CameraSelector->SetChildClassesEnabled(1);
  this->CameraSelector->SetShowHidden (1);
  this->CameraSelector->SetMRMLScene(this->GetMRMLScene());
  this->CameraSelector->GetWidget()->GetWidget()->SetWidth (24 );
  this->CameraSelector->GetWidget()->GetWidget()->IndicatorVisibilityOff();
  this->CameraSelector->SetBorderWidth(2);
  this->CameraSelector->SetPadX(2);
  this->CameraSelector->SetPadY(2);
  this->CameraSelector->SetLabelText( "Input Camera ");
  this->CameraSelector->SetBalloonHelpString("Select a camera.");
  this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                 this->CameraSelector->GetWidgetName());

  this->ROISelector = vtkSlicerNodeSelectorWidget::New();
  this->ROISelector->SetParent( f );
  this->ROISelector->Create();
  this->ROISelector->AddNodeClass("vtkMRMLROINode", NULL, NULL, NULL);
  this->ROISelector->SetChildClassesEnabled(1);
  this->ROISelector->SetNewNodeEnabled(1);
  this->ROISelector->SetShowHidden (1);
  this->ROISelector->SetMRMLScene(this->GetMRMLScene());
  this->ROISelector->GetWidget()->GetWidget()->SetWidth (24 );
  this->ROISelector->GetWidget()->GetWidget()->IndicatorVisibilityOff();
  this->ROISelector->SetBorderWidth(2);
  this->ROISelector->SetPadX(2);
  this->ROISelector->SetPadY(2);
  this->ROISelector->SetLabelText( "ROI");
  this->ROISelector->SetBalloonHelpString("Select an ROI.");
  this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                 this->ROISelector->GetWidgetName());

  this->DistanceScale = vtkKWScaleWithEntry::New();
  this->DistanceScale->SetParent( f );
  this->DistanceScale->SetLabelText("Distance from Camera");
  this->DistanceScale->Create();
  this->DistanceScale->SetRange(0,100);
  this->DistanceScale->SetResolution (1);
  this->DistanceScale->SetValue(10);
  this->Script("pack %s -side top -anchor e -padx 20 -pady 4", 
                this->DistanceScale->GetWidgetName());

  this->SizeScale = vtkKWScaleWithEntry::New();
  this->SizeScale->SetParent( f );
  this->SizeScale->SetLabelText("Size of ROI");
  this->SizeScale->Create();
  this->SizeScale->SetRange(0,50);
  this->SizeScale->SetResolution (1);
  this->SizeScale->SetValue(10);
  this->Script("pack %s -side top -anchor e -padx 20 -pady 4", 
                this->SizeScale->GetWidgetName());


  f->Delete();

  this->Init();
  this->Built = true;
}


//---------------------------------------------------------------------------
void vtkCameraBasedROIGUI::Init ( )
{
  //this->CreateParameterNode();
  //this->UpdateParameterNode();
}

//---------------------------------------------------------------------------
void vtkCameraBasedROIGUI::SetSlicerText(const char *txt)
{
  if ( this->GetApplicationGUI() )
    {
    if ( this->GetApplicationGUI()->GetMainSlicerWindow() )
      {
      this->GetApplicationGUI()->GetMainSlicerWindow()->SetStatusText (txt);
      }
    }
}

//---------------------------------------------------------------------------
void vtkCameraBasedROIGUI::CreateParameterNode ( )
{
  if (this->GetMRMLScene() == NULL)
  {
    return;
  }

  vtkMRMLCameraBasedROINode *param = NULL;

  param = vtkMRMLCameraBasedROINode::SafeDownCast(
          this->GetMRMLScene()->GetNthNodeByClass(0, "vtkMRMLCameraBasedROINode"));
  if (param == NULL)
    {
    param = vtkMRMLCameraBasedROINode::New();
    this->GetMRMLScene()->AddNodeNoNotify(param);
    param->Delete();
    }
  vtkSetAndObserveMRMLNodeMacro(this->CameraBasedROINode, param);

}

//---------------------------------------------------------------------------
void vtkCameraBasedROIGUI::UpdateParameterNode ( )
{
  if (this->CameraSelector == NULL || 
      this->ROISelector == NULL)
    {
    return;
    }

  vtkMRMLCameraNode *camera = vtkMRMLCameraNode::SafeDownCast ( this->CameraSelector->GetSelected() );
  vtkSetAndObserveMRMLNodeMacro(this->CameraNode, camera);
  if (this->CameraBasedROINode && camera) 
    {
    this->CameraBasedROINode->SetCameraNodeID(camera->GetID());
    }
 
  vtkMRMLROINode *roi = vtkMRMLROINode::SafeDownCast ( this->ROISelector->GetSelected() );
  if (this->CameraBasedROINode && roi) 
    {
    this->CameraBasedROINode->SetROINodeID(roi->GetID());
    }
}
