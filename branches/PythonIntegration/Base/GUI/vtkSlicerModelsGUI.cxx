#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkCommand.h"
#include "vtkKWWidget.h"
#include "vtkSlicerModelsGUI.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerModuleLogic.h"
//#include "vtkSlicerModelsLogic.h"
#include "vtkSlicerModelDisplayWidget.h"
#include "vtkSlicerModuleCollapsibleFrame.h"

#include "vtkKWFrameWithLabel.h"
#include "vtkKWMenuButton.h"
#include "vtkKWMessageDialog.h"

//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkSlicerModelsGUI );
vtkCxxRevisionMacro ( vtkSlicerModelsGUI, "$Revision: 1.0 $");


//---------------------------------------------------------------------------
vtkSlicerModelsGUI::vtkSlicerModelsGUI ( )
{

    // classes not yet defined!
    this->Logic = NULL;
    //this->ModelNode = NULL;
    this->LoadModelButton = NULL;
    this->LoadModelDirectoryButton = NULL;
    this->SaveModelButton = NULL;
    this->ModelSelectorWidget = NULL;
    this->ModelDisplayWidget = NULL;
    this->ClipModelsWidget = NULL;
    this->LoadScalarsButton = NULL;
    NACLabel = NULL;
    NAMICLabel =NULL;
    NCIGTLabel = NULL;
    BIRNLabel = NULL;
}


//---------------------------------------------------------------------------
vtkSlicerModelsGUI::~vtkSlicerModelsGUI ( )
{
  this->RemoveGUIObservers();

  this->SetModuleLogic ( NULL );

  if (this->LoadModelButton ) 
    {
    this->LoadModelButton->SetParent(NULL);
    this->LoadModelButton->Delete ( );
    }    
  if (this->LoadModelDirectoryButton ) 
    {
    this->LoadModelDirectoryButton->SetParent(NULL);
    this->LoadModelDirectoryButton->Delete ( );
    }    
  if (this->SaveModelButton ) 
    {
    this->SaveModelButton->SetParent(NULL);
    this->SaveModelButton->Delete ( );
    }
  if (this->ModelSelectorWidget ) 
    {
    this->ModelSelectorWidget->SetParent(NULL);
    this->ModelSelectorWidget->Delete ( );
    }
  if (this->ModelDisplayWidget ) 
    {
    this->ModelDisplayWidget->SetParent(NULL);
    this->ModelDisplayWidget->Delete ( );
    }
  if (this->ClipModelsWidget ) 
    {
    this->ClipModelsWidget->SetParent(NULL);
    this->ClipModelsWidget->Delete ( );
    }
  if (this->LoadScalarsButton )
    {
    this->LoadScalarsButton->SetParent(NULL);
    this->LoadScalarsButton->Delete ( );
    }
  if ( this->NACLabel )
    {
    this->NACLabel->SetParent ( NULL );
    this->NACLabel->Delete();
    this->NACLabel = NULL;
    }
  if ( this->NAMICLabel )
    {
    this->NAMICLabel->SetParent ( NULL );
    this->NAMICLabel->Delete();
    this->NAMICLabel = NULL;
    }
  if ( this->NCIGTLabel )
    {
    this->NCIGTLabel->SetParent ( NULL );
    this->NCIGTLabel->Delete();
    this->NCIGTLabel = NULL;
    }
  if ( this->BIRNLabel )
    {
    this->BIRNLabel->SetParent ( NULL );
    this->BIRNLabel->Delete();
    this->BIRNLabel = NULL;
    }
  this->Built = false;
}


//---------------------------------------------------------------------------
void vtkSlicerModelsGUI::PrintSelf ( ostream& os, vtkIndent indent )
{
    this->vtkObject::PrintSelf ( os, indent );

    os << indent << "SlicerModelsGUI: " << this->GetClassName ( ) << "\n";
    //os << indent << "ModelNode: " << this->GetModelNode ( ) << "\n";
    //os << indent << "Logic: " << this->GetLogic ( ) << "\n";
    // print widgets?
}



//---------------------------------------------------------------------------
void vtkSlicerModelsGUI::RemoveGUIObservers ( )
{
  if (this->LoadModelButton)
    {
    this->LoadModelButton->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }
  if (this->LoadModelDirectoryButton)
    {
    this->LoadModelDirectoryButton->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }
  if (this->SaveModelButton)
    {
    this->SaveModelButton->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }
  if (this->LoadScalarsButton)
    {
    this->LoadScalarsButton->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }
}


//---------------------------------------------------------------------------
void vtkSlicerModelsGUI::AddGUIObservers ( )
{
  this->LoadModelButton->AddObserver ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
  this->LoadModelDirectoryButton->AddObserver ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
  this->SaveModelButton->AddObserver ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
  this->LoadScalarsButton->AddObserver ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
}



//---------------------------------------------------------------------------
void vtkSlicerModelsGUI::ProcessGUIEvents ( vtkObject *caller,
                                            unsigned long event, void *callData )
{
  vtkKWLoadSaveButton *filebrowse = vtkKWLoadSaveButton::SafeDownCast(caller);
  if (filebrowse == this->LoadModelButton  && event == vtkKWPushButton::InvokedEvent )
    {
    // If a file has been selected for loading...
    char *fileName = filebrowse->GetFileName();
    if ( fileName ) 
      {
      vtkSlicerModelsLogic* modelLogic = this->Logic;
      
      vtkMRMLModelNode *modelNode = modelLogic->AddModel( fileName );
      if ( modelNode == NULL ) 
        {
        vtkKWMessageDialog *dialog = vtkKWMessageDialog::New();
        dialog->SetParent ( this->UIPanel->GetPageWidget ( "Models" ) );
        dialog->SetStyleToMessage();
        std::string msg = std::string("Unable to read model file ") + std::string(fileName);
        dialog->SetText(msg.c_str());
        dialog->Create ( );
        dialog->Invoke();
        dialog->Delete();

        vtkErrorMacro("Unable to read model file " << fileName);
        // reset the file browse button text
        }
      else
        {
        filebrowse->GetLoadSaveDialog()->SaveLastPathToRegistry("OpenPath");
        
        }
      }
      this->LoadModelButton->SetText ("Load Model");
    return;
    }
    else if (filebrowse == this->LoadModelDirectoryButton  && event == vtkKWPushButton::InvokedEvent )
    {
    // If a file has been selected for loading...
    char *fileName = filebrowse->GetFileName();
    if ( fileName ) 
      {
      vtkSlicerModelsLogic* modelLogic = this->Logic;
      
      if (modelLogic->AddModels( fileName, ".vtk") == 0)
        {
        vtkKWMessageDialog *dialog = vtkKWMessageDialog::New();
        dialog->SetParent ( this->UIPanel->GetPageWidget ( "Models" ) );
        dialog->SetStyleToMessage();
        std::string msg = std::string("Unable to read models directory ") + std::string(fileName);
        dialog->SetText(msg.c_str());
        dialog->Create ( );
        dialog->Invoke();
        dialog->Delete();
        }
      else
        {
        filebrowse->GetLoadSaveDialog()->SaveLastPathToRegistry("OpenPath");
        
        }
      }
    this->LoadModelDirectoryButton->SetText ("Load Model Directory");
    return;
    }
  else if (filebrowse == this->SaveModelButton  && event == vtkKWPushButton::InvokedEvent )
      {
      // If a file has been selected for saving...
      char *fileName = filebrowse->GetFileName();
      if ( fileName ) 
      {
        vtkSlicerModelsLogic* ModelLogic = this->Logic;
        vtkMRMLModelNode *volNode = vtkMRMLModelNode::SafeDownCast(this->ModelSelectorWidget->GetSelected());
        if ( !ModelLogic->SaveModel( fileName, volNode ))
          {
         // TODO: generate an error...
          }
        else
          {
          filebrowse->GetLoadSaveDialog()->SaveLastPathToRegistry("OpenPath");           
          }
       }
       return;
    } 
  else if (filebrowse == this->LoadScalarsButton  && event == vtkKWPushButton::InvokedEvent )
    {
    // If a scalar file has been selected for loading...
    char *fileName = filebrowse->GetFileName();
    if ( fileName ) 
      {
      // get the model
      vtkMRMLModelNode *modelNode = vtkMRMLModelNode::SafeDownCast(this->ModelSelectorWidget->GetSelected());

      // load the scalars
      vtkSlicerModelsLogic* modelLogic = this->Logic;
      if (!modelLogic->AddScalar(fileName, modelNode))
        {
        vtkKWMessageDialog *dialog = vtkKWMessageDialog::New();
        dialog->SetParent ( this->UIPanel->GetPageWidget ( "Models" ) );
        dialog->SetStyleToMessage();
        std::string msg = std::string("Unable to read scalars file ") + std::string(fileName);
        dialog->SetText(msg.c_str());
        dialog->Create ( );
        dialog->Invoke();
        dialog->Delete();

        vtkErrorMacro("Error loading scalar overlay file " << fileName);
        this->LoadScalarsButton->SetText ("Load FreeSurfer Overlay");
        }
      else
        {
        filebrowse->GetLoadSaveDialog()->SaveLastPathToRegistry("OpenPath");
        }
      }
    return;
    }
}    

//---------------------------------------------------------------------------
void vtkSlicerModelsGUI::ProcessLogicEvents ( vtkObject *caller,
                                              unsigned long event, void *callData )
{
    // Fill in
}

//---------------------------------------------------------------------------
void vtkSlicerModelsGUI::ProcessMRMLEvents ( vtkObject *caller,
                                             unsigned long event, void *callData )
{
    // Fill in
}


//---------------------------------------------------------------------------
void vtkSlicerModelsGUI::CreateModuleEventBindings ( )
{
}

//---------------------------------------------------------------------------
void vtkSlicerModelsGUI::ReleaseModuleEventBindings ( )
{
  
}


//---------------------------------------------------------------------------
void vtkSlicerModelsGUI::Enter ( )
{
  if ( this->Built == false )
    {
    this->BuildGUI();
    this->Built = true;
    this->AddGUIObservers();
    }
    this->CreateModuleEventBindings();
}



//---------------------------------------------------------------------------
void vtkSlicerModelsGUI::Exit ( )
{
  this->ReleaseModuleEventBindings();
}


//---------------------------------------------------------------------------
void vtkSlicerModelsGUI::TearDownGUI ( )
{
  this->Exit();
  if ( this->Built )
    {
    this->RemoveGUIObservers();
    }
}


//---------------------------------------------------------------------------
void vtkSlicerModelsGUI::BuildGUI ( )
{

    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
  
    // ---
    // MODULE GUI FRAME 
    // configure a page for a model loading UI for now.
    // later, switch on the modulesButton in the SlicerControlGUI
    // ---
    // create a page
    this->UIPanel->AddPage ( "Models", "Models", NULL );
    
    // Define your help text and build the help frame here.
    const char *help = "The Models Module loads, saves and adjusts display parameters of models. ";
    const char *about = "This work was supported by NA-MIC, NAC, BIRN, NCIGT, and the Slicer Community. See http://www.slicer.org for details. ";
    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "Models" );
    this->BuildHelpAndAboutFrame ( page, help, about );

    this->NACLabel = vtkKWLabel::New();
    this->NACLabel->SetParent ( this->GetLogoFrame() );
    this->NACLabel->Create();
    this->NACLabel->SetImageToIcon ( vtkSlicerModuleGUI::AcknowledgementIcons->GetNACLogo() );

    this->NAMICLabel = vtkKWLabel::New();
    this->NAMICLabel->SetParent ( this->GetLogoFrame() );
    this->NAMICLabel->Create();
    this->NAMICLabel->SetImageToIcon ( vtkSlicerModuleGUI::AcknowledgementIcons->GetNAMICLogo() );    

    this->NCIGTLabel = vtkKWLabel::New();
    this->NCIGTLabel->SetParent ( this->GetLogoFrame() );
    this->NCIGTLabel->Create();
    this->NCIGTLabel->SetImageToIcon ( vtkSlicerModuleGUI::AcknowledgementIcons->GetNCIGTLogo() );
    
    this->BIRNLabel = vtkKWLabel::New();
    this->BIRNLabel->SetParent ( this->GetLogoFrame() );
    this->BIRNLabel->Create();
    this->BIRNLabel->SetImageToIcon ( vtkSlicerModuleGUI::AcknowledgementIcons->GetBIRNLogo() );
    app->Script ( "grid %s -row 0 -column 0 -padx 2 -pady 2 -sticky w", this->NAMICLabel->GetWidgetName());
    app->Script ("grid %s -row 0 -column 1 -padx 2 -pady 2 -sticky w",  this->NACLabel->GetWidgetName());
    app->Script ( "grid %s -row 1 -column 0 -padx 2 -pady 2 -sticky w",  this->BIRNLabel->GetWidgetName());
    app->Script ( "grid %s -row 1 -column 1 -padx 2 -pady 2 -sticky w",  this->NCIGTLabel->GetWidgetName());                  

    // ---
    // LOAD FRAME            
    vtkSlicerModuleCollapsibleFrame *modLoadFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    modLoadFrame->SetParent ( this->UIPanel->GetPageWidget ( "Models" ) );
    modLoadFrame->Create ( );
    modLoadFrame->SetLabelText ("Load");
    modLoadFrame->ExpandFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  modLoadFrame->GetWidgetName(), this->UIPanel->GetPageWidget("Models")->GetWidgetName());

    // add a file browser 
    this->LoadModelButton = vtkKWLoadSaveButton::New ( );
    this->LoadModelButton->SetParent ( modLoadFrame->GetFrame() );
    this->LoadModelButton->Create ( );
    this->LoadModelButton->SetText ("Load Model");
    this->LoadModelButton->GetLoadSaveDialog()->RetrieveLastPathFromRegistry("OpenPath");
    this->LoadModelButton->GetLoadSaveDialog()->SetFileTypes(
                                                             "{ {model} {*.*} }");
    app->Script("pack %s -side left -anchor w -padx 2 -pady 4", 
                this->LoadModelButton->GetWidgetName());

   // add a file browser 
    this->LoadModelDirectoryButton = vtkKWLoadSaveButton::New ( );
    this->LoadModelDirectoryButton->SetParent ( modLoadFrame->GetFrame() );
    this->LoadModelDirectoryButton->Create ( );
    this->LoadModelDirectoryButton->SetText ("Load Model Directory");
    this->LoadModelDirectoryButton->GetLoadSaveDialog()->ChooseDirectoryOn();
    app->Script("pack %s -side left -anchor w -padx 2 -pady 4", 
                this->LoadModelDirectoryButton->GetWidgetName());

  
    // DISPLAY FRAME            
    vtkSlicerModuleCollapsibleFrame *modDisplayFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    modDisplayFrame->SetParent ( this->UIPanel->GetPageWidget ( "Models" ) );
    modDisplayFrame->Create ( );
    modDisplayFrame->SetLabelText ("Display");
    modDisplayFrame->CollapseFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  modDisplayFrame->GetWidgetName(), this->UIPanel->GetPageWidget("Models")->GetWidgetName());

    this->LoadScalarsButton = vtkKWLoadSaveButton::New();
    this->LoadScalarsButton->SetParent ( modDisplayFrame->GetFrame() );
    this->LoadScalarsButton->Create ( );
    this->LoadScalarsButton->SetText ("Load FreeSurfer Overlay");
    this->LoadScalarsButton->GetLoadSaveDialog()->RetrieveLastPathFromRegistry("OpenPath");
    this->LoadScalarsButton->GetLoadSaveDialog()->SetFileTypes("{ {Thickness} {*.thickness} } { {Curve} {*.curv} } { {Average Curve} {*.avg_curv} } { {Sulc} {*.sulc} } { {Area} {*.area} } { {W} {*.w} }");
    app->Script("pack %s -side top -anchor nw -padx 2 -pady 4", 
                this->LoadScalarsButton->GetWidgetName());

    this->ModelDisplayWidget = vtkSlicerModelDisplayWidget::New ( );
    this->ModelDisplayWidget->SetMRMLScene(this->GetMRMLScene() );
    this->ModelDisplayWidget->SetParent ( modDisplayFrame->GetFrame() );
    this->ModelDisplayWidget->Create ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  this->ModelDisplayWidget->GetWidgetName(), 
                  modDisplayFrame->GetFrame()->GetWidgetName());

    // Clip FRAME  
    vtkSlicerModuleCollapsibleFrame *clipFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    clipFrame->SetParent ( this->UIPanel->GetPageWidget ( "Models" ) );
    clipFrame->Create ( );
    clipFrame->SetLabelText ("Clipping");
    clipFrame->CollapseFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  clipFrame->GetWidgetName(), this->UIPanel->GetPageWidget("Models")->GetWidgetName());

    this->ClipModelsWidget = vtkSlicerClipModelsWidget::New ( );
    this->ClipModelsWidget->SetMRMLScene(this->GetMRMLScene() );
    this->ClipModelsWidget->SetParent ( clipFrame->GetFrame() );
    this->ClipModelsWidget->Create ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  this->ClipModelsWidget->GetWidgetName(), 
                  clipFrame->GetFrame()->GetWidgetName());

    // ---
    // Save FRAME            
    vtkSlicerModuleCollapsibleFrame *modelSaveFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    modelSaveFrame->SetParent ( this->UIPanel->GetPageWidget ( "Models" ) );
    modelSaveFrame->Create ( );
    modelSaveFrame->SetLabelText ("Save");
    modelSaveFrame->CollapseFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  modelSaveFrame->GetWidgetName(), 
                  this->UIPanel->GetPageWidget ( "Models" )->GetWidgetName());

    // selector for save
    this->ModelSelectorWidget = vtkSlicerNodeSelectorWidget::New() ;
    this->ModelSelectorWidget->SetParent ( modelSaveFrame->GetFrame() );
    this->ModelSelectorWidget->Create ( );
    this->ModelSelectorWidget->SetNodeClass("vtkMRMLModelNode", NULL, NULL, NULL);
    this->ModelSelectorWidget->SetMRMLScene(this->GetMRMLScene());
    this->ModelSelectorWidget->SetBorderWidth(2);
    this->ModelSelectorWidget->SetPadX(2);
    this->ModelSelectorWidget->SetPadY(2);
    this->ModelSelectorWidget->GetWidget()->GetWidget()->IndicatorVisibilityOff();
    this->ModelSelectorWidget->GetWidget()->GetWidget()->SetWidth(24);
    this->ModelSelectorWidget->SetLabelText( "Model To Save: ");
    this->ModelSelectorWidget->SetBalloonHelpString("select a Model from the current  scene.");
    this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                  this->ModelSelectorWidget->GetWidgetName());

    this->SaveModelButton = vtkKWLoadSaveButton::New ( );
    this->SaveModelButton->SetParent ( modelSaveFrame->GetFrame() );
    this->SaveModelButton->Create ( );
    this->SaveModelButton->SetText ("Save Model");
    this->SaveModelButton->GetLoadSaveDialog()->SaveDialogOn();
    this->SaveModelButton->GetLoadSaveDialog()->SetFileTypes(
                                                              "{ {Model} {*.*} }");
    this->SaveModelButton->GetLoadSaveDialog()->RetrieveLastPathFromRegistry(
      "OpenPath");
     app->Script("pack %s -side top -anchor w -padx 2 -pady 4", 
                this->SaveModelButton->GetWidgetName());
    
    modLoadFrame->Delete ( );
    modDisplayFrame->Delete ( );
    clipFrame->Delete ( );
    modelSaveFrame->Delete();
}





