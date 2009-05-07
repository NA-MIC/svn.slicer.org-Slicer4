#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkCommand.h"
#include <itksys/SystemTools.hxx> 
#include "vtkKWWidget.h"
#include "vtkSlicerModelsGUI.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerModuleLogic.h"
#include "vtkSlicerModelDisplayWidget.h"
#include "vtkSlicerModelHierarchyWidget.h"
#include "vtkSlicerModuleCollapsibleFrame.h"
#include "vtkSlicerModelInfoWidget.h"

// for pick events
//#include "vtkSlicerViewerWidget.h"
//#include "vtkSlicerViewerInteractorStyle.h"

#include "vtkKWFrameWithLabel.h"
#include "vtkKWMenuButton.h"
#include "vtkKWMessageDialog.h"
#include "vtkKWTkUtilities.h"
#include "vtkKWTopLevel.h"

// for scalars
#include "vtkPointData.h"

#include "vtkSlicerApplicationGUI.h"

//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkSlicerModelsGUI );
vtkCxxRevisionMacro ( vtkSlicerModelsGUI, "$Revision: 1.0 $");


//---------------------------------------------------------------------------
vtkSlicerModelsGUI::vtkSlicerModelsGUI ( )
{

  // classes not yet defined!
  this->Logic = NULL;
  this->ModelHierarchyLogic = NULL;
  this->LoadDirectory = NULL;

  //this->ModelNode = NULL;
  this->LoadModelButton = NULL;
  this->ModelDisplayWidget = NULL;
  this->ClipModelsWidget = NULL;
  this->LoadScalarsButton = NULL;
  this->ModelDisplaySelectorWidget = NULL;
  this->ModelHierarchyWidget = NULL;
  this->ModelDisplayFrame = NULL;
  this->ModelInfoWidget = NULL;

  this->AddModelDialogButton = NULL;
  this->AddModelDirectoryDialogButton = NULL;
  this->AddModelWindow = NULL;

  this->ModelSelector = NULL;
  this->AddOverlayDialogButton = NULL;
  this->AddOverlayWindow = NULL;
  this->SelectedModelNode = NULL;

  NACLabel = NULL;
  NAMICLabel =NULL;
  NCIGTLabel = NULL;
  BIRNLabel = NULL;

  // for picking
//  this->ViewerWidget = NULL;
//  this->InteractorStyle = NULL;
}


//---------------------------------------------------------------------------
vtkSlicerModelsGUI::~vtkSlicerModelsGUI ( )
{
  this->RemoveGUIObservers();

  this->SetModuleLogic ( NULL );
  this->SetModelHierarchyLogic ( NULL );
  this->SetLoadDirectory ( NULL );
  
  if (this->ModelDisplaySelectorWidget)
    {
    this->ModelDisplaySelectorWidget->SetParent(NULL);
    this->ModelDisplaySelectorWidget->Delete();
    this->ModelDisplaySelectorWidget = NULL;
    }

  if (this->ModelInfoWidget)
    {
    this->ModelInfoWidget->SetParent(NULL);
    this->ModelInfoWidget->Delete();
    this->ModelInfoWidget = NULL;
    }

  if (this->ModelHierarchyWidget)
    {
    this->ModelHierarchyWidget->SetParent(NULL);
    this->ModelHierarchyWidget->Delete();
    this->ModelHierarchyWidget = NULL;
    }

  if (this->LoadModelButton ) 
    {
    this->LoadModelButton->SetParent(NULL);
    this->LoadModelButton->Delete ( );
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

  //--- widgets in temporary raised windows.
  vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast ( this->GetApplication() );
  if ( this->AddModelDialogButton)
    {
    this->AddModelDialogButton->SetParent ( NULL );
    this->AddModelDialogButton->Delete();
    this->AddModelDialogButton = NULL;    
    }
  if ( this->AddModelDirectoryDialogButton)
    {
    this->AddModelDirectoryDialogButton->SetParent ( NULL );
    this->AddModelDirectoryDialogButton->Delete();
    this->AddModelDirectoryDialogButton = NULL;    
    }
  if ( this->AddModelWindow )
    {
    if ( app )
      {
      app->Script ( "grab release %s", this->AddModelWindow->GetWidgetName() );
      }
    this->AddModelWindow->Withdraw();
    this->AddModelWindow->Delete();
    }

  if ( this->AddOverlayDialogButton )
    {
    this->AddOverlayDialogButton->SetParent ( NULL );
    this->AddOverlayDialogButton->Delete();
    this->AddOverlayDialogButton = NULL;    
    }
  if ( this->ModelSelector)
    {
    this->ModelSelector->SetParent ( NULL );
    this->ModelSelector->Delete();
    this->ModelSelector = NULL;    
    }
  if ( this->AddOverlayWindow )
    {
    if ( app )
      {
      app->Script ( "grab release %s", this->AddOverlayWindow->GetWidgetName() );
      }
    this->AddOverlayWindow->Withdraw();
    this->AddOverlayWindow->Delete();
    }

  if (this->ModelDisplayFrame)
    {
    this->ModelDisplayFrame->SetParent ( NULL );
    this->ModelDisplayFrame->Delete();
    }

//  this->SetViewerWidget(NULL);   
//  this->SetInteractorStyle(NULL);
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
  if (this->LoadScalarsButton)
    {
    this->LoadScalarsButton->RemoveObservers( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );   
    }
  if (this->ModelDisplaySelectorWidget)
    {
    this->ModelDisplaySelectorWidget->RemoveObservers (vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );
    }
  if (this->ModelHierarchyWidget)
    { 
    this->ModelHierarchyWidget->RemoveObservers(vtkSlicerModelHierarchyWidget::SelectedEvent, (vtkCommand *)this->GUICallbackCommand );
    }

}


//---------------------------------------------------------------------------
void vtkSlicerModelsGUI::AddGUIObservers ( )
{
  this->LoadModelButton->AddObserver ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
  this->LoadScalarsButton->AddObserver ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
  //this->ModelDisplaySelectorWidget->AddObserver (vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->ModelHierarchyWidget->AddObserver(vtkSlicerModelHierarchyWidget::SelectedEvent, (vtkCommand *)this->GUICallbackCommand );
}



//---------------------------------------------------------------------------
void vtkSlicerModelsGUI::ProcessGUIEvents ( vtkObject *caller,
                                            unsigned long event, void *callData )
{

  vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast (this->GetApplication() );
  if ( !app )
    {
    vtkErrorMacro ( "ProcessGUIEvents: got Null SlicerApplication" );
    return;
    }
  vtkSlicerApplicationGUI *appGUI = app->GetApplicationGUI();
  if ( !appGUI )
    {
    vtkErrorMacro ( "ProcessGUIEvents: got Null SlicerApplicationGUI" );
    return;    
    }
  vtkSlicerWindow *win = appGUI->GetMainSlicerWindow();
  if ( !win )
    {
    vtkErrorMacro ( "ProcessGUIEvents: got NULL Slicer Window" );
    return;
    }


  if (vtkSlicerModelHierarchyWidget::SafeDownCast(caller) == this->ModelHierarchyWidget && 
      event == vtkSlicerModelHierarchyWidget::SelectedEvent)
    {
    vtkMRMLModelNode *model = reinterpret_cast<vtkMRMLModelNode *>(callData);
    if (model != NULL && model->GetDisplayNode() != NULL)
      {
      //this->ModelDisplaySelectorWidget->SetSelected(model);
      if (this->ModelDisplayFrame)
        {
        this->ModelDisplayFrame->ExpandFrame();
        this->ModelDisplayFrame->Raise();
        }
      //this->ModelDisplayWidget->SetModelDisplayNode(model->GetDisplayNode());
      //this->ModelDisplayWidget->SetModelNode(model);
      }
    return;
    }

  //--- buttons
  vtkKWPushButton *b = vtkKWPushButton::SafeDownCast ( caller );
  if ( b != NULL && event == vtkKWPushButton::InvokedEvent )
    {
    if (b == this->LoadModelButton)
      {
      this->RaiseAddModelWindow();
      }
    else if ( b == this->LoadScalarsButton)
      {
      this->RaiseAddScalarOverlayWindow();
      }
    }

  //--- look for file browser dialogs
  //--- these are all pop-ups associated with adding data types. temporary solution for centralized load.
  vtkKWLoadSaveDialog *d = vtkKWLoadSaveDialog::SafeDownCast ( caller );
  if ( d != NULL &&  event == vtkKWTopLevel::WithdrawEvent )
    {
    if ( this->AddModelDialogButton != NULL && d == this->AddModelDialogButton->GetLoadSaveDialog() )
      {
      // If a file has been selected for loading...
      const char *fileName = this->AddModelDialogButton->GetFileName();
      if ( fileName ) 
        {
        win->SetStatusText ( "Reading and loading model file..." );
        app->Script ( "update idletasks" );

        vtkMRMLModelNode *modelNode = this->Logic->AddModel( fileName );
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
          }
        else
          {
          win->SetStatusText ( "" );
          app->Script ( "update idletasks" );
          this->AddModelDialogButton->GetLoadSaveDialog()->SaveLastPathToRegistry("OpenPath");
          const vtksys_stl::string fname(fileName);
          vtksys_stl::string name = vtksys::SystemTools::GetFilenameName(fname);
          // set it to be the active model
          // set the display model
          this->SelectedModelNode = modelNode;
          if ( this->ModelSelector )
            {
            this->ModelSelector->SetSelected(this->SelectedModelNode);
            }
          }

        if ( this->AddModelDialogButton->GetText() )
          {
          this->AddModelDialogButton->SetText("");
          }
        //--- Withdraw and destroy the AddModelWindow
        this->WithdrawAddModelWindow();
        }
      return;
      }

    else if ( this->AddModelDirectoryDialogButton != NULL && d == this->AddModelDirectoryDialogButton->GetLoadSaveDialog() )
      {
      // If a file has been selected for loading...
      const char *fileName = this->AddModelDirectoryDialogButton->GetFileName();
      if ( fileName ) 
        {
        if ( this->Logic != NULL )
          {
          vtkKWMessageDialog *dialog0 = vtkKWMessageDialog::New();
          dialog0->SetParent ( this->UIPanel->GetPageWidget ( "Models" ) );
          dialog0->SetStyleToMessage();
          std::string msg0 = std::string("Reading *.vtk from models directory ") + std::string(fileName);
          dialog0->SetText(msg0.c_str());
          dialog0->Create ( );
          dialog0->Invoke();
          dialog0->Delete();

          win->SetStatusText ( "Reading and loading files..." );
          app->Script ( "update idletasks" );
      
          if (this->Logic->AddModels( fileName, ".vtk") == 0)
            {
            vtkKWMessageDialog *dialog = vtkKWMessageDialog::New();
            dialog->SetParent ( this->UIPanel->GetPageWidget ( "Models" ) );
            dialog->SetStyleToMessage();
            std::string msg = std::string("Unable to read all models from directory ") + std::string(fileName);
            dialog->SetText(msg.c_str());
            dialog->Create ( );
            dialog->Invoke();
            dialog->Delete();
            vtkErrorMacro("ProcessGUIEvents: unable to read all models from directory " << fileName);
            }
          else
            {
            this->AddModelDirectoryDialogButton->GetLoadSaveDialog()->SaveLastPathToRegistry("OpenPath");
            vtkKWMessageDialog *dialog = vtkKWMessageDialog::New();
            dialog->SetParent ( this->UIPanel->GetPageWidget ( "Models" ) );
            dialog->SetStyleToMessage();
            dialog->SetText("Done reading models...");
            dialog->Create ( );
            dialog->Invoke();
            dialog->Delete();
            }
          }
        //--- Withdraw and destroy the AddModelWindow
        if ( this->AddModelDirectoryDialogButton->GetText() )
          {
          this->AddModelDirectoryDialogButton->SetText("");
          }
        this->WithdrawAddModelWindow();
        }

      win->SetStatusText ( "" );
      app->Script ( "update idletasks" );
      return;      
      }
    
    else if ( this->AddOverlayDialogButton != NULL && d == this->AddOverlayDialogButton->GetLoadSaveDialog() )
      {
      // If a scalar file has been selected for loading...
      const char *fileName = this->AddOverlayDialogButton->GetFileName();
      if ( fileName ) 
        {
        if ( this->SelectedModelNode == NULL )
          {
          vtkKWMessageDialog *dialog = vtkKWMessageDialog::New();
          dialog->SetParent ( this->UIPanel->GetPageWidget ( "Models" ) );
          dialog->SetStyleToMessage();
          std::string msg = std::string("Please select a model to which the scalar overlay will be applied before selecting the overlay.");
          dialog->SetText(msg.c_str());
          dialog->Create ( );
          dialog->Invoke();
          dialog->Delete();
          return;
          }

        vtkMRMLModelNode *modelNode = this->SelectedModelNode;
        if (modelNode != NULL)
          {
          vtkDebugMacro("ProcessGUIEvents: loading scalar for model " << modelNode->GetName());
          // load the scalars

          win->SetStatusText ( "Reading and loading scalar overlay..." );
          app->Script ( "update idletasks" );
      
          if ( !(this->Logic->AddScalar(fileName, modelNode)) )
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
            }
          else
            {
            this->AddOverlayDialogButton->GetLoadSaveDialog()->SaveLastPathToRegistry("OpenPath");
            }
          }

        //--- Withdraw and destroy the AddOverlayWindow
        if ( this->AddOverlayDialogButton->GetText())
          {
          this->AddOverlayDialogButton->SetText ("");
          }
        this->WithdrawAddScalarOverlayWindow();
        }
      win->SetStatusText ( "" );
      app->Script ( "update idletasks" );
      return;      
      }
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
void vtkSlicerModelsGUI::Enter ( vtkMRMLNode *node )
{
  if ( this->Built == false )
    {
    this->BuildGUI();
    this->Built = true;
    this->AddGUIObservers();
    }
  this->CreateModuleEventBindings();
  if (node)
    {
    this->ModelHierarchyWidget->UpdateTreeFromMRML();
    this->ModelHierarchyWidget->GetModelDisplaySelectorWidget()->UnconditionalUpdateMenu();
    this->ModelHierarchyWidget->SelectNode(node);
    }
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
    vtkSlicerApplicationGUI *appGUI = app->GetApplicationGUI();

    // ---
    // MODULE GUI FRAME 
    // configure a page for a model loading UI for now.
    // later, switch on the modulesButton in the SlicerControlGUI
    // ---
    // create a page
    this->UIPanel->AddPage ( "Models", "Models", NULL );
    
    // Define your help text and build the help frame here.
    const char *help = "The Models Module loads and adjusts display parameters of models.\n<a>http://wiki.slicer.org/slicerWiki/index.php/Modules:Models-Documentation-3.4</a>\nSave models via the File menu, Save button.\nThe Add 3D model or a model directory button will allow you to load any model that Slicer can read, as well as all the VTK models in a directory. Add Scalar Overlay will load a scalar file and associate it with the currently active model.\nYou can adjust the display properties of the models in the Display pane. Select the model you wish to work on from the model selector drop down menu. Scalar overlays are loaded with a default colour look up table, but can be reassigned manually. Once a new scalar overlay is chosen, currently the old color map is still used, so that must be adjusted in conjunction with the overlay.\nClipping is turned on for a model in the Display pane, and the slice planes that will clip the model are selected in the Clipping pane.\nThe Model Hierarchy pane allows you to group models together and set the group's properties.";
    const char *about = "This module was contributed by Nicole Aucoin, SPL, BWH (Ron Kikinis), and Alex Yarmarkovich, Isomics Inc. (Steve Pieper).\nThis work was supported by NA-MIC, NAC, BIRN, NCIGT, and the Slicer Community. See <a>http://www.slicer.org</a> for details. ";
    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "Models" );
    this->BuildHelpAndAboutFrame ( page, help, about );

    this->NACLabel = vtkKWLabel::New();
    this->NACLabel->SetParent ( this->GetLogoFrame() );
    this->NACLabel->Create();
    this->NACLabel->SetImageToIcon ( this->GetAcknowledgementIcons()->GetNACLogo() );

    this->NAMICLabel = vtkKWLabel::New();
    this->NAMICLabel->SetParent ( this->GetLogoFrame() );
    this->NAMICLabel->Create();
    this->NAMICLabel->SetImageToIcon ( this->GetAcknowledgementIcons()->GetNAMICLogo() );    

    this->NCIGTLabel = vtkKWLabel::New();
    this->NCIGTLabel->SetParent ( this->GetLogoFrame() );
    this->NCIGTLabel->Create();
    this->NCIGTLabel->SetImageToIcon ( this->GetAcknowledgementIcons()->GetNCIGTLogo() );
    
    this->BIRNLabel = vtkKWLabel::New();
    this->BIRNLabel->SetParent ( this->GetLogoFrame() );
    this->BIRNLabel->Create();
    this->BIRNLabel->SetImageToIcon ( this->GetAcknowledgementIcons()->GetBIRNLogo() );
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
    this->LoadModelButton = vtkKWPushButton::New();
    this->LoadModelButton->SetParent ( modLoadFrame->GetFrame() );
    this->LoadModelButton->Create ( );
    this->LoadModelButton->SetBorderWidth ( 0 );
    this->LoadModelButton->SetReliefToFlat();  
    this->LoadModelButton->SetCompoundModeToLeft ();
    this->LoadModelButton->SetImageToIcon ( appGUI->GetSlicerFoundationIcons()->GetSlicerLoadModelIcon() );
    this->LoadModelButton->SetText (" Add 3D model or a model directory" );
    this->LoadModelButton->SetWidth ( 300 );
    this->LoadModelButton->SetAnchorToWest();
    this->LoadModelButton->SetBalloonHelpString("Use this model loading option to raise a dialog with options to add models to the current scene.");
  
    app->Script("pack %s -side top -anchor nw -padx 2 -pady 4 -ipadx 0 -ipady 0", 
                this->LoadModelButton->GetWidgetName());


    this->LoadScalarsButton = vtkKWPushButton::New();
    this->LoadScalarsButton->SetParent ( modLoadFrame->GetFrame() );
    this->LoadScalarsButton->Create ( );
    this->LoadScalarsButton->SetBorderWidth ( 0 );
    this->LoadScalarsButton->SetReliefToFlat();  
    this->LoadScalarsButton->SetCompoundModeToLeft ();
    this->LoadScalarsButton->SetImageToIcon ( appGUI->GetSlicerFoundationIcons()->GetSlicerLoadScalarOverlayIcon() );
    this->LoadScalarsButton->SetText (" Add scalar overlay" );
    this->LoadScalarsButton->SetWidth ( 300 );
    this->LoadScalarsButton->SetAnchorToWest();
    this->LoadScalarsButton->SetBalloonHelpString("Use this option to add a (FreeSurfer) scalar overlay to an existing model in the scene.");

    // this->LoadScalarsButton->GetWidget()->GetLoadSaveDialog()->SetFileTypes("{ {All} {.*} } { {Thickness} {.thickness} } { {Curve} {.curv} } { {Average Curve} {.avg_curv} } { {Sulc} {.sulc} } { {Area} {.area} } { {W} {.w} } { {Parcellation Annotation} {.annot} } { {Volume} {.mgz .mgh} } { {Label} {.label} }");
    app->Script("pack %s -side top -anchor nw -padx 2 -pady 4 -ipadx 0 -ipady 0", 
                this->LoadScalarsButton->GetWidgetName());
    
    // DISPLAY FRAME            
    this->ModelDisplayFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    this->ModelDisplayFrame->SetParent ( this->UIPanel->GetPageWidget ( "Models" ) );
    this->ModelDisplayFrame->Create ( );
    this->ModelDisplayFrame->SetLabelText ("Hierarchy & Display");
    this->ModelDisplayFrame->ExpandFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  this->ModelDisplayFrame->GetWidgetName(), this->UIPanel->GetPageWidget("Models")->GetWidgetName());

 
    this->ModelHierarchyWidget = vtkSlicerModelHierarchyWidget::New ( );
    this->ModelHierarchyWidget->SetAndObserveMRMLScene(this->GetMRMLScene() );
    this->ModelHierarchyWidget->SetModelHierarchyLogic(this->GetModelHierarchyLogic());
    this->ModelHierarchyWidget->SetParent ( this->ModelDisplayFrame->GetFrame() );
    this->ModelHierarchyWidget->Create ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  this->ModelHierarchyWidget->GetWidgetName(), 
                  this->ModelDisplayFrame->GetFrame()->GetWidgetName());


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

    // Info FRAME  
    vtkSlicerModuleCollapsibleFrame *infoFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    infoFrame->SetParent ( this->UIPanel->GetPageWidget ( "Models" ) );
    infoFrame->Create ( );
    infoFrame->SetLabelText ("Info");
    infoFrame->CollapseFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  infoFrame->GetWidgetName(), this->UIPanel->GetPageWidget("Models")->GetWidgetName());

    this->ModelInfoWidget = vtkSlicerModelInfoWidget::New ( );
    this->ModelInfoWidget->SetAndObserveMRMLScene(this->GetMRMLScene() );
    this->ModelInfoWidget->SetParent ( infoFrame->GetFrame() );
    this->ModelInfoWidget->Create ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  this->ModelInfoWidget->GetWidgetName(), 
                  infoFrame->GetFrame()->GetWidgetName());

   //this->ProcessGUIEvents (this->ModelDisplaySelectorWidget,
                          //vtkSlicerNodeSelectorWidget::NodeSelectedEvent, NULL );

    modLoadFrame->Delete ( );
    clipFrame->Delete ( );    
    infoFrame->Delete ( );

    // set up picking
    this->Init();
}

/*
//----------------------------------------------------------------------------
void vtkSlicerModelsGUI::SetViewerWidget ( vtkSlicerViewerWidget *viewerWidget )
{
  this->ViewerWidget = viewerWidget;
}

//----------------------------------------------------------------------------
void vtkSlicerModelsGUI::SetInteractorStyle( vtkSlicerViewerInteractorStyle *interactorStyle )
{
  // note: currently the GUICallbackCommand calls ProcessGUIEvents
  // remove observers
  if (this->InteractorStyle != NULL &&
      this->InteractorStyle->HasObserver(vtkSlicerViewerInteractorStyle::SelectRegionEvent, this->GUICallbackCommand) == 1)
    {
    this->InteractorStyle->RemoveObservers(vtkSlicerViewerInteractorStyle::SelectRegionEvent, (vtkCommand *)this->GUICallbackCommand);
    }
  
  this->InteractorStyle = interactorStyle;

  // add observers
  if (this->InteractorStyle)
    {
    vtkDebugMacro("SetInteractorStyle: Adding observer on interactor style");
    this->InteractorStyle->AddObserver(vtkSlicerViewerInteractorStyle::SelectRegionEvent, (vtkCommand *)this->GUICallbackCommand);
    }
}

//----------------------------------------------------------------------------
void vtkSlicerModelsGUI::Init(void)
{
  vtkSlicerApplicationGUI *appGUI = this->GetApplicationGUI();

  if (appGUI == NULL)
    {
    return;
    }
  
  // get the viewer widget
  this->SetViewerWidget(appGUI->GetViewerWidget());

  // get the interactor style, to set up plotting events
  if (appGUI->GetViewerWidget() != NULL &&
      appGUI->GetViewerWidget()->GetMainViewer() != NULL &&
      appGUI->GetViewerWidget()->GetMainViewer()->GetRenderWindowInteractor() != NULL &&
      appGUI->GetViewerWidget()->GetMainViewer()->GetRenderWindowInteractor()->GetInteractorStyle() != NULL)
    {
    this->SetInteractorStyle(vtkSlicerViewerInteractorStyle::SafeDownCast(appGUI->GetViewerWidget()->GetMainViewer()->GetRenderWindowInteractor()->GetInteractorStyle()));
    }
  else
    {
    vtkErrorMacro("Init: unable to get the interactor style, picking will not work.");
    }
}
*/





//---------------------------------------------------------------------------
void vtkSlicerModelsGUI::WithdrawAddModelWindow ( )
{
  vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast ( this->GetApplication() );
  if ( app && this->AddModelWindow )
    {
    app->Script ( "grab release %s", this->AddModelWindow->GetWidgetName() );
    }

  if ( this->AddModelDialogButton )
    {
    this->AddModelDialogButton->GetLoadSaveDialog()->RemoveObservers ( vtkKWTopLevel::WithdrawEvent, (vtkCommand *)this->GUICallbackCommand );
    }
  if ( this->AddModelDirectoryDialogButton )
    {
    this->AddModelDirectoryDialogButton->GetLoadSaveDialog()->RemoveObservers ( vtkKWTopLevel::WithdrawEvent, (vtkCommand *)this->GUICallbackCommand );
    }
  if ( this->AddModelWindow )
    {
    this->AddModelWindow->Withdraw();
    }
}

//---------------------------------------------------------------------------
void vtkSlicerModelsGUI::RaiseAddModelWindow ( )
{
  //--- create window if not already created.
  vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast ( this->GetApplication() );
  if ( app == NULL )
    {
    vtkErrorMacro ( "RaiseAddModelWindow: got NULL SlicerApplication");
    return;
    }
  vtkSlicerApplicationGUI *appGUI = app->GetApplicationGUI ( );
  if ( appGUI == NULL )
    {
    vtkErrorMacro ( "RaiseAddModelWindow: got NULL SlicerApplicationGUI");
    return;
    }
  vtkSlicerWindow *win = appGUI->GetMainSlicerWindow ();
  if ( win == NULL )
    {
    vtkErrorMacro ( "RaiseAddModelWindow: got NULL MainSlicerWindow");
    return;
    }

  if ( this->LoadDirectory == NULL )
    {
    if ( app )
      {
      if ( app->GetTemporaryDirectory() )
        {
        this->SetLoadDirectory(app->GetTemporaryDirectory() );
        }
      }
    }
  if ( this->AddModelWindow == NULL )
    {
    //-- top level container.
    this->AddModelWindow = vtkKWTopLevel::New();
    this->AddModelWindow->SetMasterWindow (win );
    if ( this->LoadModelButton )
      {
      this->AddModelWindow->SetParent ( this->LoadModelButton);
      }
    this->AddModelWindow->SetApplication ( this->GetApplication() );
    this->AddModelWindow->Create();
    if ( this->GetLoadModelButton() )
      {
      int px, py;
      vtkKWTkUtilities::GetWidgetCoordinates( this->GetLoadModelButton(), &px, &py );
      this->AddModelWindow->SetPosition ( px + 10, py + 10 );
      }
    this->AddModelWindow->SetBorderWidth ( 1 );
    this->AddModelWindow->SetReliefToFlat();
    this->AddModelWindow->SetTitle ( "Add a 3D model");
    this->AddModelWindow->SetSize ( 250, 100 );
    this->AddModelWindow->Withdraw();
    this->AddModelWindow->SetDeleteWindowProtocolCommand ( this, "DestroyAddModelWindow");

    //--- Add model button
    vtkKWLabel *l0 = vtkKWLabel::New();
    l0->SetParent ( this->AddModelWindow );
    l0->Create();
    l0->SetText ( "Select model:" );

    this->AddModelDialogButton = vtkKWLoadSaveButton::New();
    this->AddModelDialogButton->SetParent ( this->AddModelWindow );
    this->AddModelDialogButton->Create();
    if ( this->GetLoadDirectory() == NULL )
      {
      this->AddModelDialogButton->GetLoadSaveDialog()->RetrieveLastPathFromRegistry ("OpenPath");
      const char *lastpath = this->AddModelDialogButton->GetLoadSaveDialog()->GetLastPath();
      if ( lastpath != NULL && !(strcmp(lastpath, "" )) )
        {
//        this->AddModelDialogButton->SetInitialFileName (lastpath);
        }
      }
    else
      {
      this->AddModelDialogButton->GetLoadSaveDialog()->SetLastPath ( this->GetLoadDirectory() );
//      this->AddModelDialogButton->SetInitialFileName ( this->GetLoadDirectory() );
      }
    this->AddModelDialogButton->TrimPathFromFileNameOff();
    this->AddModelDialogButton->SetMaximumFileNameLength (128 );
    this->AddModelDialogButton->GetLoadSaveDialog()->ChooseDirectoryOff();
    this->AddModelDialogButton->GetLoadSaveDialog()->SetFileTypes(
                                                             "{ {model} {*.*} }");    
    this->AddModelDialogButton->SetBalloonHelpString ( "Select a 3D model from a pop-up file browser." );

    //--- Add model directory button
    vtkKWLabel *l1 = vtkKWLabel::New();
    l1->SetParent ( this->AddModelWindow );
    l1->Create();
    l1->SetText ( "Select model directory:" );
    this->AddModelDirectoryDialogButton = vtkKWLoadSaveButton::New();
    this->AddModelDirectoryDialogButton->SetParent ( this->AddModelWindow );
    this->AddModelDirectoryDialogButton->Create();
    if ( this->GetLoadDirectory() == NULL )
      {
      this->AddModelDirectoryDialogButton->GetLoadSaveDialog()->RetrieveLastPathFromRegistry ("OpenPath");
      const char *lastpath = this->AddModelDirectoryDialogButton->GetLoadSaveDialog()->GetLastPath();
      if ( lastpath != NULL && !(strcmp(lastpath, "" )) )
        {
//        this->AddModelDirectoryDialogButton->SetInitialFileName (lastpath);
        }
      }
    else
      {
      this->AddModelDirectoryDialogButton->GetLoadSaveDialog()->SetLastPath ( this->GetLoadDirectory() );
//      this->AddModelDirectoryDialogButton->SetInitialFileName ( this->GetLoadDirectory() );
      }
    this->AddModelDirectoryDialogButton->TrimPathFromFileNameOff();
    this->AddModelDirectoryDialogButton->SetMaximumFileNameLength (128 );
    this->AddModelDirectoryDialogButton->GetLoadSaveDialog()->ChooseDirectoryOn();
    this->AddModelDirectoryDialogButton->SetBalloonHelpString ( "Select a directory from which 3D models (*.vtk) will be loaded." );

    this->Script ( "grid %s -row 0 -column 0 -padx 2 -pady 2 -sticky e", l0->GetWidgetName() );
    this->Script ( "grid %s -row 0 -column 1   -padx 2 -pady 2 -ipadx 2 -ipady 2 -sticky w", this->AddModelDialogButton->GetWidgetName() );
    this->Script ( "grid %s -row 1 -column 0 -padx 2 -pady 2 -sticky e", l1->GetWidgetName() );
    this->Script ( "grid %s -row 1 -column 1   -padx 2 -pady 2 -ipadx 2 -ipady 2 -sticky w", this->AddModelDirectoryDialogButton->GetWidgetName() );
    this->Script ( "grid columnconfigure %s 0 -weight 0", this->AddModelWindow->GetWidgetName() );
    this->Script ( "grid columnconfigure %s 1 -weight 1", this->AddModelWindow->GetWidgetName() );
    
    l0->Delete();
    l1->Delete();
    }

  //--- add observers.
  this->AddModelDialogButton->GetLoadSaveDialog()->AddObserver ( vtkKWTopLevel::WithdrawEvent, (vtkCommand *)this->GUICallbackCommand );
  this->AddModelDirectoryDialogButton->GetLoadSaveDialog()->AddObserver ( vtkKWTopLevel::WithdrawEvent, (vtkCommand *)this->GUICallbackCommand );

  // display
  this->AddModelWindow->DeIconify();
  this->AddModelWindow->Raise();
  if ( app )
    {
    app->Script ( "grab %s", this->AddModelWindow->GetWidgetName() );
    app->ProcessIdleTasks();
    }
  this->Script ( "update idletasks");

}


//---------------------------------------------------------------------------
void vtkSlicerModelsGUI::WithdrawAddScalarOverlayWindow ( )
{
  vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast ( this->GetApplication() );
  if ( app && this->AddOverlayWindow )
    {
    app->Script ( "grab release %s", this->AddOverlayWindow->GetWidgetName() );
    }

  if ( this->AddOverlayDialogButton )
    {
    this->AddOverlayDialogButton->GetLoadSaveDialog()->RemoveObservers ( vtkKWTopLevel::WithdrawEvent, (vtkCommand *)this->GUICallbackCommand );
    }
  if ( this->ModelSelector )
    {
    this->ModelSelector->RemoveObservers ( vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );
    }
  if ( this->AddOverlayWindow )
    {
    this->AddOverlayWindow->Withdraw();
    }
}


//---------------------------------------------------------------------------
void vtkSlicerModelsGUI::RaiseAddScalarOverlayWindow ( )
{
  //--- create window if not already created.
  vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast ( this->GetApplication() );
  if ( app == NULL )
    {
    vtkErrorMacro ( "RaiseAddScalarOverlayWindow: got NULL SlicerApplication");
    return;
    }
  vtkSlicerApplicationGUI *appGUI = app->GetApplicationGUI ( );
  if ( appGUI == NULL )
    {
    vtkErrorMacro ( "RaiseAddScalarOverlayWindow: got NULL SlicerApplicationGUI");
    return;
    }
  vtkSlicerWindow *win = appGUI->GetMainSlicerWindow ();
  if ( win == NULL )
    {
    vtkErrorMacro ( "RaiseAddScalarOverlayWindow: got NULL MainSlicerWindow");
    return;
    }

  if ( this->LoadDirectory == NULL )
    {
    if ( app )
      {
      if ( app->GetTemporaryDirectory() )
        {
        this->SetLoadDirectory(app->GetTemporaryDirectory() );
        }
      }
    }
  
    if ( this->AddOverlayWindow == NULL )
    {
    //-- top level container.
    this->AddOverlayWindow = vtkKWTopLevel::New();
    this->AddOverlayWindow->SetMasterWindow (win );
    this->AddOverlayWindow->SetApplication ( this->GetApplication() );
    this->AddOverlayWindow->Create();
    if ( this->LoadScalarsButton )
      {
      int px, py;
      vtkKWTkUtilities::GetWidgetCoordinates( this->LoadScalarsButton, &px, &py );
      this->AddOverlayWindow->SetPosition ( px + 10, py + 10 );
      }
    this->AddOverlayWindow->SetBorderWidth ( 1 );
    this->AddOverlayWindow->SetReliefToFlat();
    this->AddOverlayWindow->SetTitle ( "Add a scalar overlay to a 3D model");
    this->AddOverlayWindow->SetSize ( 380, 150 );
    this->AddOverlayWindow->Withdraw();
    this->AddOverlayWindow->SetDeleteWindowProtocolCommand ( this, "DestroyAddScalarOverlayWindow");

    //-- create node selector
    this->ModelSelector = vtkSlicerNodeSelectorWidget::New();
    this->ModelSelector->SetParent( this->AddOverlayWindow );
    this->ModelSelector->Create();
    this->ModelSelector->AddNodeClass("vtkMRMLModelNode", NULL, NULL, NULL);
    this->ModelSelector->SetChildClassesEnabled(0);
    this->ModelSelector->SetShowHidden (1);
    this->ModelSelector->SetMRMLScene(this->GetMRMLScene());
    this->ModelSelector->GetWidget()->GetWidget()->SetWidth (24 );
    this->ModelSelector->GetWidget()->GetWidget()->IndicatorVisibilityOff();
    this->ModelSelector->SetBorderWidth(2);
    this->ModelSelector->SetPadX(2);
    this->ModelSelector->SetPadY(2);
    this->ModelSelector->SetLabelText( "Select model for overlay:");
    this->ModelSelector->UnconditionalUpdateMenu();
    this->ModelSelector->SetBalloonHelpString("Select a model (from the scene) to which the overlay will be applied.");

    //--- Add model button
    vtkKWLabel *l1 = vtkKWLabel::New();
    l1->SetParent ( this->AddOverlayWindow );
    l1->Create();
    l1->SetText ( "  Select a scalar overlay:" );
    this->AddOverlayDialogButton = vtkKWLoadSaveButton::New();
    this->AddOverlayDialogButton->SetParent ( this->AddOverlayWindow );
    this->AddOverlayDialogButton->Create();
    if ( this->GetLoadDirectory() == NULL )
      {
      this->AddOverlayDialogButton->GetLoadSaveDialog()->RetrieveLastPathFromRegistry ("OpenPath");
      const char *lastpath = this->AddOverlayDialogButton->GetLoadSaveDialog()->GetLastPath();
      if ( lastpath != NULL && !(strcmp(lastpath, "" )) )
        {
//        this->AddOverlayDialogButton->SetInitialFileName (lastpath);
        }
      }
    else
      {
      this->AddOverlayDialogButton->GetLoadSaveDialog()->SetLastPath ( this->GetLoadDirectory() );
//      this->AddOverlayDialogButton->SetInitialFileName ( this->GetLoadDirectory() );
      }
    this->AddOverlayDialogButton->TrimPathFromFileNameOff();
    this->AddOverlayDialogButton->SetMaximumFileNameLength (128 );
    this->AddOverlayDialogButton->GetLoadSaveDialog()->ChooseDirectoryOff();
    this->AddOverlayDialogButton->GetLoadSaveDialog()->SetFileTypes("{ {All} {.*} } { {Thickness} {.thickness} } { {Curve} {.curv} } { {Average Curve} {.avg_curv} } { {Sulc} {.sulc} } { {Area} {.area} } { {W} {.w} } { {Parcellation Annotation} {.annot} } { {Volume} {.mgz .mgh} } { {Label} {.label} }");
    this->AddOverlayDialogButton->SetBalloonHelpString ( "Select a scalar overlay and apply it to the selected model." );

    this->Script ( "grid %s -row 0 -column 0 -columnspan 2  -padx 2 -pady 2 -ipadx 2 -ipady 2 -sticky w", this->ModelSelector->GetWidgetName() );
    this->Script ( "grid %s -row 1 -column 0 -padx 2 -pady 2 -sticky w", l1->GetWidgetName() );
    this->Script ( "grid %s -row 1 -column 1   -padx 2 -pady 2 -ipadx 2 -ipady 2 -sticky w", this->AddOverlayDialogButton->GetWidgetName() );
    this->Script ( "grid columnconfigure %s 0 -weight 0", this->AddOverlayWindow->GetWidgetName() );
    this->Script ( "grid columnconfigure %s 1 -weight 1", this->AddOverlayWindow->GetWidgetName() );
    
    l1->Delete();
    }

  //--- add observers.
  this->AddOverlayDialogButton->GetLoadSaveDialog()->AddObserver ( vtkKWTopLevel::WithdrawEvent, (vtkCommand *)this->GUICallbackCommand );
  this->ModelSelector->AddObserver (vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );

  // display
  this->AddOverlayWindow->DeIconify();
  this->AddOverlayWindow->Raise();
  if ( app )
    {
    app->Script ( "grab %s", this->AddOverlayWindow->GetWidgetName() );
    app->ProcessIdleTasks();
    }
  this->Script ( "update idletasks");

}
  

//---------------------------------------------------------------------------
  void vtkSlicerModelsGUI::DestroyAddModelWindow ( )
{
  if ( !this->AddModelWindow )
    {
    return;
    }
  if ( ! (this->AddModelWindow->IsCreated()) )
    {
    vtkErrorMacro ( "DestroyAddModelWindow: AddModelWindow is not created." );
    return;
    }
  vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast ( this->GetApplication() );
  if ( app )
    {
    app->Script ( "grab release %s", this->AddModelWindow->GetWidgetName() );
    }
  this->AddModelWindow->Withdraw();

  if ( this->AddModelDialogButton )
    {
    this->AddModelDialogButton->GetLoadSaveDialog()->RemoveObservers ( vtkKWTopLevel::WithdrawEvent, (vtkCommand *)this->GUICallbackCommand );
    this->AddModelDialogButton->SetParent ( NULL );
    this->AddModelDialogButton->Delete();
    this->AddModelDialogButton = NULL;    
    }
  if ( this->AddModelDirectoryDialogButton )
    {
    this->AddModelDirectoryDialogButton->GetLoadSaveDialog()->RemoveObservers ( vtkKWTopLevel::WithdrawEvent, (vtkCommand *)this->GUICallbackCommand );
    this->AddModelDirectoryDialogButton->SetParent ( NULL);
    this->AddModelDirectoryDialogButton->Delete();
    this->AddModelDirectoryDialogButton = NULL;    
    }
  if ( this->AddModelWindow )
    {
    this->AddModelWindow->Delete();
    this->AddModelWindow = NULL;
    }
}

//---------------------------------------------------------------------------
void vtkSlicerModelsGUI::DestroyAddScalarOverlayWindow ( )
{
  if ( !this->AddOverlayWindow )
    {
    return;
    }
  if ( ! (this->AddOverlayWindow->IsCreated()) )
    {
    vtkErrorMacro ( "DestroyAddOverlayWindow: AddOverlayWindow is not created." );
    return;
    }
  vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast ( this->GetApplication() );
  if ( app )
    {
    app->Script ( "grab release %s", this->AddOverlayWindow->GetWidgetName() );
    }
  this->AddOverlayWindow->Withdraw();

  if ( this->ModelSelector  )
    {
    this->ModelSelector->RemoveObservers (vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->ModelSelector->SetParent ( NULL );
    this->ModelSelector->Delete();
    this->ModelSelector = NULL;
    }

  if ( this->AddOverlayDialogButton )
    {
    this->AddOverlayDialogButton->GetLoadSaveDialog()->RemoveObservers ( vtkKWTopLevel::WithdrawEvent, (vtkCommand *)this->GUICallbackCommand );
    this->AddOverlayDialogButton->SetParent (NULL);
    this->AddOverlayDialogButton->Delete();
    this->AddOverlayDialogButton = NULL;
    }

  if ( this->AddOverlayWindow )
    {
    this->AddOverlayWindow->Delete();
    this->AddOverlayWindow = NULL;
    }
}

