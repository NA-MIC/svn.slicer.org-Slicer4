#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkCommand.h"
#include "vtkKWWidget.h"
#include "vtkSlicerUnstructuredGridsGUI.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerModuleLogic.h"
//#include "vtkSlicerUnstructuredGridsLogic.h"
#include "vtkSlicerUnstructuredGridDisplayWidget.h"
#include "vtkSlicerModuleCollapsibleFrame.h"

#include "vtkKWFrameWithLabel.h"
#include "vtkKWMenuButton.h"
#include "vtkKWMessageDialog.h"

// for scalars
#include "vtkPointData.h"

//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkSlicerUnstructuredGridsGUI );
vtkCxxRevisionMacro ( vtkSlicerUnstructuredGridsGUI, "$Revision: 1.0 $");


//---------------------------------------------------------------------------
vtkSlicerUnstructuredGridsGUI::vtkSlicerUnstructuredGridsGUI ( )
{

  // classes not yet defined!
  this->Logic = NULL;
  //this->UnstructuredGridNode = NULL;
  this->LoadUnstructuredGridButton = NULL;
  this->LoadUnstructuredGridDirectoryButton = NULL;
  this->SaveUnstructuredGridButton = NULL;
  this->UnstructuredGridSelectorWidget = NULL;
  this->UnstructuredGridDisplayWidget = NULL;
  this->LoadScalarsButton = NULL;
  this->UnstructuredGridDisplaySelectorWidget = NULL;
  this->UnstructuredGridDisplayFrame = NULL;

  NACLabel = NULL;
  NAMICLabel =NULL;
  NCIGTLabel = NULL;
  BIRNLabel = NULL;
}


//---------------------------------------------------------------------------
vtkSlicerUnstructuredGridsGUI::~vtkSlicerUnstructuredGridsGUI ( )
{
  this->RemoveGUIObservers();

  this->SetModuleLogic ( NULL );

  if (this->UnstructuredGridDisplaySelectorWidget)
    {
    this->UnstructuredGridDisplaySelectorWidget->SetParent(NULL);
    this->UnstructuredGridDisplaySelectorWidget->Delete();
    this->UnstructuredGridDisplaySelectorWidget = NULL;
    }
  if (this->LoadUnstructuredGridButton ) 
    {
    this->LoadUnstructuredGridButton->SetParent(NULL);
    this->LoadUnstructuredGridButton->Delete ( );
    }    
  if (this->LoadUnstructuredGridDirectoryButton ) 
    {
    this->LoadUnstructuredGridDirectoryButton->SetParent(NULL);
    this->LoadUnstructuredGridDirectoryButton->Delete ( );
    }    
  if (this->SaveUnstructuredGridButton ) 
    {
    this->SaveUnstructuredGridButton->SetParent(NULL);
    this->SaveUnstructuredGridButton->Delete ( );
    }
  if (this->UnstructuredGridSelectorWidget ) 
    {
    this->UnstructuredGridSelectorWidget->SetParent(NULL);
    this->UnstructuredGridSelectorWidget->Delete ( );
    }
  if (this->UnstructuredGridDisplayWidget ) 
    {
    this->UnstructuredGridDisplayWidget->SetParent(NULL);
    this->UnstructuredGridDisplayWidget->Delete ( );
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
  if (this->UnstructuredGridDisplayFrame)
    {
    this->UnstructuredGridDisplayFrame->SetParent ( NULL );
    this->UnstructuredGridDisplayFrame->Delete();
    }
  this->Built = false;
}


//---------------------------------------------------------------------------
void vtkSlicerUnstructuredGridsGUI::PrintSelf ( ostream& os, vtkIndent indent )
{
    this->vtkObject::PrintSelf ( os, indent );

    os << indent << "SlicerUnstructuredGridsGUI: " << this->GetClassName ( ) << "\n";
}



//---------------------------------------------------------------------------
void vtkSlicerUnstructuredGridsGUI::RemoveGUIObservers ( )
{
  if (this->LoadUnstructuredGridButton)
    {
    this->LoadUnstructuredGridButton->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }
  if (this->LoadUnstructuredGridDirectoryButton)
    {
    this->LoadUnstructuredGridDirectoryButton->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }
  if (this->SaveUnstructuredGridButton)
    {
    this->SaveUnstructuredGridButton->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }
  if (this->LoadScalarsButton)
    {
    this->LoadScalarsButton->GetWidget()->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }
  if (this->UnstructuredGridDisplaySelectorWidget)
    {
    this->UnstructuredGridDisplaySelectorWidget->RemoveObservers (vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );
    }
}


//---------------------------------------------------------------------------
void vtkSlicerUnstructuredGridsGUI::AddGUIObservers ( )
{
  this->LoadUnstructuredGridButton->AddObserver ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
  this->LoadUnstructuredGridDirectoryButton->AddObserver ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
  this->SaveUnstructuredGridButton->AddObserver ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
  this->LoadScalarsButton->GetWidget()->AddObserver ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
  this->UnstructuredGridDisplaySelectorWidget->AddObserver (vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );
}



//---------------------------------------------------------------------------
void vtkSlicerUnstructuredGridsGUI::ProcessGUIEvents ( vtkObject *caller,
                                            unsigned long event, void *callData )
{


  if (vtkSlicerNodeSelectorWidget::SafeDownCast(caller) == this->UnstructuredGridDisplaySelectorWidget && 
        event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent ) 
    {
    vtkMRMLUnstructuredGridNode *UnstructuredGrid = 
        vtkMRMLUnstructuredGridNode::SafeDownCast(this->UnstructuredGridDisplaySelectorWidget->GetSelected());

    if (UnstructuredGrid != NULL && UnstructuredGrid->GetDisplayNode() != NULL)
      {
      this->UnstructuredGridDisplayWidget->SetUnstructuredGridDisplayNode(UnstructuredGrid->GetUnstructuredGridDisplayNode());
      this->UnstructuredGridDisplayWidget->SetUnstructuredGridNode(UnstructuredGrid);
      }
    return;
    }

  vtkKWLoadSaveButton *filebrowse = vtkKWLoadSaveButton::SafeDownCast(caller);
  if (filebrowse == this->LoadUnstructuredGridButton  && event == vtkKWPushButton::InvokedEvent )
    {
    // If a file has been selected for loading...
    const char *fileName = filebrowse->GetFileName();
    if ( fileName ) 
      {
      vtkSlicerUnstructuredGridsLogic* UnstructuredGridLogic = this->Logic;
      
      vtkMRMLUnstructuredGridNode *UnstructuredGridNode = UnstructuredGridLogic->AddUnstructuredGrid( fileName );
      if ( UnstructuredGridNode == NULL ) 
        {
        vtkKWMessageDialog *dialog = vtkKWMessageDialog::New();
        dialog->SetParent ( this->UIPanel->GetPageWidget ( "UnstructuredGrids" ) );
        dialog->SetStyleToMessage();
        std::string msg = std::string("Unable to read UnstructuredGrid file ") + std::string(fileName);
        dialog->SetText(msg.c_str());
        dialog->Create ( );
        dialog->Invoke();
        dialog->Delete();

        vtkErrorMacro("Unable to read UnstructuredGrid file " << fileName);
        // reset the file browse button text
        }
      else
        {
        filebrowse->GetLoadSaveDialog()->SaveLastPathToRegistry("OpenPath");
        
        }
      }
      this->LoadUnstructuredGridButton->SetText ("Load UnstructuredGrid");
    return;
    }
    else if (filebrowse == this->LoadUnstructuredGridDirectoryButton  && event == vtkKWPushButton::InvokedEvent )
    {
    // If a file has been selected for loading...
    const char *fileName = filebrowse->GetFileName();
    if ( fileName ) 
      {
        vtkSlicerUnstructuredGridsLogic* UnstructuredGridLogic = this->Logic;
      
      if (UnstructuredGridLogic->AddUnstructuredGrids( fileName, ".vtk") == 0)
        {
        vtkKWMessageDialog *dialog = vtkKWMessageDialog::New();
        dialog->SetParent ( this->UIPanel->GetPageWidget ( "UnstructuredGrids" ) );
        dialog->SetStyleToMessage();
        std::string msg = std::string("Unable to read UnstructuredGrids directory ") + std::string(fileName);
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
    this->LoadUnstructuredGridDirectoryButton->SetText ("Load UnstructuredGrid Directory");
    return;
    }
  else if (filebrowse == this->SaveUnstructuredGridButton  && event == vtkKWPushButton::InvokedEvent )
      {
      // If a file has been selected for saving...
      const char *fileName = filebrowse->GetFileName();
      if ( fileName ) 
      {
          vtkSlicerUnstructuredGridsLogic* UnstructuredGridLogic = this->Logic;
        vtkMRMLUnstructuredGridNode *volNode = vtkMRMLUnstructuredGridNode::SafeDownCast(this->UnstructuredGridSelectorWidget->GetSelected());
        if ( !UnstructuredGridLogic->SaveUnstructuredGrid( fileName, volNode ))
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
  else if (filebrowse == this->LoadScalarsButton->GetWidget()  && event == vtkKWPushButton::InvokedEvent )
    {
    // If a scalar file has been selected for loading...
    const char *fileName = filebrowse->GetFileName();
    if ( fileName ) 
      {
      // get the UnstructuredGrid from the display widget rather than this gui's save
      // UnstructuredGrid selector
      vtkMRMLUnstructuredGridNode *UnstructuredGridNode = vtkMRMLUnstructuredGridNode::SafeDownCast(this->UnstructuredGridDisplaySelectorWidget->GetSelected());
      if (UnstructuredGridNode != NULL)
        {
        vtkDebugMacro("vtkSlicerUnstructuredGridsGUI: loading scalar for UnstructuredGrid " << UnstructuredGridNode->GetName());
        // load the scalars
        vtkSlicerUnstructuredGridsLogic* UnstructuredGridLogic = this->Logic;
        if (!UnstructuredGridLogic->AddScalar(fileName, UnstructuredGridNode))
          {
          vtkKWMessageDialog *dialog = vtkKWMessageDialog::New();
          dialog->SetParent ( this->UIPanel->GetPageWidget ( "UnstructuredGrids" ) );
          dialog->SetStyleToMessage();
          std::string msg = std::string("Unable to read scalars file ") + std::string(fileName);
          dialog->SetText(msg.c_str());
          dialog->Create ( );
          dialog->Invoke();
          dialog->Delete();
          
          vtkErrorMacro("Error loading scalar overlay file " << fileName);
          this->LoadScalarsButton->GetWidget()->SetText ("None");
          }
        else
          {
          filebrowse->GetLoadSaveDialog()->SaveLastPathToRegistry("OpenPath");
          // set the active scalar in the display node to this one
          // - is done in the UnstructuredGrid storage node         
          }
        }
      }
    return;
    }
}    

//---------------------------------------------------------------------------
void vtkSlicerUnstructuredGridsGUI::ProcessLogicEvents ( vtkObject *caller,
                                              unsigned long event, void *callData )
{
    // Fill in
}

//---------------------------------------------------------------------------
void vtkSlicerUnstructuredGridsGUI::ProcessMRMLEvents ( vtkObject *caller,
                                             unsigned long event, void *callData )
{
    // Fill in
}


//---------------------------------------------------------------------------
void vtkSlicerUnstructuredGridsGUI::CreateModuleEventBindings ( )
{
}

//---------------------------------------------------------------------------
void vtkSlicerUnstructuredGridsGUI::ReleaseModuleEventBindings ( )
{
  
}


//---------------------------------------------------------------------------
void vtkSlicerUnstructuredGridsGUI::Enter ( )
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
void vtkSlicerUnstructuredGridsGUI::Exit ( )
{
  this->ReleaseModuleEventBindings();
}


//---------------------------------------------------------------------------
void vtkSlicerUnstructuredGridsGUI::TearDownGUI ( )
{
  this->Exit();
  if ( this->Built )
    {
    this->RemoveGUIObservers();
    }
}


//---------------------------------------------------------------------------
void vtkSlicerUnstructuredGridsGUI::BuildGUI ( )
{

    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
  
    // ---
    // MODULE GUI FRAME 
    // configure a page for a UnstructuredGrid loading UI for now.
    // later, switch on the modulesButton in the SlicerControlGUI
    // ---
    // create a page
    this->UIPanel->AddPage ( "UnstructuredGrids", "UnstructuredGrids", NULL );
    
    // Define your help text and build the help frame here.
    const char *help = "The UnstructuredGrids Module loads, saves and adjusts display parameters of UnstructuredGrids. ";
    const char *about = "This work was supported by NA-MIC, NAC, BIRN, NCIGT, and the Slicer Community. See http://www.slicer.org for details. ";
    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "UnstructuredGrids" );
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
    modLoadFrame->SetParent ( this->UIPanel->GetPageWidget ( "UnstructuredGrids" ) );
    modLoadFrame->Create ( );
    modLoadFrame->SetLabelText ("Load");
    modLoadFrame->ExpandFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  modLoadFrame->GetWidgetName(), this->UIPanel->GetPageWidget("UnstructuredGrids")->GetWidgetName());

    // add a file browser 
    this->LoadUnstructuredGridButton = vtkKWLoadSaveButton::New ( );
    this->LoadUnstructuredGridButton->SetParent ( modLoadFrame->GetFrame() );
    this->LoadUnstructuredGridButton->Create ( );
    this->LoadUnstructuredGridButton->SetText ("Load UnstructuredGrid");
    this->LoadUnstructuredGridButton->GetLoadSaveDialog()->SetTitle("Open UnstructuredGrid");
    this->LoadUnstructuredGridButton->GetLoadSaveDialog()->RetrieveLastPathFromRegistry("OpenPath");
    this->LoadUnstructuredGridButton->GetLoadSaveDialog()->SetFileTypes(
                                                             "{ {UnstructuredGrid} {*.*} }");
    app->Script("pack %s -side left -anchor w -padx 2 -pady 4", 
                this->LoadUnstructuredGridButton->GetWidgetName());

   // add a file browser 
    this->LoadUnstructuredGridDirectoryButton = vtkKWLoadSaveButton::New ( );
    this->LoadUnstructuredGridDirectoryButton->SetParent ( modLoadFrame->GetFrame() );
    this->LoadUnstructuredGridDirectoryButton->Create ( );
    this->LoadUnstructuredGridDirectoryButton->SetText ("Load UnstructuredGrid Directory");
    this->LoadUnstructuredGridDirectoryButton->GetLoadSaveDialog()->ChooseDirectoryOn();
    app->Script("pack %s -side left -anchor w -padx 2 -pady 4", 
                this->LoadUnstructuredGridDirectoryButton->GetWidgetName());

  
    // DISPLAY FRAME            
    this->UnstructuredGridDisplayFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    this->UnstructuredGridDisplayFrame->SetParent ( this->UIPanel->GetPageWidget ( "UnstructuredGrids" ) );
    this->UnstructuredGridDisplayFrame->Create ( );
    this->UnstructuredGridDisplayFrame->SetLabelText ("Display");
    this->UnstructuredGridDisplayFrame->CollapseFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  this->UnstructuredGridDisplayFrame->GetWidgetName(), this->UIPanel->GetPageWidget("UnstructuredGrids")->GetWidgetName());

    this->LoadScalarsButton = vtkKWLoadSaveButtonWithLabel::New();
    this->LoadScalarsButton->SetParent ( this->UnstructuredGridDisplayFrame->GetFrame() );
    this->LoadScalarsButton->Create ( );
    this->LoadScalarsButton->SetLabelText ("Load FreeSurfer Overlay: ");
    this->LoadScalarsButton->GetWidget()->SetText ("None");
    this->LoadScalarsButton->GetWidget()->GetLoadSaveDialog()->SetTitle("Open FreeSurfer Overlay");
    this->LoadScalarsButton->GetWidget()->GetLoadSaveDialog()->RetrieveLastPathFromRegistry("OpenPath");
    this->LoadScalarsButton->GetWidget()->GetLoadSaveDialog()->SetFileTypes("{ {All} {.*} } { {Thickness} {.thickness} } { {Curve} {.curv} } { {Average Curve} {.avg_curv} } { {Sulc} {.sulc} } { {Area} {.area} } { {W} {.w} } { {Parcellation Annotation} {.annot} } { {Volume} {.mgz .mgh} }");
    app->Script("pack %s -side top -anchor nw -padx 2 -pady 4", 
                this->LoadScalarsButton->GetWidgetName());

    this->UnstructuredGridDisplaySelectorWidget = vtkSlicerNodeSelectorWidget::New() ;
    this->UnstructuredGridDisplaySelectorWidget->SetParent ( this->UnstructuredGridDisplayFrame->GetFrame() );
    this->UnstructuredGridDisplaySelectorWidget->Create ( );
    this->UnstructuredGridDisplaySelectorWidget->SetNodeClass("vtkMRMLUnstructuredGridNode", NULL, NULL, NULL);
    // CRL - added to see the FE mesh surfaces
    this->UnstructuredGridDisplaySelectorWidget->SetChildClassesEnabled(1);
    this->UnstructuredGridDisplaySelectorWidget->SetMRMLScene(this->GetMRMLScene());
    this->UnstructuredGridDisplaySelectorWidget->SetBorderWidth(2);
    // this->UnstructuredGridDisplaySelectorWidget->SetReliefToGroove();
    this->UnstructuredGridDisplaySelectorWidget->SetPadX(2);
    this->UnstructuredGridDisplaySelectorWidget->SetPadY(2);
    this->UnstructuredGridDisplaySelectorWidget->GetWidget()->GetWidget()->IndicatorVisibilityOff();
    this->UnstructuredGridDisplaySelectorWidget->GetWidget()->GetWidget()->SetWidth(24);
    this->UnstructuredGridDisplaySelectorWidget->SetLabelText( "UnstructuredGrid Select: ");
    this->UnstructuredGridDisplaySelectorWidget->SetBalloonHelpString("select a UnstructuredGrid from the current mrml scene.");
    this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                   this->UnstructuredGridDisplaySelectorWidget->GetWidgetName());


    this->UnstructuredGridDisplayWidget = vtkSlicerUnstructuredGridDisplayWidget::New ( );
    this->UnstructuredGridDisplayWidget->SetMRMLScene(this->GetMRMLScene() );
    this->UnstructuredGridDisplayWidget->SetParent ( this->UnstructuredGridDisplayFrame->GetFrame() );
    this->UnstructuredGridDisplayWidget->Create ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  this->UnstructuredGridDisplayWidget->GetWidgetName(), 
                  this->UnstructuredGridDisplayFrame->GetFrame()->GetWidgetName());

    // Clip FRAME  
    vtkSlicerModuleCollapsibleFrame *clipFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    clipFrame->SetParent ( this->UIPanel->GetPageWidget ( "UnstructuredGrids" ) );
    clipFrame->Create ( );
    clipFrame->SetLabelText ("Clipping");
    clipFrame->CollapseFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  clipFrame->GetWidgetName(), this->UIPanel->GetPageWidget("UnstructuredGrids")->GetWidgetName());


    

    // ---
    // Save FRAME            
    vtkSlicerModuleCollapsibleFrame *UnstructuredGridSaveFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    UnstructuredGridSaveFrame->SetParent ( this->UIPanel->GetPageWidget ( "UnstructuredGrids" ) );
    UnstructuredGridSaveFrame->Create ( );
    UnstructuredGridSaveFrame->SetLabelText ("Save");
    UnstructuredGridSaveFrame->CollapseFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  UnstructuredGridSaveFrame->GetWidgetName(), 
                  this->UIPanel->GetPageWidget ( "UnstructuredGrids" )->GetWidgetName());

    // selector for save
    this->UnstructuredGridSelectorWidget = vtkSlicerNodeSelectorWidget::New() ;
    this->UnstructuredGridSelectorWidget->SetParent ( UnstructuredGridSaveFrame->GetFrame() );
    this->UnstructuredGridSelectorWidget->Create ( );
    this->UnstructuredGridSelectorWidget->SetNodeClass("vtkMRMLUnstructuredGridNode", NULL, NULL, NULL);
    this->UnstructuredGridSelectorWidget->SetMRMLScene(this->GetMRMLScene());
    this->UnstructuredGridSelectorWidget->SetBorderWidth(2);
    this->UnstructuredGridSelectorWidget->SetPadX(2);
    this->UnstructuredGridSelectorWidget->SetPadY(2);
    this->UnstructuredGridSelectorWidget->GetWidget()->GetWidget()->IndicatorVisibilityOff();
    this->UnstructuredGridSelectorWidget->GetWidget()->GetWidget()->SetWidth(24);
    this->UnstructuredGridSelectorWidget->SetLabelText( "UnstructuredGrid To Save: ");
    this->UnstructuredGridSelectorWidget->SetBalloonHelpString("select a UnstructuredGrid from the current  scene.");
    this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                  this->UnstructuredGridSelectorWidget->GetWidgetName());

    this->SaveUnstructuredGridButton = vtkKWLoadSaveButton::New ( );
    this->SaveUnstructuredGridButton->SetParent ( UnstructuredGridSaveFrame->GetFrame() );
    this->SaveUnstructuredGridButton->Create ( );
    this->SaveUnstructuredGridButton->SetText ("Save UnstructuredGrid");
    this->SaveUnstructuredGridButton->GetLoadSaveDialog()->SaveDialogOn();
    this->SaveUnstructuredGridButton->GetLoadSaveDialog()->SetFileTypes(
                                                              "{ {UnstructuredGrid} {.*} }");
    this->SaveUnstructuredGridButton->GetLoadSaveDialog()->RetrieveLastPathFromRegistry(
      "OpenPath");
     app->Script("pack %s -side top -anchor w -padx 2 -pady 4", 
                this->SaveUnstructuredGridButton->GetWidgetName());
    

   this->ProcessGUIEvents (this->UnstructuredGridDisplaySelectorWidget,
                          vtkSlicerNodeSelectorWidget::NodeSelectedEvent, NULL );


    modLoadFrame->Delete ( );
    clipFrame->Delete ( );
    UnstructuredGridSaveFrame->Delete();
    hierFrame->Delete ( );
}





