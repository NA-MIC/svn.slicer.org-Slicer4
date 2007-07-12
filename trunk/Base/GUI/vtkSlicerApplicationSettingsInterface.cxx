#include "vtkSlicerApplicationSettingsInterface.h"
#include "vtkObjectFactory.h"
#include "vtkKWWidget.h"
#include "vtkKWFrame.h"
#include "vtkKWMenu.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWEntryWithLabel.h"
#include "vtkKWEntry.h"
#include "vtkKWLoadSaveDialog.h"
#include "vtkKWLoadSaveButton.h"
#include "vtkKWLoadSaveButtonWithLabel.h"
#include "vtkKWCheckButton.h"
#include "vtkSlicerApplication.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkSlicerApplicationSettingsInterface );
vtkCxxRevisionMacro(vtkSlicerApplicationSettingsInterface, "$Revision: 1.0 $");

//----------------------------------------------------------------------------
vtkSlicerApplicationSettingsInterface::vtkSlicerApplicationSettingsInterface()
{
  this->SlicerSettingsFrame = NULL;
  this->ConfirmDeleteCheckButton = NULL;
    
  this->ModuleSettingsFrame = NULL;
  this->ModulePathEntry = NULL;
  this->HomeModuleEntry = NULL;
  this->TemporaryDirectoryButton = NULL;
  this->LoadCommandLineModulesCheckButton = NULL;
  this->EnableDaemonCheckButton = NULL;
}

//----------------------------------------------------------------------------
vtkSlicerApplicationSettingsInterface::~vtkSlicerApplicationSettingsInterface()
{
  if (this->SlicerSettingsFrame)
    {
    this->SlicerSettingsFrame->Delete();
    this->SlicerSettingsFrame = 0;
    }

  if (this->ConfirmDeleteCheckButton)
    {
    this->ConfirmDeleteCheckButton->Delete();
    this->ConfirmDeleteCheckButton = NULL;
    }
  
  if (this->ModuleSettingsFrame)
    {
    this->ModuleSettingsFrame->Delete();
    this->ModuleSettingsFrame = 0;
    }

  if (this->ModulePathEntry)
    {
    this->ModulePathEntry->Delete();
    this->ModulePathEntry = 0;
    }

  if (this->HomeModuleEntry)
    {
    this->HomeModuleEntry->Delete();
    this->HomeModuleEntry = 0;
    }

  if (this->TemporaryDirectoryButton)
    {
    this->TemporaryDirectoryButton->Delete();
    this->TemporaryDirectoryButton = 0;
    }

  if (this->LoadCommandLineModulesCheckButton)
    {
    this->LoadCommandLineModulesCheckButton->Delete();
    this->LoadCommandLineModulesCheckButton = NULL;
    }

  if (this->EnableDaemonCheckButton)
    {
    this->EnableDaemonCheckButton->Delete();
    this->EnableDaemonCheckButton = NULL;
    }
}

//----------------------------------------------------------------------------
void vtkSlicerApplicationSettingsInterface::Create()
{
  if (this->IsCreated())
    {
    vtkErrorMacro("The panel is already created.");
    return;
    }

  // Create the superclass instance (and set the application)

  this->Superclass::Create();

  ostrstream tk_cmd;
  vtkKWWidget *page;
  vtkKWFrame *frame;

  int label_width = 20;

  // --------------------------------------------------------------
  // Add a "Preferences" page

  this->AddPage(this->GetName());
  page = this->GetPageWidget(this->GetName());

  // --------------------------------------------------------------
  // Slicer Interface settings : main frame
  if (!this->SlicerSettingsFrame)
    {
    this->SlicerSettingsFrame = vtkKWFrameWithLabel::New();
    }
  this->SlicerSettingsFrame->SetParent(this->GetPagesParentWidget());
  this->SlicerSettingsFrame->Create();
  this->SlicerSettingsFrame->SetLabelText("Slicer Settings");

  tk_cmd << "pack " << this->SlicerSettingsFrame->GetWidgetName()
         << " -side top -anchor nw -fill x -padx 2 -pady 2 " 
         << " -in " << page->GetWidgetName() << endl;
  
  frame = this->SlicerSettingsFrame->GetFrame();
  
  // --------------------------------------------------------------
  // Slicer interface settings : Confirm on delete ?

  if (!this->ConfirmDeleteCheckButton)
    {
    this->ConfirmDeleteCheckButton = vtkKWCheckButton::New();
    }
  this->ConfirmDeleteCheckButton->SetParent(frame);
  this->ConfirmDeleteCheckButton->Create();
  this->ConfirmDeleteCheckButton->SetText("Confirm delete");
  this->ConfirmDeleteCheckButton->SetCommand(this, "ConfirmDeleteCallback");
  this->ConfirmDeleteCheckButton->SetBalloonHelpString(
    "A confirmation dialog will be presented to the user on deleting nodes.");

  tk_cmd << "pack " << this->ConfirmDeleteCheckButton->GetWidgetName()
         << "  -side top -anchor w -expand no -fill none" << endl;

  // --------------------------------------------------------------
  // Slicer interface settings : Load Daemon?

  if (!this->EnableDaemonCheckButton)
    {
    this->EnableDaemonCheckButton = vtkKWCheckButton::New();
    }
  this->EnableDaemonCheckButton->SetParent(frame);
  this->EnableDaemonCheckButton->Create();
  this->EnableDaemonCheckButton->SetText("Enable Slicer Daemon");
  this->EnableDaemonCheckButton->SetCommand(this, "EnableDaemonCallback");
  this->EnableDaemonCheckButton->SetBalloonHelpString(
    "The Slicer Daemon will be enabled at startup.\nThis feature allows external programs to connect to a network port opened by Slicer.\nA dialog box will appear when the first connection is made giving you the option to allow connections or not.");

  tk_cmd << "pack " << this->EnableDaemonCheckButton->GetWidgetName()
         << "  -side top -anchor w -expand no -fill none" << endl;
  
  // --------------------------------------------------------------
  // Module Interface settings : main frame

  if (!this->ModuleSettingsFrame)
    {
    this->ModuleSettingsFrame = vtkKWFrameWithLabel::New();
    }
  this->ModuleSettingsFrame->SetParent(this->GetPagesParentWidget());
  this->ModuleSettingsFrame->Create();
  this->ModuleSettingsFrame->SetLabelText("Module Settings");
    
  tk_cmd << "pack " << this->ModuleSettingsFrame->GetWidgetName()
         << " -side top -anchor nw -fill x -padx 2 -pady 2 " 
         << " -in " << page->GetWidgetName() << endl;
  
  frame = this->ModuleSettingsFrame->GetFrame();

  // --------------------------------------------------------------
  // Module settings : Load modules on startup ?

  if (!this->LoadCommandLineModulesCheckButton)
    {
    this->LoadCommandLineModulesCheckButton = vtkKWCheckButton::New();
    }
  this->LoadCommandLineModulesCheckButton->SetParent(frame);
  this->LoadCommandLineModulesCheckButton->Create();
  this->LoadCommandLineModulesCheckButton->SetText(
    "Load Command-Line Modules");
  this->LoadCommandLineModulesCheckButton->SetCommand(
    this, "LoadCommandLineModulesCallback");
  this->LoadCommandLineModulesCheckButton->SetBalloonHelpString(
    "Control if modules should be loaded at startup.");

  tk_cmd << "pack " << this->LoadCommandLineModulesCheckButton->GetWidgetName()
         << "  -side top -anchor w -expand no -fill none" << endl;
  
  // --------------------------------------------------------------
  // Module settings : Home Module

  if ( !this->HomeModuleEntry )
    {
    this->HomeModuleEntry = vtkKWEntryWithLabel::New ( );
    }
  this->HomeModuleEntry->SetParent ( frame );
  this->HomeModuleEntry->Create ( );  
  this->HomeModuleEntry->SetLabelText( "Home Module:" );
  this->HomeModuleEntry->SetLabelWidth(label_width);
  this->HomeModuleEntry->GetWidget()->SetCommand ( 
    this, "HomeModuleCallback" );
  this->HomeModuleEntry->SetBalloonHelpString ( 
    "Module displayed at startup and when 'home' icon is clicked." );

  tk_cmd << "pack " << this->HomeModuleEntry->GetWidgetName()
         << "  -side top -anchor w -expand no -fill x -padx 2 -pady 2" << endl;

  // --------------------------------------------------------------
  // Module settings : Module Path

  if (!this->ModulePathEntry)
    {
    this->ModulePathEntry = vtkKWEntryWithLabel::New();
    }

  this->ModulePathEntry->SetParent(frame);
  this->ModulePathEntry->Create();
  this->ModulePathEntry->SetLabelText("Module Path:");
  this->ModulePathEntry->SetLabelWidth(label_width);
  this->ModulePathEntry->GetWidget()->SetCommand(this, "ModulePathCallback");
  this->ModulePathEntry->SetBalloonHelpString("Search path for modules.");

  tk_cmd << "pack " << this->ModulePathEntry->GetWidgetName()
         << "  -side top -anchor w -expand no -fill x -padx 2 -pady 2" << endl;

  // --------------------------------------------------------------
  // Module settings : TemporaryDirectory

  if (!this->TemporaryDirectoryButton)
    {
    this->TemporaryDirectoryButton = vtkKWLoadSaveButtonWithLabel::New();
    }

  this->TemporaryDirectoryButton->SetParent(frame);
  this->TemporaryDirectoryButton->Create();
  this->TemporaryDirectoryButton->SetLabelText("Temporary Directory:");
  this->TemporaryDirectoryButton->SetLabelWidth(label_width);
  this->TemporaryDirectoryButton->GetWidget()->TrimPathFromFileNameOff();
  this->TemporaryDirectoryButton->GetWidget()
    ->SetCommand(this, "TemporaryDirectoryCallback");
  this->TemporaryDirectoryButton->GetWidget()
    ->GetLoadSaveDialog()->ChooseDirectoryOn();
  this->TemporaryDirectoryButton->GetWidget()
    ->GetLoadSaveDialog()->SaveDialogOff();
  this->TemporaryDirectoryButton->GetWidget()
    ->GetLoadSaveDialog()->SetTitle("Select a directory for temporary files");
  this->TemporaryDirectoryButton->SetBalloonHelpString(
    "Temporary directory for intermediate files.");

  tk_cmd << "pack " << this->TemporaryDirectoryButton->GetWidgetName()
         << "  -side top -anchor w -expand no -padx 2 -pady 2" << endl;
  
  // --------------------------------------------------------------
  // Pack 

  tk_cmd << ends;
  this->Script(tk_cmd.str());
  tk_cmd.rdbuf()->freeze(0);

  // Update

  this->Update();
}

//----------------------------------------------------------------------------
void vtkSlicerApplicationSettingsInterface::ConfirmDeleteCallback(int state)
{
  vtkSlicerApplication *app
    = vtkSlicerApplication::SafeDownCast(this->GetApplication());
  if (app)
    {
    app->SetConfirmDelete(state ? "1" : "0");       
    }
}

//----------------------------------------------------------------------------
void vtkSlicerApplicationSettingsInterface::LoadCommandLineModulesCallback(int state)
{
  vtkSlicerApplication *app
    = vtkSlicerApplication::SafeDownCast(this->GetApplication());
  if (app)
    {
    app->SetLoadCommandLineModules(state ? 1 : 0);       
    }
}

//----------------------------------------------------------------------------
void vtkSlicerApplicationSettingsInterface::EnableDaemonCallback(int state)
{
  vtkSlicerApplication *app
    = vtkSlicerApplication::SafeDownCast(this->GetApplication());
  if (app)
    {
    app->SetEnableDaemon(state ? 1 : 0);       
    }
}

//----------------------------------------------------------------------------
void vtkSlicerApplicationSettingsInterface::HomeModuleCallback(char *name)
{
  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
  if ( app && name )
    {
    app->SetHomeModule ( name );
    }
}

//----------------------------------------------------------------------------
void vtkSlicerApplicationSettingsInterface::ModulePathCallback(char *path)
{
  vtkSlicerApplication *app
    = vtkSlicerApplication::SafeDownCast(this->GetApplication());

  if (app)
    {
    // Store the setting in the application object
    app->SetModulePath(path);
    }
}

//----------------------------------------------------------------------------
void vtkSlicerApplicationSettingsInterface::TemporaryDirectoryCallback()
{
  vtkSlicerApplication *app
    = vtkSlicerApplication::SafeDownCast(this->GetApplication());

  if (app)
    {
    // Store the setting in the application object
    app->SetTemporaryDirectory(this->TemporaryDirectoryButton->GetWidget()->GetLoadSaveDialog()->GetFileName());
    }
}

//----------------------------------------------------------------------------
void vtkSlicerApplicationSettingsInterface::Update()
{
  vtkSlicerApplication *app
    = vtkSlicerApplication::SafeDownCast(this->GetApplication());

  if (app)
    {
    // Pull values from the application object and put them in the
    // settings interface widgets
    if (this->ConfirmDeleteCheckButton)
      {
      this->ConfirmDeleteCheckButton->SetSelectedState(
        (strncmp(app->GetConfirmDelete(), "1", 1) == 0) ? 1 : 0);
      }
    if (this->EnableDaemonCheckButton)
      {
      this->EnableDaemonCheckButton->SetSelectedState(
        app->GetEnableDaemon() ? 1 : 0);
      }
    if (this->LoadCommandLineModulesCheckButton)
      {
      this->LoadCommandLineModulesCheckButton->SetSelectedState(
        app->GetLoadCommandLineModules() ? 1 : 0);
      }
    if (this->HomeModuleEntry)
      {
      this->HomeModuleEntry->GetWidget()->SetValue(app->GetHomeModule());
      }

    if (this->ModulePathEntry)
      {
      this->ModulePathEntry->GetWidget()->SetValue(app->GetModulePath());
      }

    if (this->TemporaryDirectoryButton)
      {
      this->TemporaryDirectoryButton->GetWidget()
        ->SetText(app->GetTemporaryDirectory());
      this->TemporaryDirectoryButton->GetWidget()
        ->GetLoadSaveDialog()->SetLastPath(app->GetTemporaryDirectory());
      }
    }
}
