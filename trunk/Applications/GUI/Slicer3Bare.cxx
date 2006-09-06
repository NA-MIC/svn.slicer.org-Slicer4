
#include "vtkRenderWindow.h"

#include "vtkKWApplication.h"
#include "vtkKWBalloonHelpManager.h"
#include "vtkKWWindow.h"
#include "vtkKWNotebook.h"
#include "vtkKWRegistryHelper.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerApplicationLogic.h"
#include "vtkSlicerSliceLogic.h"
#include "vtkSlicerVolumesLogic.h"
#include "vtkSlicerModelsLogic.h"
#include "vtkSlicerFiducialsLogic.h"
#include "vtkMRMLScene.h"
#include "vtkSlicerComponentGUI.h"
#include "vtkSlicerApplicationGUI.h"
#include "vtkSlicerSlicesGUI.h"
#include "vtkKWUserInterfacePanel.h"
#include "vtkKWUserInterfaceManager.h"
#include "vtkKWUserInterfaceManagerNotebook.h"
#include "vtkSlicerGUICollection.h"
#include "vtkSlicerVolumesGUI.h"
#include "vtkSlicerModelsGUI.h"
#include "vtkSlicerFiducialsGUI.h"
#include "vtkSlicerDataGUI.h"
#include "vtkSlicerTransformsGUI.h"
#include "vtkSlicerTheme.h"

#include "vtkSlicerConfigure.h" // for VTKSLICER_CONFIGURATION_TYPES

#include "vtkGradientAnisotropicDiffusionFilterLogic.h"
#include "vtkGradientAnisotropicDiffusionFilterGUI.h"

#include "vtkCommandLineModuleLogic.h"
#include "vtkCommandLineModuleGUI.h"

#include "vtkScriptedModuleLogic.h"
#include "vtkScriptedModuleGUI.h"

#include "ModuleFactory.h"

#include <vtksys/SystemTools.hxx>


// Get the automated arg parsing code
#include "Slicer3CLP.h"

extern "C" int Slicerbasegui_Init(Tcl_Interp *interp);
extern "C" int Slicerbaselogic_Init(Tcl_Interp *interp);
extern "C" int Mrml_Init(Tcl_Interp *interp);
extern "C" int Vtkitk_Init(Tcl_Interp *interp);

//TODO added temporary
extern "C" int Gradientanisotropicdiffusionfilter_Init(Tcl_Interp *interp);
extern "C" int Slicerdaemon_Init(Tcl_Interp *interp);
extern "C" int Commandlinemodule_Init(Tcl_Interp *interp);
extern "C" int Scriptedmodule_Init(Tcl_Interp *interp);

struct SpacesToUnderscores
{
  char operator() (char in)
    {
      if (in == ' ' )
        {
        return '_';
        }

      return in;
    }
};


int Slicer3_Tcl_Eval ( Tcl_Interp *interp, const char *script )
{
  if ( Tcl_Eval (interp, script) != TCL_OK )
    {
    cerr << "Error: " << Tcl_GetStringResult( interp ) << "\n";
    return 1;
    }
  return 0;
}

#define MODULES_DEBUG
#define SLICES_DEBUG
#define MODELS_DEBUG
#define VOLUMES_DEBUG
//#define APPGUI_DEBUG

int Slicer3_main(int argc, char *argv[])
{
  
    // Initialize Tcl
    // -- create the interp
    // -- set up initial global variables
    // -- later in this function tcl scripts are executed 
    //    and some additional global variables are defined

    Tcl_Interp *interp = vtkKWApplication::InitializeTcl(argc, argv, &cerr);
    if (!interp)
        {
            cerr << "Error: InitializeTcl failed" << endl ;
            return 1;
        }


    // Tell KWWidgets to make names like .vtkKWPushButton10 instead of .10 
    vtkKWWidget::UseClassNameInWidgetNameOn();

    //
    // Set a global variable so modules that use tcl can find the 
    // binary dir (where Slicer3.exe is) and the build dir (where tcl scripts are stored)
    //
    std::string config_types(VTKSLICER_CONFIGURATION_TYPES);
    std::string one_up;
    if (config_types.size())
      {
      one_up = "/..";
      }
    char cmd[2048];
    sprintf(cmd, "                                                   \
      set ::SLICER_BIN [file dirname [info nameofexecutable]];       \
      if { $::tcl_platform(platform) == \"windows\"} {               \
        set ::SLICER_BUILD [file normalize $SLICER_BIN/..%s]         \
      } else {                                                       \
        set ::SLICER_BUILD [file normalize $SLICER_BIN/..]           \
      };                                                             \
    ", one_up.c_str());

    Slicer3_Tcl_Eval( interp, cmd);

    PARSE_ARGS;  // relies on the header file auto generated by Slicer3.xml

    if (VTKDebugOn)
      {
      vtkObject::GlobalWarningDisplayOn();
      }

    //
    // load the gui tcl package (for the interactive widgets)
    //
    {    
      std::string cmd;
      int returnCode;

      // Pass arguments to the Tcl script
      cmd =  "lappend auto_path $::SLICER_BUILD/SlicerBaseGUITcl; ";
      cmd += "package require SlicerBaseGUITcl; ";
      returnCode = Slicer3_Tcl_Eval( interp, cmd.c_str() );
      if ( returnCode )
        {
        return ( returnCode );
        }
    }

    //
    // Initialize our Tcl library (i.e. our classes wrapped in Tcl)
    //
    Slicerbasegui_Init(interp);
    Slicerbaselogic_Init(interp);
    Mrml_Init(interp);
    Vtkitk_Init(interp);
    //TODO added temporary
    Gradientanisotropicdiffusionfilter_Init(interp);
    Slicerdaemon_Init(interp);
    Commandlinemodule_Init(interp);
    Scriptedmodule_Init(interp);


    // Create SlicerGUI application, style, and main window 
    vtkSlicerApplication *slicerApp = vtkSlicerApplication::New ( );
    slicerApp->InstallTheme( slicerApp->GetSlicerTheme() );

    // Create MRML scene
    vtkMRMLScene *scene = vtkMRMLScene::New();
    vtkMRMLScene::SetActiveScene( scene );
    
    // Create the application Logic object, 
    // Create the application GUI object
    // and have it observe the Logic
    vtkSlicerApplicationLogic *appLogic = vtkSlicerApplicationLogic::New( );
    appLogic->SetMRMLScene ( scene );
    // pass through event handling once without observing the scene
    // -- allows any dependent nodes to be created
    appLogic->ProcessMRMLEvents (scene, vtkCommand::ModifiedEvent, NULL);  
    appLogic->SetAndObserveMRMLScene ( scene );

    // CREATE APPLICATION GUI, including the main window
#ifndef APPGUI_DEBUG
    vtkSlicerApplicationGUI *appGUI = vtkSlicerApplicationGUI::New ( );
    appGUI->SetApplication ( slicerApp );
    appGUI->SetAndObserveApplicationLogic ( appLogic );
    appGUI->SetAndObserveMRMLScene ( scene );
    appGUI->BuildGUI ( );
    appGUI->AddGUIObservers ( );
#endif

    // ------------------------------
    // CREATE MODULE LOGICS & GUIS; add to GUI collection
    // (presumably these will be auto-detected, not listed out as below...)
    // ---
    // Note on vtkSlicerApplication's ModuleGUICollection:
    // right now the vtkSlicerApplication's ModuleGUICollection
    // collects ONLY module GUIs which populate the shared
    // ui panel in the main user interface. Other GUIs, like
    // the vtkSlicerSliceGUI which contains one SliceWidget
    // the vtkSlicerApplicationGUI which manages the entire
    // main user interface and viewer, any future GUI that
    // creates toplevel widgets are not added to this collection.
    // If we need to collect them at some point, we should define 
    // other collections in the vtkSlicerApplication class.

    // ADD INDIVIDUAL MODULES
    // (these require appGUI to be built):
    // --- Volumes module

#ifndef VOLUMES_DEBUG
    vtkSlicerVolumesLogic *volumesLogic = vtkSlicerVolumesLogic::New ( );
    volumesLogic->SetAndObserveMRMLScene ( scene );
    vtkSlicerVolumesGUI *volumesGUI = vtkSlicerVolumesGUI::New ( );
    volumesGUI->SetApplication ( slicerApp );
    volumesGUI->SetAndObserveApplicationLogic ( appLogic );
    volumesGUI->SetAndObserveMRMLScene ( scene );
    volumesGUI->SetModuleLogic ( volumesLogic );
    volumesGUI->SetGUIName( "Volumes" );
    volumesGUI->GetUIPanel()->SetName ( volumesGUI->GetGUIName ( ) );
    volumesGUI->GetUIPanel()->SetUserInterfaceManager (appGUI->GetMainSlicerWin()->GetMainUserInterfaceManager ( ) );
    volumesGUI->GetUIPanel()->Create ( );
    slicerApp->AddModuleGUI ( volumesGUI );
    volumesGUI->BuildGUI ( );
    volumesGUI->AddGUIObservers ( );
#endif

#ifndef MODELS_DEBUG

    // --- Models module    
    vtkSlicerModelsLogic *modelsLogic = vtkSlicerModelsLogic::New ( );
    modelsLogic->SetAndObserveMRMLScene ( scene );
    vtkSlicerModelsGUI *modelsGUI = vtkSlicerModelsGUI::New ( );
    modelsGUI->SetApplication ( slicerApp );
    modelsGUI->SetAndObserveApplicationLogic ( appLogic );
    modelsGUI->SetAndObserveMRMLScene ( scene );
    modelsGUI->SetModuleLogic ( modelsLogic );
    modelsGUI->SetGUIName( "Models" );
    modelsGUI->GetUIPanel()->SetName ( modelsGUI->GetGUIName ( ) );
    modelsGUI->GetUIPanel()->SetUserInterfaceManager (appGUI->GetMainSlicerWin()->GetMainUserInterfaceManager ( ) );
    modelsGUI->GetUIPanel()->Create ( );
    slicerApp->AddModuleGUI ( modelsGUI );
    modelsGUI->BuildGUI ( );
    modelsGUI->AddGUIObservers ( );
#endif

#ifndef SLICES_DEBUG

    // --- Slices module
    // - set up each of the slice logics (these initialize their
    //   helper classes and nodes the first time the process MRML and
    //   Logic events)
    vtkIntArray *events = vtkIntArray::New();
    events->InsertNextValue(vtkCommand::ModifiedEvent);
    events->InsertNextValue(vtkMRMLScene::NewSceneEvent);
    events->InsertNextValue(vtkMRMLScene::SceneCloseEvent);

    vtkSlicerSliceLogic *sliceLogic0 = vtkSlicerSliceLogic::New ( );
    sliceLogic0->SetName("Red");
    sliceLogic0->SetMRMLScene ( scene );
    sliceLogic0->ProcessLogicEvents ();
    sliceLogic0->ProcessMRMLEvents (scene, vtkCommand::ModifiedEvent, NULL);
    sliceLogic0->SetAndObserveMRMLSceneEvents ( scene, events );
    vtkSlicerSliceLogic *sliceLogic1 = vtkSlicerSliceLogic::New ( );
    sliceLogic1->SetName("Yellow");
    sliceLogic1->SetMRMLScene ( scene );
    sliceLogic1->ProcessLogicEvents ();
    sliceLogic1->ProcessMRMLEvents (scene, vtkCommand::ModifiedEvent, NULL);
    sliceLogic1->SetAndObserveMRMLSceneEvents ( scene, events );
    vtkSlicerSliceLogic *sliceLogic2 = vtkSlicerSliceLogic::New ( );
    sliceLogic2->SetName("Green");
    sliceLogic2->SetMRMLScene ( scene );
    sliceLogic2->ProcessLogicEvents ();
    sliceLogic2->ProcessMRMLEvents (scene, vtkCommand::ModifiedEvent, NULL);
    sliceLogic2->SetAndObserveMRMLSceneEvents ( scene, events );

    vtkSlicerSlicesGUI *slicesGUI = vtkSlicerSlicesGUI::New ();
    slicesGUI->SetApplication ( slicerApp );
    slicesGUI->SetAndObserveApplicationLogic ( appLogic );
    slicesGUI->SetAndObserveMRMLScene ( scene );
    slicesGUI->SetGUIName( "Slices" );
    slicesGUI->GetUIPanel()->SetName ( slicesGUI->GetGUIName ( ) );
    slicesGUI->GetUIPanel()->SetUserInterfaceManager ( appGUI->GetMainSlicerWin( )->GetMainUserInterfaceManager( ) );
    slicesGUI->GetUIPanel( )->Create( );
    slicerApp->AddModuleGUI ( slicesGUI );
    slicesGUI->BuildGUI ();
    slicesGUI->AddGUIObservers ( );
#endif

    //
    // get the Tcl name so the vtk class will be registered in the interpreter as a byproduct
    // - set some handy variables so it will be easy to access these classes from
    //   the tkcon
    // - all the variables are put in the slicer3 tcl namespace for easy access
    //
    const char *name;
    name = slicerApp->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set Application %s", name);
#ifndef APPGUI_DEBUG
    name = appGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set ApplicationGUI %s", name);
#endif
#ifndef SLICES_DEBUG
    name = slicesGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set SlicesGUI %s", name);
#endif
#ifndef MODELS_DEBUG
    name = modelsGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set ModelsGUI %s", name);
#endif

#ifndef APPGUI_DEBUG
    //name = appGUI->GetViewerWidget()->GetTclName();
    //slicerApp->Script ("namespace eval slicer3 set ViewerWidget %s", name);
#endif

    slicerApp->Script ("namespace eval slicer3 set ApplicationLogic [$::slicer3::ApplicationGUI GetApplicationLogic]");
    slicerApp->Script ("namespace eval slicer3 set MRMLScene [$::slicer3::ApplicationLogic GetMRMLScene]");


    //
    // create the three main slice viewers after slicesGUI is created
    //
#ifndef APPGUI_DEBUG
    appGUI->PopulateModuleChooseList ( );
#endif
#ifndef SLICES_DEBUG
    appGUI->SetSliceGUICollection ( slicesGUI->GetSliceGUICollection() );
    appGUI->SetAndObserveMainSliceLogic ( sliceLogic0, sliceLogic1, sliceLogic2 );
    appGUI->AddMainSliceViewersToCollection ( );
    appGUI->ConfigureMainSliceViewers ( );
#endif
    
    // ------------------------------
    // CONFIGURE SlICER'S SHARED GUI PANEL
    // Additional Modules GUI panel configuration.
#ifndef APPGUI_DEBUG
    vtkKWUserInterfaceManagerNotebook *mnb = vtkKWUserInterfaceManagerNotebook::SafeDownCast (appGUI->GetMainSlicerWin()->GetMainUserInterfaceManager());
    mnb->GetNotebook()->AlwaysShowTabsOff();
    mnb->GetNotebook()->ShowOnlyPagesWithSameTagOn();    

    
    // ------------------------------
    // DISPLAY WINDOW AND RUN
    appGUI->DisplayMainSlicerWindow ( );
#endif

    int res; // return code (exit code)

    //
    // Run!  - this will return when the user exits
    //
    res = slicerApp->StartApplication();

#ifndef APPGUI_DEBUG
    appGUI->GetMainSlicerWin()->PrepareForDelete();
    appGUI->GetMainSlicerWin()->Close();
#endif

    // ------------------------------
    // REMOVE OBSERVERS and references to MRML and Logic
#ifndef VOLUMES_DEBUG
    volumesGUI->RemoveGUIObservers ( );
#endif
#ifndef MODELS_DEBUG
    modelsGUI->RemoveGUIObservers ( );
#endif
#ifndef SLICES_DEBUG
    slicesGUI->RemoveGUIObservers ( );
#endif
#ifndef APPGUI_DEBUG
    appGUI->RemoveGUIObservers ( );
#endif
#ifndef SLICES_DEBUG
    appGUI->SetAndObserveMainSliceLogic ( NULL, NULL, NULL );

    // remove all from the slicesGUI collection of sliceGUIs
    slicesGUI->GetSliceGUICollection()->RemoveAllItems ( );
    appGUI->SetSliceGUICollection ( NULL );
#endif

    // ------------------------------
    // Remove References to Module GUIs
    slicerApp->GetModuleGUICollection ( )->RemoveAllItems ( );
    slicerApp->GetTheme()->SetApplication( NULL );
    slicerApp->SetTheme( NULL );

    // ------------------------------
    // EXIT 
    slicerApp->Exit();

    // ------------------------------
    // DELETE 
    
    //--- delete gui first, removing Refs to Logic and MRML

#ifndef VOLUMES_DEBUG
    volumesGUI->Delete ();
#endif
#ifndef MODELS_DEBUG
    modelsGUI->Delete ();
#endif
#ifndef SLICES_DEBUG
    slicesGUI->Delete ();
#endif
#ifndef APPGUI_DEBUG
    appGUI->Delete ();
#endif

    
    //--- delete logic next, removing Refs to MRML
    appLogic->ClearCollections ( );
#ifndef VOLUMES_DEBUG
    volumesLogic->SetAndObserveMRMLScene ( NULL );
    volumesLogic->Delete();
#endif
#ifndef MODELS_DEBUG
    modelsLogic->SetAndObserveMRMLScene ( NULL );
    modelsLogic->Delete();
#endif
#ifndef MODULES_DEBUG
    fiducialsLogic->SetAndObserveMRMLScene ( NULL );
    fiducialsLogic->Delete();
#endif
#ifndef SLICES_DEBUG
    sliceLogic2->SetAndObserveMRMLScene ( NULL );
    sliceLogic2->Delete ();
    sliceLogic1->SetAndObserveMRMLScene ( NULL );
    sliceLogic1->Delete ();
    sliceLogic0->SetAndObserveMRMLScene ( NULL );
    sliceLogic0->Delete ();
#endif
    appLogic->SetAndObserveMRMLScene ( NULL );
    appLogic->Delete ();

    //--- scene next;
    scene->Delete ();

    //--- application last
    if ( slicerApp->GetBalloonHelpManager() )
      {
      slicerApp->GetBalloonHelpManager()->SetApplication(NULL);
      }
    slicerApp->Delete ();

    return res;
}

#ifdef _WIN32
#include <windows.h>
int __stdcall WinMain(HINSTANCE, HINSTANCE, LPSTR lpCmdLine, int)
{
    int argc;
    char **argv;
    vtksys::SystemTools::ConvertWindowsCommandLineToUnixArguments(
                                                                  lpCmdLine, &argc, &argv);
    int ret = Slicer3_main(argc, argv);
    for (int i = 0; i < argc; i++) { delete [] argv[i]; }
    delete [] argv;
    return ret;
}

int main(int argc, char *argv[])
{
    return Slicer3_main(argc, argv);
}

#else
int main(int argc, char *argv[])
{
    return Slicer3_main(argc, argv);
}
#endif



