
#include "vtkRenderWindow.h"

#include "vtkKWApplication.h"
#include "vtkKWWindow.h"
#include "vtkKWNotebook.h"
#include "vtkKWRegistryHelper.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerApplicationLogic.h"
#include "vtkSlicerSliceLogic.h"
#include "vtkSlicerVolumesLogic.h"
#include "vtkSlicerModelsLogic.h"
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
#include "vtkSlicerDataGUI.h"
#include "vtkSlicerTransformsGUI.h"
#include "vtkSlicerTheme.h"

#include "vtkSlicerConfigure.h" // for VTKSLICER_CONFIGURATION_TYPES

#include "vtkGradientAnisotropicDiffusionFilterLogic.h"
#include "vtkGradientAnisotropicDiffusionFilterGUI.h"

#include "vtkCommandLineModuleLogic.h"
#include "vtkCommandLineModuleGUI.h"

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

int Slicer3_main(int argc, char *argv[])
{
  // Append the path to the slicer executable to the ITK_AUTOLOAD_PATH
  // so that Slicer specific ITK factories will be available by
  // default. We assume any factories to be loaded will be in the same
  // directory as the slicer executable
  std::string itkAutoLoadPath;
  vtksys::SystemTools::GetEnv("ITK_AUTOLOAD_PATH", itkAutoLoadPath);

  std::string ptemp;
  ptemp = vtksys::SystemTools::CollapseFullPath(argv[0]);
  ptemp = vtksys::SystemTools::GetFilenamePath(ptemp);
  ptemp = vtksys::SystemTools::ConvertToOutputPath(ptemp.c_str());
#if WIN32
  itkAutoLoadPath = ptemp + ";" + itkAutoLoadPath;
#else
  itkAutoLoadPath = ptemp + ":" + itkAutoLoadPath;
#endif
  itkAutoLoadPath = "ITK_AUTOLOAD_PATH=" + itkAutoLoadPath;
  putenv(const_cast <char *> (itkAutoLoadPath.c_str()));
  
  
  
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
    char cmd[512];
    sprintf(cmd, "                                                   \
      set ::SLICER_BIN [file dirname [info nameofexecutable]];       \
      if { $::tcl_platform(platform) == \"windows\"} {               \
        set ::SLICER_BUILD [file normalize $SLICER_BIN/..%s]         \
      } else {                                                       \
        set ::SLICER_BUILD [file normalize $SLICER_BIN/..]           \
      }                                                              \
    ", one_up.c_str());

    Slicer3_Tcl_Eval( interp, cmd);

    PARSE_ARGS;  // relies on the header file auto generated by Slicer3.xml


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

    //
    // use the startup script passed on command line if it exists
    //
    if ( File != "" )
      {    

      std::string cmd;
#if 0

// TODO: this works to pass args to the tcl script, but I can't get
the xml to do what I need...

      cmd = "set args \"\"; ";
      std::vector<std::string>::const_iterator argit = Args.begin();
      while (argit != Args.end())
        {
        cmd += " lappend args " + *argit + "; ";
        ++argit;
        }
      Slicer3_Tcl_Eval( interp, cmd.c_str() );
#endif

      cmd = "source " + File;
      return ( Slicer3_Tcl_Eval( interp, cmd.c_str() ) );
      }

    //
    // use the startup code passed on command line if it exists
    //
    if ( Eval != "" )
      {    
      std::string cmd = "set ::SLICER(eval) \"" + Eval + "\";";
      cmd += "regsub -all {\\.,} $::SLICER(eval) \";\" ::SLICER(eval);";
      cmd += "regsub -all {,\\.} $::SLICER(eval) \";\" ::SLICER(eval);";
      cmd += "eval $::SLICER(eval);";
      return ( Slicer3_Tcl_Eval( interp, cmd.c_str() ) );
      }

    // Create SlicerGUI application, style, and main window 
    vtkSlicerApplication *slicerApp = vtkSlicerApplication::New ( );
    slicerApp->InstallTheme( slicerApp->GetSlicerTheme() );

    // Create MRML scene
    vtkMRMLScene *scene = vtkMRMLScene::New();
    vtkMRMLScene::SetActiveScene( scene );
    vtkMRMLScalarVolumeNode::CreateNoneNode(scene);
    
    // Create the application Logic object, 
    // Create the application GUI object
    // and have it observe the Logic
    vtkSlicerApplicationLogic *appLogic = vtkSlicerApplicationLogic::New( );
    appLogic->SetMRMLScene ( scene );
    // pass through event handling once without observing the scene
    // -- allows any dependent nodes to be created
    appLogic->ProcessMRMLEvents ();  
    appLogic->SetAndObserveMRMLScene ( scene );

    // CREATE APPLICATION GUI, including the main window
    vtkSlicerApplicationGUI *appGUI = vtkSlicerApplicationGUI::New ( );
    appGUI->SetApplication ( slicerApp );
    appGUI->SetAndObserveApplicationLogic ( appLogic );
    appGUI->SetAndObserveMRMLScene ( scene );
    appGUI->BuildGUI ( );
    appGUI->AddGUIObservers ( );

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

    // --- Transforms module
    vtkSlicerTransformsGUI *transformsGUI = vtkSlicerTransformsGUI::New ( );
    transformsGUI->SetApplication ( slicerApp );
    transformsGUI->SetAndObserveApplicationLogic ( appLogic );
    transformsGUI->SetAndObserveMRMLScene ( scene );
    transformsGUI->SetGUIName( "Transforms" );
    transformsGUI->GetUIPanel()->SetName ( transformsGUI->GetGUIName ( ) );
    transformsGUI->GetUIPanel()->SetUserInterfaceManager (appGUI->GetMainSlicerWin()->GetMainUserInterfaceManager ( ) );
    transformsGUI->GetUIPanel()->Create ( );
    slicerApp->AddModuleGUI ( transformsGUI );
    transformsGUI->BuildGUI ( );
    transformsGUI->AddGUIObservers ( );

    //--- Data module
    //vtkSlicerDataLogic *dataLogic = vtkSlicerDataLogic::New ( );
    //dataLogic->SetAndObserveMRMLScene ( scene );
    //dataLogic->SetApplicationLogic ( appLogic );
    vtkSlicerDataGUI *dataGUI = vtkSlicerDataGUI::New ( );
    dataGUI->SetApplication ( slicerApp );
    dataGUI->SetAndObserveApplicationLogic ( appLogic );
    dataGUI->SetAndObserveMRMLScene ( scene );
    //dataGUI->SetModuleLogic ( dataLogic );
    dataGUI->SetGUIName( "Data" );
    dataGUI->GetUIPanel()->SetName ( dataGUI->GetGUIName ( ) );
    dataGUI->GetUIPanel()->SetUserInterfaceManager (appGUI->GetMainSlicerWin()->GetMainUserInterfaceManager ( ) );
    dataGUI->GetUIPanel()->Create ( );    
    slicerApp->AddModuleGUI ( dataGUI );
    dataGUI->BuildGUI ( );
    dataGUI->AddGUIObservers ( );

    // --- Slices module
    // - set up each of the slice logics (these initialize their
    //   helper classes and nodes the first time the process MRML and
    //   Logic events)
    vtkSlicerSliceLogic *sliceLogic0 = vtkSlicerSliceLogic::New ( );
    sliceLogic0->SetName("Red");
    sliceLogic0->SetMRMLScene ( scene );
    sliceLogic0->ProcessLogicEvents ();
    sliceLogic0->ProcessMRMLEvents ();
    sliceLogic0->SetAndObserveMRMLScene ( scene );
    vtkSlicerSliceLogic *sliceLogic1 = vtkSlicerSliceLogic::New ( );
    sliceLogic1->SetName("Yellow");
    sliceLogic1->SetMRMLScene ( scene );
    sliceLogic1->ProcessLogicEvents ();
    sliceLogic1->ProcessMRMLEvents ();
    sliceLogic1->SetAndObserveMRMLScene ( scene );
    vtkSlicerSliceLogic *sliceLogic2 = vtkSlicerSliceLogic::New ( );
    sliceLogic2->SetName("Green");
    sliceLogic2->SetMRMLScene ( scene );
    sliceLogic2->ProcessLogicEvents ();
    sliceLogic2->ProcessMRMLEvents ();
    sliceLogic2->SetAndObserveMRMLScene ( scene );

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


    // --- Gradient anisotropic diffusion filter module
    vtkGradientAnisotropicDiffusionFilterGUI *gradientAnisotropicDiffusionFilterGUI = vtkGradientAnisotropicDiffusionFilterGUI::New ( );
    vtkGradientAnisotropicDiffusionFilterLogic *gradientAnisotropicDiffusionFilterLogic  = vtkGradientAnisotropicDiffusionFilterLogic::New ( );
    gradientAnisotropicDiffusionFilterLogic->SetAndObserveMRMLScene ( scene );
    gradientAnisotropicDiffusionFilterLogic->SetApplicationLogic ( appLogic );
    //    gradientAnisotropicDiffusionFilterLogic->SetMRMLScene(scene);
    gradientAnisotropicDiffusionFilterGUI->SetLogic ( gradientAnisotropicDiffusionFilterLogic );
    gradientAnisotropicDiffusionFilterGUI->SetApplication ( slicerApp );
    gradientAnisotropicDiffusionFilterGUI->SetApplicationLogic ( appLogic );
    gradientAnisotropicDiffusionFilterGUI->SetGUIName( "GradientAnisotropicDiffusionFilter" );
    gradientAnisotropicDiffusionFilterGUI->GetUIPanel()->SetName ( gradientAnisotropicDiffusionFilterGUI->GetGUIName ( ) );
    gradientAnisotropicDiffusionFilterGUI->GetUIPanel()->SetUserInterfaceManager (appGUI->GetMainSlicerWin()->GetMainUserInterfaceManager ( ) );
    gradientAnisotropicDiffusionFilterGUI->GetUIPanel()->Create ( );
    slicerApp->AddModuleGUI ( gradientAnisotropicDiffusionFilterGUI );
    gradientAnisotropicDiffusionFilterGUI->BuildGUI ( );
    gradientAnisotropicDiffusionFilterGUI->AddGUIObservers ( );


    // --- SlicerDaemon Module
    // need to source the slicerd.tcl script here
    Slicer3_Tcl_Eval( interp, "                                              \
      source $::SLICER_BUILD/slicerd.tcl; slicerd_start              \
    ");


    // --- Scan for command line and shared object modules
    //
    //
    ModuleFactory moduleFactory;
    moduleFactory.SetSearchPath( slicerApp->GetModulePath() );
    moduleFactory.Scan();

    // add the modules to the available modules
    std::vector<std::string> moduleNames = moduleFactory.GetModuleNames();
    std::vector<std::string>::const_iterator mit = moduleNames.begin();
    while (mit != moduleNames.end())
      {
      // std::cout << moduleFactory.GetModuleDescription(*mit) << std::endl;

      // For now, create vtkCommandLineModule* items. When the
      // ModuleFactory can discover shared object modules, then we'll
      // come back and generalize this.
      vtkCommandLineModuleGUI *commandLineModuleGUI
        = vtkCommandLineModuleGUI::New();
      vtkCommandLineModuleLogic *commandLineModuleLogic
        = vtkCommandLineModuleLogic::New ( );
      
      // Set the ModuleDescripton on the gui
      commandLineModuleGUI
        ->SetModuleDescription( moduleFactory.GetModuleDescription(*mit) );

      // Configure the Logic, GUI, and add to app
      commandLineModuleLogic->SetAndObserveMRMLScene ( scene );
      commandLineModuleLogic->SetApplicationLogic (appLogic);
      commandLineModuleGUI->SetLogic ( commandLineModuleLogic );
      commandLineModuleGUI->SetApplication ( slicerApp );
      commandLineModuleGUI->SetApplicationLogic ( appLogic );
      commandLineModuleGUI->SetGUIName( moduleFactory.GetModuleDescription(*mit).GetTitle().c_str() );
      commandLineModuleGUI->GetUIPanel()->SetName ( commandLineModuleGUI->GetGUIName ( ) );
      commandLineModuleGUI->GetUIPanel()->SetUserInterfaceManager (appGUI->GetMainSlicerWin()->GetMainUserInterfaceManager ( ) );
      commandLineModuleGUI->GetUIPanel()->Create ( );
      slicerApp->AddModuleGUI ( commandLineModuleGUI );

      ++mit;
      }
    // -- Build the factory discovered modules gui and observers
    mit = moduleNames.begin();
    while ( mit != moduleNames.end() )
      {
      vtkSlicerModuleGUI *module;
      module = slicerApp->GetModuleGUIByName( (*mit).c_str() );

      module->BuildGUI();
      module->AddGUIObservers();
      
      ++mit;
      }

    // create the three main slice viewers after slicesGUI is created
    appGUI->PopulateModuleChooseList ( );
    appGUI->SetSliceGUICollection ( slicesGUI->GetSliceGUICollection() );
    appGUI->AddMainSliceViewersToCollection ( );
    appGUI->AddMainSliceViewerObservers ( );
    appGUI->SetAndObserveMainSliceLogic ( sliceLogic0, sliceLogic1, sliceLogic2 );
    appGUI->ConfigureMainSliceViewers ( );
    // Initialize the event handling code for slice viewers
    Slicer3_Tcl_Eval (interp, "source $::SLICER_BUILD/SliceViewerInteractor.tcl");
    
    // ------------------------------
    // CONFIGURE SlICER'S SHARED GUI PANEL
    // Additional Modules GUI panel configuration.
    vtkKWUserInterfaceManagerNotebook *mnb = vtkKWUserInterfaceManagerNotebook::SafeDownCast (appGUI->GetMainSlicerWin()->GetMainUserInterfaceManager());
    mnb->GetNotebook()->AlwaysShowTabsOff();
    mnb->GetNotebook()->ShowOnlyPagesWithSameTagOn();    

    //
    // get the Tcl name so the vtk class will be registered in the interpreter as a byproduct
    // - set some handy variables so it will be easy to access these classes from
    //   the tkcon
    // - all the variables are put in the slicer3 tcl namespace for easy access
    //
    const char *name;
    name = slicerApp->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set Application %s", name);
    name = appGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set ApplicationGUI %s", name);
    name = slicesGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set SlicesGUI %s", name);

    name = volumesGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set VolumesGUI %s", name);
    name = modelsGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set ModelsGUI %s", name);
    name = transformsGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set VolumesGUI %s", name);

    slicerApp->Script ("namespace eval slicer3 set ApplicationLogic [$::slicer3::ApplicationGUI GetApplicationLogic]");
    slicerApp->Script ("namespace eval slicer3 set MRMLScene [$::slicer3::ApplicationLogic GetMRMLScene]");


    mit = moduleNames.begin();
    while ( mit != moduleNames.end() )
      {
      vtkSlicerModuleGUI *module;
      module = slicerApp->GetModuleGUIByName( (*mit).c_str() );

      name = module->GetTclName();

      std::string title = *mit;
      std::transform(title.begin(), title.end(), title.begin(), SpacesToUnderscores());

      slicerApp->Script ("namespace eval slicer3 set CommandLineModuleGUI_%s %s", title.c_str(), name);
      
      ++mit;
      }

    
    // ------------------------------
    // DISPLAY WINDOW AND RUN
    appGUI->DisplayMainSlicerWindow ( );

    // More command line arguments:
    // use the startup script passed on command line if it exists
    if ( Script != "" )
      {    
      std::string cmd = "source " + Script;
      Slicer3_Tcl_Eval( interp, cmd.c_str() ) ;
      }

    // use the startup code passed on command line if it exists
    if ( Exec != "" )
      {    
      std::string cmd = "set ::SLICER(exec) \"" + Exec + "\";";
      cmd += "regsub -all {\\.,} $::SLICER(exec) \";\" ::SLICER(exec);";
      cmd += "regsub -all {,\\.} $::SLICER(exec) \";\" ::SLICER(exec);";
      cmd += "eval $::SLICER(exec);";
      Slicer3_Tcl_Eval( interp, cmd.c_str() );
      }

    int res = slicerApp->StartApplication();


    // ------------------------------
    // REMOVE OBSERVERS and references to MRML and Logic
    gradientAnisotropicDiffusionFilterGUI->RemoveGUIObservers ( );
    volumesGUI->RemoveGUIObservers ( );
    modelsGUI->RemoveGUIObservers ( );
    transformsGUI->RemoveGUIObservers ( );
    dataGUI->RemoveGUIObservers ( );
    slicesGUI->RemoveGUIObservers ( );
    appGUI->RemoveGUIObservers ( );

    // remove all from the slicesGUI collection of sliceGUIs
    slicesGUI->GetSliceGUICollection()->RemoveAllItems ( );
    appGUI->SetSliceGUICollection ( NULL );

    // remove the observers from the factory discovered modules
    // (as we remove the observers, cache the GUIs in a vector so we
    // can delete them later).
    mit = moduleNames.begin();
    std::vector<vtkSlicerModuleGUI*> moduleGUIs;
    while ( mit != moduleNames.end() )
      {
      vtkSlicerModuleGUI *module;
      module = slicerApp->GetModuleGUIByName( (*mit).c_str() );

      module->RemoveGUIObservers();

      moduleGUIs.push_back(module);
      
      ++mit;
      }
    

    // ------------------------------
    // Remove References to Module GUIs
    slicerApp->GetModuleGUICollection ( )->RemoveAllItems ( );

    // ------------------------------
    // EXIT 
    slicerApp->Exit();

    // ------------------------------
    // DELETE 
    
    //--- delete gui first, removing Refs to Logic and MRML

    gradientAnisotropicDiffusionFilterGUI->Delete ();
    volumesGUI->Delete ();
    modelsGUI->Delete ();
    transformsGUI->Delete ();
    dataGUI->Delete ();
    slicesGUI->Delete ();
    appGUI->Delete ();


    // delete the factory discovered module GUIs (as we delete the
    // GUIs, cache the associated logic instances so we can delete
    // them later).
    std::vector<vtkSlicerModuleGUI*>::iterator git;
    std::vector<vtkSlicerModuleLogic*> moduleLogics;
    for (git = moduleGUIs.begin(); git != moduleGUIs.end(); ++git)
      {
      moduleLogics.push_back(dynamic_cast<vtkCommandLineModuleGUI*>((*git))->GetLogic());

      (*git)->Delete();
      }
    moduleGUIs.clear();
    
    //--- delete logic next, removing Refs to MRML
    appLogic->ClearCollections ( );
    gradientAnisotropicDiffusionFilterLogic->Delete ();
    volumesLogic->Delete();
    modelsLogic->Delete();
    sliceLogic0->Delete ();
    sliceLogic1->Delete ();
    sliceLogic2->Delete ();
    appLogic->Delete ();


    // delete the factory discovered module Logics
    std::vector<vtkSlicerModuleLogic*>::iterator lit;
    for (lit = moduleLogics.begin(); lit != moduleLogics.end(); ++lit)
      {
      (*lit)->Delete();
      }
    moduleLogics.clear();
    
    //--- scene next;
    scene->Delete ();

    //--- application last
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

