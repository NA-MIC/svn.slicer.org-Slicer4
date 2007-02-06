
#include "vtkRenderWindow.h"

#include "vtkKWApplication.h"
#include "vtkKWWindow.h"
#include "vtkKWNotebook.h"
#include "vtkKWRegistryHelper.h"
#include "vtkKWTkUtilities.h"
#include "vtkKWResourceUtilities.h"
#include "vtkKWSplashScreen.h"
#include "vtkSlicerLogoIcons.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerApplicationLogic.h"
#include "vtkSlicerSliceLogic.h"
#include "vtkSlicerVolumesLogic.h"
#include "vtkSlicerModelsLogic.h"
#include "vtkSlicerFiducialsLogic.h"
#include "vtkSlicerColorLogic.h"
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
#include "vtkSlicerColorGUI.h"
#include "vtkSlicerDataGUI.h"
#include "vtkSlicerTransformsGUI.h"
#include "vtkSlicerCamerasGUI.h"
#include "vtkSlicerTheme.h"
#include "vtkSlicerWindow.h"
#include "vtkSlicerApplicationSettingsInterface.h"

#include "vtkSlicerConfigure.h" // for VTKSLICER_CONFIGURATION_TYPES

#include "vtkGradientAnisotropicDiffusionFilterLogic.h"
#include "vtkGradientAnisotropicDiffusionFilterGUI.h"

//#include "vtkSlicerTractographyDisplayLogic.h"
//#include "vtkSlicerTractographyDisplayGUI.h"

#include "vtkQueryAtlasLogic.h"
#include "vtkQueryAtlasGUI.h"

#include "vtkCommandLineModuleLogic.h"
#include "vtkCommandLineModuleGUI.h"

#include "vtkScriptedModuleLogic.h"
#include "vtkScriptedModuleGUI.h"

#include "ModuleFactory.h"

#include <vtksys/SystemTools.hxx>
#include <vtksys/stl/string>

// for data prov
#include <time.h>
#include <stdlib.h>

#include "Resources/vtkSlicerSplashScreen_ImageData.h"

// uncomment these lines to disable a particular module (handy for debugging)
//#define CLIMODULES_DEBUG
//#define TCLMODULES_DEBUG
//#define SLICES_DEBUG
//#define MODELS_DEBUG
//#define VOLUMES_DEBUG
//#define QUERYATLAS_DEBUG
//#define COLORS_DEBUG
#define RealTimeImaging_DEBUG
//#define FIDUCIALS_DEBUG
//#define CAMERA_DEBUG
#define EMSEG_DEBUG
#define REALTIMEIMAGING_DEBUG
//#define MRABLATION_DEBUG
//#define NEURONAV_DEBUG
#define TRACTOGRAPHY_DEBUG


#ifndef EMSEG_DEBUG
#include "vtkEMSegmentLogic.h"
#include "vtkEMSegmentGUI.h"
#endif

#ifndef REALTIMEIMAGING_DEBUG
#include "vtkRealTimeImagingLogic.h"
#include "vtkRealTimeImagingGUI.h"
#endif

#ifndef MRABLATION_DEBUG
#include "vtkMRAblationLogic.h"
#include "vtkMRAblationGUI.h"
#endif

#ifndef NEURONAV_DEBUG
#include "vtkNeuroNavLogic.h"
#include "vtkNeuroNavGUI.h"
#endif

#ifndef WFENGINE_DEBUG
#include "vtkWFEngineModuleLogic.h"
#include "vtkWFEngineModuleGUI.h"
#endif

#ifdef _WIN32
#  define slicerCerr(x) \
  do { \
    vtkstd::ostringstream str; \
    str << x; \
    MessageBox(NULL, str.str().c_str(), "Slicer Error", MB_OK);\
  } while (0)
#else
#  define slicerCerr(x) \
  cerr << x
#endif


// Get the automated arg parsing code
#include "Slicer3CLP.h"

extern "C" int Slicerbasegui_Init(Tcl_Interp *interp);
extern "C" int Slicerbaselogic_Init(Tcl_Interp *interp);
extern "C" int Mrml_Init(Tcl_Interp *interp);
extern "C" int Vtkitk_Init(Tcl_Interp *interp);
extern "C" int Freesurfer_Init(Tcl_Interp *interp);
extern "C" int Igt_Init(Tcl_Interp *interp);


//TODO added temporary
#ifndef EMSEG_DEBUG
extern "C" int Emsegment_Init(Tcl_Interp *interp);
#endif
#ifndef REALTIMEIMAGING_DEBUG
extern "C" int Realtimeimaging_Init(Tcl_Interp *interp);
#endif
extern "C" int Gradientanisotropicdiffusionfilter_Init(Tcl_Interp *interp);
// extern "C" int Slicertractographydisplay_Init(Tcl_Interp *interp);
extern "C" int Queryatlas_Init(Tcl_Interp *interp);
extern "C" int Slicerdaemon_Init(Tcl_Interp *interp);
extern "C" int Commandlinemodule_Init(Tcl_Interp *interp);
extern "C" int Scriptedmodule_Init(Tcl_Interp *interp);
#ifndef WFENGINE_DEBUG
extern "C" int Wfenginemodule_Init(Tcl_Interp *interp);
#endif

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
    // TODO: need to figure out how to turn on the stdio channels
    // so puts will work from scripts in windows...
    slicerCerr("Error: " << Tcl_GetStringResult( interp ) << "\n");
    return 1;
    }
  return 0;
}

void printAllInfo(int argc, char **argv)
{
  int i;
  struct tm * timeInfo;
  time_t rawtime;
  // mm/dd/yy-hh-mm-ss-TZ
  // plus one for 3 char time zone
  char timeStr[22];
  
  fprintf(stdout, "ProgramName: %s", argv[0]);
  fprintf(stdout, " ProgramArguments:");
  for (i = 1; i < argc; i++)
    {
    fprintf(stdout, " %s", argv[i]);
    }
  fprintf(stdout, " ProgramVersion: $Revision$ CVS: $Id$ TimeStamp: ");
  time ( &rawtime );
  timeInfo = localtime (&rawtime);
  strftime (timeStr, 22, "%D-%k-%M-%S-%Z", timeInfo);
  fprintf(stdout, "%s", timeStr);
  fprintf(stdout, " User: %s", getenv("USER"));
  
  fprintf(stdout, " Machine: %s", getenv("HOSTNAME"));
  fprintf(stdout, " Platform: ");
#if defined(linux) || defined(__linux)
  fprintf(stdout, "Linux");
#endif 
#if defined(macintosh) || defined(Macintosh)
  fprintf(stdout, "MAC OS 9");
#endif
#ifdef __MACOSX__
  fprintf(stdout, "MAC OS X");
#endif
#if defined(sun) || defined(__sun)
# if defined(__SVR4) || defined(__svr4__)
  fprintf(stdout, "Solaris");
# else
  fprintf(stdout, "SunOS");
# endif
#endif
#if defined(_WIN32) || defined(__WIN32__)
  fprintf(stdout, "Windows");
#endif
  fprintf(stdout, " PlatformVersion: ");
  
#if defined(sun) || defined(__sun)
#if defined(__SunOS_5_7)
  fprintf(stdout, "2.7");
#endif
#if defined(__SunOS_5_8)
  fprintf(stdout, "8");
#endif
#endif
#if defined(linux) || defined(__linux)
  fprintf(stdout, "unknown");
#endif
  fprintf(stdout, " CompilerName: ");
#if defined(__GNUC__)
  fprintf(stdout, "GCC");
#else
#if defined(_MSC_VER)
  fprintf(stdout, "MSC");
#else
  fprintf(stdout, "UKNOWN");
#endif
#endif
  fprintf(stdout, " CompilerVersion: ");
#if defined(__GNUC__)
#if defined(__GNU_PATCHLEVEL__)
  fprintf(stdout, "%d", (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__));
#else
  fprintf(stdout, "%d", (__GNUC__ * 10000 + __GNUC_MINOR__ * 100));
#endif
#endif
#if defined(_MSC_VER)
  fprintf(stdout, "%d", (_MSC_VER));
#endif
  
  fprintf(stdout, " LibName: VTK LibVersion: %s LibName: ITK LibVersion: unknown", VTK_VERSION);
  int major, minor, patchLevel;
  Tcl_GetVersion(&major, &minor, &patchLevel, NULL);
  fprintf(stdout, " LibName: TCL LibVersion: %d.%d.%d", major, minor, patchLevel);
  //fprintf(stdout, " LibName: TK LibVersion: %d.%d.%d", major, minor, patchLevel);
  fprintf(stdout, "\n");
  
}


static void WarningMessage(const char *msg)
{
  vtkSlicerApplication::GetInstance()->WarningMessage(msg);
}

static void ErrorMessage(const char *msg)
{
  vtkSlicerApplication::GetInstance()->ErrorMessage(msg);
}

static void InformationMessage(const char *msg)
{
  vtkSlicerApplication::GetInstance()->InformationMessage(msg);
}

static void DebugMessage(const char *msg)
{
  vtkSlicerApplication::GetInstance()->DebugMessage(msg);
}

static void SplashMessage(const char *msg)
{
  vtkSlicerApplication::GetInstance()->GetSplashScreen()->SetProgressMessage(msg);
}


int Slicer3_main(int argc, char *argv[])
{
  // Append the path to the slicer executable to the ITK_AUTOLOAD_PATH
  // so that Slicer specific ITK factories will be available by
  // default. We assume any factories to be loaded will be in the same
  // directory as the slicer executable. Also, set up the TCL_LIBRARY
  // environment variable.
  std::string itkAutoLoadPath;
  vtksys::SystemTools::GetEnv("ITK_AUTOLOAD_PATH", itkAutoLoadPath);

  std::string programPath;
  std::string errorMessage;
  if ( !vtksys::SystemTools::FindProgramPath(argv[0],
      programPath, errorMessage) )
    {
    slicerCerr("Error: Cannot find Slicer3 executable" << endl);
    return 1;
    }
  cerr << "Using slicer executable: " << programPath.c_str() << endl;
  
  std::string slicerBinDir
    = vtksys::SystemTools::GetFilenamePath(programPath.c_str());
  std::string ptemp;
  bool hasIntDir = false;
  std::string intDir = "";
  
  std::string tmpName = slicerBinDir + "/../lib/Slicer3/slicerd.tcl";
  if ( !vtksys::SystemTools::FileExists(tmpName.c_str()) )
    {
    // Handle Visual Studio IntDir
    std::vector<std::string> pathComponents;
    vtksys::SystemTools::SplitPath(slicerBinDir.c_str(), pathComponents);

    slicerBinDir = slicerBinDir + "/..";
    tmpName = slicerBinDir + "/../lib/Slicer3/slicerd.tcl";
    if ( !vtksys::SystemTools::FileExists(tmpName.c_str()) )
      {
      slicerCerr("Error: Cannot find Slicer3 libraries" << endl);
      return 1;
      }

    if (pathComponents.size() > 0)
      {
      hasIntDir = true;
      intDir = pathComponents[pathComponents.size()-1];
      }
    }
  std::string tclEnv = "TCL_LIBRARY=";
  tclEnv += slicerBinDir + "/../lib/Slicer3/tcl/lib/tcl8.4";
  cerr << "Set environment: " << tclEnv.c_str() << endl;
  putenv(const_cast <char *> (tclEnv.c_str()));

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
            slicerCerr("Error: InitializeTcl failed" << endl );
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
    // load itcl package (needed for interactive widgets)
    //
    {    
      std::string cmd;
      int returnCode;

      // Pass arguments to the Tcl script
      cmd =  "package require Itcl;";
      returnCode = Slicer3_Tcl_Eval( interp, cmd.c_str() );
      if ( returnCode )
        {
        slicerCerr("Error: slicer requires the Itcl package (" << cmd.c_str() << ")" << endl);
        return ( returnCode );
        }
    }

    //
    // load the gui tcl package (for the interactive widgets)
    //
    {    
      std::string cmd;
      int returnCode;

      // Pass arguments to the Tcl script
      cmd =  "lappend auto_path \"" + slicerBinDir + "/../"
        SLICER_INSTALL_LIBRARIES_DIR "\"; ";
      cmd +=  "lappend auto_path \"" + slicerBinDir + "/../../"
        SLICER_INSTALL_LIBRARIES_DIR "\"; ";
      //cmd += "puts $auto_path; ";
      cmd += "package require SlicerBaseGUITcl; ";
      returnCode = Slicer3_Tcl_Eval( interp, cmd.c_str() );
      if ( returnCode )
        {
        slicerCerr("Load SlicerBaseGUITcl: " << cmd.c_str() << endl);
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
    Freesurfer_Init(interp);
    Igt_Init(interp);

    //TODO added
#ifndef EMSEG_DEBUG
    Emsegment_Init(interp);
#endif
#ifndef REALTIMEIMAGING_DEBUG
    Realtimeimaging_Init(interp);
#endif

    Gradientanisotropicdiffusionfilter_Init(interp);
//    Slicertractographydisplay_Init(interp);
    Queryatlas_Init(interp);
    Slicerdaemon_Init(interp);
    Commandlinemodule_Init(interp);
    Scriptedmodule_Init(interp);
#ifndef WFENGINE_DEBUG
    Wfenginemodule_Init(interp);
#endif

    //
    // use the startup script passed on command line if it exists
    //
    if ( File != "" )
      {    

      std::string cmd;

      // Pass arguments to the Tcl script
      cmd = "set args \"\"; ";
      std::vector<std::string>::const_iterator argit = Args.begin();
      while (argit != Args.end())
        {
        cmd += " lappend args \"" + *argit + "\"; ";
        ++argit;
        }
      Slicer3_Tcl_Eval( interp, cmd.c_str() );

      cmd = "source " + File;
      return ( Slicer3_Tcl_Eval( interp, cmd.c_str() ) );
      }

    //
    // print out data provenance information if requested
    //
    if (AllInfo != 0)
      {
      printAllInfo(argc, argv);
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
    //  - note: use the singleton application 
    vtkSlicerApplication *slicerApp = vtkSlicerApplication::GetInstance ( );
    slicerApp->InstallTheme( slicerApp->GetSlicerTheme() );

    // copy the image from the header file into memory

    unsigned char *buffer = 
      new unsigned char [image_S3SplashScreen_length];

    unsigned int i;
    unsigned char *curPos = buffer;
    for (i = 0; i < image_S3SplashScreen_nb_sections; i++)
      {
      size_t len = strlen((const char*)image_S3SplashScreen_sections[i]);
      memcpy(curPos, image_S3SplashScreen_sections[i], len);
      curPos += len;
      }

    vtkKWTkUtilities::UpdatePhoto(
      slicerApp->GetMainInterp(),
      "S3SplashScreen", 
      buffer, 
      image_S3SplashScreen_width, 
      image_S3SplashScreen_height,
      image_S3SplashScreen_pixel_size,
      image_S3SplashScreen_length);
    delete [] buffer; 

    slicerApp->GetSplashScreen()->SetImageName("S3SplashScreen");
    slicerApp->SupportSplashScreenOn();
    slicerApp->SplashScreenVisibilityOn();
    slicerApp->GetSplashScreen()->SetProgressMessageVerticalOffset(-25);
    slicerApp->GetSplashScreen()->SetProgressMessage(
      "Initializing Window...");

    // Create MRML scene
    vtkMRMLScene *scene = vtkMRMLScene::New();
    vtksys_stl::string root = vtksys::SystemTools::GetCurrentWorkingDirectory();
    scene->SetRootDirectory(root.c_str());
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
    appLogic->CreateProcessingThread();

    slicerApp->GetSplashScreen()->SetProgressMessage(
      "Creating Application GUI...");

    // CREATE APPLICATION GUI, including the main window
    vtkSlicerApplicationGUI *appGUI = vtkSlicerApplicationGUI::New ( );
    appGUI->SetApplication ( slicerApp );
    appGUI->SetAndObserveApplicationLogic ( appLogic );
    appGUI->SetAndObserveMRMLScene ( scene );
    appGUI->BuildGUI ( );
    appGUI->AddGUIObservers ( );
    slicerApp->SetApplicationGUI ( appGUI );

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
    slicerApp->GetSplashScreen()->SetProgressMessage(
      "Initializing Volumes Module...");

    vtkSlicerVolumesLogic *volumesLogic = vtkSlicerVolumesLogic::New ( );
    volumesLogic->SetAndObserveMRMLScene ( scene );
    vtkSlicerVolumesGUI *volumesGUI = vtkSlicerVolumesGUI::New ( );
    volumesGUI->SetApplication ( slicerApp );
    volumesGUI->SetApplicationGUI ( appGUI );
    volumesGUI->SetAndObserveApplicationLogic ( appLogic );
    volumesGUI->SetAndObserveMRMLScene ( scene );
    volumesGUI->SetModuleLogic ( volumesLogic );
    volumesGUI->SetGUIName( "Volumes" );
    volumesGUI->GetUIPanel()->SetName ( volumesGUI->GetGUIName ( ) );
    volumesGUI->GetUIPanel()->SetUserInterfaceManager (appGUI->GetMainSlicerWindow()->GetMainUserInterfaceManager ( ) );
    volumesGUI->GetUIPanel()->Create ( );
    slicerApp->AddModuleGUI ( volumesGUI );
    volumesGUI->BuildGUI ( );
    volumesGUI->AddGUIObservers ( );
#endif

#ifndef MODELS_DEBUG
    slicerApp->GetSplashScreen()->SetProgressMessage(
      "Initializing Models Module...");

    // --- Models module    
    vtkSlicerModelsLogic *modelsLogic = vtkSlicerModelsLogic::New ( );
    modelsLogic->SetAndObserveMRMLScene ( scene );
    vtkSlicerModelsGUI *modelsGUI = vtkSlicerModelsGUI::New ( );
    modelsGUI->SetApplication ( slicerApp );
    modelsGUI->SetApplicationGUI ( appGUI );
    modelsGUI->SetAndObserveApplicationLogic ( appLogic );
    modelsGUI->SetAndObserveMRMLScene ( scene );
    modelsGUI->SetModuleLogic ( modelsLogic );
    modelsGUI->SetGUIName( "Models" );
    modelsGUI->GetUIPanel()->SetName ( modelsGUI->GetGUIName ( ) );
    modelsGUI->GetUIPanel()->SetUserInterfaceManager (appGUI->GetMainSlicerWindow()->GetMainUserInterfaceManager ( ) );
    modelsGUI->GetUIPanel()->Create ( );
    slicerApp->AddModuleGUI ( modelsGUI );
    modelsGUI->BuildGUI ( );
    modelsGUI->AddGUIObservers ( );
#endif


#ifndef FIDUCIALS_DEBUG
    slicerApp->GetSplashScreen()->SetProgressMessage(
      "Initializing Fiducials Module...");

    // --- Fiducials module    
    vtkSlicerFiducialsLogic *fiducialsLogic = vtkSlicerFiducialsLogic::New ( );
    fiducialsLogic->SetAndObserveMRMLScene ( scene );
    vtkSlicerFiducialsGUI *fiducialsGUI = vtkSlicerFiducialsGUI::New ( );
    fiducialsGUI->SetApplication ( slicerApp );
    fiducialsGUI->SetApplicationGUI ( appGUI );
    fiducialsGUI->SetAndObserveApplicationLogic ( appLogic );
    fiducialsGUI->SetAndObserveMRMLScene ( scene );
    fiducialsGUI->SetModuleLogic ( fiducialsLogic );
    fiducialsGUI->SetGUIName( "Fiducials" );
    fiducialsGUI->GetUIPanel()->SetName ( fiducialsGUI->GetGUIName ( ) );
    fiducialsGUI->GetUIPanel()->SetUserInterfaceManager (appGUI->GetMainSlicerWindow()->GetMainUserInterfaceManager ( ) );
    fiducialsGUI->GetUIPanel()->Create ( );
    slicerApp->AddModuleGUI ( fiducialsGUI );
    fiducialsGUI->BuildGUI ( );
    fiducialsGUI->AddGUIObservers ( );
#endif

#ifndef COLORS_DEBUG
    slicerApp->GetSplashScreen()->SetProgressMessage(
      "Initializing Colors Module...");

    // -- Color module
    vtkSlicerColorLogic *colorLogic = vtkSlicerColorLogic::New ( );
    // observe the scene's new scene event
    vtkIntArray *colorEvents = vtkIntArray::New();
    colorEvents->InsertNextValue( vtkMRMLScene::NewSceneEvent );
    colorEvents->InsertNextValue( vtkMRMLScene::SceneCloseEvent );
    colorLogic->SetAndObserveMRMLSceneEvents ( scene,  colorEvents);
    colorEvents->Delete();
    // this should be triggered somehow by a new scene event, but for now,
    // call it explicitly
    colorLogic->AddDefaultColorNodes();
    vtkSlicerColorGUI *colorGUI = vtkSlicerColorGUI::New ( );
    colorGUI->SetApplication ( slicerApp );
    colorGUI->SetApplicationGUI ( appGUI );
    colorGUI->SetAndObserveApplicationLogic ( appLogic );
    colorGUI->SetAndObserveMRMLScene ( scene );
    colorGUI->SetModuleLogic ( colorLogic );
    colorGUI->SetGUIName( "Color" );
    colorGUI->GetUIPanel()->SetName ( colorGUI->GetGUIName ( ) );
    colorGUI->GetUIPanel()->SetUserInterfaceManager (appGUI->GetMainSlicerWindow()->GetMainUserInterfaceManager ( ) );
    colorGUI->GetUIPanel()->Create ( );
    slicerApp->AddModuleGUI ( colorGUI );
    colorGUI->BuildGUI ( );
    colorGUI->AddGUIObservers ( );
#endif


#ifndef REALTIMEIMAGING_DEBUG
    // -- Real Time Imaging module
    vtkRealTimeImagingLogic *realtimeimagingLogic = vtkRealTimeImagingLogic::New ( );
    realtimeimagingLogic->SetAndObserveMRMLScene ( scene );
    realtimeimagingLogic->AddRealTimeVolumeNode ("RealTime");
    vtkRealTimeImagingGUI *realtimeimagingGUI = vtkRealTimeImagingGUI::New ( );

    realtimeimagingGUI->SetApplication ( slicerApp );
    realtimeimagingGUI->SetApplicationGUI ( appGUI );
    realtimeimagingGUI->SetAndObserveApplicationLogic ( appLogic );
    realtimeimagingGUI->SetAndObserveMRMLScene ( scene );
    realtimeimagingGUI->SetModuleLogic ( realtimeimagingLogic );
    realtimeimagingGUI->SetGUIName( "Real Time Imaging" );
    realtimeimagingGUI->GetUIPanel()->SetName ( realtimeimagingGUI->GetGUIName ( ) );
    realtimeimagingGUI->GetUIPanel()->SetUserInterfaceManager(appGUI->GetMainSlicerWindow()->GetMainUserInterfaceManager ( ) );
    realtimeimagingGUI->GetUIPanel()->Create ( );
    slicerApp->AddModuleGUI ( realtimeimagingGUI );
    realtimeimagingGUI->BuildGUI ( );
    realtimeimagingGUI->AddGUIObservers ( );
#endif 

#ifndef MRABLATION_DEBUG
    // -- MRAblation module
    vtkMRAblationLogic *ablationLogic = vtkMRAblationLogic::New(); 
    ablationLogic->SetAndObserveMRMLScene ( scene );
    vtkMRAblationGUI *ablationGUI = vtkMRAblationGUI::New();

    ablationGUI->SetApplication ( slicerApp );
    ablationGUI->SetApplicationGUI ( appGUI );
    ablationGUI->SetAndObserveApplicationLogic ( appLogic );
    ablationGUI->SetAndObserveMRMLScene ( scene );
    ablationGUI->SetModuleLogic ( ablationLogic );
    ablationGUI->SetGUIName( "MRAblation" );
    ablationGUI->GetUIPanel()->SetName ( ablationGUI->GetGUIName ( ) );
    ablationGUI->GetUIPanel()->SetUserInterfaceManager (appGUI->GetMainSlicerWindow()->GetMainUserInterfaceManager ( ) );
    ablationGUI->GetUIPanel()->Create ( );
    slicerApp->AddModuleGUI ( ablationGUI );
    ablationGUI->BuildGUI ( );
    ablationGUI->AddGUIObservers ( );
#endif 


#ifndef NEURONAV_DEBUG
    // -- NeuroNav module
    vtkNeuroNavLogic *neuronavLogic = vtkNeuroNavLogic::New(); 
    neuronavLogic->SetAndObserveMRMLScene ( scene );
    vtkNeuroNavGUI *neuronavGUI = vtkNeuroNavGUI::New();

    neuronavGUI->SetApplication ( slicerApp );
    neuronavGUI->SetApplicationGUI ( appGUI );
    neuronavGUI->SetAndObserveApplicationLogic ( appLogic );
    neuronavGUI->SetAndObserveMRMLScene ( scene );
    neuronavGUI->SetModuleLogic ( neuronavLogic );
    neuronavGUI->SetGUIName( "NeuroNav" );
    neuronavGUI->GetUIPanel()->SetName ( neuronavGUI->GetGUIName ( ) );
    neuronavGUI->GetUIPanel()->SetUserInterfaceManager (appGUI->GetMainSlicerWindow()->GetMainUserInterfaceManager ( ) );
    neuronavGUI->GetUIPanel()->Create ( );
    slicerApp->AddModuleGUI ( neuronavGUI );
    neuronavGUI->BuildGUI ( );
    neuronavGUI->AddGUIObservers ( );
    neuronavGUI->Init();
#endif 

    // --- Transforms module
    slicerApp->GetSplashScreen()->SetProgressMessage(
      "Initializing Transforms Module...");

    vtkSlicerTransformsGUI *transformsGUI = vtkSlicerTransformsGUI::New ( );
    transformsGUI->SetApplication ( slicerApp );
    transformsGUI->SetApplicationGUI ( appGUI );
    transformsGUI->SetAndObserveApplicationLogic ( appLogic );
    transformsGUI->SetAndObserveMRMLScene ( scene );
    transformsGUI->SetGUIName( "Transforms" );
    transformsGUI->GetUIPanel()->SetName ( transformsGUI->GetGUIName ( ) );
    transformsGUI->GetUIPanel()->SetUserInterfaceManager (appGUI->GetMainSlicerWindow()->GetMainUserInterfaceManager ( ) );
    transformsGUI->GetUIPanel()->Create ( );
    slicerApp->AddModuleGUI ( transformsGUI );
    transformsGUI->BuildGUI ( );
    transformsGUI->AddGUIObservers ( );

    //--- Data module
    slicerApp->GetSplashScreen()->SetProgressMessage(
      "Initializing Data Module...");

    //vtkSlicerDataLogic *dataLogic = vtkSlicerDataLogic::New ( );
    //dataLogic->SetAndObserveMRMLScene ( scene );
    //dataLogic->SetApplicationLogic ( appLogic );
    vtkSlicerDataGUI *dataGUI = vtkSlicerDataGUI::New ( );
    dataGUI->SetApplication ( slicerApp );
    dataGUI->SetApplicationGUI ( appGUI );
    dataGUI->SetAndObserveApplicationLogic ( appLogic );
    dataGUI->SetAndObserveMRMLScene ( scene );
    //dataGUI->SetModuleLogic ( dataLogic );
    dataGUI->SetGUIName( "Data" );
    dataGUI->GetUIPanel()->SetName ( dataGUI->GetGUIName ( ) );
    dataGUI->GetUIPanel()->SetUserInterfaceManager (appGUI->GetMainSlicerWindow()->GetMainUserInterfaceManager ( ) );
    dataGUI->GetUIPanel()->Create ( );    
    slicerApp->AddModuleGUI ( dataGUI );
    dataGUI->BuildGUI ( );
    dataGUI->AddGUIObservers ( );
    dataGUI->AddObserver (vtkSlicerModuleGUI::ModuleSelectedEvent, (vtkCommand *)appGUI->GetGUICallbackCommand() );
  
#ifndef CAMERA_DEBUG
    slicerApp->GetSplashScreen()->SetProgressMessage(
      "Initializing Camera Module...");

    // --- Camera module
    vtkSlicerCamerasGUI *cameraGUI = vtkSlicerCamerasGUI::New ( );
    cameraGUI->SetApplication ( slicerApp );
    cameraGUI->SetApplicationGUI ( appGUI );
    cameraGUI->SetAndObserveApplicationLogic ( appLogic );
    cameraGUI->SetAndObserveMRMLScene ( scene );
    cameraGUI->SetGUIName( "Cameras" );
    cameraGUI->GetUIPanel()->SetName ( cameraGUI->GetGUIName ( ) );
    cameraGUI->GetUIPanel()->SetUserInterfaceManager (appGUI->GetMainSlicerWindow()->GetMainUserInterfaceManager ( ) );
    cameraGUI->GetUIPanel()->Create ( );
    slicerApp->AddModuleGUI ( cameraGUI );
    cameraGUI->BuildGUI ( );
    cameraGUI->AddGUIObservers ( );
    cameraGUI->UpdateCameraSelector();
#endif

    // --- Slices module
    slicerApp->GetSplashScreen()->SetProgressMessage(
      "Initializing Slices Module...");
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
    events->Delete();

#ifndef SLICES_DEBUG
    vtkSlicerSlicesGUI *slicesGUI = vtkSlicerSlicesGUI::New ();
    slicesGUI->SetApplication ( slicerApp );
    slicesGUI->SetApplicationGUI ( appGUI );
    slicesGUI->SetAndObserveApplicationLogic ( appLogic );
    slicesGUI->SetAndObserveMRMLScene ( scene );
    slicesGUI->SetGUIName( "Slices" );
    slicesGUI->GetUIPanel()->SetName ( slicesGUI->GetGUIName ( ) );
    slicesGUI->GetUIPanel()->SetUserInterfaceManager ( appGUI->GetMainSlicerWindow( )->GetMainUserInterfaceManager( ) );
    slicesGUI->GetUIPanel( )->Create( );
    slicerApp->AddModuleGUI ( slicesGUI );
    slicesGUI->BuildGUI ();
    slicesGUI->AddGUIObservers ( );
#endif


   

    // --- Gradient anisotropic diffusion filter module
    slicerApp->GetSplashScreen()->SetProgressMessage(
      "Initializing Gradient Anisotropic Module...");
    vtkGradientAnisotropicDiffusionFilterGUI *gradientAnisotropicDiffusionFilterGUI = vtkGradientAnisotropicDiffusionFilterGUI::New ( );
    vtkGradientAnisotropicDiffusionFilterLogic *gradientAnisotropicDiffusionFilterLogic  = vtkGradientAnisotropicDiffusionFilterLogic::New ( );
    gradientAnisotropicDiffusionFilterLogic->SetAndObserveMRMLScene ( scene );
    gradientAnisotropicDiffusionFilterLogic->SetApplicationLogic ( appLogic );
    //    gradientAnisotropicDiffusionFilterLogic->SetMRMLScene(scene);
    gradientAnisotropicDiffusionFilterGUI->SetLogic ( gradientAnisotropicDiffusionFilterLogic );
    gradientAnisotropicDiffusionFilterGUI->SetApplication ( slicerApp );
    gradientAnisotropicDiffusionFilterGUI->SetApplicationLogic ( appLogic );
    gradientAnisotropicDiffusionFilterGUI->SetApplicationGUI ( appGUI );
    gradientAnisotropicDiffusionFilterGUI->SetGUIName( "GradientAnisotropicDiffusionFilter" );
    gradientAnisotropicDiffusionFilterGUI->GetUIPanel()->SetName ( gradientAnisotropicDiffusionFilterGUI->GetGUIName ( ) );
    gradientAnisotropicDiffusionFilterGUI->GetUIPanel()->SetUserInterfaceManager (appGUI->GetMainSlicerWindow()->GetMainUserInterfaceManager ( ) );
    gradientAnisotropicDiffusionFilterGUI->GetUIPanel()->Create ( );
    slicerApp->AddModuleGUI ( gradientAnisotropicDiffusionFilterGUI );
    gradientAnisotropicDiffusionFilterGUI->BuildGUI ( );
    gradientAnisotropicDiffusionFilterGUI->AddGUIObservers ( );

#ifndef TRACTOGRAPHY_DEBUG
    // --- Tractography Display module
    slicerApp->GetSplashScreen()->SetProgressMessage(
      "Initializing Tractography Display Module...");
    vtkSlicerTractographyDisplayGUI *slicerTractographyDisplayGUI = vtkSlicerTractographyDisplayGUI::New ( );
    //vtkSlicerTractographyDisplayLogic *slicerTractographyDisplayLogic  = vtkSlicerTractographyDisplayLogic::New ( );
    //slicerTractographyDisplayLogic->SetAndObserveMRMLScene ( scene );
    //slicerTractographyDisplayLogic->SetApplicationLogic ( appLogic );
    ////    slicerTractographyDisplayLogic->SetMRMLScene(scene);
    //slicerTractographyDisplayGUI->SetLogic ( slicerTractographyDisplayLogic );
    slicerTractographyDisplayGUI->SetApplication ( slicerApp );
    //slicerTractographyDisplayGUI->SetApplicationLogic ( appLogic );
    slicerTractographyDisplayGUI->SetApplicationGUI ( appGUI );
    slicerTractographyDisplayGUI->SetGUIName( "Tractography" );
    slicerTractographyDisplayGUI->GetUIPanel()->SetName ( slicerTractographyDisplayGUI->GetGUIName ( ) );
    slicerTractographyDisplayGUI->GetUIPanel()->SetUserInterfaceManager (appGUI->GetMainSlicerWindow()->GetMainUserInterfaceManager ( ) );
    slicerTractographyDisplayGUI->GetUIPanel()->Create ( );
    slicerApp->AddModuleGUI ( slicerTractographyDisplayGUI );
    slicerTractographyDisplayGUI->BuildGUI ( );
    slicerTractographyDisplayGUI->AddGUIObservers ( );
#endif


#ifndef EMSEG_DEBUG
    //
    // --- EMSegment Template builder module
    //
    vtkEMSegmentGUI *emSegmentGUI = 
      vtkEMSegmentGUI::New ( );
    vtkEMSegmentLogic *emSegmentLogic = 
      vtkEMSegmentLogic::New ( );

    emSegmentLogic->SetAndObserveMRMLScene(scene);
    vtkIntArray *emsEvents = vtkIntArray::New();
    emsEvents->InsertNextValue(vtkMRMLScene::NodeAddedEvent);
    emsEvents->InsertNextValue(vtkMRMLScene::NodeRemovedEvent);
    emSegmentLogic->SetAndObserveMRMLSceneEvents(scene, emsEvents);
    emSegmentLogic->SetApplicationLogic(appLogic);

    emSegmentGUI->SetAndObserveMRMLScene (scene);
    emSegmentGUI->SetLogic(emSegmentLogic);
    emSegmentGUI->SetApplication(slicerApp);
    emSegmentGUI->SetApplicationLogic(appLogic);
    emSegmentGUI->SetApplicationGUI(appGUI);
    emSegmentGUI->SetGUIName("EMSegment");
    emSegmentGUI->GetUIPanel()->
      SetName(emSegmentGUI->GetGUIName());
    emSegmentGUI->GetUIPanel()->
      SetUserInterfaceManager(appGUI->GetMainSlicerWindow()->
                              GetMainUserInterfaceManager());
    emSegmentGUI->GetUIPanel()->Create();
    slicerApp->AddModuleGUI(emSegmentGUI);
    emSegmentGUI->BuildGUI();
    emSegmentGUI->AddGUIObservers();
    //
    // --- END EMSegment Template builder module
    //
#endif

#ifndef QUERYATLAS_DEBUG
    slicerApp->GetSplashScreen()->SetProgressMessage(
      "Initializing Query Atlas Module...");
    //--- Query Atlas Module
    vtkQueryAtlasGUI *queryAtlasGUI = vtkQueryAtlasGUI::New ( );
    vtkQueryAtlasLogic *queryAtlasLogic  = vtkQueryAtlasLogic::New ( );
    queryAtlasLogic->SetAndObserveMRMLScene ( scene );
    queryAtlasLogic->SetApplicationLogic ( appLogic );
    queryAtlasLogic->SetMRMLScene(scene);
    queryAtlasGUI->SetApplication ( slicerApp );
    queryAtlasGUI->SetApplicationLogic ( appLogic );
    queryAtlasGUI->SetApplicationGUI ( appGUI );
    queryAtlasGUI->SetGUIName( "QueryAtlas" );
    queryAtlasGUI->GetUIPanel()->SetName ( queryAtlasGUI->GetGUIName ( ) );
    queryAtlasGUI->GetUIPanel()->SetUserInterfaceManager (appGUI->GetMainSlicerWindow()->GetMainUserInterfaceManager ( ) );
    queryAtlasGUI->GetUIPanel()->Create ( );
    slicerApp->AddModuleGUI ( queryAtlasGUI );
    queryAtlasGUI->BuildGUI ( );
    queryAtlasGUI->AddGUIObservers ( );
#endif
    
#ifndef WFENGINE_DEBUG
    slicerApp->GetSplashScreen()->SetProgressMessage(
      "Initializing WFEngine Module...");
    //--- WFEngine Module
    vtkWFEngineModuleGUI *wfEngineModuleGUI = vtkWFEngineModuleGUI::New ( );
    vtkWFEngineModuleLogic *wfEngineModuleLogic  = vtkWFEngineModuleLogic::New ( );
    wfEngineModuleLogic->SetAndObserveMRMLScene ( scene );
    wfEngineModuleLogic->SetApplicationLogic ( appLogic );
    wfEngineModuleLogic->SetMRMLScene(scene);
    wfEngineModuleGUI->SetApplication ( slicerApp );
    wfEngineModuleGUI->SetApplicationLogic ( appLogic );
    wfEngineModuleGUI->SetApplicationGUI ( appGUI );
    wfEngineModuleGUI->SetGUIName( "WFEngineModule" );
    wfEngineModuleGUI->GetUIPanel()->SetName ( wfEngineModuleGUI->GetGUIName ( ) );
    wfEngineModuleGUI->GetUIPanel()->SetUserInterfaceManager (appGUI->GetMainSlicerWindow()->GetMainUserInterfaceManager ( ) );
    wfEngineModuleGUI->GetUIPanel()->Create ( );
    slicerApp->AddModuleGUI ( wfEngineModuleGUI );
    wfEngineModuleGUI->BuildGUI ( );
    wfEngineModuleGUI->AddGUIObservers ( );
#endif

    // --- SlicerDaemon Module
    // need to source the slicerd.tcl script here
    // - only start if selected on command line, since 
    //   windows firewall will complain. 
    //   TODO: this could be changed to registry option in the future

    if ( Daemon )
      {
      std::string cmd;
      cmd =  "source \"" + slicerBinDir + "/../"
        SLICER_INSTALL_LIBRARIES_DIR "/slicerd.tcl\"; slicerd_start; ";
      Slicer3_Tcl_Eval(interp, cmd.c_str());
      }


#ifndef CLIMODULES_DEBUG
    std::vector<std::string> moduleNames;
    std::vector<std::string>::const_iterator mit;
    if (slicerApp->GetLoadCommandLineModules())
      {
      // --- Scan for command line and shared object modules
      //
      // - set the module path the the default if there isn't something set
      // already
      //
      // NB: don't want to use the registry value of the module path, because 
      // there is only one value even if you have multiple versions of 
      // slicer installed, you may get conflicts.
      //
      //
#ifdef _WIN32
      std::string delim(";");
#else
      std::string delim(":");
#endif
      std::string packagePath;
      std::string defaultPackageDir;
      std::string userPackagePath;

      // define a default plugin path based on the slicer installation
      // or build tree
      defaultPackageDir = slicerBinDir + "/../lib/Slicer3/Plugins";
      if (hasIntDir)
        {
        defaultPackageDir += "/" + intDir;
        }
    
      // get the path of that the user has configured
      if (slicerApp->GetModulePath())
        {
        userPackagePath = slicerApp->GetModulePath();
        }

      // add the default plugin directory (based on the slicer
      // installation or build tree) to the user path
      packagePath = userPackagePath + delim + defaultPackageDir;

      // Search for modules
      ModuleFactory moduleFactory;
      moduleFactory.SetSearchPath( packagePath );
      moduleFactory.SetWarningMessageCallback( WarningMessage );
      moduleFactory.SetErrorMessageCallback( ErrorMessage );
      moduleFactory.SetInformationMessageCallback( InformationMessage );
      moduleFactory.SetModuleDiscoveryMessageCallback( SplashMessage );
      moduleFactory.Scan();

      // add the modules to the available modules
      moduleNames = moduleFactory.GetModuleNames();
      mit = moduleNames.begin();
      while (mit != moduleNames.end())
        {
        // slicerCerr(moduleFactory.GetModuleDescription(*mit) << endl);

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
        commandLineModuleGUI->SetApplicationGUI ( appGUI );
        commandLineModuleGUI->SetGUIName( moduleFactory.GetModuleDescription(*mit).GetTitle().c_str() );
        commandLineModuleGUI->GetUIPanel()->SetName ( commandLineModuleGUI->GetGUIName ( ) );
        commandLineModuleGUI->GetUIPanel()->SetUserInterfaceManager (appGUI->GetMainSlicerWindow()->GetMainUserInterfaceManager ( ) );
        commandLineModuleGUI->GetUIPanel()->Create ( );
        slicerApp->AddModuleGUI ( commandLineModuleGUI );

        ++mit;
        }
      // -- Build the factory discovered modules gui and observers
      mit = moduleNames.begin();
      while ( mit != moduleNames.end() )
        {
        std::string progress_msg("Initializing ");
        progress_msg +=  (*mit) ;
        progress_msg += " Module...";
        slicerApp->GetSplashScreen()->SetProgressMessage(progress_msg.c_str());

        vtkSlicerModuleGUI *module;
        module = slicerApp->GetModuleGUIByName( (*mit).c_str() );

        module->BuildGUI();
        module->AddGUIObservers();
      
        ++mit;
        }
      }
#endif

    //
    // get the Tcl name so the vtk class will be registered in the interpreter
    // as a byproduct
    // - set some handy variables so it will be easy to access these classes
    ///  from   the tkcon
    // - all the variables are put in the slicer3 tcl namespace for easy access
    //
    const char *name;
    name = slicerApp->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set Application %s", name);
    name = appGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set ApplicationGUI %s", name);
#ifndef SLICES_DEBUG
    name = slicesGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set SlicesGUI %s", name);
#endif

#ifndef VOLUMES_DEBUG
    name = volumesGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set VolumesGUI %s", name);
#endif
#ifndef MODELS_DEBUG
    name = modelsGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set ModelsGUI %s", name);
#endif
#ifndef FIDUCIALS_DEBUG
    name = fiducialsGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set FiducialsGUI %s", name);
#endif
#ifndef COLORS_DEBUG
    name = colorGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set ColorGUI %s", name);
#endif
#ifndef RealTimeImaging_DEBUG
    name = realtimeimagingGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set RealTimeImagingGUI %s", name);
#endif
#ifndef MRABLATION_DEBUG
    name = ablationGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set MRAblationGUI %s", name);
#endif

#ifndef NEURONAV_DEBUG
    name = neuronavGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set NeuroNavGUI %s", name);
#endif

    name = transformsGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set TransformsGUI %s", name);
#ifndef QUERYATLAS_DEBUG
    name = queryAtlasGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set QueryAtlasGUI %s", name);
#endif
    
#ifndef WFENGINE_DEBUG
    name = wfEngineModuleGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set WFEngineModuleGUI %s", name);
#endif


    name = appGUI->GetViewerWidget()->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set ViewerWidget %s", name);

    slicerApp->Script ("namespace eval slicer3 set ApplicationLogic [$::slicer3::ApplicationGUI GetApplicationLogic]");
    slicerApp->Script ("namespace eval slicer3 set MRMLScene [$::slicer3::ApplicationLogic GetMRMLScene]");


#ifndef CLIMODULES_DEBUG
    mit = moduleNames.begin();
    while ( mit != moduleNames.end() )
      {
      vtkSlicerModuleGUI *module;
      module = slicerApp->GetModuleGUIByName( (*mit).c_str() );

      name = module->GetTclName();

      std::string title = *mit;
      std::transform(
        title.begin(), title.end(), title.begin(), SpacesToUnderscores());

      slicerApp->Script ("namespace eval slicer3 set CommandLineModuleGUI_%s %s", title.c_str(), name);
      
      ++mit;
      }
#endif

#ifndef TCLMODULES_DEBUG
    //
    // process any ScriptedModules
    // - scan for pkgIndex.tcl files 
    // - load the corresponding packages
    // - create scripted logic and scripted gui instances for the module
    // - have the GUI construct itself
    //

    std::string tclCommand = "set ::SLICER_PACKAGES(list) {};";
    tclCommand += "set dirs [glob \"" + slicerBinDir + "/../"
      SLICER_INSTALL_LIBRARIES_DIR "/Modules/Packages/*\"]; ";
    tclCommand += "foreach d $dirs { ";
    tclCommand += "  if { [file exists $d/pkgIndex.tcl] } {";
    tclCommand += "    lappend ::SLICER_PACKAGES(list) [file tail $d];";
    tclCommand += "    lappend ::auto_path $d;";
    tclCommand += "  }";
    tclCommand += "} ";
    Slicer3_Tcl_Eval( interp, tclCommand.c_str() );

    tclCommand = "";
    tclCommand += "foreach package $::SLICER_PACKAGES(list) { ";
    tclCommand += "  package require $package;";
    tclCommand += "  set ::SLICER_PACKAGES($package,logic) [vtkScriptedModuleLogic New];";
    tclCommand += "  set logic $::SLICER_PACKAGES($package,logic);";
    tclCommand += "  $logic SetModuleName $package;";
    tclCommand += "  $logic SetAndObserveMRMLScene $::slicer3::MRMLScene;";
    tclCommand += "  $logic SetApplicationLogic $::slicer3::ApplicationLogic;";
    tclCommand += "  set ::SLICER_PACKAGES($package,gui) [vtkScriptedModuleGUI New];";
    tclCommand += "  set gui $::SLICER_PACKAGES($package,gui);";
    tclCommand += "  $gui SetModuleName $package;";
    tclCommand += "  $gui SetLogic $logic;";
    tclCommand += "  $gui SetApplicationGUI $::slicer3::ApplicationGUI;";
    tclCommand += "  $gui SetApplication $::slicer3::Application;";
    tclCommand += "  $gui SetGUIName $package;";
    tclCommand += "  [$gui GetUIPanel] SetName $package;";
    tclCommand += "  [$gui GetUIPanel] SetUserInterfaceManager [[$::slicer3::ApplicationGUI GetMainSlicerWindow] GetMainUserInterfaceManager];";
    tclCommand += "  [$gui GetUIPanel] Create;";
    tclCommand += "  $::slicer3::Application AddModuleGUI $gui;" ;
    tclCommand += "  $gui BuildGUI;";
    tclCommand += "  $gui AddGUIObservers;";
    tclCommand += "";
    tclCommand += "}";
    Slicer3_Tcl_Eval( interp, tclCommand.c_str() );
#endif

    //
    // create the three main slice viewers after slicesGUI is created
    //
    appGUI->PopulateModuleChooseList ( );
#ifndef SLICES_DEBUG
    appGUI->SetSliceGUICollection ( slicesGUI->GetSliceGUICollection() );
    appGUI->SetAndObserveMainSliceLogic ( sliceLogic0, sliceLogic1, sliceLogic2 );
    appGUI->AddMainSliceViewersToCollection ( );
    appGUI->ConfigureMainSliceViewers ( );
#endif
    
    // ------------------------------
    // CONFIGURE SlICER'S SHARED GUI PANEL
    // Additional Modules GUI panel configuration.
    vtkKWUserInterfaceManagerNotebook *mnb = vtkKWUserInterfaceManagerNotebook::SafeDownCast (appGUI->GetMainSlicerWindow()->GetMainUserInterfaceManager());
    mnb->GetNotebook()->AlwaysShowTabsOff();
    mnb->GetNotebook()->ShowOnlyPagesWithSameTagOn();    

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

    int res; // return code (exit code)

    // use the startup code passed on command line if it exists
    // Example for csh to exit after startup:
    //
    // ./Slicer3 --exec 'after idle $::slicer3::Application Exit' 
    //
    if ( Exec != "" )
      {    
      std::string cmd = "set ::SLICER(exec) \"" + Exec + "\";";
      cmd += "regsub -all {\\.,} $::SLICER(exec) \";\" ::SLICER(exec);";
      cmd += "regsub -all {,\\.} $::SLICER(exec) \";\" ::SLICER(exec);";
      cmd += "eval $::SLICER(exec);";
      res = Slicer3_Tcl_Eval( interp, cmd.c_str() );
      }

    //
    // if there was no script file, the Args should hold a mrml file
    //
    if ( File == ""  && Args.size() != 0)
    {
        std::vector<std::string>::const_iterator argit = Args.begin();
        while (argit != Args.end())
        {
            cerr << "Arg =  " << *argit << endl;
            std::string fileName;
            fileName.append(*argit);
            // is it a MRML or XML file?
            if (fileName.find(".mrml",0) != std::string::npos ||
                fileName.find(".xml",0) != std::string::npos)
            {
                cerr << fileName << " is a MRML or XML file, setting MRML scene file name and connecting\n";
                appLogic->GetMRMLScene()->SetURL(fileName.c_str());
                // and then load it
                int errorCode = appLogic->GetMRMLScene()->Connect();
                if (errorCode != 1)
                {
                    slicerCerr("ERROR loading MRML file " << fileName << ", error code = " << errorCode << endl);
                }
            }                    
            ++argit;
        }
    }

    //--- set home module based on registry settings
    if ( slicerApp->GetHomeModule() )
      {
      appGUI->SelectModule ( slicerApp->GetHomeModule() );
      }
    else
      {
      appGUI->SelectModule("Volumes");
      }

    //
    // Run!  - this will return when the user exits
    //
    res = slicerApp->StartApplication();

    appGUI->GetMainSlicerWindow()->PrepareForDelete();
    appGUI->GetMainSlicerWindow()->Close();

    // ------------------------------
    // REMOVE OBSERVERS and references to MRML and Logic
    gradientAnisotropicDiffusionFilterGUI->RemoveGUIObservers ( );

#ifndef TRACTOGRAPHY_DEBUG
    slicerTractographyDisplayGUI->RemoveGUIObservers ( );
#endif

#ifndef EMSEG_DEBUG
    emSegmentGUI->RemoveGUIObservers();
#endif

#ifndef QUERYATLAS_DEBUG
    queryAtlasGUI->RemoveGUIObservers ( );
#endif

#ifndef VOLUMES_DEBUG
    volumesGUI->RemoveGUIObservers ( );
#endif
#ifndef MODELS_DEBUG
    modelsGUI->RemoveGUIObservers ( );
#endif
#ifndef FIDUCIALS_DEBUG
    fiducialsGUI->RemoveGUIObservers ( );
#endif
#ifndef COLORS_DEBUG
    colorGUI->RemoveGUIObservers ( );
#endif
#ifndef RealTimeImaging_DEBUG
    realtimeimagingGUI->RemoveGUIObservers ( );
#endif
#ifndef MRABLATION_DEBUG
    ablationGUI->RemoveGUIObservers ( );
#endif
#ifndef NEURONAV_DEBUG
    neuronavGUI->RemoveGUIObservers ( );
#endif
    
#ifndef WFENGINE_DEBUG
    wfEngineModuleGUI->RemoveGUIObservers ( );
#endif

    transformsGUI->RemoveGUIObservers ( );
#ifndef CAMERA_DEBUG
    cameraGUI->RemoveGUIObservers ( );
#endif
    dataGUI->RemoveGUIObservers ( );
#ifndef SLICES_DEBUG
    slicesGUI->RemoveGUIObservers ( );
#endif
    appGUI->RemoveGUIObservers ( );
#ifndef SLICES_DEBUG
    appGUI->SetAndObserveMainSliceLogic ( NULL, NULL, NULL );

    // remove all from the slicesGUI collection of sliceGUIs
    slicesGUI->GetSliceGUICollection()->RemoveAllItems ( );
    appGUI->SetSliceGUICollection ( NULL );
#endif

#ifndef CLIMODULES_DEBUG
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
#endif    

#ifndef TCLMODULES_DEBUG
    // remove the observers from the scripted modules
    tclCommand = "";
    tclCommand += "foreach package $::SLICER_PACKAGES(list) { ";
    tclCommand += "  $::SLICER_PACKAGES($package,gui) RemoveGUIObservers;";
    tclCommand += "  $::SLICER_PACKAGES($package,gui) TearDownGUI;";
    tclCommand += "}";
    Slicer3_Tcl_Eval( interp, tclCommand.c_str() );
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

    gradientAnisotropicDiffusionFilterGUI->Delete ();

#ifndef TRACTOGRAPHY_DEBUG
    slicerTractographyDisplayGUI->Delete ();
#endif

#ifndef EMSEG_DEBUG
    emSegmentGUI->Delete();
#endif

#ifndef QUERYATLAS_DEBUG
    queryAtlasGUI->Delete ( );
#endif
    
#ifndef VOLUMES_DEBUG
    volumesGUI->Delete ();
#endif
#ifndef MODELS_DEBUG
    modelsGUI->Delete ();
#endif
#ifndef FIDUCIALS_DEBUG
    fiducialsGUI->Delete ();
#endif
#ifndef COLORS_DEBUG
    colorGUI->Delete();
#endif
#ifndef RealTimeImaging_DEBUG
    realtimeimagingGUI->Delete();
#endif    
#ifndef MRABLATION_DEBUG
    ablationGUI->Delete();
#endif    
#ifndef NEURONAV_DEBUG
    neuronavGUI->Delete();
#endif
    
#ifndef WFENGINE_DEBUG
    wfEngineModuleGUI->Delete ( );
#endif

    transformsGUI->Delete ();
#ifndef CAMERA_DEBUG
    cameraGUI->Delete ();
#endif
    dataGUI->Delete ();
#ifndef SLICES_DEBUG
    slicesGUI->Delete ();
#endif

#ifndef TCLMODULES_DEBUG
    tclCommand = "";
    tclCommand += "foreach package $::SLICER_PACKAGES(list) { ";
    tclCommand += "  $::SLICER_PACKAGES($package,gui) Delete;";
    tclCommand += "}";
    Slicer3_Tcl_Eval( interp, tclCommand.c_str() );
#endif

    // Release reference to applicaiton GUI
    // and delete it.
   slicerApp->SetApplicationGUI ( NULL );
   appGUI->DeleteComponentGUIs();
   appGUI->Delete ();

#ifndef CLIMODULES_DEBUG
    // delete the factory discovered module GUIs (as we delete the
    // GUIs, cache the associated logic instances so we can delete
    std::vector<vtkSlicerModuleGUI*>::iterator git;
    std::vector<vtkSlicerModuleLogic*> moduleLogics;
    for (git = moduleGUIs.begin(); git != moduleGUIs.end(); ++git)
      {
      moduleLogics.push_back(
        dynamic_cast<vtkCommandLineModuleGUI*>((*git))->GetLogic());

      (*git)->Delete();
      }
    moduleGUIs.clear();
#endif
    
    //--- delete logic next, removing Refs to MRML
    appLogic->ClearCollections ( );

    gradientAnisotropicDiffusionFilterLogic->SetAndObserveMRMLScene ( NULL );
    gradientAnisotropicDiffusionFilterLogic->Delete ();

    //slicerTractographyDisplayLogic->SetAndObserveMRMLScene ( NULL );
    //slicerTractographyDisplayLogic->Delete ();
        
#ifndef QUERYATLAS_DEBUG
    queryAtlasLogic->SetAndObserveMRMLScene ( NULL );
    queryAtlasLogic->Delete ( );
#endif
    
#ifndef VOLUMES_DEBUG
    volumesLogic->SetAndObserveMRMLScene ( NULL );
    volumesLogic->Delete();
#endif
#ifndef MODELS_DEBUG
    modelsLogic->SetAndObserveMRMLScene ( NULL );
    modelsLogic->Delete();
#endif
#ifndef FIDUCIALS_DEBUG
    fiducialsLogic->SetAndObserveMRMLScene ( NULL );
    fiducialsLogic->Delete();
#endif
#ifndef COLORS_DEBUG
    colorLogic->SetAndObserveMRMLScene ( NULL );
    colorLogic->Delete();
#endif

#ifndef RealTimeImaging_DEBUG
    realtimeimagingLogic->SetAndObserveMRMLScene ( NULL );
    realtimeimagingLogic->Delete();
#endif

#ifndef MRABLATION_DEBUG
    ablationLogic->SetAndObserveMRMLScene ( NULL );
    ablationLogic->Delete();
#endif    
#ifndef NEURONAV_DEBUG
    neuronavLogic->SetAndObserveMRMLScene ( NULL );
    neuronavLogic->Delete();
#endif
    
#ifndef WFENGINE_DEBUG
    wfEngineModuleLogic->SetAndObserveMRMLScene ( NULL );
    wfEngineModuleLogic->Delete ( );
#endif

    sliceLogic2->SetAndObserveMRMLScene ( NULL );
    sliceLogic2->Delete ();
    sliceLogic1->SetAndObserveMRMLScene ( NULL );
    sliceLogic1->Delete ();
    sliceLogic0->SetAndObserveMRMLScene ( NULL );
    sliceLogic0->Delete ();
#ifndef SLICES_DEBUG
#endif
    appLogic->SetAndObserveMRMLScene ( NULL );
    appLogic->TerminateProcessingThread();
    appLogic->Delete ();

#ifndef CLIMODULES_DEBUG
    // delete the factory discovered module Logics
    std::vector<vtkSlicerModuleLogic*>::iterator lit;
    for (lit = moduleLogics.begin(); lit != moduleLogics.end(); ++lit)
      {
      (*lit)->SetAndObserveMRMLScene ( NULL );
      (*lit)->Delete();
      }
    moduleLogics.clear();
#endif

#ifndef TCLMODULES_DEBUG
    // delete the scripted logics
    tclCommand = "";
    tclCommand += "foreach package $::SLICER_PACKAGES(list) { ";
    tclCommand += "  $::SLICER_PACKAGES($package,logic) SetAndObserveMRMLScene {};";
    tclCommand += "  $::SLICER_PACKAGES($package,logic) Delete;";
    tclCommand += "}";
    Slicer3_Tcl_Eval( interp, tclCommand.c_str() );
#endif
    

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







