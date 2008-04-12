#include "vtkOpenGLRenderWindow.h"
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
#include "vtkSlicerSlicesGUI.h"
#include "vtkMRMLScene.h"
#include "vtkSlicerComponentGUI.h"
#include "vtkSlicerApplicationGUI.h"
#include "vtkKWUserInterfacePanel.h"
#include "vtkKWUserInterfaceManager.h"
#include "vtkKWUserInterfaceManagerNotebook.h"
#include "vtkKWLogDialog.h"
#include "vtkSlicerGUICollection.h"
#include "vtkSlicerDataGUI.h"
#include "vtkSlicerTransformsGUI.h"
#include "vtkSlicerTheme.h"
#include "vtkSlicerWindow.h"
#include "vtkSlicerApplicationSettingsInterface.h"


#include "vtkSlicerConfigure.h" // for VTKSLICER_CONFIGURATION_TYPES

#include "vtkMRMLCommandLineModuleNode.h"
#include "vtkCommandLineModuleGUI.h"
#include "ModuleFactory.h"

#ifdef USE_PYTHON
// If debug, Python wants pythonxx_d.lib, so fake it out
#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif

extern "C" {
  void init_mytkinter( Tcl_Interp* );
  void init_slicer(void );
}
#include "vtkTclUtil.h"

#endif

#include <vtksys/SystemTools.hxx>
#include <vtksys/Directory.hxx>
#include <vtksys/stl/string>

// for data prov
#include <time.h>
#include <stdlib.h>

#include "Resources/vtkSlicerSplashScreen_ImageData.h"

#include <vtkSlicerModuleLogic.h>
#include <vtkSlicerModuleGUI.h>
#include <LoadableModuleFactory.h>

// uncomment these lines to disable a particular module (handy for debugging)
//#define CLIMODULES_DEBUG
//#define TCLMODULES_DEBUG
//#define SLICES_DEBUG

#if !defined(TCLMODULES_DEBUG) && defined(BUILD_MODULES)
#include "vtkScriptedModuleLogic.h"
#include "vtkScriptedModuleGUI.h"
#endif

#if !defined(COMMANDLINE_DEBUG) && defined(BUILD_MODULES)
#include "vtkCommandLineModuleLogic.h"
#include "vtkCommandLineModuleGUI.h"
#endif

//
// note: always write to cout rather than cerr so log messages will
// appear in the correct order when running tests.  Normally cerr preempts cout
// when output is buffered going to a terminal, but in the case of the launcher
// cerr is collected in a separate buffer and printed at the end.   Either
// way, it's better for all logging to go the the same channel so it shows
// up in the correct order.
//
#ifdef _WIN32
#  define slicerCerr(x) \
  do { \
    cout << x; \
    vtkstd::ostringstream str; \
    str << x; \
    MessageBox(NULL, str.str().c_str(), "Slicer Error", MB_OK);\
  } while (0)
#else
#  define slicerCerr(x) \
  cout << x
#endif


// Get the automated arg parsing code
#include "Slicer3CLP.h"

extern "C" int Slicerbasegui_Init(Tcl_Interp *interp);
extern "C" int Slicerbaselogic_Init(Tcl_Interp *interp);
extern "C" int Mrml_Init(Tcl_Interp *interp);
extern "C" int Vtkitk_Init(Tcl_Interp *interp);
extern "C" int Freesurfer_Init(Tcl_Interp *interp);
extern "C" int Igt_Init(Tcl_Interp *interp);
extern "C" int Vtkteem_Init(Tcl_Interp *interp);


#if !defined(DAEMON_DEBUG) && defined(BUILD_MODULES)
extern "C" int Slicerdaemon_Init(Tcl_Interp *interp);
#endif
#if !defined(COMMANDLINE_DEBUG) && defined(BUILD_MODULES)
extern "C" int Commandlinemodule_Init(Tcl_Interp *interp);
#endif
#if !defined(TCLMODULES_DEBUG) && defined(BUILD_MODULES)
extern "C" int Scriptedmodule_Init(Tcl_Interp *interp);
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
    slicerCerr("Error: " << Tcl_GetStringResult( interp ) << "\n" 
      << "while executing:\n" << script << "\n\n" 
      << Tcl_GetVar2( interp, "errorInfo", NULL, TCL_GLOBAL_ONLY | TCL_LEAVE_ERR_MSG ) );
    return 1;
    }
  return 0;
}

void printAllInfo(int argc, char **argv)
{
  int i;
  struct tm * timeInfo;
  time_t rawtime;
  // yyyy/mm/dd-hh-mm-ss-ms-TZ
  // plus one for 3 char time zone
  char timeStr[27];

  fprintf(stdout, "<ProcessStep>\n");
  fprintf(stdout, "<ProgramName version=\"$Revision$\">%s</ProgramName>\n", argv[0]);
  fprintf(stdout, "<ProgramArguments>");
  for (i = 1; i < argc; i++)
    {
    fprintf(stdout, " %s", argv[i]);
    }
  fprintf(stdout, "</ProgramArguments>\n");
  fprintf(stdout, "<CVS>$Id$</CVS> <TimeStamp>");
  time ( &rawtime );
  timeInfo = localtime (&rawtime);
  strftime (timeStr, 27, "%Y/%m/%d-%H-%M-%S-00-%Z", timeInfo);
  fprintf(stdout, "%s</TimeStamp>\n", timeStr);
  fprintf(stdout, "<User>%s</User>\n", getenv("USER"));
  
  fprintf(stdout, "<HostName>");
  if (getenv("HOSTNAME") == NULL)
    {
    if (getenv("HOST") == NULL)
      {
      fprintf(stdout, "(unknown)");
      }
    else
      {
      fprintf(stdout, "%s", getenv("HOST"));
      }
    }
  else
    {
    fprintf(stdout, "%s", getenv("HOSTNAME"));
    }
  fprintf(stdout, "</HostName><platform version=\"");
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
  
  fprintf(stdout, "\">");
  // now the platform name
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
  fprintf(stdout, "</platform>\n");

  if (getenv("MACHTYPE") != NULL)
    {
    fprintf(stdout, "<Architecture>%s</Architecture>\n", getenv("MACHTYPE"));
    }
  
  fprintf(stdout, "<Compiler version=\"");
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
  // now the compiler name
  fprintf(stdout, ">");
#if defined(__GNUC__)
  fprintf(stdout, "GCC");
#else
#if defined(_MSC_VER)
  fprintf(stdout, "MSC");
#else
  fprintf(stdout, "UKNOWN");
#endif
#endif
  fprintf(stdout, "</Compiler>\n");

  
  fprintf(stdout, "<library version=\"%s\">VTK</librarary><library version=\"unknown\">ITK</library><library version=\"%s\">KWWidgets</library>\n", VTK_VERSION, KWWidgets_VERSION);
  int major, minor, patchLevel;
  Tcl_GetVersion(&major, &minor, &patchLevel, NULL);
  fprintf(stdout, "<library version=\"%d.%d.%d\">TCL</library>\n", major, minor, patchLevel);
  //fprintf(stdout, "<library version=\"%d.%d.%d\">TK</library>\n", major,
  //minor, patchLevel);
#ifdef USE_PYTHON
  fprintf(stdout, "<library version=\"%s\">Python</library>\n", PY_VERSION);
#endif
  fprintf(stdout, "<Repository>$HeadURL$</Repository>\n");
  fprintf(stdout, "<ProcessStep>\n");
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
  vtkSlicerApplication::GetInstance()->SplashMessage(msg);
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

  // set the SLICER_HOME variable if it doesn't already exist from the launcher
  vtksys_stl::string slicerHome;
  if ( !vtksys::SystemTools::GetEnv("SLICER_HOME", slicerHome) )
    {
    std::string homeEnv = "SLICER_HOME=";
    homeEnv += slicerBinDir + "/../";
    cout << "Set environment: " << homeEnv.c_str() << endl;
    vtkKWApplication::PutEnv(const_cast <char *> (homeEnv.c_str()));
    vtksys::SystemTools::GetEnv("SLICER_HOME", slicerHome);
    }

  std::string tclEnv = "TCL_LIBRARY=";
  tclEnv += slicerBinDir + "/../lib/Slicer3/tcl/lib/tcl8.4";
  vtkKWApplication::PutEnv(const_cast <char *> (tclEnv.c_str()));


  ptemp = vtksys::SystemTools::CollapseFullPath(argv[0]);
  ptemp = vtksys::SystemTools::GetFilenamePath(ptemp);
#if WIN32
  itkAutoLoadPath = ptemp + ";" + itkAutoLoadPath;
#else
  // shared libs are in bin dir in the build tree, but get
  // moved to the lib directory in the INSTALL, so put
  // both paths into the path
  if ( itkAutoLoadPath.size() == 0 )
    {
    itkAutoLoadPath = ptemp; // add the bin dir
    }
  else 
    {
    itkAutoLoadPath = ptemp + ":" + itkAutoLoadPath; // add the bin dir
    }
  itkAutoLoadPath = ptemp + "/../lib:" + itkAutoLoadPath; // add the lib dir
#endif
  itkAutoLoadPath = "ITK_AUTOLOAD_PATH=" + itkAutoLoadPath;
  int putSuccess = 
    vtkKWApplication::PutEnv(const_cast <char *> (itkAutoLoadPath.c_str()));
  if (!putSuccess)
    {
    cout << "Unable to set ITK_AUTOLOAD_PATH. " << itkAutoLoadPath << endl;
    }

  
  // Initialize Tcl
  // -- create the interp
  // -- set up initial global variables
  // -- later in this function tcl scripts are executed 
  //    and some additional global variables are defined
  Tcl_Interp *interp = NULL;
  interp = vtkKWApplication::InitializeTcl(argc, argv, &cout);
  if (!interp)
    {
    slicerCerr("Error: InitializeTcl failed" << endl );
    return 1;
    }

  //
  // turn off hardware antialiasing by default
  // - the QueryAtlas picker relies on this
  //   
  vtkOpenGLRenderWindow::SetGlobalMaximumNumberOfMultiSamples(0);

#ifdef USE_PYTHON

#if WIN32
#define PathSep ";"
#else
#define PathSep ":"
#endif
    // Initialize Python
    
    // Set up the search path
    std::string pythonEnv = "PYTHONPATH=";
    
    const char* existingPythonEnv = vtksys::SystemTools::GetEnv("PYTHONPATH");
    if ( existingPythonEnv )
      {
      pythonEnv += std::string ( existingPythonEnv ) + PathSep;
      }

    pythonEnv += slicerBinDir + "/../../Slicer3/Base/GUI/Python" + PathSep;
    pythonEnv += slicerBinDir + "/../lib/Slicer3/Plugins" + PathSep;
    pythonEnv += slicerBinDir + "/../Base/GUI/Python";
    vtkKWApplication::PutEnv(const_cast <char *> (pythonEnv.c_str()));
  
    Py_Initialize();
    PySys_SetArgv(argc, argv);
    PyObject* PythonModule = PyImport_AddModule("__main__");
    if (PythonModule == NULL)
      {
      std::cout << "Warning: Failed to initialize python" << std::endl;
      }
    PyObject* PythonDictionary = PyModule_GetDict(PythonModule);

    // Intercept _tkinter, and use ours...
    init_mytkinter(interp);
    init_slicer();
    PyObject* v;

    std::string TkinitString = "import Tkinter, sys;"
      "tk = Tkinter.Tk();"
      "sys.path.append ( \""
      + slicerBinDir + "/../../Slicer3/Base/GUI/Python"
      + "\" );\n"
      "sys.path.append ( \""
      + slicerBinDir + "/../Base/GUI/Python"
      + "\" );\n";
      "sys.path.append ( \""
      + slicerBinDir + "/../lib/Slicer3/Plugins"
      + "\" );\n";
  
    v = PyRun_String( TkinitString.c_str(),
                      Py_file_input,
                      PythonDictionary,
                      PythonDictionary );
    if (v == NULL)
      {
      PyErr_Print();
      }
    else
      {
      if (Py_FlushLine())
        {
        PyErr_Clear();
        }
      }
#endif

    // TODO: get rid of fixed size buffer
    char cmd[2048];

    // Make sure SLICER_HOME is available

    sprintf(cmd, "set ::env(SLICER_HOME) {%s};", slicerHome.c_str());
    Slicer3_Tcl_Eval(interp, cmd);
  
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
    // load the custom icons
    //
#ifdef _WIN32
    {    
      std::string cmd;
      int returnCode;

      cmd =  "wm iconbitmap . -default "+ slicerHome + "/lib/slicer3.ico";
      returnCode = Slicer3_Tcl_Eval( interp, cmd.c_str() );
      if ( returnCode )
        {
        slicerCerr("Load Custom Icons: " << cmd.c_str() << endl);
        }
    }
#endif


    //
    // Initialize our Tcl library (i.e. our classes wrapped in Tcl)
    //
    Slicerbasegui_Init(interp);
    Slicerbaselogic_Init(interp);
    Mrml_Init(interp);
    Vtkitk_Init(interp);
    Freesurfer_Init(interp);
    Igt_Init(interp);
    Vtkteem_Init(interp);

#if !defined(COMMANDLINE_DEBUG) && defined(BUILD_MODULES)
    Commandlinemodule_Init(interp);
#endif
#if !defined(TCLMODULES_DEBUG) && defined(BUILD_MODULES)
    Scriptedmodule_Init(interp);
#endif

  // first call to GetInstance will create the Application
  // 
  // make a substitute tcl proc for the 'exit' built in that
  // goes through the vtkKWApplication methodology for exiting 
  // cleanly
  //
  vtkSlicerApplication *slicerApp = vtkSlicerApplication::GetInstance ( );

    {
    std::string cmd, slicerAppName;

    slicerAppName = slicerApp->GetTclName();

    slicerApp->Script ("namespace eval slicer3 set Application %s", slicerAppName.c_str());

    cmd = "rename exit tcl_exit; ";

    cmd += 
     "proc exit {args} { \
        if { $args != {} && [string is integer $args] == \"1\" } { \
          " + slicerAppName + " SetExitStatus $args \
        } else { \
          " + slicerAppName + " SetExitStatus 0 \
        } ;\
        after idle {" + slicerAppName + " Exit}; \
      }";
      Slicer3_Tcl_Eval( interp, cmd.c_str() );
    }

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
      int returnCode = Slicer3_Tcl_Eval( interp, cmd.c_str() );
      Slicer3_Tcl_Eval( interp, "update" );
      slicerApp->Delete();
      return ( returnCode );
      }
    
    //
    // use the startup code passed on command line if it exists
    //
    if ( Eval != "" )
      {    
      std::string cmd = "set ::SLICER(eval) \"" + Eval + "\" ; ";
      cmd += "regsub -all {\\.,} $::SLICER(eval) \";\" ::SLICER(eval); ";
      cmd += "regsub -all {,\\.} $::SLICER(eval) \";\" ::SLICER(eval); ";
      cmd += "eval $::SLICER(eval);";
      int returnCode = Slicer3_Tcl_Eval( interp, cmd.c_str() );
      Slicer3_Tcl_Eval( interp, "update" );
      slicerApp->Delete();
      return ( returnCode );
      }

    // Create SlicerGUI application, style, and main window 
    //  - note: use the singleton application 
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
    if ( NoSplash )
      {
      slicerApp->SetUseSplashScreen(0);
      slicerApp->SupportSplashScreenOff();
      slicerApp->SplashScreenVisibilityOff();
      }
    else
      {
      slicerApp->SetUseSplashScreen(1);
      slicerApp->SupportSplashScreenOn();
      slicerApp->SplashScreenVisibilityOn();
      }
    slicerApp->GetSplashScreen()->SetProgressMessageVerticalOffset(-25);
    slicerApp->SplashMessage("Initializing Window...");

    cout << "Starting Slicer: " << slicerHome.c_str() << endl;

    if ( Stereo )
      {
        slicerApp->SetStereoEnabled(1);
      } 
    else 
      {
        slicerApp->SetStereoEnabled(0);
      }
    

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

    slicerApp->SplashMessage("Creating Application GUI...");

    // CREATE APPLICATION GUI, including the main window 
    vtkSlicerApplicationGUI *appGUI = vtkSlicerApplicationGUI::New ( );
    appGUI->SetApplication ( slicerApp );
    appGUI->SetAndObserveApplicationLogic ( appLogic );
    appGUI->SetAndObserveMRMLScene ( scene );

    // set fonts from registry before building GUI...
    /*
      if ( appGUI->GetMainSlicerWindow()->GetApplicationSettingsInterface() )
      {
      slicerApp->GetSlicerTheme()->InstallFonts();
      }
    */
    slicerApp->SaveUserInterfaceGeometryOn();

    appGUI->BuildGUI ( );
    appGUI->AddGUIObservers ( );
    slicerApp->SetApplicationGUI ( appGUI );

    // Set cache paths for modules


    std::string cachePath;
    std::string defaultCachePath;
    std::string userCachePath;

    // define a default cache for module information
    defaultCachePath = slicerBinDir + "/../lib/Slicer3/PluginsCache";
    if (hasIntDir)
      {
        defaultCachePath += "/" + intDir;
      }
      
    // get the cache path that the user has configured
    if (slicerApp->GetModuleCachePath())
      {
        userCachePath = slicerApp->GetModuleCachePath();
      }

    // if user cache path is set and we can write to it, use it.
    // if user cache path is not set or we cannot write to it, try
    // the default cache path.
    // if we cannot write to the default cache path, then warn and
    // don't use a cache.
    vtksys::Directory directory;
    if (userCachePath != "")
      {
        if (!vtksys::SystemTools::FileExists(userCachePath.c_str()))
          {
            vtksys::SystemTools::MakeDirectory(userCachePath.c_str());
          }
        if (vtksys::SystemTools::FileExists(userCachePath.c_str())
            && vtksys::SystemTools::FileIsDirectory(userCachePath.c_str()))
          {
            std::ofstream tst((userCachePath + "/tstCache.txt").c_str());
            if (tst)
              {
                cachePath = userCachePath;
              }
          }
      }
    if (cachePath == "")
      {
        if (!vtksys::SystemTools::FileExists(defaultCachePath.c_str()))
          {
            vtksys::SystemTools::MakeDirectory(defaultCachePath.c_str());
          }
        if (vtksys::SystemTools::FileExists(defaultCachePath.c_str())
            && vtksys::SystemTools::FileIsDirectory(defaultCachePath.c_str()))
          {
            std::ofstream tst((defaultCachePath + "/tstCache.txt").c_str());
            if (tst)
              {
                cachePath = defaultCachePath;
              }
          }
      }
    if (ClearModuleCache)
      {
        std::string cacheFile = cachePath + "/ModuleCache.csv";
        vtksys::SystemTools::RemoveFile(cacheFile.c_str());
      }
    if (NoModuleCache)
      {
        cachePath = "";
      }
    if (cachePath == "")
      {
        if (NoModuleCache)
          {
            std::cout << "Module cache disabled by command line argument."
                      << std::endl;
          }
        else
          {
            std::cout << "Module cache disabled. Slicer application directory may not be writable, a user level cache directory may not be set, or the user level cache directory may not be writable." << std::endl;
          }
      }

    
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

    std::string slicerModulePath = slicerBinDir;

    if (hasIntDir)
    {
      slicerModulePath += "/" + intDir;
    }

    LoadableModuleFactory loadableModuleFactory;
    loadableModuleFactory.SetName("Slicer");
    loadableModuleFactory.SetSearchPath( slicerModulePath );
    loadableModuleFactory.SetWarningMessageCallback( WarningMessage );
    loadableModuleFactory.SetErrorMessageCallback( ErrorMessage );
    loadableModuleFactory.SetInformationMessageCallback( InformationMessage );
    loadableModuleFactory.SetModuleDiscoveryMessageCallback( SplashMessage );
    loadableModuleFactory.Scan();
    
    std::vector<std::string> loadableModuleNames = 
      loadableModuleFactory.GetModuleNames();
    std::vector<std::string>::const_iterator lmit =
      loadableModuleNames.begin();
    
    while (lmit != loadableModuleNames.end())
      {
        LoadableModuleDescription desc = loadableModuleFactory.GetModuleDescription(*lmit);

        slicerApp->SplashMessage(desc.GetMessage().c_str());

        vtkSlicerModuleLogic* logic = desc.GetLogicPtr();
        logic->SetAndObserveMRMLScene( scene );
        vtkSlicerModuleGUI* gui = desc.GetGUIPtr();

        gui->SetApplication( slicerApp );
        gui->SetApplicationGUI( appGUI );
        gui->SetAndObserveApplicationLogic( appLogic );
        gui->SetAndObserveMRMLScene( scene );
        gui->SetModuleLogic ( logic );
        gui->SetGUIName( desc.GetGUIName().c_str() );
        gui->GetUIPanel()->SetName( gui->GetGUIName ( ) );
        gui->GetUIPanel()->SetUserInterfaceManager( appGUI->GetMainSlicerWindow()->GetMainUserInterfaceManager ( ) );
        gui->GetUIPanel()->Create( );
        slicerApp->AddModuleGUI( gui );
        gui->BuildGUI ( );
        gui->AddGUIObservers ( );
        gui->Init();

        // Initialize TCL

        TclInit Tcl_Init = desc.GetTclInitFunction();

        if (Tcl_Init != 0)
        {
          (*Tcl_Init)(interp);
        }

        lmit++;
      }

    // --- Transforms module
    slicerApp->SplashMessage("Initializing Transforms Module...");

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

    //--- Data module
    slicerApp->SplashMessage( "Initializing Data Module...");

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
    
    // --- Slices module
    slicerApp->SplashMessage("Initializing Slices Module...");
    // - set up each of the slice logics (these initialize their
    //   helper classes and nodes the first time the process MRML and
    //   Logic events)
    vtkIntArray *events = vtkIntArray::New();
    events->InsertNextValue(vtkMRMLScene::NewSceneEvent);
    events->InsertNextValue(vtkMRMLScene::SceneCloseEvent);
    events->InsertNextValue(vtkMRMLScene::NodeAddedEvent);
    events->InsertNextValue(vtkMRMLScene::NodeRemovedEvent);

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
    if (appLogic->GetSlices())
      {
      appLogic->GetSlices()->AddItem(sliceLogic0);
      appLogic->GetSlices()->AddItem(sliceLogic1);
      appLogic->GetSlices()->AddItem(sliceLogic2);
      }

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

    appGUI->InitializeSlicesControlGUI();
    appGUI->InitializeViewControlGUI();
//    appGUI->InitializeNavigationWidget();

#if !defined(DAEMON_DEBUG) && defined(BUILD_MODULES)
    //
    // --- SlicerDaemon Module
    // need to source the slicerd.tcl script here
    // - only start if selected on command line or enabled in registry, since 
    //   windows firewall will complain. 
    //

    if ( Daemon || slicerApp->GetEnableDaemon() )
      {
      slicerApp->SplashMessage("Initializing Slicer Daemon...");
      std::string cmd;
      cmd =  "source \"" + slicerBinDir + "/../"
        SLICER_INSTALL_LIBRARIES_DIR "/slicerd.tcl\"; slicerd_start; ";
      Slicer3_Tcl_Eval(interp, cmd.c_str());
      }
#endif

#if !defined(CLIMODULES_DEBUG) && defined(BUILD_MODULES)
    std::vector<std::string> moduleNames;
    std::vector<std::string>::const_iterator mit;
    if ( slicerApp->GetLoadCommandLineModules() && !NoModules )
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
    
      // get the path that the user has configured
      if (slicerApp->GetModulePath())
        {
        userPackagePath = slicerApp->GetModulePath();
        }

      // add the default plugin directory (based on the slicer
      // installation or build tree) to the user path
      packagePath = userPackagePath + delim + defaultPackageDir;

      // Search for modules
      ModuleFactory moduleFactory;
      moduleFactory.SetName("Slicer");
      moduleFactory.SetSearchPath( packagePath );
      moduleFactory.SetCachePath( cachePath );
      moduleFactory.SetWarningMessageCallback( WarningMessage );
      moduleFactory.SetErrorMessageCallback( ErrorMessage );
      moduleFactory.SetInformationMessageCallback( InformationMessage );
      moduleFactory.SetModuleDiscoveryMessageCallback( SplashMessage );
      moduleFactory.Scan();

      // Register the node type for the command line modules
      vtkMRMLCommandLineModuleNode* clmNode = vtkMRMLCommandLineModuleNode::New();
      scene->RegisterNodeClass(clmNode);
      clmNode->Delete();
      
      // add the modules to the available modules
      moduleNames = moduleFactory.GetModuleNames();
      mit = moduleNames.begin();
      while (mit != moduleNames.end())
        {
        // slicerCerr(moduleFactory.GetModuleDescription(*mit) << endl);

        vtkCommandLineModuleGUI *commandLineModuleGUI
          = vtkCommandLineModuleGUI::New();
        vtkCommandLineModuleLogic *commandLineModuleLogic
          = vtkCommandLineModuleLogic::New ( );
      
        // Set the ModuleDescripton on the gui
        commandLineModuleGUI
          ->SetModuleDescription( moduleFactory.GetModuleDescription(*mit) );

        // Register the module description in the master list
        vtkMRMLCommandLineModuleNode::RegisterModuleDescription( moduleFactory.GetModuleDescription(*mit) );
        
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
      }
#endif // CLIMODULES_DEBUG

    //
    // get the Tcl name so the vtk class will be registered in the interpreter
    // as a byproduct
    // - set some handy variables so it will be easy to access these classes
    ///  from   the tkcon
    // - all the variables are put in the slicer3 tcl namespace for easy access
    // - note, slicerApp is loaded as $::slicer3::Application above, so it
    //   can be used from scripts that run without the GUI
    //
    const char *name;
    name = appGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set ApplicationGUI %s", name);

    lmit = loadableModuleNames.begin();
    
    while (lmit != loadableModuleNames.end())
      {
        LoadableModuleDescription desc = loadableModuleFactory.GetModuleDescription(*lmit);

        vtkSlicerModuleGUI* gui = desc.GetGUIPtr();

        name = gui->GetTclName();
        std::string format("namespace eval slicer3 set ");
        format += desc.GetShortName();
        format += "GUI %s";

        slicerApp->Script (format.c_str(), name);        

        lmit++;
      }

#ifndef SLICES_DEBUG
    name = slicesGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set SlicesGUI %s", name);
#endif

    name = transformsGUI->GetTclName();
    slicerApp->Script ("namespace eval slicer3 set TransformsGUI %s", name);

    if ( appGUI->GetViewerWidget() )
      {
      name = appGUI->GetViewerWidget()->GetTclName();
      slicerApp->Script ("namespace eval slicer3 set ViewerWidget %s", name);
      }

    slicerApp->Script ("namespace eval slicer3 set ApplicationLogic [$::slicer3::ApplicationGUI GetApplicationLogic]");
    slicerApp->Script ("namespace eval slicer3 set MRMLScene [$::slicer3::ApplicationLogic GetMRMLScene]");

#if !defined(TCLMODULES_DEBUG) && defined(BUILD_MODULES)
    //
    // process any ScriptedModules
    // - scan for pkgIndex.tcl files 
    // - load the corresponding packages
    // - create scripted logic and scripted gui instances for the module
    // - have the GUI construct itself
    //

    slicerApp->SplashMessage("Initializing Scripted Modules...");
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
    slicerApp->SplashMessage("Configuring Slices and Modules...");
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
    slicerApp->SplashMessage("Finalizing Startup...");
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
    // ./Slicer3 --exec exit' 
    //
    if ( Exec != "" )
      {    
      std::string cmd = "set ::SLICER(exec) \"" + Exec + "\" ; ";
      cmd += "regsub -all {\\.,} $::SLICER(exec) \";\" ::SLICER(exec); ";
      cmd += "regsub -all {,\\.} $::SLICER(exec) \";\" ::SLICER(exec); ";
      cmd += "eval $::SLICER(exec);";
      res = Slicer3_Tcl_Eval( interp, cmd.c_str() );
      }

    //
    // if there was no script file, the Args should hold a mrml file
    // 
    //
    if ( File == ""  && Args.size() != 0)
      {
      std::vector<std::string>::const_iterator argit = Args.begin();
      while (argit != Args.end())
        {
        std::string fileName;
        // strip backslashes
        char buffer[2]; buffer[1] = '\0';
        for (unsigned int i = 0; i < (*argit).size(); i++)
          {
          buffer[0] = (*argit)[i];
          if ( buffer[0] != '\\' )
            {
            fileName.append(std::string(buffer));
            }
          }
        // is it a MRML or XML file?
        if (fileName.find(".mrml",0) != std::string::npos
            || fileName.find(".MRML",0) != std::string::npos)
          {
          appLogic->GetMRMLScene()->SetURL(fileName.c_str());
          // and then load it
          slicerApp->SplashMessage("Set scene url, connecting...");
          int errorCode = appLogic->GetMRMLScene()->Connect();
          if (errorCode != 1)
            {
            slicerCerr("ERROR loading MRML file " << fileName << ", error code = " << errorCode << endl);
            }
          else
            {
            slicerApp->SplashMessage("Connected to scene.");
            }
          }                    
        else if (fileName.find(".xml",0) != std::string::npos
                 || fileName.find(".XML",0) != std::string::npos)
          {
          // if it's an xml file, load it
          std::string cmd = "after idle {ImportSlicer2Scene \"" + *argit + "\"}";
          slicerApp->SplashMessage("Importing Slicer2 scene...");
          res = Slicer3_Tcl_Eval( interp, cmd.c_str() );
          slicerApp->SplashMessage("Imported scene.");
          }
        else if (fileName.find(".xcat",0) != std::string::npos
                 || fileName.find(".XCAT",0) != std::string::npos)
          {
          // if it's an xcede file, load it
          slicerApp->SplashMessage("Importing Xcede catalog ...");
          std::string cmd = "after idle {XcedeCatalogImport \"" + *argit + "\"}";
          res = Slicer3_Tcl_Eval( interp, cmd.c_str() );
          slicerApp->SplashMessage("Imported catalog.");
          }
        else if (fileName.find(".tcl",0) != std::string::npos)
          {
          // if it's a tcl file source it after the app starts
          std::string cmd = "after idle {source " + *argit + "}";
          res = Slicer3_Tcl_Eval( interp, cmd.c_str() );
          }
        else
          {
          // if we're not sure, assume it is data to load...
          std::string cmd = "::Loader::ShowDialog \"" + *argit + "\"";
          res = Slicer3_Tcl_Eval( interp, cmd.c_str() );
          }
        ++argit;
        }
    }

    //
    // --load_dicom <directory | filename>
    //
    if ( LoadDicomDir != "" )
      {    
      // load either a directory or an archetype
      std::string cmd = "::Loader::LoadArchetype " + LoadDicomDir;
      res = Slicer3_Tcl_Eval( interp, cmd.c_str() );
      }


    //--- set home module based on registry settings
    const char *homeModule = slicerApp->GetHomeModule();
    if ( homeModule && *homeModule )
      {
        appGUI->SelectModule ( homeModule );
      }
    else
      {
      appGUI->SelectModule("Data");
      }
#ifdef USE_PYTHON
    vtkSlicerApplication::GetInstance()->InitializePython ( PythonModule, PythonDictionary );
#endif    

    //
    // Run!  - this will return when the user exits
    //
    res = slicerApp->StartApplication();

    appGUI->GetMainSlicerWindow()->PrepareForDelete();
    appGUI->GetMainSlicerWindow()->Close();

    // ------------------------------
    // REMOVE OBSERVERS and references to MRML and Logic

  lmit = loadableModuleNames.begin();
  while (lmit != loadableModuleNames.end()) {
    LoadableModuleDescription desc = loadableModuleFactory.GetModuleDescription(*lmit);

    desc.GetGUIPtr()->TearDownGUI();
    desc.GetGUIPtr()->RemoveGUIObservers();

    lmit++;
  }

    transformsGUI->TearDownGUI ( );

    dataGUI->TearDownGUI ( );
    
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

#if !defined(CLIMODULES_DEBUG) && defined(BUILD_MODULES)
    // remove the observers from the factory discovered modules
    // (as we remove the observers, cache the GUIs in a vector so we
    // can delete them later).
    mit = moduleNames.begin();
    std::vector<vtkSlicerModuleGUI*> moduleGUIs;
    while ( mit != moduleNames.end() )
      {
      vtkSlicerModuleGUI *module;
      module = slicerApp->GetModuleGUIByName( (*mit).c_str() );

/*      module->RemoveGUIObservers();*/
      module->TearDownGUI ( );

      moduleGUIs.push_back(module);
      
      ++mit;
      }
#endif    

#if !defined(TCLMODULES_DEBUG) && defined(BUILD_MODULES)
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

  lmit = loadableModuleNames.begin();
  while (lmit != loadableModuleNames.end()) {
    LoadableModuleDescription desc = loadableModuleFactory.GetModuleDescription(*lmit);

    desc.GetGUIPtr()->Delete();

    lmit++;
  }

    transformsGUI->Delete ();

    dataGUI->Delete ();

#if !defined(TCLMODULES_DEBUG) && defined(BUILD_MODULES)
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
//cout << "vtkSlicerApplicationGUI deleting app GUI\n";
//   appGUI->Delete ();

#if !defined(CLIMODULES_DEBUG) && defined(BUILD_MODULES)
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

    // Release reference to applicaiton GUI
    // and delete it.
   slicerApp->SetApplicationGUI ( NULL );
   appGUI->DeleteComponentGUIs();
   appGUI->Delete ();
    
    //--- delete logic next, removing Refs to MRML
    appLogic->ClearCollections ( );

  lmit = loadableModuleNames.begin();
  while (lmit != loadableModuleNames.end()) {
    LoadableModuleDescription desc = loadableModuleFactory.GetModuleDescription(*lmit);

    vtkSlicerModuleLogic* logic = desc.GetLogicPtr();

    logic->SetAndObserveMRMLScene( NULL );
    logic->Delete();

    lmit++;
  }

    sliceLogic2->SetAndObserveMRMLScene ( NULL );
    sliceLogic2->Delete ();
    sliceLogic1->SetAndObserveMRMLScene ( NULL );
    sliceLogic1->Delete ();
    sliceLogic0->SetAndObserveMRMLScene ( NULL );
    sliceLogic0->Delete ();

    appLogic->SetAndObserveMRMLScene ( NULL );
    appLogic->TerminateProcessingThread();
    appLogic->Delete ();

#if !defined(CLIMODULES_DEBUG) && defined(BUILD_MODULES)
    // delete the factory discovered module Logics
    std::vector<vtkSlicerModuleLogic*>::iterator lit;
    for (lit = moduleLogics.begin(); lit != moduleLogics.end(); ++lit)
      {
      (*lit)->SetAndObserveMRMLScene ( NULL );
      (*lit)->Delete();
      }
    moduleLogics.clear();
#endif

#if !defined(TCLMODULES_DEBUG) && defined(BUILD_MODULES)
    // delete the scripted logics
    tclCommand = "";
    tclCommand += "foreach package $::SLICER_PACKAGES(list) { ";
    tclCommand += "  $::SLICER_PACKAGES($package,logic) SetAndObserveMRMLScene {};";
    tclCommand += "  $::SLICER_PACKAGES($package,logic) Delete;";
    tclCommand += "}";
    Slicer3_Tcl_Eval( interp, tclCommand.c_str() );
#endif
    

    //--- scene next;
    scene->Clear(1);
    scene->Delete ();

    //--- application last
    slicerApp->Delete ();

#ifdef USE_PYTHON
    // Shutdown python interpreter
    Py_Finalize();
#endif


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










