#include <sstream>
#include <vtksys/stl/string>
#include <vtksys/SystemTools.hxx>

#include "vtkObjectFactory.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerGUICollection.h"
#include "vtkSlicerModuleGUI.h"
#include "vtkKWNotebook.h"
#include "vtkKWFrame.h"
#include "vtkKWUserInterfacePanel.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWWindowBase.h"
#include "vtkKWTkUtilities.h"
#include "vtkKWLogDialog.h"
#include "vtkKWLogWidget.h"
#include "vtkKWInternationalization.h"
#include "vtkKWTclInteractor.h"
#include "vtkKWSplashScreen.h"
#include "vtkKWSplitFrame.h"
#include "vtkKWDirectoryPresetSelector.h"

#include "vtkSlicerBaseGUIWin32Header.h"
#include "vtkKWRegistryHelper.h"
#include "vtkSlicerGUILayout.h"
#include "vtkMRMLLayoutNode.h"
#include "vtkSlicerGUICollection.h"
#include "vtkSlicerTheme.h"
#include "vtkSlicerFont.h"

#include "vtkOutputWindow.h"
#include "itkOutputWindow.h"

#ifdef WIN32
#include "vtkKWWin32RegistryHelper.h"
#endif

#ifndef _WIN32
#include <unistd.h>
#endif

#include "itksys/SystemTools.hxx"
#include <queue>
#include <algorithm>
#include "vtkKWMessageDialog.h"

#include "vtkSlicerConfigure.h" /* Slicer3_USE_* */

#ifdef Slicer3_USE_PYTHON
#include "slicerPython.h"
#endif

const char *vtkSlicerApplication::ModulePathsRegKey = "ModulePaths";
const char *vtkSlicerApplication::PotentialModulePathsRegKey = "PotentialModulePaths";
const char *vtkSlicerApplication::ColorFilePathsRegKey = "ColorFilePaths";
const char *vtkSlicerApplication::PotentialColorFilePathsRegKey = "PotentialColorFilePaths";
const char *vtkSlicerApplication::ModuleCachePathRegKey = "ModuleCachePath";
const char *vtkSlicerApplication::TemporaryDirectoryRegKey = "TemporaryDirectory";
const char *vtkSlicerApplication::WebBrowserRegKey = "WebBrowser";
const char *vtkSlicerApplication::UnzipRegKey = "Unzip";
const char *vtkSlicerApplication::ZipRegKey = "Zip";
const char *vtkSlicerApplication::RmRegKey = "Rm";
const char *vtkSlicerApplication::ConfirmDeleteRegKey = "ConfirmDelete";
const char *vtkSlicerApplication::HomeModuleRegKey = "HomeModule";
const char *vtkSlicerApplication::LoadCommandLineModulesRegKey = "LoadCommandLineModules";
const char *vtkSlicerApplication::LoadModulesRegKey = "LoadModules";
const char *vtkSlicerApplication::IgnoreModulesRegKey = "IgnoreModules";
const char *vtkSlicerApplication::EnableDaemonRegKey = "EnableDaemon";
const char *vtkSlicerApplication::ApplicationFontSizeRegKey = "ApplicationFontSize";
const char *vtkSlicerApplication::ApplicationFontFamilyRegKey = "ApplicationFontFamily";
const char *vtkSlicerApplication::ApplicationWindowWidthRegKey = "ApplicationWindowWidth";
const char *vtkSlicerApplication::ApplicationWindowHeightRegKey = "ApplicationWindowHeight";
const char *vtkSlicerApplication::ApplicationSlicesFrameHeightRegKey = "ApplicationSlicesFrameHeight";
const char *vtkSlicerApplication::ApplicationLayoutTypeRegKey = "ApplicationLayoutType";
const char *vtkSlicerApplication::EnableAsynchronousIORegKey = "EnableAsynchronousIO";
const char *vtkSlicerApplication::UseWelcomeModuleAtStartupRegKey = "UseWelcomeModuleAtStartup";
const char *vtkSlicerApplication::EnableForceRedownloadRegKey = "EnableForceRedownload";
//const char *vtkSlicerApplication::EnableRemoteCacheOverwritingRegKey = "EnableRemoteCacheOverwriting";
const char *vtkSlicerApplication::RemoteCacheDirectoryRegKey = "RemoteCacheDirectory";
const char *vtkSlicerApplication::RemoteCacheLimitRegKey = "RemoteCacheLimit";
const char *vtkSlicerApplication::RemoteCacheFreeBufferSizeRegKey = "RemoteCacheFreeBufferSize";

vtkSlicerApplication *vtkSlicerApplication::Instance = NULL;

//---------------------------------------------------------------------------
vtkCxxRevisionMacro(vtkSlicerApplication, "$Revision: 1.0 $");

typedef std::pair<std::string, std::string> AddRecordType;
class DisplayMessageQueue : public std::queue<AddRecordType> {};

//----------------------------------------------------------------------------
// Slicer needs its own version of itk::OutputWindow to ensure that
// only the application thread controlling the gui tries to display a
// message. 
namespace itk {

class SlicerOutputWindow : public OutputWindow
{
public:
  /** Standard class typedefs. */
  typedef SlicerOutputWindow        Self;
  typedef OutputWindow  Superclass;
  typedef SmartPointer<Self>  Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(SlicerOutputWindow,OutputWindow);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  void DisplayDebugText(const char* t)
    { 
      vtkSlicerApplication::GetInstance()->DebugMessage(t); 
    }
  void DisplayWarningText(const char* t)
    { 
      vtkSlicerApplication::GetInstance()->WarningMessage(t); 
    }
  void DisplayErrorText(const char* t)
    { 
      vtkSlicerApplication::GetInstance()->ErrorMessage(t); 
    }
  void DisplayText(const char* t)
    { 
      vtkSlicerApplication::GetInstance()->InformationMessage(t); 
    }
  void DisplayGenericWarningText(const char* t)
    { 
      this->DisplayWarningText(t); 
    }
  
protected:
  SlicerOutputWindow()
    { 
    }

private:
  SlicerOutputWindow(const SlicerOutputWindow&);
  void operator=(const SlicerOutputWindow&);
};

}; // end namespace itk

//----------------------------------------------------------------------------
// Slicer needs its own version of vtkKWOutputWindow to ensure that
// only the application thread controlling the gui tries to display a
// message.
//
// NOTE: it looks as though as long as Slicer constructs the dialog
// window early enough, then we do not need our own version of the
// output window for VTK.  The virtual overrides of
// InformationMessage(), WarningMessage(), etc. in
// vtkSlicerApplication are enough to ensure that we display messages
// in a thread safe manner.
//
// 
// class vtkSlicerOutputWindow : public vtkOutputWindow
// {
// public:
//   vtkTypeMacro(vtkSlicerOutputWindow,vtkOutputWindow);
//   static vtkSlicerOutputWindow* New();
//
//   void DisplayDebugText(const char* t)
//     { 
//       this->Application->DebugMessage(t); 
//     }
//   void DisplayWarningText(const char* t)
//     { 
//       this->Application->WarningMessage(t); 
//     }
//   void DisplayErrorText(const char* t)
//     { 
//       this->Application->ErrorMessage(t); 
//     }
//   void DisplayText(const char* t)
//     { 
//       this->Application->InformationMessage(t); 
//     }
//   void DisplayGenericWarningText(const char* t)
//     { 
//       this->DisplayWarningText(t); 
//     }
//  
//   void SetApplication(vtkSlicerApplication *app)
//     { 
//       this->Application = app; 
//     }
//
// protected:
//   vtkSlicerOutputWindow()
//     { 
//       this->Application = NULL; 
//     }
//   vtkSlicerApplication *Application;
//
// private:
//   vtkSlicerOutputWindow(const vtkSlicerOutputWindow&);
//   void operator=(const vtkSlicerOutputWindow&);
// };
//
//
// vtkStandardNewMacro(vtkSlicerOutputWindow);

//---------------------------------------------------------------------------
vtkSlicerApplication::vtkSlicerApplication ( ) {

    this->ApplicationGUI = NULL;

    strcpy(this->ConfirmDelete, "");
    
    strcpy(this->ModulePaths, "");
    strcpy(this->PotentialModulePaths, "");
    strcpy(this->ColorFilePaths, "");
    strcpy(this->PotentialColorFilePaths, "");
    strcpy(this->ModuleCachePath, "");
    strcpy ( this->HomeModule, "");
    this->LoadCommandLineModules = 1;
    this->LoadModules = 1;
    this->IgnoreModules = vtkStringArray::New();
    this->LoadableModules = vtkStringArray::New();
    strcpy(this->IgnoreModuleNames, "");
    this->NameSeparator = ";";
    this->EnableDaemon = 0;
   
    this->DefaultGeometry = vtkSlicerGUILayout::New ( );
    // defaults
    strcpy (this->ApplicationFontSize, "small" );
    strcpy ( this->ApplicationFontFamily, "Arial" );
    this->ApplicationWindowWidth = 0;
    this->ApplicationWindowHeight = 0;
    this->ApplicationLayoutType = vtkMRMLLayoutNode::SlicerLayoutConventionalView;
    this->ApplicationSlicesFrameHeight = this->DefaultGeometry->GetDefaultSliceGUIFrameHeight();

    // Remote data handling settings
    strcpy ( this->RemoteCacheDirectory, "");
    this->EnableAsynchronousIO = 0;
    this->EnableForceRedownload = 0;
//    this->EnableRemoteCacheOverwriting = 1;
    this->RemoteCacheLimit = 200;
    this->RemoteCacheFreeBufferSize = 10;
    
    strcpy( this->ExtensionsDownloadDir, "" );
    
    this->UseWelcomeModuleAtStartup = 1;

    // configure the application before creating
    this->SetName ( "3D Slicer Version 3.4" );

#ifdef _WIN32
    vtkKWWin32RegistryHelper *regHelper = 
        vtkKWWin32RegistryHelper::SafeDownCast( this->GetRegistryHelper() );
    regHelper->SetOrganization("NA-MIC");
#endif

    this->RestoreApplicationSettingsFromRegistry ( );
    
    // FOR NOW!
    // this->ApplicationLayoutType = vtkMRMLLayoutNode::SlicerLayoutConventionalView;

    this->SetHelpDialogStartingPage ( "http://www.slicer.org" );

    this->ModuleGUICollection = vtkSlicerGUICollection::New ( );
    vtkKWFrameWithLabel::SetDefaultLabelFontWeightToNormal( );
    this->SlicerTheme = vtkSlicerTheme::New ( );
    this->ApplicationGUI = NULL;

    this->DisplayMessageQueueActive = false;
    this->DisplayMessageQueueActiveLock = itk::MutexLock::New();
    this->DisplayMessageQueueLock = itk::MutexLock::New();
    
    this->InternalDisplayMessageQueue = new DisplayMessageQueue;

    this->DisplayMessageQueueActiveLock->Lock();
    this->DisplayMessageQueueActive = true;
    this->DisplayMessageQueueActiveLock->Unlock();

    vtkKWTkUtilities::CreateTimerHandler(
      this, 100, this, "ProcessDisplayMessage");

    // Override the type of output windows used for VTK and ITK.  Note
    // that in the VTK case, we are currently bypassing the output
    // window mechanism provided by KWWidgets.  In KWWidgets, there
    // are calls to InstallOutputWindow()/RestoreOutputWindow() to
    // manage a KWWidget specific output window.  In the Slicer case,
    // we need a different type of output window to ensure that only
    // the main thread updates the gui.
    //
    //
    // NOTE: it looks as though as long as Slicer constructs the dialog
    // window early enough, then we do not need our own version of the
    // output window for VTK.  The virtual overrides of
    // InformationMessage(), WarningMessage(), etc. in
    // vtkSlicerApplication are enough to ensure that we display messages
    // in a thread safe manner.
    
    //vtkSlicerOutputWindow *vtkoutput = vtkSlicerOutputWindow::New();
    //vtkoutput->SetApplication(this);
    //vtkOutputWindow::SetInstance( vtkoutput );
    
    itk::SlicerOutputWindow::SetInstance( itk::SlicerOutputWindow::New() );

    // Use the splash screen by default - needs to be overridden before
    // the splash is first used
    this->UseSplashScreen = 1;

    // Disable stereo render capability by default
    this->SetStereoEnabled(0);

#ifdef Slicer3_USE_PYTHON
    this->PythonModule = NULL; 
    this->PythonDictionary = NULL; 
#endif
}

//---------------------------------------------------------------------------
vtkSlicerApplication::~vtkSlicerApplication ( ) {

    if ( this->DefaultGeometry )
      {
      this->DefaultGeometry->Delete ( );
      this->DefaultGeometry = NULL;
      }
    if ( this->SlicerTheme )
      {
      this->SlicerTheme->Delete ( );
      this->SlicerTheme = NULL;
      }
    if ( this->ModuleGUICollection )
      {
      this->ModuleGUICollection->RemoveAllItems ( );
      this->ModuleGUICollection->Delete ( );
      this->ModuleGUICollection = NULL;
      }
    this->ApplicationGUI = NULL;
    
    delete this->InternalDisplayMessageQueue;
    this->InternalDisplayMessageQueue = 0;

    this->DisplayMessageQueueActiveLock->Lock();
    this->DisplayMessageQueueActive = false;
    this->DisplayMessageQueueActiveLock->Unlock();

    this->IgnoreModules->Delete();
    this->LoadableModules->Delete();
}

//----------------------------------------------------------------------------
// Up the reference count so it behaves like New
vtkSlicerApplication* vtkSlicerApplication::New()
{
  vtkSlicerApplication* ret = vtkSlicerApplication::GetInstance();
  ret->Register(NULL);
  return ret;
}




//----------------------------------------------------------------------------
// Return the single instance of the vtkSlicerApplication
vtkSlicerApplication* vtkSlicerApplication::GetInstance()
{
  if(!vtkSlicerApplication::Instance)
    {
    // Try the factory first
    vtkSlicerApplication::Instance = (vtkSlicerApplication*)
      vtkObjectFactory::CreateInstance("vtkSlicerApplication");
    // if the factory did not provide one, then create it here
    if(!vtkSlicerApplication::Instance)
      {
      vtkSlicerApplication::Instance = new vtkSlicerApplication;
      }
    }
  // return the instance
  return vtkSlicerApplication::Instance;
}

//---------------------------------------------------------------------------
const char *vtkSlicerApplication::Evaluate(const char *expression) {
    return (this->Script(expression));
}




//---------------------------------------------------------------------------
int vtkSlicerApplication::FullFileSystemCheck ( ) {
  
  //---
  //--- temporary directory first
  //---
  std::string temporaryDirectory;
  std::string testFile;
  if (this->GetTemporaryDirectory() )
    {
    temporaryDirectory = this->GetTemporaryDirectory();
    }
  this->SetTemporaryDirectory(temporaryDirectory.c_str());
  if (temporaryDirectory != "")
    {
    //--- make sure directory is present
    if (!vtksys::SystemTools::FileExists(temporaryDirectory.c_str()))
      {
      vtksys::SystemTools::MakeDirectory(temporaryDirectory.c_str());
      }
    //--- if there, make sure it's actually a dir
    if (vtksys::SystemTools::FileExists(temporaryDirectory.c_str())
        && vtksys::SystemTools::FileIsDirectory(temporaryDirectory.c_str()))
      {
      testFile = temporaryDirectory;
      testFile += "/SlicerDiskSpaceCheck.txt";
      std::ofstream tst ( testFile.c_str() );
      if (tst.good() == false || tst.fail() == true || tst.bad() == true )
        {
        std::string msg = "Your file system appears to be full: Do you want to clear your Temporary Directory (" + std::string(this->TemporaryDirectory) + "')?\n";        
        vtkKWMessageDialog *message = vtkKWMessageDialog::New();
        message->SetParent(this->ApplicationGUI->GetViewerWidget());
        message->SetOptions(vtkKWMessageDialog::ErrorIcon);
        message->SetIcon();
        message->SetStyleToOkCancel();
        message->SetText(msg.c_str());
        message->Create();
        message->Invoke();
        message->Delete();
        int clearDir = message->Invoke();
        message->Delete();
        if ( clearDir )
          {
          //--- empty temp directory
          vtksys::SystemTools::RemoveADirectory ( temporaryDirectory.c_str() );
          if ( vtksys::SystemTools::MakeDirectory ( temporaryDirectory.c_str() ) == false )
            {
            vtkWarningMacro ( "Temporary Directory cleared: Error: unable to recreate Temporary Directory after deleting its contents." );      
            }
          }
        }
      else
        {
        tst.close();
        //--- clean up
        vtksys::SystemTools::RemoveFile ( testFile.c_str() );
        }
      }
    }

  //--- now cache directory....
  std::string cacheDirectory;
  if (this->GetRemoteCacheDirectory() )
    {
    cacheDirectory = this->GetRemoteCacheDirectory();
    }
  this->SetRemoteCacheDirectory(cacheDirectory.c_str());
  
  if (cacheDirectory != "")
    {
    //--- make sure directory is present
    if (!vtksys::SystemTools::FileExists(cacheDirectory.c_str()))
      {
      vtksys::SystemTools::MakeDirectory(cacheDirectory.c_str());
      }
    //--- if there, make sure it's actually a dir
    if (vtksys::SystemTools::FileExists(cacheDirectory.c_str())
        && vtksys::SystemTools::FileIsDirectory(cacheDirectory.c_str()))
      {
      //--- make a test write and see if it works.
      testFile.clear();
      testFile = cacheDirectory;
      testFile += "/SlicerDiskSpaceCheck.txt";
      std::ofstream tst2( testFile.c_str() );
      tst2 << "Disk space check";
      if (tst2.good() == false || tst2.fail() == true || tst2.bad() == true )
        {
        std::string msg = "Your file system still appears to be full: Do you want to clear your Cache Directory (" + std::string(this->RemoteCacheDirectory) + "')?\n";
        vtkKWMessageDialog *message = vtkKWMessageDialog::New();
        message->SetParent(this->ApplicationGUI->GetViewerWidget());
        message->SetOptions(vtkKWMessageDialog::ErrorIcon);
        message->SetIcon();
        message->SetStyleToOkCancel();
        message->SetText(msg.c_str());
        message->Create();
        int clearDir = message->Invoke();
        message->Delete();
        if ( clearDir )
          {
          //--- empty temp directory
          vtksys::SystemTools::RemoveADirectory ( cacheDirectory.c_str() );
          if ( vtksys::SystemTools::MakeDirectory ( cacheDirectory.c_str() ) == false )
            {
            vtkWarningMacro ( "Remote Cache Directory cleared: Error: unable to recreate Remote Cache Directory after deleting its contents." );      
            }
          }
        }
      else
        {
        tst2.close();
        //--- clean up
        vtksys::SystemTools::RemoveFile ( testFile.c_str() );
        }
      }
    }

  //--- now final test in the temporary directory
  //--- after chance to clear temporary and cache directories,
  //--- see if there's disk space now...
  int diskFull = 1;
  if (temporaryDirectory != "")
    {
    //--- make sure directory is present
    if (!vtksys::SystemTools::FileExists(temporaryDirectory.c_str()))
      {
      vtksys::SystemTools::MakeDirectory(temporaryDirectory.c_str());
      }
    //--- if there, make sure it's actually a dir
    if (vtksys::SystemTools::FileExists(temporaryDirectory.c_str())
        && vtksys::SystemTools::FileIsDirectory(temporaryDirectory.c_str()))
      {
      //--- make a test write and see if it works.
      testFile.clear();
      testFile = temporaryDirectory;
      testFile += "/SlicerDiskSpaceCheck.txt";
      std::ofstream tst3( testFile.c_str() );
      tst3 << "Disk space check";
      if (tst3.good() == false || tst3.fail() == true || tst3.bad() == true )
        {
        std::string msg = "Your file system still appears to be full. Slicer will not start up with its full set of modules. It is recommended that you close the application, create some additional disk space, and then restart Slicer.";
        vtkKWMessageDialog *message = vtkKWMessageDialog::New();
        message->SetParent(this->ApplicationGUI->GetViewerWidget());
        message->SetOptions(vtkKWMessageDialog::ErrorIcon);
        message->SetIcon();
        message->SetStyleToMessage();
        message->SetText(msg.c_str());
        message->Create();
        message->Invoke();
        message->Delete();
        diskFull = 1;
        }
      else
        {
        diskFull = 0;
        tst3.close();
        //--- clean up
        vtksys::SystemTools::RemoveFile ( testFile.c_str() );
        }
      }
    }

  if ( diskFull )
    {
    return 1;
    }
  else
    {
    return 0;
    }
}




//---------------------------------------------------------------------------
void vtkSlicerApplication::AddModuleGUI ( vtkSlicerModuleGUI *gui ) {

    // Create if it doesn't exist already
    if ( this->ModuleGUICollection == NULL ) {
        this->ModuleGUICollection = vtkSlicerGUICollection::New ( );
    } 
    // Add a gui
    this->ModuleGUICollection->AddItem ( gui );
}

//---------------------------------------------------------------------------
void vtkSlicerApplication::RemoveModuleGUI ( vtkSlicerModuleGUI *gui ) {

    // Create if it doesn't exist already
    if ( this->ModuleGUICollection == NULL ) {
        this->ModuleGUICollection = vtkSlicerGUICollection::New ( );
    } 
    // Remove a gui
    this->ModuleGUICollection->RemoveItem ( gui );
}

//---------------------------------------------------------------------------
vtkSlicerModuleGUI* vtkSlicerApplication::GetModuleGUIByName ( const char *name )
{
    if ( this->ModuleGUICollection != NULL ) {
        int n = this->ModuleGUICollection->GetNumberOfItems ( );
        int i;
        for (i = 0; i < n; i ++ ) {
            vtkSlicerModuleGUI *m = vtkSlicerModuleGUI::SafeDownCast(
              this->ModuleGUICollection->GetItemAsObject(i) );
            if ( !strcmp (m->GetGUIName(), name) ) {
                return (m);
            }
        }
    }
    return ( NULL );
}

//---------------------------------------------------------------------------
void vtkSlicerApplication::ConfigureApplication ( ) {

    this->PromptBeforeExitOn ( );
    this->SupportSplashScreenOn ( );
    this->SplashScreenVisibilityOn ( );
    this->SaveUserInterfaceGeometryOn ( );
}

//---------------------------------------------------------------------------
void vtkSlicerApplication::InstallTheme ( vtkKWTheme *theme )
{
    if ( theme != NULL ) {
        if ( vtkSlicerTheme::SafeDownCast (theme) == this->SlicerTheme ) {
            this->SetTheme (this->SlicerTheme );
        } else {
            this->SetTheme ( theme );
        }
    }
}

//---------------------------------------------------------------------------
void vtkSlicerApplication::CloseAllWindows ( ) {
    int n, i;
    vtkKWWindowBase *win;
    
    n= this->GetNumberOfWindows ( );
    for (i=0; i<n; i++) {
        win = this->GetNthWindow ( n );
        win->Close ( );
    }
}

//---------------------------------------------------------------------------
int vtkSlicerApplication::StartApplication ( ) {

    int ret = 0;

    // Start the application & event loop here
    this->Start ( );
    this->CloseAllWindows ( );

    // Clean up and exit
    ret = this->GetExitStatus ( );
    return ret;
}

//---------------------------------------------------------------------------
void vtkSlicerApplication::DoOneTclEvent ( ) 
{
  //
  // First, handle system-level events such as mouse moves, keys,
  // socket connections, etc
  //
  Tcl_DoOneEvent(0);

  //
  // Then handle application-level events that were queued in 
  // response to the system events
  // - only do this if event broker is in asynchronous mode
  // - have tcl first handle all its pending events (e.g. click and drag events)
  //
  vtkEventBroker *broker = vtkEventBroker::GetInstance();
  if ( broker->GetEventMode() == vtkEventBroker::Asynchronous )
    {
    Tcl_ServiceAll();
    broker->ProcessEventQueue();
    }
}

//----------------------------------------------------------------------------
//  access to registry values

int vtkSlicerApplication::HasRegistry(const char *key)
{
  return this->HasRegistryValue( 2, "RunTime", key);
}

void vtkSlicerApplication::RequestRegistry(const char *key)
{
  if (this->HasRegistryValue( 2, "RunTime", key))
    {
    this->GetRegistryValue( 2, "RunTime", key, this->RegistryHolder);
    }
  else
    {
    *this->RegistryHolder = '\0';
    }
}

const char *vtkSlicerApplication::GetRegistryHolder()
{
  return this->RegistryHolder;
}

void vtkSlicerApplication::SetRegistry(const char *key, char *value)
{
  this->SetRegistryValue(2, "RunTime", key, value);
}


//----------------------------------------------------------------------------
void vtkSlicerApplication::RestoreApplicationSettingsFromRegistry()
{
  // Make a good guess before we read from the registry.  Default to a
  // subdirectory called Slicer3 in a standard temp location.
#ifdef _WIN32
  GetTempPath(vtkKWRegistryHelper::RegistryKeyValueSizeMax,
              this->TemporaryDirectory);
#else
  strcpy(this->TemporaryDirectory, "~");
#endif


  // Tk does not understand Windows short path names, so convert to
  // long path names and unix slashes
  std::string temporaryDirectory = this->TemporaryDirectory;
  temporaryDirectory
    = itksys::SystemTools::GetActualCaseForPath(temporaryDirectory.c_str());
  itksys::SystemTools::ConvertToUnixSlashes( temporaryDirectory );
  
  std::vector<std::string> pathcomponents;
  std::string pathWithSlicer;
  itksys::SystemTools::SplitPath(temporaryDirectory.c_str(), pathcomponents);
#ifdef _WIN32
  pathcomponents.push_back("Slicer3");
#else
  std::string dirName("Slicer3");
  if ( getenv("USER") != NULL )
    {
    dirName = dirName + getenv("USER");
    }
  pathcomponents.push_back(dirName);
#endif
  pathWithSlicer = itksys::SystemTools::JoinPath(pathcomponents);
  
  itksys::SystemTools::MakeDirectory(pathWithSlicer.c_str());
  if (pathWithSlicer.size() < vtkKWRegistryHelper::RegistryKeyValueSizeMax)
    {
    strcpy(this->TemporaryDirectory, pathWithSlicer.c_str());
    }
  else
    {
    // path with "Slicer3" attached is too long. Try it without
    // "Slicer3". If still too long, use the original path. (This path
    // may have short names in it and hence will not work with Tk).
    if (temporaryDirectory.size()
        < vtkKWRegistryHelper::RegistryKeyValueSizeMax)
      {
      strcpy(this->TemporaryDirectory, temporaryDirectory.c_str());
      }
    vtkWarningMacro("Default temporary directory path " << pathWithSlicer.c_str()
                    << " is too long to be stored in the registry."
                    << " Using unmodified temporary directory path "
                    << this->TemporaryDirectory);
    }
  
  //--- Set up the cache directory -- use the temporary directory initially.
  strcpy ( this->RemoteCacheDirectory, this->TemporaryDirectory);

  strcpy( this->ExtensionsDownloadDir, this->TemporaryDirectory );

  //--- web browser
  // start with no browser...
  strcpy(this->WebBrowser, "");
  // Tk does not understand Windows short path names, so convert to
  // long path names and unix slashes
  std::string webBrowser = this->WebBrowser;
/*
  webBrowser 
    = itksys::SystemTools::GetActualCaseForPath(webBrowser.c_str());
  itksys::SystemTools::ConvertToUnixSlashes( webBrowser );
    strcpy(this->WebBrowser, webBrowser.c_str());
  if (webBrowser.size() > vtkKWRegistryHelper::RegistryKeyValueSizeMax)
    {
    vtkWarningMacro("Path to firefox: " << this->WebBrowser
                    << " is too long to be stored in the registry."
                    << " You will have to set the browser location each time you launch Slicer to use modules that require it.");
    }
*/  

  //--- unzip
  strcpy(this->Unzip, "");
  //--- zip
  strcpy(this->Zip, "");
  //--- rm
  strcpy(this->Rm, "");

  Superclass::RestoreApplicationSettingsFromRegistry();
  
  char temp_reg_value[vtkKWRegistryHelper::RegistryKeyValueSizeMax];
  
  if (this->HasRegistryValue(
    2, "RunTime", vtkSlicerApplication::ConfirmDeleteRegKey))
    {
    this->GetRegistryValue(
      2, "RunTime", vtkSlicerApplication::ConfirmDeleteRegKey,
      this->ConfirmDelete);
    }
  
  if (this->HasRegistryValue(
    2, "RunTime", vtkSlicerApplication::HomeModuleRegKey))
    {
    this->GetRegistryValue(
      2, "RunTime", vtkSlicerApplication::HomeModuleRegKey,
      this->HomeModule);
    }

  if (this->HasRegistryValue(
    2, "RunTime", vtkSlicerApplication::PotentialModulePathsRegKey))
    {
    this->GetRegistryValue(
      2, "RunTime", vtkSlicerApplication::PotentialModulePathsRegKey,
      this->PotentialModulePaths);
    }

  if (this->HasRegistryValue(
    2, "RunTime", vtkSlicerApplication::PotentialColorFilePathsRegKey))
    {
    this->GetRegistryValue(
      2, "RunTime", vtkSlicerApplication::PotentialColorFilePathsRegKey,
      this->PotentialColorFilePaths);
    }

  if (this->HasRegistryValue(
    2, "RunTime", vtkSlicerApplication::ModulePathsRegKey))
    {
    this->GetRegistryValue(
      2, "RunTime", vtkSlicerApplication::ModulePathsRegKey,
      temp_reg_value);
    // Use SetModulePaths so that PotentialModulePaths is updated too
    this->SetModulePaths(temp_reg_value); 
    }
  if (this->HasRegistryValue(
    2, "RunTime", vtkSlicerApplication::ColorFilePathsRegKey))
    {
    this->GetRegistryValue(
      2, "RunTime", vtkSlicerApplication::ColorFilePathsRegKey,
      temp_reg_value);
    // Use SetColorFilePaths so that PotentialColorFilePaths is updated too
    this->SetColorFilePaths(temp_reg_value); 
    }

  if (this->HasRegistryValue(
    2, "RunTime", vtkSlicerApplication::ModuleCachePathRegKey))
    {
    this->GetRegistryValue(
      2, "RunTime", vtkSlicerApplication::ModuleCachePathRegKey,
      this->ModuleCachePath);
    }

  if (this->HasRegistryValue(
    2, "RunTime", vtkSlicerApplication::TemporaryDirectoryRegKey))
    {
    this->GetRegistryValue(
      2, "RunTime", vtkSlicerApplication::TemporaryDirectoryRegKey,
      this->TemporaryDirectory);
    }

  if (this->HasRegistryValue(
    2, "RunTime", vtkSlicerApplication::WebBrowserRegKey))
    {
    this->GetRegistryValue(
      2, "RunTime", vtkSlicerApplication::WebBrowserRegKey,
      this->WebBrowser);
    }

  if (this->HasRegistryValue(2, "RunTime", vtkSlicerApplication::UnzipRegKey))
    {
    this->GetRegistryValue(2, "RunTime", vtkSlicerApplication::UnzipRegKey,
                           this->Unzip);
    }
  if (this->HasRegistryValue(2, "RunTime", vtkSlicerApplication::ZipRegKey))
    {
    this->GetRegistryValue(2, "RunTime", vtkSlicerApplication::ZipRegKey,
                           this->Zip);
    }
  if (this->HasRegistryValue(2, "RunTime", vtkSlicerApplication::RmRegKey))
    {
    this->GetRegistryValue(2, "RunTime", vtkSlicerApplication::RmRegKey,
                           this->Rm);
    }
      
  if (this->HasRegistryValue(
    2, "RunTime", vtkSlicerApplication::LoadCommandLineModulesRegKey))
    {
    this->LoadCommandLineModules = this->GetIntRegistryValue(
      2, "RunTime", vtkSlicerApplication::LoadCommandLineModulesRegKey);
    }

  if (this->HasRegistryValue(
    2, "RunTime", vtkSlicerApplication::IgnoreModulesRegKey))
    {
    this->GetRegistryValue(2, "RunTime", vtkSlicerApplication::IgnoreModulesRegKey,
                           this->IgnoreModuleNames);
    this->StringToArray(std::string(this->IgnoreModuleNames), this->NameSeparator.c_str()[0], this->IgnoreModules);
    }

  if (this->HasRegistryValue(
    2, "RunTime", vtkSlicerApplication::LoadModulesRegKey))     
    {
    this->LoadModules = this->GetIntRegistryValue(
      2, "RunTime", vtkSlicerApplication::LoadModulesRegKey);
    }

  if (this->HasRegistryValue(
    2, "RunTime", vtkSlicerApplication::EnableDaemonRegKey))
    {
    this->EnableDaemon = this->GetIntRegistryValue(
      2, "RunTime", vtkSlicerApplication::EnableDaemonRegKey);
    }

  if ( this->HasRegistryValue(2, "RunTime", vtkSlicerApplication::ApplicationFontSizeRegKey))
    {
    this->GetRegistryValue (
      2, "RunTime", vtkSlicerApplication::ApplicationFontSizeRegKey, this->ApplicationFontSize);
    }

  if ( this->HasRegistryValue(2, "RunTime", vtkSlicerApplication::ApplicationFontFamilyRegKey))
    {
    this->GetRegistryValue (
       2, "RunTime", vtkSlicerApplication::ApplicationFontFamilyRegKey, this->ApplicationFontFamily );
    }

  if ( this->HasRegistryValue (2, "RunTime", vtkSlicerApplication::ApplicationWindowWidthRegKey))
    {
    this->ApplicationWindowWidth = this->GetIntRegistryValue(
        2, "RunTime", vtkSlicerApplication::ApplicationWindowWidthRegKey);
    }
  if ( this->HasRegistryValue (2, "RunTime", vtkSlicerApplication::ApplicationWindowHeightRegKey))
    {
    this->ApplicationWindowHeight = this->GetIntRegistryValue(
        2, "RunTime", vtkSlicerApplication::ApplicationWindowHeightRegKey);
    }
  if ( this->HasRegistryValue (2, "RunTime", vtkSlicerApplication::ApplicationSlicesFrameHeightRegKey))
    {
    this->ApplicationSlicesFrameHeight = this->GetIntRegistryValue(
        2, "RunTime", vtkSlicerApplication::ApplicationSlicesFrameHeightRegKey);
    }
  if ( this->HasRegistryValue (2, "RunTime", vtkSlicerApplication::ApplicationLayoutTypeRegKey))
    {
    this->ApplicationLayoutType = this->GetIntRegistryValue(
        2, "RunTime", vtkSlicerApplication::ApplicationLayoutTypeRegKey);
    }

  if ( this->HasRegistryValue(
        2, "RunTime", vtkSlicerApplication::UseWelcomeModuleAtStartupRegKey))
    {
    this->UseWelcomeModuleAtStartup = this->GetIntRegistryValue (
        2, "RunTime", vtkSlicerApplication::UseWelcomeModuleAtStartupRegKey);
    }
   if (this->HasRegistryValue(
         2, "RunTime", vtkSlicerApplication::EnableAsynchronousIORegKey))
    {
    this->EnableAsynchronousIO = this->GetIntRegistryValue(
      2, "RunTime", vtkSlicerApplication::EnableAsynchronousIORegKey);
    }
   if (this->HasRegistryValue(
         2, "RunTime", vtkSlicerApplication::EnableForceRedownloadRegKey))
    {
    this->EnableForceRedownload = this->GetIntRegistryValue(
      2, "RunTime", vtkSlicerApplication::EnableForceRedownloadRegKey);
    }


  if (this->HasRegistryValue(
    2, "RunTime", vtkSlicerApplication::RemoteCacheDirectoryRegKey))
    {
    this->GetRegistryValue(
      2, "RunTime", vtkSlicerApplication::RemoteCacheDirectoryRegKey,
      this->RemoteCacheDirectory);
    }
/*
   if (this->HasRegistryValue(
         2, "RunTime", vtkSlicerApplication::EnableRemoteCacheOverwritingRegKey))
    {
    this->EnableRemoteCacheOverwriting = this->GetIntRegistryValue(
      2, "RunTime", vtkSlicerApplication::EnableRemoteCacheOverwritingRegKey);
    }
*/
   if (this->HasRegistryValue(
         2, "RunTime", vtkSlicerApplication::RemoteCacheLimitRegKey))
    {
    this->RemoteCacheLimit = this->GetIntRegistryValue(
      2, "RunTime", vtkSlicerApplication::RemoteCacheLimitRegKey);
    }
   if (this->HasRegistryValue(
         2, "RunTime", vtkSlicerApplication::RemoteCacheFreeBufferSizeRegKey))
    {
    this->RemoteCacheFreeBufferSize = this->GetIntRegistryValue(
      2, "RunTime", vtkSlicerApplication::RemoteCacheFreeBufferSizeRegKey);
    }
}


//----------------------------------------------------------------------------
void vtkSlicerApplication::ConfigureRemoteIOSettingsFromRegistry()
{
  if ( this->ApplicationGUI )
    {
    //--- propagate this value through the GUI into MRML's CacheManager
    this->ApplicationGUI->ConfigureRemoteIOSettings();
    }
}



//----------------------------------------------------------------------------
void vtkSlicerApplication::UpdateRemoteIOSettingsForRegistry()
{
  if ( this->ApplicationGUI )
    {
    //--- update values from MRML, through the GUI 
    this->ApplicationGUI->UpdateRemoteIOConfigurationForRegistry();
    }
}


//----------------------------------------------------------------------------
void vtkSlicerApplication::SaveApplicationWindowConfiguration()
{
  if ( this->ApplicationGUI )
    {
    if ( this->ApplicationGUI->GetMainSlicerWindow() )
      {
      this->SetApplicationWindowWidth (this->ApplicationGUI->GetMainSlicerWindow()->GetWidth());
      this->SetApplicationWindowHeight (this->ApplicationGUI->GetMainSlicerWindow()->GetHeight());
      this->SetApplicationSlicesFrameHeight ( this->ApplicationGUI->GetMainSlicerWindow()->GetSecondarySplitFrame()->GetFrame1Size() );
      if ( this->ApplicationGUI->GetGUILayoutNode() )
        {
        // Compare View not supported as a saved layout yet (need to
        // put the number of compare views into the Layout node
        if (this->ApplicationGUI->GetGUILayoutNode()->GetViewArrangement()
            != vtkMRMLLayoutNode::SlicerLayoutCompareView)
          {
          this->SetApplicationLayoutType ( this->ApplicationGUI->GetGUILayoutNode()->GetViewArrangement() );
          }
        }
      }
    }
}

//----------------------------------------------------------------------------
void vtkSlicerApplication::SaveApplicationSettingsToRegistry()
{ 
  Superclass::SaveApplicationSettingsToRegistry();

  this->SetRegistryValue(
    2, "RunTime", vtkSlicerApplication::ConfirmDeleteRegKey, "%s", 
    this->ConfirmDelete);
  
  this->SetRegistryValue(
    2, "RunTime", vtkSlicerApplication::HomeModuleRegKey, "%s", 
    this->HomeModule);

  this->SetRegistryValue ( 2, "RunTime", vtkSlicerApplication::ApplicationFontFamilyRegKey, "%s",
                           this->ApplicationFontFamily);
  this->SetRegistryValue ( 2, "RunTime", vtkSlicerApplication::ApplicationFontSizeRegKey, "%s",
                           this->ApplicationFontSize);

  this->SetRegistryValue(
    2, "RunTime", vtkSlicerApplication::PotentialModulePathsRegKey, "%s", 
    this->PotentialModulePaths);

  this->SetRegistryValue(
    2, "RunTime", vtkSlicerApplication::PotentialColorFilePathsRegKey, "%s", 
    this->PotentialColorFilePaths);

  this->SetRegistryValue(
    2, "RunTime", vtkSlicerApplication::ModulePathsRegKey, "%s", 
    this->ModulePaths);

  this->SetRegistryValue(
    2, "RunTime", vtkSlicerApplication::ColorFilePathsRegKey, "%s", 
    this->ColorFilePaths);

  this->SetRegistryValue(
    2, "RunTime", vtkSlicerApplication::ModuleCachePathRegKey, "%s", 
    this->ModuleCachePath);
  
  this->SetRegistryValue(
    2, "RunTime", vtkSlicerApplication::TemporaryDirectoryRegKey, "%s", 
    this->TemporaryDirectory);

  this->SetRegistryValue(
    2, "RunTime", vtkSlicerApplication::WebBrowserRegKey, "%s", 
    this->WebBrowser);

  this->SetRegistryValue(
    2, "RunTime", vtkSlicerApplication::UnzipRegKey, "%s", 
    this->Unzip);
  this->SetRegistryValue(
    2, "RunTime", vtkSlicerApplication::ZipRegKey, "%s", 
    this->Zip);
  this->SetRegistryValue(
    2, "RunTime", vtkSlicerApplication::RmRegKey, "%s", 
    this->Rm);

  this->SetRegistryValue(
    2, "RunTime", vtkSlicerApplication::LoadCommandLineModulesRegKey, "%d", 
    this->LoadCommandLineModules);

  this->SetRegistryValue(
    2, "RunTime", vtkSlicerApplication::LoadModulesRegKey, "%d", 
    this->LoadModules);

  
  this->ArrayToString(this->IgnoreModules, this->NameSeparator, this->IgnoreModuleNames, 
                      vtkKWRegistryHelper::RegistryKeyValueSizeMax);
  this->SetRegistryValue(
    2, "RunTime", vtkSlicerApplication::IgnoreModulesRegKey, "%s", 
    this->IgnoreModuleNames);
  

  this->SetRegistryValue(
    2, "RunTime", vtkSlicerApplication::EnableDaemonRegKey, "%d", 
    this->EnableDaemon);

  this->SaveApplicationWindowConfiguration();
  this->SetRegistryValue(
                         2, "RunTime", vtkSlicerApplication::ApplicationWindowWidthRegKey, "%d",
                         this->ApplicationWindowWidth );
  this->SetRegistryValue(
                         2, "RunTime", vtkSlicerApplication::ApplicationWindowHeightRegKey, "%d",
                         this->ApplicationWindowHeight );
  this->SetRegistryValue(
                         2, "RunTime", vtkSlicerApplication::ApplicationLayoutTypeRegKey, "%d",
                         this->ApplicationLayoutType );
  this->SetRegistryValue(
                         2, "RunTime", vtkSlicerApplication::ApplicationSlicesFrameHeightRegKey, "%d",
                         this->ApplicationSlicesFrameHeight );

  this->SetRegistryValue(
                         2, "RunTime", vtkSlicerApplication::RemoteCacheDirectoryRegKey, "%s", 
                         this->RemoteCacheDirectory);
  this->SetRegistryValue(
                         2, "RunTime", vtkSlicerApplication::EnableAsynchronousIORegKey, "%d", 
                         this->EnableAsynchronousIO);
  this->SetRegistryValue(
                         2, "RunTime", vtkSlicerApplication::EnableForceRedownloadRegKey, "%d", 
                         this->EnableForceRedownload);
/*
  this->SetRegistryValue(
                         2, "RunTime", vtkSlicerApplication::EnableRemoteCacheOverwritingRegKey, "%d", 
                         this->EnableRemoteCacheOverwriting);
*/
  this->SetRegistryValue(
                         2, "RunTime", vtkSlicerApplication::RemoteCacheLimitRegKey, "%d", 
                         this->RemoteCacheLimit);
  this->SetRegistryValue(
                         2, "RunTime", vtkSlicerApplication::RemoteCacheFreeBufferSizeRegKey, "%d", 
                         this->RemoteCacheFreeBufferSize);
  this->SetRegistryValue (
                          2, "RunTime", vtkSlicerApplication::UseWelcomeModuleAtStartupRegKey, "%d",
                          this->UseWelcomeModuleAtStartup);

}


//----------------------------------------------------------------------------
void vtkSlicerApplication::SetApplicationWindowSize (int width, int height )
{
  this->ApplicationWindowWidth = width;
  this->ApplicationWindowHeight = height;
}



//----------------------------------------------------------------------------
void vtkSlicerApplication::SetApplicationFontFamily ( const char *family)
{
    if (family)
    {
        if (strcmp(this->ApplicationFontFamily, family) != 0
        && strlen(family) < vtkKWRegistryHelper::RegistryKeyValueSizeMax)
        {
            strcpy(this->ApplicationFontFamily, family);
            this->Modified();
        }
    }
}
//----------------------------------------------------------------------------
const char *vtkSlicerApplication::GetApplicationFontFamily () const
{
  return this->ApplicationFontFamily;
}


//----------------------------------------------------------------------------
void vtkSlicerApplication::SetApplicationFontSize ( const char *size)
{
    if (size )
    {
        if (strcmp(this->ApplicationFontSize, size) != 0
        && strlen(size) < vtkKWRegistryHelper::RegistryKeyValueSizeMax)
        {
            strcpy(this->ApplicationFontSize, size);
            this->Modified();
        }
    }
}
//----------------------------------------------------------------------------
const char *vtkSlicerApplication::GetApplicationFontSize () const
{
  return this->ApplicationFontSize;
}




//----------------------------------------------------------------------------
void vtkSlicerApplication::SetConfirmDelete(const char* state)
{
    if (state)
    {
        if (strcmp(this->ConfirmDelete, state) != 0
        && strlen(state) < vtkKWRegistryHelper::RegistryKeyValueSizeMax)
        {
            strcpy(this->ConfirmDelete, state);
            this->Modified();
        }
    }
}


//----------------------------------------------------------------------------
const char *vtkSlicerApplication::GetConfirmDelete() const
{
    return this->ConfirmDelete;
}

//----------------------------------------------------------------------------
void vtkSlicerApplication::SetHomeModule ( const char *name )
{
  if (name)
    {
    if (strcmp(this->HomeModule, name) != 0
        && strlen(name) < vtkKWRegistryHelper::RegistryKeyValueSizeMax)
      {
      strcpy(this->HomeModule, name);
      this->Modified();

      }
    }
}

//----------------------------------------------------------------------------
const char *vtkSlicerApplication::GetHomeModule () const
{
  return this->HomeModule;
}

//----------------------------------------------------------------------------
void vtkSlicerApplication::SetModulePaths(const char* paths)
{
#if WIN32
  const char delim = ';';
#else
  const char delim = ':';
#endif
  if (paths)
    {
    if (strcmp(this->ModulePaths, paths) != 0
        && strlen(paths) < vtkKWRegistryHelper::RegistryKeyValueSizeMax)
      {
      strcpy(this->ModulePaths, paths);
      char *str = new char [strlen(this->PotentialModulePaths) + 1];
      strcpy(str, this->PotentialModulePaths);
      int count = vtkKWDirectoryPresetSelector::
        UpdatePresetDirectoriesFromEnabledPresetDirectories(
          &str, '|', this->ModulePaths, delim);
      if (count)
        {
        strcpy(this->PotentialModulePaths, str);
        }
      delete [] str;
      this->Modified();
      }
    }
}

//----------------------------------------------------------------------------
const char* vtkSlicerApplication::GetModulePaths() const
{
  return this->ModulePaths;
}

//----------------------------------------------------------------------------
void vtkSlicerApplication::SetPotentialModulePaths(const char* paths)
{
#if WIN32
  const char delim = ';';
#else
  const char delim = ':';
#endif
  if (paths)
    {
    if (strcmp(this->PotentialModulePaths, paths) != 0
        && strlen(paths) < vtkKWRegistryHelper::RegistryKeyValueSizeMax)
      {
      strcpy(this->PotentialModulePaths, paths);
      char *str = NULL;
      vtkKWDirectoryPresetSelector::
        GetEnabledPresetDirectoriesFromPresetDirectories(
          &str, delim, this->PotentialModulePaths, '|');
      strcpy(this->ModulePaths, str ? str : "");
      delete [] str;
      this->Modified();
      }
    }
}

//----------------------------------------------------------------------------
const char* vtkSlicerApplication::GetPotentialModulePaths() const
{
  return this->PotentialModulePaths;
}

//----------------------------------------------------------------------------
void vtkSlicerApplication::SetColorFilePaths(const char* paths)
{
#if WIN32
  const char delim = ';';
#else
  const char delim = ':';
#endif
  if (paths)
    {
    if (strcmp(this->ColorFilePaths, paths) != 0
        && strlen(paths) < vtkKWRegistryHelper::RegistryKeyValueSizeMax)
      {
      strcpy(this->ColorFilePaths, paths);      
      char *str = new char [strlen(this->PotentialColorFilePaths) + 1];
      strcpy(str, this->PotentialColorFilePaths);
      int count = vtkKWDirectoryPresetSelector::
        UpdatePresetDirectoriesFromEnabledPresetDirectories(
          &str, '|', this->ColorFilePaths, delim);
      if (count)
        {
        strcpy(this->PotentialColorFilePaths, str);       
        }
      delete [] str;
      this->Modified();
      // let the colour module know?
      }
    }
}
//----------------------------------------------------------------------------
const char* vtkSlicerApplication::GetColorFilePaths() const
{
  return this->ColorFilePaths;
}
//----------------------------------------------------------------------------
void vtkSlicerApplication::SetPotentialColorFilePaths(const char* paths)
{
#if WIN32
  const char delim = ';';
#else
  const char delim = ':';
#endif
  if (paths)
    {
    if (strcmp(this->PotentialColorFilePaths, paths) != 0
        && strlen(paths) < vtkKWRegistryHelper::RegistryKeyValueSizeMax)
      {
      strcpy(this->PotentialColorFilePaths, paths);
      char *str = NULL;
      vtkKWDirectoryPresetSelector::
        GetEnabledPresetDirectoriesFromPresetDirectories(
          &str, delim, this->PotentialColorFilePaths, '|');
      strcpy(this->ColorFilePaths, str ? str : "");
      delete [] str;
      this->Modified();
      }
    }
}

//----------------------------------------------------------------------------
const char* vtkSlicerApplication::GetPotentialColorFilePaths() const
{
  return this->PotentialColorFilePaths;
}

//----------------------------------------------------------------------------
void vtkSlicerApplication::SetModuleCachePath(const char* path)
{
  if (path)
    {
    if (strcmp(this->ModuleCachePath, path) != 0
        && strlen(path) < vtkKWRegistryHelper::RegistryKeyValueSizeMax)
      {
      strcpy(this->ModuleCachePath, path);
      this->Modified();
      }
    }
}

//----------------------------------------------------------------------------
const char* vtkSlicerApplication::GetModuleCachePath() const
{
  return this->ModuleCachePath;
}

//----------------------------------------------------------------------------
void vtkSlicerApplication::SetWebBrowser(const char* path)
{
  if (path)
    {
    if (strcmp(this->WebBrowser, path) != 0
        && strlen(path) < vtkKWRegistryHelper::RegistryKeyValueSizeMax)
      {
      strcpy(this->WebBrowser, path);
      this->Modified();
      }
    }

}

//----------------------------------------------------------------------------
const char* vtkSlicerApplication::GetWebBrowser() const
{
  // should check if this is an executable file...
  return this->WebBrowser;
}


//----------------------------------------------------------------------------
void vtkSlicerApplication::SetUnzip(const char* path)
{
  if (path)
    {
    if (strcmp(this->Unzip, path) != 0
        && strlen(path) < vtkKWRegistryHelper::RegistryKeyValueSizeMax)
      {
      strcpy(this->Unzip, path);
      this->Modified();
      }
    }
}

//----------------------------------------------------------------------------
const char* vtkSlicerApplication::GetUnzip() const
{
  // should check if this is an executable file...
  return this->Unzip;
}

//----------------------------------------------------------------------------
void vtkSlicerApplication::SetZip(const char* path)
{
  if (path)
    {
    if (strcmp(this->Zip, path) != 0
        && strlen(path) < vtkKWRegistryHelper::RegistryKeyValueSizeMax)
      {
      strcpy(this->Zip, path);
      this->Modified();
      }
    }
}

//----------------------------------------------------------------------------
const char* vtkSlicerApplication::GetZip() const
{
  // should check if this is an executable file...
  return this->Zip;
}

//----------------------------------------------------------------------------
void vtkSlicerApplication::SetRm(const char* path)
{
  if (path)
    {
    if (strcmp(this->Rm, path) != 0
        && strlen(path) < vtkKWRegistryHelper::RegistryKeyValueSizeMax)
      {
      strcpy(this->Rm, path);
      this->Modified();
      }
    }
}

//----------------------------------------------------------------------------
const char* vtkSlicerApplication::GetRm() const
{
  // should check if this is an executable file...
  return this->Rm;
}

//----------------------------------------------------------------------------
void vtkSlicerApplication::SetTemporaryDirectory(const char* path)
{
  if (path)
    {
    if (strcmp(this->TemporaryDirectory, path) != 0
        && strlen(path) < vtkKWRegistryHelper::RegistryKeyValueSizeMax)
      {
      strcpy(this->TemporaryDirectory, path);
      this->Modified();
      }
    }
}

//----------------------------------------------------------------------------
const char* vtkSlicerApplication::GetTemporaryDirectory() const
{
  if (this->TemporaryDirectory)
    {
    // does the path exist?
    if (!itksys::SystemTools::MakeDirectory(this->TemporaryDirectory))
      {
      // error making sure that the dir exists
      std::cout << "vtkSlicerApplication::GetTemporaryDirectory: Unable to make temporary directory: '" << this->TemporaryDirectory << "'\n\tYou can change the Temporary Directory under View->Application Settings->Module Settings." << std::endl;
      // pop up a window if we've got something to set for the parent
      if (this->ApplicationGUI && this->ApplicationGUI->GetViewerWidget())
        {
        std::string msg = std::string("ERROR\nUnable to make temporary directory: '") + std::string(this->TemporaryDirectory) + std::string("'\nYou can change the Temporary Directory under View->Application Settings->Module Settings.");
        vtkKWMessageDialog *message = vtkKWMessageDialog::New();
        message->SetParent(this->ApplicationGUI->GetViewerWidget());
        message->SetOptions(vtkKWMessageDialog::ErrorIcon);
        message->SetIcon();
        message->SetStyleToCancel();
        message->SetText(msg.c_str());
        message->Create();
        message->Invoke();
        message->Delete();
        }
      }
    else
      {
      // check that can write to it
      std::vector<std::string> tempPath;
      tempPath.push_back("");
      // no slash between first two elements
      tempPath.push_back(this->TemporaryDirectory);
      tempPath.push_back("testWrite.txt");
      std::string tempFile = itksys::SystemTools::JoinPath(tempPath);
      FILE *fp = fopen(tempFile.c_str(), "w");
      if (!fp)
        {
        std::cerr << "WARNING: Unable to write files in TemporaryDirectory: '" << this->TemporaryDirectory << "'" << std::endl;
        // pop up a window if we've got something to set for the parent
        if (this->ApplicationGUI && this->ApplicationGUI->GetViewerWidget())
          {
          std::string msg = std::string("WARNING\nUnable to write files in TemporaryDirectory:\n'") + std::string(this->TemporaryDirectory) + std::string("'");
          vtkKWMessageDialog *message = vtkKWMessageDialog::New();
          message->SetParent(this->ApplicationGUI->GetViewerWidget());
          message->SetOptions(vtkKWMessageDialog::ErrorIcon);
          message->SetIcon();
          message->SetStyleToCancel();
          message->SetText(msg.c_str());
          message->Create();
          message->Invoke();
          message->Delete();
          }
        }
      else
        {
        fclose(fp);
        // delete it
        remove(tempFile.c_str());
        }
      }
    }
  return this->TemporaryDirectory;
}

//----------------------------------------------------------------------------
void vtkSlicerApplication::SetBinDir(const char* path)
{
  if (path)
    {
    if (strcmp(this->BinDir, path) != 0
        && strlen(path) < vtkKWRegistryHelper::RegistryKeyValueSizeMax)
      {
      strcpy(this->BinDir, path);
      this->Modified();
      }
    }
}

//----------------------------------------------------------------------------
const char* vtkSlicerApplication::GetBinDir() const
{
  return this->BinDir;
}

//----------------------------------------------------------------------------
bool vtkSlicerApplication::RequestDisplayMessage( const char *type, const char *message )
{
  bool active;

  //std::cout << "Requesting a message be put on the logger " << type << ": " << message << std::endl;

  // only request to add a record to the log if the log queue is up
  this->DisplayMessageQueueActiveLock->Lock();
  active = this->DisplayMessageQueueActive;
  this->DisplayMessageQueueActiveLock->Unlock();

  if (active)
    {
    this->DisplayMessageQueueLock->Lock();
    (*this->InternalDisplayMessageQueue).push( AddRecordType(type, message) );
//     std::cout << " [" << (*this->InternalDisplayMessageQueue).size()
//               << "] " << std::endl;
    this->DisplayMessageQueueLock->Unlock();
    
    return true;
    }

  // could not request the record be added to the queue
  return false;
}

//----------------------------------------------------------------------------
void vtkSlicerApplication::ProcessDisplayMessage()
{
  bool active = true;
  AddRecordType record;
  record.first = "";
  
  // Check to see if we should be shutting down
  this->DisplayMessageQueueActiveLock->Lock();
  active = this->DisplayMessageQueueActive;
  this->DisplayMessageQueueActiveLock->Unlock();
  
  if (active)
    {
    // pull an object off the queue 
    this->DisplayMessageQueueLock->Lock();
    if ((*this->InternalDisplayMessageQueue).size() > 0)
      {
      record = (*this->InternalDisplayMessageQueue).front();
      (*this->InternalDisplayMessageQueue).pop();
      }
    this->DisplayMessageQueueLock->Unlock();

    // post the message
    if (vtkSlicerApplication::GetInstance()->CreateLogDialog())
      {
      if (record.first == "Error")
        {
        vtkSlicerApplication::GetInstance()->GetLogDialog()->GetLogWidget()->AddErrorRecord( record.second.c_str() );
        }
      else if (record.first == "Warning")
        {
        vtkSlicerApplication::GetInstance()->GetLogDialog()->GetLogWidget()->AddWarningRecord( record.second.c_str() );
        }
      else if (record.first == "Information")
        {
        vtkSlicerApplication::GetInstance()->GetLogDialog()->GetLogWidget()->AddInformationRecord( record.second.c_str() );
        }
      else if (record.first == "Debug")
        {
        vtkSlicerApplication::GetInstance()->GetLogDialog()->GetLogWidget()->AddDebugRecord( record.second.c_str() );
        }
      }
    }

  if ((*this->InternalDisplayMessageQueue).size() > 0)
    {
    // schedule the next timer sooner in case there's stuff in the queue
    vtkKWTkUtilities::CreateTimerHandler(this, 5, this, "ProcessDisplayMessage");
    }
  else
    {
    // schedule the next timer for a while later
    vtkKWTkUtilities::CreateTimerHandler(this, 100, this, "ProcessDisplayMessage");
    }
}

//----------------------------------------------------------------------------
void
vtkSlicerApplication::WarningMessage(const char* message)
{
#ifdef _WIN32
  ::OutputDebugString(message);
#endif
#ifndef NDEBUG
  cout << message;
  cout.flush();
#endif

  this->RequestDisplayMessage("Warning", message);
}

//----------------------------------------------------------------------------
void
vtkSlicerApplication::ErrorMessage(const char* message)
{
#ifdef _WIN32
  ::OutputDebugString(message);
#endif
#ifndef NDEBUG
  cout << message;
  cout.flush();
#endif

  this->RequestDisplayMessage("Error", message);
}

//----------------------------------------------------------------------------
void
vtkSlicerApplication::DebugMessage(const char* message)
{
#ifdef _WIN32
  ::OutputDebugString(message);
#endif
#ifndef NDEBUG
  cout << message;
  cout.flush();
#endif

  this->RequestDisplayMessage("Debug", message);
}

//----------------------------------------------------------------------------
void
vtkSlicerApplication::InformationMessage(const char* message)
{
#ifdef _WIN32
  ::OutputDebugString(message);
#endif
#ifndef NDEBUG
  cout << message;
  cout.flush();
#endif

  this->RequestDisplayMessage("Information", message);
}

//----------------------------------------------------------------------------
void vtkSlicerApplication::AddAboutCopyrights(ostream &os)
{
  os << "Slicer is NOT an FDA approved medical device." << endl << endl;
  os << "See www.slicer.org for license details." << endl << endl;
  os << "See http://www.na-mic.org/Wiki/index.php/Slicer3:Acknowledgements" << endl << endl;
  os << "VTK http://www.vtk.org/copyright.php" << endl;
  os << "ITK http://www.itk.org/HTML/Copyright.htm" << endl;
  os << "KWWidgets http://www.kitware.com/Copyright.htm" << endl;
  os << "Tcl/Tk http://www.tcl.tk" << endl;
  os << "Teem:  http://teem.sf.net" << endl;
  os << "Supported by: NA-MIC, NAC, BIRN, NCIGT and the Slicer Community." << endl;
  os << "Special thanks to the NIH and our other supporters." << endl;
  os << "This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149. Information on the National Centers for Biomedical Computing can be obtained from http://nihroadmap.nih.gov/bioinformatics." << endl;

#if 0
  // example of the extra detail needed:
  //
     << tcl_major << "." << tcl_minor << "." << tcl_patch_level << endl
     << "  - Copyright (c) 1989-1994 The Regents of the University of "
     << "California." << endl
     << "  - Copyright (c) 1994 The Australian National University." << endl
     << "  - Copyright (c) 1994-1998 Sun Microsystems, Inc." << endl
     << "  - Copyright (c) 1998-2000 Ajuba Solutions." << endl;
#endif
}

//----------------------------------------------------------------------------
//  override default behavior of KWWidgets so that toplevel window 
//  can be on top of the tcl interactor (i.e. so it's not 'transient')
//
void vtkSlicerApplication::DisplayTclInteractor(vtkKWTopLevel *master)
{
  vtkKWTclInteractor *tcl_interactor = this->GetTclInteractor();
  if (tcl_interactor)
    {
    if (!master)
      {
      master = this->GetNthWindow(0);
      }
    if (master)
      {
      vtksys_stl::string title;
      if (master->GetTitle())
        {
        title += master->GetTitle();
        title += " : ";
        }
      title += ks_("Tcl Interactor Dialog|Title|Tcl Interactor");
      tcl_interactor->SetTitle(title.c_str());
      // The change:
      // tcl_interactor->SetMasterWindow(master);
      }
    tcl_interactor->Display();
    }
}

//----------------------------------------------------------------------------
//  override default behavior of KWWidgets so that toplevel window 
//  can be on top of the log dialog (i.e. so it's not 'transient')
//
void vtkSlicerApplication::DisplayLogDialog(vtkKWTopLevel* master)
{
  if (this->CreateLogDialog())
    {
    // The change:
    //this->LogDialog->SetMasterWindow(master ? master : this->GetNthWindow(0));
    this->LogDialog->Display();
    }
}


//----------------------------------------------------------------------------
void vtkSlicerApplication::SplashMessage (const char *message)
{
  if (this->GetUseSplashScreen())
    {
    this->GetSplashScreen()->SetProgressMessage(message);
    }
}

//----------------------------------------------------------------------------
void vtkSlicerApplication::SetUseWelcomeModuleAtStartup ( int val )
{
  if ( val != this->UseWelcomeModuleAtStartup )
    {
    if ( val == 0 || val ==1 )
      {
      this->UseWelcomeModuleAtStartup = val;
      }
    }
  return;
}


//----------------------------------------------------------------------------
void vtkSlicerApplication::SetEnableAsynchronousIO ( int val )
{
  if ( val != this->EnableAsynchronousIO )
    {
    if ( val == 0 || val == 1 )
      {
      this->EnableAsynchronousIO = val;
      this->ConfigureRemoteIOSettingsFromRegistry();
      }
    }
  return;
}

//----------------------------------------------------------------------------
void vtkSlicerApplication::SetEnableForceRedownload ( int val )
{
  if ( val != this->EnableForceRedownload )
    {
    if ( val == 0 || val == 1 )
      {
      this->EnableForceRedownload = val;
      this->ConfigureRemoteIOSettingsFromRegistry();    
      }
    }
  return;

}

//----------------------------------------------------------------------------
void vtkSlicerApplication::SetEnableRemoteCacheOverwriting ( int val )
{
  if ( val != this->EnableRemoteCacheOverwriting )
    {
    if ( val == 0 || val == 1 )
      {
      this->EnableRemoteCacheOverwriting = val;
      this->ConfigureRemoteIOSettingsFromRegistry();    
      }
    }
  return;
}

//----------------------------------------------------------------------------
void vtkSlicerApplication::SetRemoteCacheLimit ( int val )
{
  if ( val != this->RemoteCacheLimit )
    {
    if ( val > 0 )
      {
      this->RemoteCacheLimit = val;
      this->ConfigureRemoteIOSettingsFromRegistry();
      }
    }
  return;
}

//----------------------------------------------------------------------------
void vtkSlicerApplication::SetRemoteCacheFreeBufferSize ( int val )
{

  if ( val != this->RemoteCacheFreeBufferSize )
    {
    if ( val >= 0 )
      {
      this->RemoteCacheFreeBufferSize = val;
      this->ConfigureRemoteIOSettingsFromRegistry();
      }
    }
  return;
}




//----------------------------------------------------------------------------
void vtkSlicerApplication::SetRemoteCacheDirectory(const char* path)
{
  if (path)
    {
    if ( strlen(path) < vtkKWRegistryHelper::RegistryKeyValueSizeMax)
      {
      if ( strcmp ( this->RemoteCacheDirectory, path ) )
        {
        strcpy(this->RemoteCacheDirectory, path);
        //--- propagate this value through the GUI into MRML's CacheManager
        this->ConfigureRemoteIOSettingsFromRegistry();
        this->Modified();
        }
      }
    }
  return;
}

//----------------------------------------------------------------------------
const char* vtkSlicerApplication::GetRemoteCacheDirectory() const
{
  if (this->RemoteCacheDirectory != NULL )
    {
    // does the path exist?
    if (!itksys::SystemTools::MakeDirectory(this->RemoteCacheDirectory))
      {
      // error making sure that the dir exists
      std::cout << "vtkSlicerApplication::GetRemoteCacheDirectory: Unable to make remote cache directory: '" << this->RemoteCacheDirectory << "'\n\tYou can change the Remote Cache Directory under View->Application Settings->Remote Data Handling Settings." << std::endl;
      // pop up a window if we've got something to set for the parent
      if (this->ApplicationGUI && this->ApplicationGUI->GetViewerWidget())
        {
        std::string msg = std::string("ERROR\nUnable to make  remote cache directory: '") + std::string(this->RemoteCacheDirectory) + std::string("'\nYou can change the Remote Cache Directory under View->Application Settings->Remote Data Handling Settings.");
        vtkKWMessageDialog *message = vtkKWMessageDialog::New();
        message->SetParent(this->ApplicationGUI->GetViewerWidget());
        message->SetOptions(vtkKWMessageDialog::ErrorIcon);
        message->SetIcon();
        message->SetStyleToCancel();
        message->SetText(msg.c_str());
        message->Create();
        message->Invoke();
        message->Delete();
        }
      }
     else
      {
      // check that can write to it
      std::vector<std::string> tempPath;
      tempPath.push_back("");
      // no slash between first two elements
      tempPath.push_back(this->RemoteCacheDirectory);
      tempPath.push_back("testWrite.txt");
      std::string tempFile = itksys::SystemTools::JoinPath(tempPath);
      FILE *fp = fopen(tempFile.c_str(), "w");
      if (!fp)
        {
        std::cerr << "WARNING: Unable to write files in RemoteCacheDirectory: '" << this->RemoteCacheDirectory << "'" << std::endl;
        // pop up a window if we've got something to set for the parent
        if (this->ApplicationGUI && this->ApplicationGUI->GetViewerWidget())
          {
          std::string msg = std::string("WARNING\nUnable to write files in RemoteCacheDirectory:\n'") + std::string(this->RemoteCacheDirectory) + std::string("'");
          vtkKWMessageDialog *message = vtkKWMessageDialog::New();
          message->SetParent(this->ApplicationGUI->GetViewerWidget());
          message->SetOptions(vtkKWMessageDialog::ErrorIcon);
          message->SetIcon();
          message->SetStyleToCancel();
          message->SetText(msg.c_str());
          message->Create();
          message->Invoke();
          message->Delete();
          }
        }
      else
        {
        fclose(fp);
        // delete it
        remove(tempFile.c_str());
        }
      }
    }
  return this->RemoteCacheDirectory;
}


//----------------------------------------------------------------------------
void vtkSlicerApplication::SetExtensionsDownloadDirectory(const char* path)
{
  if (path)
    {
    if ( strlen(path) < vtkKWRegistryHelper::RegistryKeyValueSizeMax)
      {
      if ( strcmp ( this->ExtensionsDownloadDir, path ) )
        {
        strcpy(this->ExtensionsDownloadDir, path);
        //--- propagate this value through the GUI into MRML's CacheManager
        this->ConfigureRemoteIOSettingsFromRegistry();
        this->Modified();
        }
      }
    }
  return;
}


//----------------------------------------------------------------------------
const char* vtkSlicerApplication::GetExtensionsDownloadDirectory() const
{
  if (this->ExtensionsDownloadDir != NULL )
    {
    // does the path exist?
    if (!itksys::SystemTools::MakeDirectory(this->ExtensionsDownloadDir))
      {
      // error making sure that the dir exists
      std::cout << "vtkSlicerApplication::GetExtensionsDownloadDirectory: Unable to make extensions download directory: '" << this->ExtensionsDownloadDir << "'\n\tYou can change the Extensions Download Directory under View->Extensions Manager." << std::endl;
      // pop up a window if we've got something to set for the parent
      if (this->ApplicationGUI && this->ApplicationGUI->GetViewerWidget())
        {
        std::string msg = std::string("ERROR\nUnable to make extensions download directory: '") + std::string(this->ExtensionsDownloadDir) + std::string("'\nYou can change the Extensions Download Directory under View->Extensions Manager.");
        vtkKWMessageDialog *message = vtkKWMessageDialog::New();
        message->SetParent(this->ApplicationGUI->GetViewerWidget());
        message->SetOptions(vtkKWMessageDialog::ErrorIcon);
        message->SetIcon();
        message->SetStyleToCancel();
        message->SetText(msg.c_str());
        message->Create();
        message->Invoke();
        message->Delete();
        }
      }
     else
      {
      // check that can write to it
      std::vector<std::string> tempPath;
      tempPath.push_back("");
      // no slash between first two elements
      tempPath.push_back(this->ExtensionsDownloadDir);
      tempPath.push_back("testWrite.txt");
      std::string tempFile = itksys::SystemTools::JoinPath(tempPath);
      FILE *fp = fopen(tempFile.c_str(), "w");
      if (!fp)
        {
        std::cerr << "WARNING: Unable to write files in ExtensionsDownloadDirectory: '" << this->ExtensionsDownloadDir << "'" << std::endl;
        // pop up a window if we've got something to set for the parent
        if (this->ApplicationGUI && this->ApplicationGUI->GetViewerWidget())
          {
          std::string msg = std::string("WARNING\nUnable to write files in ExtensionsDownloadDirectory:\n'") + std::string(this->ExtensionsDownloadDir) + std::string("'");
          vtkKWMessageDialog *message = vtkKWMessageDialog::New();
          message->SetParent(this->ApplicationGUI->GetViewerWidget());
          message->SetOptions(vtkKWMessageDialog::ErrorIcon);
          message->SetIcon();
          message->SetStyleToCancel();
          message->SetText(msg.c_str());
          message->Create();
          message->Invoke();
          message->Delete();
          }
        }
      else
        {
        fclose(fp);
        // delete it
        remove(tempFile.c_str());
        }
      }
    }
  return this->ExtensionsDownloadDir;
}

//----------------------------------------------------------------------------
void vtkSlicerApplication::InitializePython(void* mod, void* dict)
{ 
  this->PythonModule = mod; 
  this->PythonDictionary = dict; 
}

//----------------------------------------------------------------------------
void* vtkSlicerApplication::GetPythonModule()
{ 
  return this->PythonModule; 
}

//----------------------------------------------------------------------------
void* vtkSlicerApplication::GetPythonDictionary()
{ 
  return this->PythonDictionary; 
}

//----------------------------------------------------------------------------
void vtkSlicerApplication::StringToArray(std::string s, char separator, vtkStringArray *array)
{ 
  std::string sr = s;
  std::replace( sr.begin(), sr.end(), ' ', '?' );
  std::replace( sr.begin(), sr.end(), separator, ' ' );
  std::istringstream stream( sr );
  for( ;; )
    {
    std::string word;
    if( !( stream >> word ) ) 
      { 
      break; 
      }
    std::replace( word.begin(), word.end(), '?', ' ');
    array->InsertNextValue(word);
    }
}

void vtkSlicerApplication::ArrayToString(vtkStringArray *array, std::string sep, char *string, int maxLength )
{ 
  int len=0;
  std::string res = "";
  for (int i=0; i<array->GetNumberOfValues(); i++)
    {
    if (len >= maxLength)
      {
      break;
      }
    std::string s(array->GetValue(i).c_str());
    if (i) 
      {
      res = res + sep;
      }
    res = res + s;
    len += (int)(s.size()+1);
    }
  strcpy(string, res.c_str());
}

//----------------------------------------------------------------------------
void vtkSlicerApplication::PrintSelf ( ostream& os, vtkIndent indent )
{
  this->Superclass::PrintSelf ( os, indent );

  vtkIndent nextIndent = indent.GetNextIndent();
  
  os << indent << "Registry:\n";
  os << nextIndent << "ConfirmDelete: " << ConfirmDelete << "\n";
  os << nextIndent << "ModulePaths: " << ModulePaths << "\n";
  os << nextIndent << "ColorFilePaths: " << ColorFilePaths << "\n";
  os << nextIndent << "PotentialModulePaths: " << PotentialModulePaths << "\n";
  os << nextIndent << "PotentialColorFilePaths: " << PotentialColorFilePaths << "\n";
  os << nextIndent << "ModuleCachePath: " << ModuleCachePath << "\n";
  os << nextIndent << "WebBrowser: " << WebBrowser << "\n";
  os << nextIndent << "Unzip: " << Unzip << "\n";
  os << nextIndent << "Zip: " << Zip << "\n";
  os << nextIndent << "Rm: " << Rm << "\n";
  os << nextIndent << "TemporaryDirectory: " << TemporaryDirectory << "\n";
  os << nextIndent << "HomeModule: " << HomeModule << "\n";
  os << nextIndent << "ApplicationFontSize: " << ApplicationFontSize << "\n";
  os << nextIndent << "ApplicationFontFamily: " << ApplicationFontFamily << "\n";
  os << nextIndent << "IgnoreModuleNames: " << IgnoreModuleNames << "\n";

  os << nextIndent << "ApplicationWindowWidth: " << ApplicationWindowWidth << "\n";
  os << nextIndent << "ApplicationWindowHeight: " << ApplicationWindowHeight << "\n";
  os << nextIndent << "ApplicationSlicesFrameHeight: " << ApplicationSlicesFrameHeight << "\n";
  os << nextIndent << "ApplicationLayoutType: " << ApplicationLayoutType << "\n";

  os << nextIndent << "RegistryHolder: " << RegistryHolder << "\n";

  os << indent << "Ignore Modules: \n";
  for (int i = 0; i < IgnoreModules->GetNumberOfValues(); i++)
    {
    os << nextIndent << i << " " << IgnoreModules->GetValue(i).c_str() << "\n";
    }
  os << indent << "LoadableModules: \n";
  for (int i = 0; i < LoadableModules->GetNumberOfValues(); i++)
    {
    os << nextIndent << i << " " << LoadableModules->GetValue(i).c_str() << "\n";
    }

  os << indent << "Flags: \n";
  os << nextIndent << "LoadModules: " << LoadModules << "\n";
  os << nextIndent << "LoadCommandLineModules: " << LoadCommandLineModules <<  "\n";
  os << nextIndent << "EnableDaemon: " << EnableDaemon <<  "\n";

  os << indent << "Remote I/O:\n";
  os << nextIndent << "EnableAsynchronousIO: " << EnableAsynchronousIO<< "\n";
  os << nextIndent << "EnableForceRedownload: " << EnableForceRedownload <<  "\n";
  os << nextIndent << "EnableRemoteCacheOverwriting: " << EnableRemoteCacheOverwriting <<  "\n";
  os << nextIndent << "RemoteCacheDirectory: " << RemoteCacheDirectory <<  "\n";
  os << nextIndent << "RemoteCacheLimit: " << RemoteCacheLimit <<  "\n";
  os << nextIndent << "RemoteCacheFreeBufferSize: " << RemoteCacheFreeBufferSize <<  "\n";

  os << nextIndent << "UseWelcomeModuleAtStartup: " << UseWelcomeModuleAtStartup << "\n";
  os << indent << "UseSplashScreen: " << UseSplashScreen <<  "\n";
  os << indent << "StereoEnabled: " << StereoEnabled <<  "\n";
}
