#include <sstream>
#include "vtkObjectFactory.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerGUICollection.h"
#include "vtkSlicerModuleGUI.h"
#include "vtkKWNotebook.h"
#include "vtkKWFrame.h"
#include "vtkKWUserInterfacePanel.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWWindowBase.h"
#include "vtkKwTkUtilities.h"

#ifdef WIN32
#include "vtkKWWin32RegistryHelper.h"
#endif

#include "itksys/SystemTools.hxx"

#include <queue>

#include "vtkSlicerTask.h"

const char *vtkSlicerApplication::ModulePathRegKey = "ModulePath";
const char *vtkSlicerApplication::TemporaryDirectoryRegKey = "TemporaryDirectory";
const char *vtkSlicerApplication::ConfirmDeleteRegKey = "ConfirmDelete";

//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkSlicerApplication);
vtkCxxRevisionMacro(vtkSlicerApplication, "$Revision: 1.0 $");

class ProcessingTaskQueue : public std::queue<vtkSmartPointer<vtkSlicerTask> > {};
class ModifiedQueue : public std::queue<vtkSmartPointer<vtkObject> > {};

//---------------------------------------------------------------------------
vtkSlicerApplication::vtkSlicerApplication ( ) {

    strcpy(this->ModulePath, "");

    strcpy(this->ConfirmDelete, "");
    
    // configure the application before creating
    this->SetName ( "3D Slicer Version 3.0 Alpha" );

#ifdef _WIN32
    vtkKWWin32RegistryHelper *regHelper = 
        vtkKWWin32RegistryHelper::SafeDownCast( this->GetRegistryHelper() );
    regHelper->SetOrganization("NA-MIC");
#endif

    this->RestoreApplicationSettingsFromRegistry ( );
    this->SetHelpDialogStartingPage ( "http://www.slicer.org" );

    this->ModuleGUICollection = vtkSlicerGUICollection::New ( );
    vtkKWFrameWithLabel::SetDefaultLabelFontWeightToNormal( );
    this->MainLayout = vtkSlicerGUILayout::New ( );
    this->SlicerTheme = vtkSlicerTheme::New ( );

    this->ProcessingThreader = itk::MultiThreader::New();
    this->ProcessingThreadId = -1;
    this->ProcessingThreadActive = false;
    this->ProcessingThreadActiveLock = itk::MutexLock::New();
    this->ProcessingTaskQueueLock = itk::MutexLock::New();

    this->ModifiedQueueActive = false;
    this->ModifiedQueueActiveLock = itk::MutexLock::New();
    this->ModifiedQueueLock = itk::MutexLock::New();
    
    this->InternalTaskQueue = new ProcessingTaskQueue;
    this->InternalModifiedQueue = new ModifiedQueue;
    
}



//---------------------------------------------------------------------------
vtkSlicerApplication::~vtkSlicerApplication ( ) {

    if ( this->MainLayout )
      {
      this->MainLayout->Delete ( );
      this->MainLayout = NULL;
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

    // Note that TerminateThread does not kill a thread, it only waits
    // for the thread to finish.  We need to signal the thread that we
    // want to terminate
    if (this->ProcessingThreadId != -1 && this->ProcessingThreader)
      {
      // Signal the processingThread that we are terminating. 
      this->ProcessingThreadActiveLock->Lock();
      this->ProcessingThreadActive = false;
      this->ProcessingThreadActiveLock->Unlock();
      
      // Wait for the thread to finish and clean up the state of the threader
      this->ProcessingThreader->TerminateThread( this->ProcessingThreadId );
      
      this->ProcessingThreadId = -1;
      }
    
    delete this->InternalTaskQueue;
    this->InternalTaskQueue = 0;
    
    delete this->InternalModifiedQueue;
    this->InternalModifiedQueue = 0;
}


//---------------------------------------------------------------------------
const char *vtkSlicerApplication::Evaluate(const char *expression) {
    return (this->Script(expression));
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


//----------------------------------------------------------------------------
void vtkSlicerApplication::RestoreApplicationSettingsFromRegistry()
{
  // Make a good guess before we read from the registry.  Default to a
  // subdirectory called Slicer3 in a standard temp location.
#ifdef _WIN32
  GetTempPath(vtkKWRegistryHelper::RegistryKeyValueSizeMax,
              this->TemporaryDirectory);
#else
  strcpy(this->TemporaryDirectory, "/usr/tmp");
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
  pathcomponents.push_back("Slicer3");
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

    
  Superclass::RestoreApplicationSettingsFromRegistry();

  if (this->HasRegistryValue(
    2, "RunTime", vtkSlicerApplication::ConfirmDeleteRegKey))
    {
    this->GetRegistryValue(
      2, "RunTime", vtkSlicerApplication::ConfirmDeleteRegKey,
      this->ConfirmDelete);
    }
  
  if (this->HasRegistryValue(
    2, "RunTime", vtkSlicerApplication::ModulePathRegKey))
    {
    this->GetRegistryValue(
      2, "RunTime", vtkSlicerApplication::ModulePathRegKey,
      this->ModulePath);
    }

  if (this->HasRegistryValue(
    2, "RunTime", vtkSlicerApplication::TemporaryDirectoryRegKey))
    {
    this->GetRegistryValue(
      2, "RunTime", vtkSlicerApplication::TemporaryDirectoryRegKey,
      this->TemporaryDirectory);
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
    2, "RunTime", vtkSlicerApplication::ModulePathRegKey, "%s", 
    this->ModulePath);

  this->SetRegistryValue(
    2, "RunTime", vtkSlicerApplication::TemporaryDirectoryRegKey, "%s", 
    this->TemporaryDirectory);
}

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

const char *vtkSlicerApplication::GetConfirmDelete() const
{
    return this->ConfirmDelete;
}

void vtkSlicerApplication::SetModulePath(const char* path)
{
  if (path)
    {
    if (strcmp(this->ModulePath, path) != 0
        && strlen(path) < vtkKWRegistryHelper::RegistryKeyValueSizeMax)
      {
      strcpy(this->ModulePath, path);
      this->Modified();
      }
    }
}

const char* vtkSlicerApplication::GetModulePath() const
{
  return this->ModulePath;
}


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

const char* vtkSlicerApplication::GetTemporaryDirectory() const
{
  return this->TemporaryDirectory;
}


//----------------------------------------------------------------------------
void vtkSlicerApplication::CreateProcessingThread()
{
  if (this->ProcessingThreadId == -1)
    {
    this->ProcessingThreadActiveLock->Lock();
    this->ProcessingThreadActive = true;
    this->ProcessingThreadActiveLock->Unlock();
    
    this->ProcessingThreadId
      = this->ProcessingThreader
      ->SpawnThread(vtkSlicerApplication::ProcessingThreaderCallback,
                    this);

    // Setup the communication channel back to the main thread
    this->ModifiedQueueActiveLock->Lock();
    this->ModifiedQueueActive = true;
    this->ModifiedQueueActiveLock->Unlock();

    vtkKWTkUtilities::CreateTimerHandler(this, 100, this, "ProcessModified");
    }

}

//----------------------------------------------------------------------------
void vtkSlicerApplication::TerminateProcessingThread()
{
  if (this->ProcessingThreadId != -1)
    {
    this->ModifiedQueueActiveLock->Lock();
    this->ModifiedQueueActive = false;
    this->ModifiedQueueActiveLock->Unlock();

    this->ProcessingThreadActiveLock->Lock();
    this->ProcessingThreadActive = false;
    this->ProcessingThreadActiveLock->Unlock();

    this->ProcessingThreader->TerminateThread( this->ProcessingThreadId );

    this->ProcessingThreadId = -1;
    }
}


ITK_THREAD_RETURN_TYPE
vtkSlicerApplication
::ProcessingThreaderCallback( void *arg )
{
  // pull out the reference to the app
  vtkSlicerApplication *app
    = (vtkSlicerApplication*)
    (((itk::MultiThreader::ThreadInfoStruct *)(arg))->UserData);

  // Tell the app to start processing any tasks slated for the
  // processing thread
  app->ProcessTasks();

  return ITK_THREAD_RETURN_VALUE;
}

void vtkSlicerApplication::ProcessTasks()
{
  bool active = true;
  vtkSmartPointer<vtkSlicerTask> task = 0;
  
  while (active)
    {
    // Check to see if we should be shutting down
    this->ProcessingThreadActiveLock->Lock();
    active = this->ProcessingThreadActive;
    this->ProcessingThreadActiveLock->Unlock();

    if (active)
      {
      // pull a task off the queue
      this->ProcessingTaskQueueLock->Lock();
      if ((*this->InternalTaskQueue).size() > 0)
        {
        std::cout << "Number of queued tasks: " << (*this->InternalTaskQueue).size() << std::endl;
        task = (*this->InternalTaskQueue).front();
        (*this->InternalTaskQueue).pop();
        }
      this->ProcessingTaskQueueLock->Unlock();
      
      // process the task (should this be in a separate thread?)
      if (task)
        {
        task->Execute();
        task = 0;
        }
      }

    // busy wait
    itksys::SystemTools::Delay(100);
    }
}

bool vtkSlicerApplication::ScheduleTask( vtkSlicerTask *task )
{
  bool active;

  std::cout << "Scheduling a task ";
  // only schedule a task if the processing task is up
  this->ProcessingThreadActiveLock->Lock();
  active = this->ProcessingThreadActive;
  this->ProcessingThreadActiveLock->Unlock();

  if (active)
    {
    this->ProcessingTaskQueueLock->Lock();
    (*this->InternalTaskQueue).push( task );
    std::cout << (*this->InternalTaskQueue).size() << std::endl;
    this->ProcessingTaskQueueLock->Unlock();
    
    return true;
    }

  // could not schedule the task
  return false;
}


bool vtkSlicerApplication::ScheduleModified( vtkObject *obj )
{
  bool active;

  std::cout << "Scheduling a modified on " << obj;

  // only schedule a Modified if the Modified queue is up
  this->ModifiedQueueActiveLock->Lock();
  active = this->ModifiedQueueActive;
  this->ModifiedQueueActiveLock->Unlock();

  if (active)
    {
    this->ModifiedQueueLock->Lock();
    (*this->InternalModifiedQueue).push( obj );
    std::cout << (*this->InternalModifiedQueue).size() << std::endl;
    this->ModifiedQueueLock->Unlock();
    
    return true;
    }

  // could not schedule the Modified
  return false;
}


void vtkSlicerApplication::ProcessModified()
{
  bool active = true;
  vtkSmartPointer<vtkObject> obj = 0;
  
  // Check to see if we should be shutting down
  this->ModifiedQueueActiveLock->Lock();
  active = this->ModifiedQueueActive;
  this->ModifiedQueueActiveLock->Unlock();
  
  if (active)
    {
    // pull an object off the queue to modify
    this->ModifiedQueueLock->Lock();
    if ((*this->InternalModifiedQueue).size() > 0)
      {
      obj = (*this->InternalModifiedQueue).front();
      (*this->InternalModifiedQueue).pop();
      }
    this->ModifiedQueueLock->Unlock();
    
    // Modify the object
    if (obj)
      {
      obj->Modified();
      obj = 0;
      }
    }
  
  // schedule the next timer
  vtkKWTkUtilities::CreateTimerHandler(this, 100, this, "ProcessModified");
}

