#include <string>
#include <iostream>
#include <sstream>

#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkCommand.h"

#include "vtkFetchMIGUI.h"
#include "vtkFetchMIServer.h"
#include "vtkFetchMIServerCollection.h"

#include "vtkSlicerApplication.h"
#include "vtkSlicerApplicationLogic.h"
#include "vtkSlicerNodeSelectorWidget.h"
#include "vtkSlicerPopUpHelpWidget.h"
#include "vtkSlicerGUILayout.h"
#include "vtkSlicerWaitMessageWidget.h"

#include "vtkMRMLStorageNode.h"
#include "vtkMRMLVolumeNode.h"
#include "vtkMRMLModelStorageNode.h"
#include "vtkMRMLFreeSurferModelStorageNode.h"
#include "vtkMRMLNRRDStorageNode.h"
#include "vtkMRMLVolumeArchetypeStorageNode.h"
#include "vtkMRMLDiffusionTensorVolumeNode.h"
#include "vtkMRMLDiffusionWeightedVolumeNode.h"
#include "vtkMRMLScalarVolumeNode.h"
#include "vtkDataIOManager.h"

#include "vtkXNDHandler.h"
#include "vtkXNDTagTable.h"
#include "vtkHIDTagTable.h"
#include "vtkTagTable.h"

#include "vtkKWApplication.h"
#include "vtkKWWidget.h"
#include "vtkKWEvent.h"
#include "vtkKWEntry.h"
#include "vtkKWEntryWithLabel.h"
#include "vtkKWLabel.h"
#include "vtkKWMenu.h"
#include "vtkKWMenuButton.h"
#include "vtkKWMenuButtonWithLabel.h"
#include "vtkKWFrame.h"
#include "vtkKWSplitFrame.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWPushButton.h"
#include "vtkKWMessageDialog.h"
#include "vtkKWTopLevel.h"
#include "vtkKWTkUtilities.h"
#include "vtkFetchMIIcons.h"
#include "vtkKWMultiColumnList.h"
#include "vtkKWMultiColumnListWithScrollbars.h"
#include "vtkFetchMIQueryTermWidget.h"
#include "vtkFetchMIFlatResourceWidget.h"
#include "vtkFetchMIResourceUploadWidget.h"
#include "vtkFetchMITagViewWidget.h"
#include "vtkKWNotebook.h"

#include <map>
#include <string>
#include <vector>
#include <iterator>
#include <sstream>

#define _br 0.945
#define _bg 0.945
#define _bb 0.975

#define _fr 0.75
#define _fg 0.75
#define _fb 0.75

//----------------------------------------------------------------------------
//--- a word about language:
//--- Methods and vars in this module assume that:
//--- "Tag" means a metadata element comprised of an "attribute" (or "keyword") and "value".
//--- Tags may have an attribute with many possible values.
//--- Sometimes "Tag" is used to mean "attribute".
//--- we'll change this eventually to be "Tagname"
//----------------------------------------------------------------------------


//------------------------------------------------------------------------------
vtkFetchMIGUI* vtkFetchMIGUI::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkFetchMIGUI");
  if(ret)
    {
      return (vtkFetchMIGUI*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkFetchMIGUI;
}


//----------------------------------------------------------------------------
vtkFetchMIGUI::vtkFetchMIGUI()
{
  this->Logic = NULL;
  this->FetchMINode = NULL;
  this->QueryList = NULL;
  this->ResourceList = NULL;
  this->TaggedDataList = NULL;
  this->AddServerButton = NULL;
  this->CloseNewServerButton = NULL;
  this->NewServerWindow = NULL;
  this->ServerMenuButton = NULL;
  this->AddServerEntry = NULL;
  this->FetchMIIcons = NULL;
  this->FetchMINode = NULL;
  this->UpdatingGUI = 0;
  this->UpdatingMRML = 0;
  this->DataDirectoryName = NULL;
  this->TagViewer = NULL;
  this->Notebook = NULL;
  this->SetGUIWidth(-1);
  
//  this->DebugOn();
}

//----------------------------------------------------------------------------
vtkFetchMIGUI::~vtkFetchMIGUI()
{
    this->RemoveMRMLNodeObservers ( );
    this->RemoveLogicObservers ( );
    this->SetGUIWidth(-1);
    
    if ( this->QueryList )
      {
      this->QueryList->SetParent ( NULL );
      this->QueryList->SetApplication ( NULL );
      this->QueryList->Delete();
      this->QueryList = NULL;
      }
    if ( this->ResourceList )
      {
      this->ResourceList->SetParent ( NULL );
      this->ResourceList->SetApplication ( NULL );
      if ( this->ResourceList->GetMRMLScene() != NULL )
        {
        this->ResourceList->SetMRMLScene ( NULL );
        }
      this->ResourceList->Delete();
      this->ResourceList = NULL;
      }
    if ( this->TaggedDataList )
      {
      this->TaggedDataList->SetParent ( NULL );
      this->TaggedDataList->SetApplication ( NULL );
      if ( this->TaggedDataList->GetMRMLScene() != NULL )
        {
        this->TaggedDataList->SetMRMLScene ( NULL );
        }
      this->TaggedDataList->Delete();
      this->TaggedDataList = NULL;
      }
    if ( this->AddServerButton )
      {
      this->AddServerButton->SetParent ( NULL );
      this->AddServerButton->Delete();
      this->AddServerButton = NULL;
      }
    if ( this->CloseNewServerButton )
      {
      this->CloseNewServerButton->SetParent ( NULL );
      this->CloseNewServerButton->Delete();
      this->CloseNewServerButton = NULL;
      }
    if ( this->NewServerWindow)
      {
      this->UnBindNewServerWindow();
      this->NewServerWindow->SetParent ( NULL );
      this->NewServerWindow->Delete();
      this->NewServerWindow = NULL;
      }
    if ( this->AddServerEntry )
      {
      this->AddServerEntry->SetParent ( NULL );
      this->AddServerEntry->Delete();
      this->AddServerEntry = NULL;
      }
    if ( this->ServerMenuButton )
      {
      this->ServerMenuButton->SetParent ( NULL );
      this->ServerMenuButton->Delete();
      this->ServerMenuButton = NULL;
      }
    if ( this->FetchMIIcons )
      {
      this->FetchMIIcons->Delete();
      this->FetchMIIcons = NULL;
      }
    if ( this->TagViewer)
      {
      this->TagViewer->SetParent ( NULL );
      this->TagViewer->Delete();
      this->TagViewer = NULL;
      }
    if ( this->Notebook )
      {
      this->Notebook->SetParent ( NULL );
      this->Notebook->Delete();
      this->Notebook = NULL;
      }

    this->UpdatingMRML = 0;
    this->UpdatingGUI = 0;
    this->SetDataDirectoryName ( NULL );

    this->Logic = NULL;
    vtkSetAndObserveMRMLNodeMacro( this->FetchMINode, NULL );
}

//----------------------------------------------------------------------------
void vtkFetchMIGUI::Enter()
{
  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
  if ( app )
    {
    vtkSlicerGUILayout *geom = app->GetDefaultGeometry ( );
    if ( geom )
      {
      vtkSlicerApplicationGUI *appGUI = this->GetApplicationGUI();
      if ( appGUI )
        {
        if ( appGUI->GetMainSlicerWindow() )
          {
          this->SetGUIWidth (appGUI->GetMainSlicerWindow()->GetMainSplitFrame()->GetFrame1Size ());
          appGUI->GetMainSlicerWindow()->GetMainSplitFrame()->SetFrame1Size (geom->GetDefaultGUIPanelWidth() * 1.75 );
          }
        }
      }
    }
}


//----------------------------------------------------------------------------
void vtkFetchMIGUI::Exit ( )
{
  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
  if ( app )
    {
    vtkSlicerGUILayout *geom = app->GetDefaultGeometry ( );
    if ( geom )
      {
      vtkSlicerApplicationGUI *appGUI = this->GetApplicationGUI();
      if ( appGUI )
        {
        if ( appGUI->GetMainSlicerWindow() )
          {
          if ( this->GUIWidth < 0 )
            {
            appGUI->GetMainSlicerWindow()->GetMainSplitFrame()->SetFrame1Size (geom->GetDefaultGUIPanelWidth() );
            }
          else
            {
            // restore.
            appGUI->GetMainSlicerWindow()->GetMainSplitFrame()->SetFrame1Size (this->GUIWidth );
            }
          }
        }
      }
    }
}


//----------------------------------------------------------------------------
void vtkFetchMIGUI::TearDownGUI ( )
{
  if ( this->FetchMINode )
    {
    vtkSetAndObserveMRMLNodeMacro( this->FetchMINode, NULL );
    }
  if ( this->TagViewer )
    {
    this->TagViewer->UnBind();
    }
  this->QueryList->RemoveWidgetObservers();
  this->ResourceList->RemoveWidgetObservers();
  this->TaggedDataList->RemoveWidgetObservers();
  this->RemoveGUIObservers ( );
  this->Logic->SetFetchMINode ( NULL );
  this->SetLogic ( NULL );
  this->ResourceList->SetMRMLScene( NULL );
  this->TaggedDataList->SetMRMLScene ( NULL );
  this->DestroyNewServerWindow();
  this->SetAndObserveMRMLScene ( NULL );

}


//----------------------------------------------------------------------------
void vtkFetchMIGUI::PrintSelf(ostream& os, vtkIndent indent)
{
  
}

//---------------------------------------------------------------------------
void vtkFetchMIGUI::LoadTclPackage ( )
{
  if (!this->GetLogic())
    {
    return;
    }
  std::string qaTclCommand;
  if (0)
    {
  std::string dir(this->GetLogic()->GetModuleShareDirectory());
  qaTclCommand =  "set ::FETCHMI_PACKAGE {}; ";
  qaTclCommand += "package forget FetchMI; ";
  qaTclCommand += "  set dir \"" + dir + "\";";
  qaTclCommand += "  if { [ file exists \"$dir/Tcl/pkgIndex.tcl\" ] } { ";
  qaTclCommand += "    lappend ::auto_path $dir; ";
  qaTclCommand += "    package require FetchMI ";
  qaTclCommand += "  }";
    }
  else
    {
    // it's not a package that defines a tcl scripted module, just a file with
    // some helper procs in it, so just source it
    std::string dir(this->GetLogic()->GetModuleLibDirectory());
    qaTclCommand = "  set dir \"" + dir + "\";";
    qaTclCommand += " if  { [ file exists \"$dir/Tcl/FetchMIXMLUtilities.tcl\" ] } { ";
    qaTclCommand += " source \"$dir/Tcl/FetchMIXMLUtilities.tcl\"";
    qaTclCommand += "  }";
    }
  this->Script ( qaTclCommand.c_str() ); 
}

//---------------------------------------------------------------------------
void vtkFetchMIGUI::AddGUIObservers ( ) 
{
  this->QueryList->AddWidgetObservers();
  this->QueryList->AddObserver(vtkFetchMIQueryTermWidget::TagChangedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->QueryList->AddObserver(vtkFetchMIQueryTermWidget::QuerySubmittedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->ResourceList->AddObserver (vtkFetchMIFlatResourceWidget::DeleteResourceEvent, (vtkCommand *)this->GUICallbackCommand );
  this->ResourceList->AddWidgetObservers();
  this->TaggedDataList->AddObserver(vtkFetchMIResourceUploadWidget::TagSelectedDataEvent, (vtkCommand *)this->GUICallbackCommand);
  this->TaggedDataList->AddObserver(vtkFetchMIResourceUploadWidget::RemoveTagSelectedDataEvent, (vtkCommand *)this->GUICallbackCommand);
  this->TaggedDataList->AddObserver(vtkFetchMIResourceUploadWidget::ShowAllTagViewEvent, (vtkCommand *)this->GUICallbackCommand);
  this->TaggedDataList->AddObserver(vtkFetchMIResourceUploadWidget::ShowSelectionTagViewEvent, (vtkCommand *)this->GUICallbackCommand);
  this->TaggedDataList->AddObserver(vtkFetchMIResourceUploadWidget::UploadRequestedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->TaggedDataList->AddWidgetObservers();
  this->ServerMenuButton->GetMenu()->AddObserver ( vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->Notebook->AddObserver ( vtkKWEvent::NotebookRaisePageEvent, (vtkCommand *)this->GUICallbackCommand );
  this->Notebook->AddObserver ( vtkKWEvent::NotebookShowPageEvent, (vtkCommand *)this->GUICallbackCommand );

}



//---------------------------------------------------------------------------
void vtkFetchMIGUI::RemoveGUIObservers ( )
{
  this->QueryList->RemoveWidgetObservers();
  this->QueryList->RemoveObservers(vtkFetchMIQueryTermWidget::TagChangedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->QueryList->RemoveObservers(vtkFetchMIQueryTermWidget::QuerySubmittedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->ResourceList->RemoveObservers(vtkFetchMIFlatResourceWidget::DeleteResourceEvent, (vtkCommand *)this->GUICallbackCommand );
  this->ResourceList->RemoveWidgetObservers();
  this->TaggedDataList->RemoveObservers(vtkFetchMIResourceUploadWidget::TagSelectedDataEvent, (vtkCommand *)this->GUICallbackCommand);
  this->TaggedDataList->RemoveObservers(vtkFetchMIResourceUploadWidget::RemoveTagSelectedDataEvent, (vtkCommand *)this->GUICallbackCommand);
  this->TaggedDataList->RemoveObservers(vtkFetchMIResourceUploadWidget::ShowAllTagViewEvent, (vtkCommand *)this->GUICallbackCommand);
  this->TaggedDataList->RemoveObservers(vtkFetchMIResourceUploadWidget::UploadRequestedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->TaggedDataList->RemoveWidgetObservers();
  this->ServerMenuButton->GetMenu()->RemoveObservers (vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->Notebook->RemoveObservers ( vtkKWEvent::NotebookRaisePageEvent, (vtkCommand *)this->GUICallbackCommand );
  this->Notebook->RemoveObservers ( vtkKWEvent::NotebookShowPageEvent, (vtkCommand *)this->GUICallbackCommand );

}


//---------------------------------------------------------------------------
vtkIntArray* vtkFetchMIGUI::NewObservableEvents()
{
  vtkIntArray* events = vtkIntArray::New();
  events->InsertNextValue(vtkMRMLScene::NewSceneEvent);
  events->InsertNextValue(vtkMRMLScene::SceneCloseEvent);
  events->InsertNextValue(vtkMRMLScene::NodeAddedEvent);
  events->InsertNextValue(vtkMRMLScene::NodeRemovedEvent);
  return events;
}


//---------------------------------------------------------------------------
void vtkFetchMIGUI::RemoveMRMLNodeObservers ( ) {
    // Fill in.
}

//---------------------------------------------------------------------------
void vtkFetchMIGUI::RemoveLogicObservers ( ) {
    // Fill in
}





//---------------------------------------------------------------------------
void vtkFetchMIGUI::ProcessGUIEvents ( vtkObject *caller,
                                           unsigned long event,
                                           void *callData ) 
{

  if ( this->FetchMINode == NULL )
    {
    //TODO: error macro
    return;
    }
  if ( this->Logic == NULL )
    {
    //TODO: error macro
    return;
    }

  vtkKWPushButton *b = vtkKWPushButton::SafeDownCast ( caller );
  vtkKWEntry *e = vtkKWEntry::SafeDownCast ( caller );
  vtkKWMenu *m = vtkKWMenu::SafeDownCast ( caller );
  vtkFetchMIResourceUploadWidget *w = vtkFetchMIResourceUploadWidget::SafeDownCast ( caller );
  vtkFetchMIFlatResourceWidget *f = vtkFetchMIFlatResourceWidget::SafeDownCast ( caller );
  vtkFetchMIQueryTermWidget *q= vtkFetchMIQueryTermWidget::SafeDownCast ( caller );

  if ( f != NULL )
    {
    if ( (f == this->ResourceList) && (event == vtkFetchMIFlatResourceWidget::DeleteResourceEvent) )
      {
      vtkKWMessageDialog *message = vtkKWMessageDialog::New();
      message->SetParent ( this->GetApplicationGUI()->GetMainSlicerWindow() );
      message->SetStyleToOkCancel();
      std::string msg = "This action will delete the selected resources from the selected server and is NOT undoable. \n Do you want to continue?";
      message->SetText(msg.c_str());
      message->Create();
      int ok = message->Invoke();
      message->Delete(); 
      if (ok)
        {
        this->DeleteSelectedResourcesFromServer();
        }
      }
    }
  
  if ( w != NULL )
    {
    if ( (w== this->TaggedDataList) && (event == vtkFetchMIResourceUploadWidget::TagSelectedDataEvent) )
      {
      this->TagSelectedData();
      }
    if ( (w== this->TaggedDataList) && (event == vtkFetchMIResourceUploadWidget::RemoveTagSelectedDataEvent) )
      {
      this->RemoveTagFromSelectedData();
      }
    else if ( (w== this->TaggedDataList) && (event == vtkFetchMIResourceUploadWidget::ShowSelectionTagViewEvent) )
      {
      this->ShowSelectionTagView();
      }
    else if ( (w== this->TaggedDataList) && (event == vtkFetchMIResourceUploadWidget::ShowAllTagViewEvent) )
      {
      this->ShowAllTagView();
      }
    else if ( (w== this->TaggedDataList) && (event == vtkFetchMIResourceUploadWidget::UploadRequestedEvent) )
      {
      if ( this->MRMLScene->GetDataIOManager() )
        {
        this->MRMLScene->GetDataIOManager()->InvokeEvent ( vtkDataIOManager::DisplayManagerWindowEvent );
        }
      //--- try to post a message....
      if ( this->GetApplication() )
        {
        vtkSlicerApplication* app = vtkSlicerApplication::SafeDownCast(this->GetApplication() );
        if ( app )
          {
          vtkSlicerApplicationGUI *appGUI = app->GetApplicationGUI();
          if ( appGUI )
            {
            if (appGUI->GetMainSlicerWindow() )
              {
              vtkSlicerWaitMessageWidget *wm = vtkSlicerWaitMessageWidget::New();
              wm->SetParent ( appGUI->GetMainSlicerWindow() );
              wm->Create();
              wm->SetText ("Uploading scene and its resources (may take a little while)...");
              wm->DisplayWindow();
              this->SetStatusText ("Uploading scene and its resources ( may take a little while).");
              this->Logic->RequestResourceUpload();
              wm->SetText ("Uploading scene and its resources (may take a little while)... done.");
              wm->WithdrawWindow();
              wm->Delete();
              this->SetStatusText ("");
              }
            }
          }
        }
      else
        {
        this->Logic->RequestResourceUpload();
        }
      }
    }
/*
  if ( n != NULL )
    {
    if ( (n == this->Notebook) && ((event == vtkKWEvent::NotebookRaisePageEvent) ||
                                   (event == vtkKWEvent::NotebookShowPageEvent)) )
      {
      int id = this->Notebook->GetRaisedPageId();
      //--- grey all pages fg color down
      this->Notebook->SetForegroundColor ( _fr, _fg, _fb);
      //--- then highlight the raised page.
      vtkKWFrame *nf = this->Notebook->GetFrame (id );
      if ( nf )
        {
        nf->SetForegroundColor ( 0.0, 0.0, 0.0);
        }
      }
    }
*/

  if ( q != NULL )
    {
    if ( (q== this->QueryList) && (event == vtkFetchMIQueryTermWidget::TagChangedEvent) )
      {
      this->UpdateTagTableFromGUI();
      }
    if ( (q== this->QueryList) && (event == vtkFetchMIQueryTermWidget::QuerySubmittedEvent) )
      {
      this->ResourceList->DeleteAllItems();
      vtkTagTable *t = this->FetchMINode->GetResourceDescription();
      t->ClearTagTable();
      }    
    }


  //--- add a new server if its name is not null or empty.
  if ( e && event == vtkKWEntry::EntryValueChangedEvent )
    {
    if (e == this->AddServerEntry )
      {
      if ( e->GetValue() != NULL )
        {
        if ( strcmp ( e->GetValue(), "" ) )
          {
          //--- TODO: when more service types are availabe, build out way to get them
          this->Logic->AddNewServer(e->GetValue(), "XND", "XND", "XND");
          this->FetchMINode->SetServer (e->GetValue() );
          e->SetValue ( "" );
          this->ResourceList->DeleteAllItems();
          }
        }
      }
    }
  if ( b && event == vtkKWPushButton::InvokedEvent )
    {
    if ( b == this->AddServerButton )
      {
      if ( this->GetAddServerEntry()->GetValue() != NULL )
        {
        if ( strcmp ( this->GetAddServerEntry()->GetValue(), "" ) )
          {
          //--- TODO: when more service types are availabe, build out way to get them
          this->Logic->AddNewServer (this->GetAddServerEntry()->GetValue(), "XND", "XND", "XND");
          this->FetchMINode->SetServer ( this->GetAddServerEntry()->GetValue() );        
          e->SetValue ( "" );
          this->ResourceList->DeleteAllItems();
          }
        }
      }
    }
  
  if ( m && event == vtkKWMenu::MenuItemInvokedEvent )
    {
    if ( this->ServerMenuButton != NULL )
      {
      if ( m == this->ServerMenuButton->GetMenu() )
        {
        if ( this->ServerMenuButton->GetValue() != NULL )
          {
          //--- if server is same, just being re-selected,
          //--- set state variable in logic so that all a user's
          //--- tag selections can be restored once GUI is refreshed.
          if ( (this->FetchMINode->GetSelectedServer() != NULL) && (this->ServerMenuButton->GetValue() != NULL) )
            {
            if ( !strcmp (this->FetchMINode->GetSelectedServer(), this->ServerMenuButton->GetValue() ) )
              {
              this->FetchMINode->GetTagTableCollection()->SetRestoreSelectionStateForAllTables(1);
              }
            else
              {
              this->ResourceList->DeleteAllItems();
              this->TaggedDataList->ResetCurrentTagLabel();
              }
            }
          if ( !(strcmp (this->ServerMenuButton->GetValue(), "Add new server (XNAT Desktop servers only)" )))
            {
            vtkErrorMacro ( "This is commented out for now." );
            //this->RaiseNewServerWindow ();
            return;
            }

          this->FetchMINode->SetServer ( this->ServerMenuButton->GetValue() );
          //--- this queries server for tags
          vtkDebugMacro ("--------------------GUI event calling Query.");
          this->SetStatusText ( "Querying selected server for metadata (may take a little while)..." );

          if ( this->GetApplication() )
            {
            vtkSlicerApplication* app = vtkSlicerApplication::SafeDownCast(this->GetApplication() );
            if ( app )
              {
              vtkSlicerApplicationGUI *appGUI = app->GetApplicationGUI();
              if ( appGUI )
                {
                if (appGUI->GetMainSlicerWindow() )
                  {
                  vtkSlicerWaitMessageWidget *wm = vtkSlicerWaitMessageWidget::New();
                  wm->SetParent ( appGUI->GetMainSlicerWindow() );
                  wm->Create();
                  wm->SetText ("Querying selected server for metadata (may take a little while)...");
                  wm->DisplayWindow();
                  this->Logic->QueryServerForTags();
                  this->SetStatusText ( "Querying selected server for metadata (may take a little while)......." );
                  this->Logic->QueryServerForTagValues( );
                  wm->SetText ("Querying selected server for metadata (may take a little while)... done.");
                  wm->WithdrawWindow();
                  wm->Delete();
                  // TODO: temporary fix for HID which we are
                  // not yet querying for available tags. Just
                  // repopulate from default tags in FetchMINode
                  this->SetStatusText ( "" );
                  }
                }
              }
            }
          else
            {
            this->Logic->QueryServerForTags();
            this->SetStatusText ( "Querying selected server for metadata (may take a little while)......." );
            this->Logic->QueryServerForTagValues( );
            this->SetStatusText ( "" );
            }
          }
        }
      }
    }
}


//---------------------------------------------------------------------------
void vtkFetchMIGUI::SetStatusText(const char *txt)
{
  if ( this->GetApplicationGUI() )
    {
    if ( this->GetApplicationGUI()->GetMainSlicerWindow() )
      {
      this->GetApplicationGUI()->GetMainSlicerWindow()->SetStatusText (txt);
      }
    }
}


//---------------------------------------------------------------------------
void vtkFetchMIGUI::UpdateResourceTableFromMRML ( )
{
  if ( this->GetFetchMINode() == NULL )
    {
    vtkErrorMacro ("FetchMIGUI: UpdateResourceTableFromMRML got a NULL FetchMINode." );
    return;
    }
  if ( this->ResourceList == NULL )
    {
    vtkErrorMacro ("FetchMIGUI: UpdateResourceTableFromMRML got a NULL ResourceList widget." );
    return;
    }

  //--- clear the widget
  this->ResourceList->DeleteAllItems();
  
  //--- update the Resource Table from the FetchMINode
  vtkTagTable *t = this->FetchMINode->GetResourceDescription();
  if ( t != NULL )
    {
    //--- see if we get this far ok.
    const char *uri;
    const char *val;
    int i, row;
    for (i=0; i < t->GetNumberOfTags(); i++ )
      {
      uri = t->GetTagAttribute(i);
      val = t->GetTagValue(i);
      this->ResourceList->AddNewItem (uri, val);
      row = this->ResourceList->GetRowForURI ( uri );
      if ( row >= 0 && (t->IsTagSelected(uri)) )
        {
        this->ResourceList->SelectRow(row);              
        }
      }
    }
}





//---------------------------------------------------------------------------
void vtkFetchMIGUI::UpdateSceneTableFromMRML()
{

  if ( this->GetFetchMINode() == NULL )
    {
    vtkErrorMacro ("FetchMIGUI: UpdateSceneTableFromMRML got a NULL FetchMINode." );
    return;
    }
  if ( this->TaggedDataList == NULL )
    {
    vtkErrorMacro ("FetchMIGUI: UpdateSceneTableFromMRML got a NULL TaggedDataList widget." );
    return;
    }

  this->TaggedDataList->DeleteAllItems();
  this->Logic->ClearModifiedNodes();
  this->AddMRMLSceneRow();
  this->AddVolumeNodes();
  this->AddModelNodes();
  this->AddUnstructuredGridNodes();
}


//---------------------------------------------------------------------------
void vtkFetchMIGUI::AddVolumeNodes()
{

  vtkMRMLNode *node=NULL;
  vtkMRMLStorableNode *stnode=NULL;
  vtkTagTable *t=NULL;
  
  int nnodes = this->MRMLScene->GetNumberOfNodesByClass("vtkMRMLVolumeNode");
  int n;
  int row = this->TaggedDataList->GetMultiColumnList()->GetWidget()->GetNumberOfRows();
  const char *dtype = NULL;
  for (n=0; n<nnodes; n++)
    {
    node = this->MRMLScene->GetNthNodeByClass(n, "vtkMRMLVolumeNode");
    if (node->GetHideFromEditors()) 
      {
      continue;
      }
    vtkMRMLVolumeNode *vnode = vtkMRMLVolumeNode::SafeDownCast(node);
    vtkMRMLStorageNode* snode = vnode->GetStorageNode();
    //--- if a storage node doesn't yet exist for the node, add it.
    if (snode == NULL) 
      {
      vtkMRMLStorageNode *storageNode;
      if ( vtkMRMLDiffusionTensorVolumeNode::SafeDownCast(node) || 
            vtkMRMLDiffusionWeightedVolumeNode::SafeDownCast(node) )
        {
        storageNode = vtkMRMLNRRDStorageNode::New();
        }
      else
        {
        storageNode = vtkMRMLVolumeArchetypeStorageNode::New();
        }
      storageNode->SetScene(this->GetMRMLScene());
      this->SetMRMLScene(this->GetMRMLScene());
      this->GetMRMLScene()->AddNode(storageNode);  
      this->SetAndObserveMRMLScene(this->GetMRMLScene());
      vnode->SetAndObserveStorageNodeID(storageNode->GetID());
      storageNode->Delete();
      snode = storageNode;
      }

    if (snode->GetFileName() == NULL && this->DataDirectoryName != NULL) 
      {
      std::string name (this->DataDirectoryName);
      name += std::string(node->GetName());
      name += std::string(".nrrd");
      snode->SetFileName(name.c_str());
      }

    // get absolute filename
    std::string name;
    if (this->MRMLScene->IsFilePathRelative(snode->GetFileName()))
      {
      name = this->MRMLScene->GetRootDirectory();
      if (name[name.size()-1] != '/')
        {
        name = name + std::string("/");
        }
      }
    name += snode->GetFileName();

    // Set the SlicerDataType
    dtype = "Volume";
    vtkMRMLScalarVolumeNode *vsnode = vtkMRMLScalarVolumeNode::SafeDownCast (vnode );
    stnode = vtkMRMLStorableNode::SafeDownCast ( vsnode );
    vtkMRMLDiffusionTensorVolumeNode *dtinode = vtkMRMLDiffusionTensorVolumeNode::SafeDownCast (vnode );
    vtkMRMLDiffusionWeightedVolumeNode *dwinode = vtkMRMLDiffusionWeightedVolumeNode::SafeDownCast (vnode);
    if ( vsnode != NULL )
      {
      if ( vsnode->GetLabelMap() )
        {
        dtype = "LabelMap";
        }
      else
        {
        dtype = "ScalarVolume";
        }
      }
    if ( dtinode != NULL )
      {
      dtype = "DTIVolume";
      }
    if ( dwinode != NULL )
      {
      dtype = "DWIVolume";
      }
    
    this->TaggedDataList->AddNewItem ( node->GetID(), dtype );

    //--- store node's slicerdatatype in its UserTagTable
    if ( stnode != NULL && (strcmp(dtype, "")) )
      {
      t = stnode->GetUserTagTable();
      if ( t != NULL )
        {
        t->AddOrUpdateTag ( "SlicerDataType", dtype, 1 );
        }
      }
    
    if (node->GetModifiedSinceRead()) 
      {
      this->TaggedDataList->SelectRow ( row );
      this->Logic->AddModifiedNode(node->GetID());
      //--- TODO: can we get rid of this?
      //--- this should be getting called by the TaggedDataList when SelectRow
      //--- triggers a widget event...
      this->Logic->AddSelectedStorableNode(node->GetID() );
      }
    row++;
    }
}


//---------------------------------------------------------------------------
void vtkFetchMIGUI::AddModelNodes()
{
  vtkMRMLNode *node;
  vtkMRMLStorableNode *stnode;
  vtkTagTable *t;
  
  int nnodes = this->MRMLScene->GetNumberOfNodesByClass("vtkMRMLModelNode");
  int n;
  int row = this->TaggedDataList->GetMultiColumnList()->GetWidget()->GetNumberOfRows();
  const char *dtype = "VTKModel";
  
  for (n=0; n<nnodes; n++)
    {
    node = this->MRMLScene->GetNthNodeByClass(n, "vtkMRMLModelNode");
    if (node->GetHideFromEditors()) 
      {
      continue;
      }
    stnode = vtkMRMLStorableNode::SafeDownCast ( node );
    vtkMRMLModelNode *mnode = vtkMRMLModelNode::SafeDownCast(node);
    vtkMRMLStorageNode* snode = mnode->GetStorageNode();
    if (snode == NULL && !node->GetModifiedSinceRead())
      {
      continue;
      }
    if (snode == NULL && node->GetModifiedSinceRead()) 
      {
      vtkMRMLModelStorageNode *storageNode = vtkMRMLModelStorageNode::New();
      storageNode->SetScene(this->GetMRMLScene());
      this->SetMRMLScene(this->GetMRMLScene());
      this->GetMRMLScene()->AddNode(storageNode);  
      this->SetAndObserveMRMLScene(this->GetMRMLScene());
      mnode->SetAndObserveStorageNodeID(storageNode->GetID());
      storageNode->Delete();
      snode = storageNode;
      }

    if (snode->GetFileName() == NULL && this->DataDirectoryName != NULL) {
      std::string name (this->DataDirectoryName);
      name += std::string(node->GetName());
      name += std::string(".vtk");
      snode->SetFileName(name.c_str());
    }

    // get absolute filename
    std::string name;
    if (this->MRMLScene->IsFilePathRelative(snode->GetFileName()))
      {
      name = this->MRMLScene->GetRootDirectory();
      if (name[name.size()-1] != '/')
        {
        name = name + std::string("/");
        }
      }
    name += snode->GetFileName();
    
    // Set the SlicerDataType
    vtkMRMLFreeSurferModelStorageNode *fsnode = vtkMRMLFreeSurferModelStorageNode::SafeDownCast (snode);
    if ( fsnode != NULL )
      {
      if ( snode->IsA("vtkMRMLFreeSurferModelStorageNode") )
        {
        dtype = "FreeSurferModel";
        }
      }

    this->TaggedDataList->AddNewItem ( node->GetID(), dtype );

    //--- store node's slicerdatatype in its UserTagTable.
    if ( stnode != NULL && (strcmp(dtype, "")) )
      {
      t = stnode->GetUserTagTable();
      if ( t != NULL )
        {
        t->AddOrUpdateTag ( "SlicerDataType", dtype, 1 );
        }
      }

    if (node->GetModifiedSinceRead()) 
      {
      this->TaggedDataList->SelectRow ( row );
      this->Logic->AddModifiedNode(node->GetID());
      //--- TODO: can we get rid of this?
      //--- this should be getting called by the TaggedDataList when SelectRow
      //--- triggers a widget event...
      this->Logic->AddSelectedStorableNode(node->GetID() );      
      }
    row++;
    }
}

//---------------------------------------------------------------------------
void vtkFetchMIGUI::AddUnstructuredGridNodes()
{
  //--- UNSTRUCTURED GRID NODES
#if !defined(MESHING_DEBUG) && defined(Slicer3_BUILD_MODULES)  
  // *** add UnstructuredGrid types 
  // An additional datatype, MRMLUnstructuredGrid and its subclasses are 
  // also searched in the MRML tree.  This is done so instances of FiniteElement
  // meshes and other vtkUnstructuredGrid datatypes can be stored persistently.
  // this code is gated by MESHING_DEBUG since the MEshing MRML modules 
  
  vtkMRMLNode *node;
  vtkMRMLStorableNode *stnode;
  int nnodes = this->MRMLScene->GetNumberOfNodesByClass("vtkMRMLUnstructuredGridNode");
  int n;
  int row = this->TaggedDataList->GetMultiColumnList()->GetWidget()->GetNumberOfRows();
  const char *dtype;
  
  for (n=0; n<nnodes; n++)
    {
    node = this->MRMLScene->GetNthNodeByClass(n, "vtkMRMLUnstructuredGridNode");
    if (node->GetHideFromEditors()) 
      {
      continue;
      }
    stnode = vtkMRMLStorableNode::SafeDownCast ( node );
    vtkMRMLUnstructuredGridNode *gnode = vtkMRMLUnstructuredGridNode::SafeDownCast(node);
    vtkMRMLStorageNode* snode = gnode->GetStorageNode();
    if (snode == NULL && !node->GetModifiedSinceRead())
      {
      continue;
      }
    if (snode == NULL && node->GetModifiedSinceRead()) 
      {
        vtkMRMLUnstructuredGridStorageNode *storageNode = vtkMRMLUnstructuredGridStorageNode::New();
      storageNode->SetScene(this->GetMRMLScene());
      this->SetMRMLScene(this->GetMRMLScene());
      this->GetMRMLScene()->AddNode(storageNode);  
      this->SetAndObserveMRMLScene(this->GetMRMLScene());
      gnode->SetAndObserveStorageNodeID(storageNode->GetID());
      storageNode->Delete();
      snode = storageNode;
      }
    if (snode->GetFileName() == NULL && this->DataDirectoryName != NULL) {
      std::string name (this->DataDirectoryName);
      name += std::string(node->GetName());
      name += std::string(".vtk");
      snode->SetFileName(name.c_str());
    }

    // get absolute filename
    std::string name;
    if (this->MRMLScene->IsFilePathRelative(snode->GetFileName()))
      {
      name = this->MRMLScene->GetRootDirectory();
      if (name[name.size()-1] != '/')
        {
        name = name + std::string("/");
        }
      }
    name += snode->GetFileName();


    dtype = "UnstructuredGrid";
    this->TaggedDataList->AddNewItem ( node->GetID(), dtype );
    //--- store node's slicerdatatype in its UserTagTable.
    if ( stnode != NULL && (strcmp(dtype, "")) )
      {
      t = stnode->GetUserTagTable();
      if ( t != NULL )
        {
        t->AddOrUpdateTag ( "SlicerDataType", dtype, 1 );
        }
      }

    if (node->GetModifiedSinceRead()) 
      {
      this->TaggedDataList->SelectRow(row);
      this->Logic->AddModifiedNode(node->GetID());
      //--- TODO: can we get rid of this?
      //--- this should be getting called by the TaggedDataList when SelectRow
      //--- triggers a widget event...
      this->Logic->AddSelectedStorableNode(node->GetID() );      
      }
    row++;
    }
    // end of UGrid MRML node processing
#endif  
}




//---------------------------------------------------------------------------
void vtkFetchMIGUI::AddMRMLSceneRow()
{
  if ( this->MRMLScene == NULL )
    {
    vtkErrorMacro ("FetchMIGUI: AddMRMLSceneRow got a NULL MRMLScene.");
    return;
    }


  std::string dir = this->MRMLScene->GetRootDirectory();
  if (dir[dir.size()-1] != '/')
    {
    dir += std::string("/");
    }
  this->SetDataDirectoryName ( dir.c_str());
  
  //--- make sure the scene has a selected SlicerDataType tag.
  this->MRMLScene->GetUserTagTable()->AddOrUpdateTag ( "SlicerDataType", "MRML", 0);

  std::string uriName;
  const char *url = this->MRMLScene->GetURL();
  if (!url || !(*url))
    {
    uriName = dir.append("SlicerScene1");
    }
  else
    {
    uriName = url;
    }

  if(!uriName.empty())
    {
    //--- put a row in the TaggedDataList with selected, datatype, and url.
    this->TaggedDataList->AddNewItem ( "Scene description", "MRML");
    this->TaggedDataList->SelectRow ( 0 );
    }
}




//---------------------------------------------------------------------------
void vtkFetchMIGUI::UpdateTagTableFromGUI ( )
{
  if ( this->GetFetchMINode() == NULL )
    {
    vtkErrorMacro ("FetchMIGUI: UpdateTagTableFromGUI got a NULL FetchMINode." );
    return;
    }
  if ( this->QueryList == NULL )
    {
    vtkErrorMacro ("FetchMIGUI: UpdateTagTableFromGUI got a NULL QueryList widget." );
    return;
    }
  
  if ( this->Logic->GetCurrentWebService() == NULL )
    {
    vtkErrorMacro ("FetchMIGUI: UpdateTagTableFromGUI got a NULL server.\n");
    return;
    }
  const char *svctype = this->Logic->GetCurrentWebService()->GetServiceType();
  if (! this->Logic->GetServerCollection()->IsKnownServiceType(svctype) )
    {
    vtkErrorMacro ( "UpdateTagTableFromGUI:Got unknown web service type");
    return;
    }

  int num = this->QueryList->GetMultiColumnList()->GetWidget()->GetNumberOfRows();
  std::string att;
  std::string val;
  int sel;
  //--- update the FetchMINode, depending on what service is selected.
  
  vtkTagTable *t = this->Logic->GetCurrentWebService()->GetTagTable();
  if ( t == NULL )
    {
    // TODO: vtkErrorMacro
    vtkErrorMacro ( "UpdateTagTableFromGUI:Got NULL tag table for current web service client." );
    return;
    }
  for ( int i=0; i < num; i++ )
    {
    att = this->QueryList->GetAttributeOfItem ( i );
    val = this->QueryList->GetValueOfItem ( i );
    sel = this->QueryList->IsItemSelected ( i );
    t->AddOrUpdateTag ( att.c_str(), val.c_str(), sel );
    }
}




//---------------------------------------------------------------------------
void vtkFetchMIGUI::UpdateTagTableFromMRML ( )
{

  if ( this->GetFetchMINode() == NULL )
    {
    vtkErrorMacro ("FetchMIGUI: UpdateTagTableFromMRML got a NULL FetchMINode." );
    return;
    }
  if ( this->QueryList == NULL )
    {
    vtkErrorMacro ("FetchMIGUI: UpdateTagTableFromMRML got a NULL QueryList widget." );
    return;
    }
  
  if ( this->Logic->GetCurrentWebService() == NULL )
    {
    return;
    }
  const char *svctype = this->Logic->GetCurrentWebService()->GetServiceType();
  if (! this->Logic->GetServerCollection()->IsKnownServiceType(svctype) )
    {
    vtkErrorMacro ( "UpdateTagTableFromMRML:Got unknown web service type");
    return;
    }

  //--- now restore user's selection state for all tags.
  vtkTagTable *t = this->Logic->GetCurrentWebService()->GetTagTable();
  if ( t != NULL )
    {
    const char *att;
    int i, row;
    for (i=0; i < t->GetNumberOfTags(); i++ )
      {
      att = t->GetTagAttribute(i);
      row = this->QueryList->GetRowForAttribute ( att );
      if ( row >= 0 && (t->IsTagSelected(att)) )
        {
        this->QueryList->SelectRow(row);              
        }
      }
    }
}


//---------------------------------------------------------------------------
void vtkFetchMIGUI::PopulateQueryListFromServer()
{
  //--- This flag causes QueryList to hold off on invoking any
  //--- events (via the FetchMINode that cause the GUI to
  //--- update. We do this once at the end.
  this->QueryList->SetInPopulateWidget(1);
  this->QueryList->PopulateFromServer();

  const char *ttname = this->Logic->GetCurrentWebService()->GetTagTableName();
  vtkTagTable *t = this->FetchMINode->GetTagTableCollection()->FindTagTableByName ( ttname );
  if ( t != NULL )
    {
    if (t->GetRestoreSelectionState() )
      {
      //--- This means that the same server has been
      //--- selected by user, causing the list of available
      //--- tags for query to be refreshed. If possible, we
      //--- want to restore the 'selected' status of tags
      //--- and the values that users had for each.
      //---
      //--- NOTE: this is not essential, but seems friendly... so try it out.
      //--- Check to see if previously selected value for each tag
      //--- (stored in node's tagtable) is in the new list of known
      //--- values for tag, just populated in Logic's CurrentWebServiceMetadata.
      //--- If so, select it. If this behavior is annoying, then just comment
      //--- out this block.
      //---
      this->RestoreSelectedValuesForTagsFromMRML();
      }
    }
  //--- Finally, make sure the restore flag off.
  this->FetchMINode->GetTagTableCollection()->SetRestoreSelectionStateForAllTables(0);
  //--- And the populate widget flag is off
  this->QueryList->SetInPopulateWidget(0);
}

//----------------------------------------------------------------------------
void vtkFetchMIGUI::RestoreSelectedValuesForTagsFromMRML()
{
  if ( this->FetchMINode == NULL )
    {
    vtkErrorMacro ( "RestoreSelectedValuesForTagsFromMRML: got NULL FetchMINode");
    return;
    }
  if ( this->FetchMINode->GetTagTableCollection() == NULL )
    {
    vtkErrorMacro ( "RestoreSelectedValuesForTagsFromMRML: got NULL TagTableCollection in FetchMINode");
    return;
    }
  if ( this->Logic == NULL )
    {
    vtkErrorMacro ( "RestoreSelectedValuesForTagsFromMRML: got NULL Logic");
    return;
    }
  if ( this->Logic->GetCurrentWebService() == NULL )
    {
    vtkErrorMacro ( "RestoreSelectedValuesForTagsFromMRML: got NULL current web service");
    return;
    }
  if ( this->Logic->GetCurrentWebService()->GetTagTable() == NULL )
    {
    vtkErrorMacro ( "RestoreSelectedValuesForTagsFromMRML: got NULL tagTable");
    return;
    }  

  //--- check node's tag table and see what values were stored for each attribute.
  //--- Then, go thru CurrentWebServiceMetadata and select the value if it's present.
  std::map<std::string, std::vector<std::string> >::iterator iter;
  vtkTagTable *t = this->Logic->GetCurrentWebService()->GetTagTable ( );
  if ( t != NULL )
    {
    const char *att;
    const char *val;
    int i, j;
    int numTags = t->GetNumberOfTags();
    int widgetRow = 0;
    // For each tag in mrml, get the attribute, and value.
    for (i=0; i < numTags; i++ )
      {
      att = t->GetTagAttribute(i);
      val = t->GetTagValue(i);
      //--- Search CurrentWebServiceMetadata for the attribute.
      if ( att != NULL && val != NULL )
        {
        for ( iter = this->Logic->CurrentWebServiceMetadata.begin();
              iter != this->Logic->CurrentWebServiceMetadata.end();
              iter++ )
          {
          if ( ! (strcmp (iter->first.c_str(), att ) ) )
            {
            //--- Search thru attributes values to see if the value is present.
            int numValues = iter->second.size();
            for (j=0; j< numValues; j++ )
              {
              if ( !(strcmp(val, iter->second[j].c_str()) ) )
                {
                //--- Value is present for tag. Find the corresponding
                //--- row in the GUI, and select the value.
                widgetRow = this->QueryList->GetRowForAttribute ( att );
                this->QueryList->SelectValueOfItem(widgetRow, val);
                if (t->IsTagSelected(att) )
                  {
                  this->QueryList->SelectRow(widgetRow);
                  }
                }
              }
            }
          }
        }
      }
    }
}



//---------------------------------------------------------------------------
void vtkFetchMIGUI::ProcessMRMLEvents ( vtkObject *caller,
                                            unsigned long event,
                                            void *callData ) 
{
  if ( this->FetchMINode == NULL )
    {
    vtkErrorMacro ("FetchMIGUI: ProcessMRMLEvents has a NULL FetchMINode");
    return;
    }

  // if parameter node has been changed externally, update GUI widgets with new values
  vtkMRMLFetchMINode* node = vtkMRMLFetchMINode::SafeDownCast(caller);

  if ( this->ApplicationGUI == NULL )
    {
    //TODO vtkErrorMacro();
    return;    
    }
  
  if (event == vtkMRMLFetchMINode::SelectedServerModifiedEvent )
    {
    if ( (strcmp (this->FetchMINode->GetSelectedServer(), this->ServerMenuButton->GetValue() ) ) )
      {
      this->ServerMenuButton->SetValue ( this->FetchMINode->GetSelectedServer() );
      this->ResourceList->DeleteAllItems();
      }
    }

  if ( event == vtkMRMLScene::SceneCloseEvent )
    {
    this->Logic->ClearModifiedNodes();
    this->Logic->ClearSelectedStorableNodes();
    this->UpdateSceneTableFromMRML();
    }

  if (( event == vtkMRMLScene::NodeAddedEvent) ||
      ( event == vtkMRMLScene::NodeRemovedEvent) ||
      ( event == vtkMRMLScene::NewSceneEvent ))
    {
    this->UpdateSceneTableFromMRML();
    }
  
  else if (node != NULL && this->GetFetchMINode() == node) 
    {
    if ( event == vtkMRMLFetchMINode::RemoteIOErrorEvent )
      {
          vtkKWMessageDialog *dialog = vtkKWMessageDialog::New();
          dialog->SetParent ( this->GetApplicationGUI()->GetMainSlicerWindow() );
          dialog->SetStyleToMessage();
          dialog->SetText ( this->FetchMINode->GetErrorMessage() );
          dialog->Create();
          dialog->Invoke();
          dialog->Delete();
      }
    if (event == vtkMRMLFetchMINode::TagResponseReadyEvent )
      {
      //--- check for error
      if ( this->FetchMINode->GetErrorMessage() != NULL )
        {
          vtkKWMessageDialog *dialog = vtkKWMessageDialog::New();
          dialog->SetParent ( this->GetApplicationGUI()->GetMainSlicerWindow() );
          dialog->SetStyleToMessage();
          dialog->SetText ( this->FetchMINode->GetErrorMessage() );
          dialog->Create();
          dialog->Invoke();
          dialog->Delete();
        }
      vtkDebugMacro ("--------------------Populating the Query Widget");
      this->PopulateQueryListFromServer();
      }
    if (event == vtkMRMLFetchMINode::ResourceResponseReadyEvent )
      {
      //--- check for error
      if ( this->FetchMINode->GetErrorMessage() != NULL )
        {
          vtkKWMessageDialog *dialog = vtkKWMessageDialog::New();
          dialog->SetParent ( this->GetApplicationGUI()->GetMainSlicerWindow() );
          dialog->SetStyleToMessage();
          dialog->SetText ( this->FetchMINode->GetErrorMessage() );
          dialog->Create();
          dialog->Invoke();
          dialog->Delete();
        }
      this->UpdateResourceTableFromMRML();
      this->Notebook->RaisePage ("Browse Query Results & Download");
      }
    else
      {
      this->UpdateGUI();
      }
    }
}



//---------------------------------------------------------------------------
void vtkFetchMIGUI::UpdateMRML ()
{
  // update from GUI
  if ( this->UpdatingGUI )
    {
    return;
    }
  this->UpdatingMRML = 1;
  
  vtkMRMLFetchMINode* n = this->GetFetchMINode();
  if (n == NULL)
    {
    n = vtkMRMLFetchMINode::New();
    vtkIntArray *events = vtkIntArray::New();
    events->InsertNextValue ( vtkMRMLFetchMINode::KnownServersModifiedEvent );
    events->InsertNextValue ( vtkMRMLFetchMINode::SelectedServerModifiedEvent );
    events->InsertNextValue ( vtkMRMLFetchMINode::TagsModifiedEvent );
    events->InsertNextValue ( vtkMRMLFetchMINode::SaveSelectionEvent );
    events->InsertNextValue ( vtkMRMLFetchMINode::RemoteIOErrorEvent );
    events->InsertNextValue ( vtkMRMLFetchMINode::RemoteIOErrorChoiceEvent );
    events->InsertNextValue ( vtkMRMLFetchMINode::ResourceResponseReadyEvent );
    events->InsertNextValue ( vtkMRMLFetchMINode::TagResponseReadyEvent );    
    vtkSetAndObserveMRMLNodeEventsMacro ( this->FetchMINode, n, events );
    if ( this->Logic->GetFetchMINode() != NULL )
      {
      this->Logic->SetFetchMINode ( NULL );
      this->Logic->SetFetchMINode( n );
      }
    events->Delete();
  }

  // save node parameters for Undo
  //this->GetLogic()->GetMRMLScene()->SaveStateForUndo(n);
//  n->Delete();
  this->UpdatingMRML = 0;
}



//---------------------------------------------------------------------------
void vtkFetchMIGUI::UpdateGUI ()
{
  // update from MRML
  if ( this->UpdatingMRML )
    {
    return;
    }
  this->UpdatingGUI = 1;
  
  vtkMRMLFetchMINode* n = this->GetFetchMINode();
  if (n == NULL )
    {
    n = vtkMRMLFetchMINode::New();
    vtkIntArray *events = vtkIntArray::New();
    events->InsertNextValue ( vtkMRMLFetchMINode::KnownServersModifiedEvent );
    events->InsertNextValue ( vtkMRMLFetchMINode::SelectedServerModifiedEvent );
    events->InsertNextValue ( vtkMRMLFetchMINode::TagsModifiedEvent );
    events->InsertNextValue ( vtkMRMLFetchMINode::SaveSelectionEvent );
    events->InsertNextValue ( vtkMRMLFetchMINode::RemoteIOErrorEvent );
    events->InsertNextValue ( vtkMRMLFetchMINode::RemoteIOErrorChoiceEvent );
    events->InsertNextValue ( vtkMRMLFetchMINode::ResourceResponseReadyEvent );
    events->InsertNextValue ( vtkMRMLFetchMINode::TagResponseReadyEvent );    
    vtkSetAndObserveMRMLNodeEventsMacro ( this->FetchMINode, n, events );
    if ( this->Logic->GetFetchMINode() != NULL )
      {
      this->Logic->SetFetchMINode ( NULL );
      this->Logic->SetFetchMINode( n );
      }
    events->Delete();
    }
  
  if (n != NULL)
    {
    //---  update the list of known servers in the
    if ( this->ServerMenuButton != NULL )
      {
      this->ServerMenuButton->GetMenu()->DeleteAllItems();
      vtkFetchMIServer *s;
      int l = this->Logic->GetServerCollection()->GetNumberOfItems();
      for (int i=0; i < l; i ++ )
        {
        s = static_cast<vtkFetchMIServer *>(this->Logic->GetServerCollection()->GetItemAsObject(i));
        if ( s != NULL )
          {
          this->ServerMenuButton->GetMenu()->AddRadioButton ( s->GetName() );
          }
        }
      //TODO: hook up these commands!
      this->ServerMenuButton->GetMenu()->AddSeparator();
      this->ServerMenuButton->GetMenu()->AddRadioButton("Add new server (XNAT Desktop servers only)");
      
      //--- select active server in the ServerMenuButton
      if ( this->FetchMINode->GetSelectedServer() != NULL )
        {
        this->ServerMenuButton->SetValue ( this->FetchMINode->GetSelectedServer() );
        }
      else
        {
        this->ServerMenuButton->SetValue ( "none" );
        }
      }
    
    }
  else
    {
    vtkErrorMacro ("FetchMIGUI: UpdateGUI has a NULL FetchMINode." );
    }
  // n->Delete();
  this->UpdateTagTableFromMRML();
  this->UpdateSceneTableFromMRML();
  this->UpdatingGUI = 0;
}





//---------------------------------------------------------------------------
void vtkFetchMIGUI::DestroyNewServerWindow( )
{
  if ( !this->NewServerWindow )
    {
    return;
    }
  if ( !this->NewServerWindow->IsCreated() )
    {
    vtkErrorMacro ("DestroyNewServerWindow: NewServerWindow is not created.");
    return;
    }
  this->NewServerWindow->Withdraw();
  this->UnBindNewServerWindow();

  if ( this->AddServerEntry )
    {
    this->AddServerEntry->RemoveObservers ( vtkKWEntry::EntryValueChangedEvent, (vtkCommand *)this->GUICallbackCommand);
    this->AddServerEntry->SetParent ( NULL );
    this->AddServerEntry->Delete();
    this->AddServerEntry = NULL;
    }
  if ( this->AddServerButton )
    {
    this->AddServerButton->RemoveObservers (vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->AddServerButton->SetParent ( NULL );
    this->AddServerButton->Delete();
    this->AddServerButton = NULL;
    }

  if ( this->CloseNewServerButton )
    {
    this->CloseNewServerButton->RemoveObservers (vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->CloseNewServerButton->SetParent ( NULL );
    this->CloseNewServerButton->Delete();    
    this->CloseNewServerButton = NULL;
    }
  this->NewServerWindow->Delete();
  this->NewServerWindow = NULL;
}



//---------------------------------------------------------------------------
void vtkFetchMIGUI::BindNewServerWindow ( )
{
  if (!this->CloseNewServerButton )
    {
    return;
    }
  if ( this->CloseNewServerButton->IsCreated() )
    {
    this->CloseNewServerButton->SetBinding ( "<ButtonPress>", this, "WithdrawNewServerWindow" );
//    this->CloseNewServerButton->SetBinding ( "<ButtonPress>", this, "DestroyNewServerWindow" );
    }

}

//---------------------------------------------------------------------------
void vtkFetchMIGUI::WithdrawNewServerWindow()
{
  this->NewServerWindow->Withdraw();
}


//---------------------------------------------------------------------------
void vtkFetchMIGUI::UnBindNewServerWindow ( )
{

  if ( !this->CloseNewServerButton )
    {
    return;
    }
  if (this->CloseNewServerButton->IsCreated() )
    {
    this->CloseNewServerButton->RemoveBinding ( "<ButtonPress>" );
    }
}


//---------------------------------------------------------------------------
void vtkFetchMIGUI::RaiseNewServerWindow()
{
  if ( this->Logic == NULL )
    {
    vtkErrorMacro ("RaiseNewServerWindow: Got NULL Logic.");
    return;
    }
  if ( this->Logic->GetFetchMINode() == NULL )
    {
    vtkErrorMacro ("RaiseNewServerWindow: Got NULL FetchMINode");
    return;
    }

  if ( !this->NewServerWindow->IsCreated() )
    {
    int px, py;
    //--- top level container.
    this->NewServerWindow = vtkKWTopLevel::New();
    this->NewServerWindow->SetMasterWindow (this->GetServerMenuButton() );
    this->NewServerWindow->SetApplication ( this->GetApplication() );
    this->NewServerWindow->Create();
    vtkKWTkUtilities::GetWidgetCoordinates(this->GetServerMenuButton(), &px, &py);
    this->NewServerWindow->SetPosition ( px + 10, py + 10) ;
    this->NewServerWindow->SetBorderWidth ( 1 );
    this->NewServerWindow->SetReliefToFlat();
    this->NewServerWindow->SetTitle ("Add a new server (only XNAT Desktop supported at this time.)");
    this->NewServerWindow->SetSize (450, 75);
    this->NewServerWindow->Withdraw();
    this->NewServerWindow->SetDeleteWindowProtocolCommand ( this, "DestroyNewServerWindow");

    vtkKWFrame *f1 = vtkKWFrame::New();
    f1->SetParent ( this->NewServerWindow );
    f1->Create();
    f1->SetBorderWidth ( 1 );
    this->Script ( "pack %s -side top -anchor nw -fill x -expand n -padx 0 -pady 1", f1->GetWidgetName() );

    //--- new tag entry
    vtkKWLabel *l1 = vtkKWLabel::New();
    l1->SetParent (f1);
    l1->Create();
    l1->SetText ( "Server:" );
    l1->SetWidth ( 12 );
    this->AddServerEntry = vtkKWEntry::New();
    this->AddServerEntry->SetParent ( f1 );
    this->AddServerEntry->Create();
    this->AddServerEntry->SetWidth(20);
    this->AddServerEntry->AddObserver ( vtkKWEntry::EntryValueChangedEvent, (vtkCommand *)this->GUICallbackCommand);
    this->AddServerButton = vtkKWPushButton::New();
    this->AddServerButton->SetParent ( f1);
    this->AddServerButton->Create();
    this->AddServerButton->SetReliefToFlat();
    this->AddServerButton->SetBorderWidth ( 0 );
    this->AddServerButton->SetImageToIcon ( this->FetchMIIcons->GetAddNewIcon() );
    this->AddServerButton->AddObserver ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand);

    this->Script ( "grid %s -row 0 -column 0 -sticky e -padx 2 -pady 2", l1->GetWidgetName() );
    this->Script ( "grid %s -row 0 -column 1 -sticky ew -padx 2 -pady 2", this->AddServerEntry->GetWidgetName() );
    this->Script ( "grid %s -row 0 -column 2 -sticky ew -padx 2 -pady 2", this->AddServerButton->GetWidgetName() );
    this->Script ( "grid columnconfigure %s 0 -weight 0", f1->GetWidgetName() );
    this->Script ( "grid columnconfigure %s 1 -weight 1", f1->GetWidgetName() );
    this->Script ( "grid columnconfigure %s 2 -weight 0", f1->GetWidgetName() );

    //--- close button (destroys win and widgets when closed.
    //--- TODO: this is causing an error when window is destroyed.
    //--- I think because the button is destroyed while the binding
    //--- to it is still active. Not sure how to fix, so leaving out the
    //--- close button for now; window can be closed using the 'x'
    //--- in the title bar.

    vtkKWFrame *f2 = vtkKWFrame::New();
    f2->SetParent ( this->NewServerWindow );
    f2->Create();
    f2->SetBorderWidth ( 1 );
    this->Script ( "pack %s -side top -anchor nw -fill x -expand n -padx 0 -pady 1", f2->GetWidgetName() );

    this->CloseNewServerButton = vtkKWPushButton::New();
    this->CloseNewServerButton->SetParent ( f2 );
    this->CloseNewServerButton->Create();
    this->CloseNewServerButton->SetText ( "close" );
    this->Script ( "pack %s -side top -anchor c  -expand n -padx 2 -pady 6",
                   this->CloseNewServerButton->GetWidgetName() );

    this->BindNewServerWindow();
    f1->Delete();
    f2->Delete();
    l1->Delete();
    }

  //-- display
  this->NewServerWindow->DeIconify();
  this->NewServerWindow->Raise();
}


//---------------------------------------------------------------------------
void vtkFetchMIGUI::ShowSelectionTagView()
{
  if ( this->MRMLScene == NULL )
    {
    //TODO vtkErrorMacro();
    return;
    }
  if ( this->ResourceList == NULL )
    {
    //TODO vtkErrorMacro();
    return;
    }
  if ( this->ApplicationGUI == NULL )
    {
    //TODO vtkErrorMacro();
    return;
    }

  if ( this->TagViewer == NULL )
    {
    int px, py;
    this->TagViewer = vtkFetchMITagViewWidget::New();
    this->TagViewer->SetParent ( this->GetApplicationGUI()->GetMainSlicerWindow() );
    this->TagViewer->Create();
    vtkKWTkUtilities::GetWidgetCoordinates(this->GetServerMenuButton(), &px, &py);
    this->TagViewer->GetTagViewWindow()->SetPosition ( px + 10, py + 10) ;
    this->TagViewer->SetTagTitle ("Tags for scene and all data:");
    }

  std::stringstream ss;
  vtkMRMLStorableNode *node;
  vtkTagTable *t = NULL;

  //--- figure out the text
//  int dnum = this->TaggedDataList->GetNumberOfSelectedItems();
  int dnum = this->TaggedDataList->GetNumberOfItems();
  int i, j;
  int numtags;
  const char *nodeID;
  const char *att;
  const char *val;
  for (i=0; i<dnum; i++)
    {
//    nodeID = this->TaggedDataList->GetNthSelectedDataTarget(i);
    nodeID = this->TaggedDataList->GetMultiColumnList()->GetWidget()->GetCellText(i,3);
    if ( nodeID != NULL )
      {
      //--- get tags on the data.
      ss << "\n";
      ss << "**";
      ss << nodeID;
      ss << ":**\n";
      if ( !(strcmp (nodeID, "Scene description")))
        {
        t = this->MRMLScene->GetUserTagTable();
        }
      else
        {
        node = vtkMRMLStorableNode::SafeDownCast ( this->MRMLScene->GetNodeByID(nodeID));
        if ( node != NULL )
          {
          t = node->GetUserTagTable();
          }
        }
      if ( t != NULL )
        {
        numtags = t->GetNumberOfTags();
        for ( j=0; j <numtags; j++)
          {
          att = t->GetTagAttribute(j);
          val = t->GetTagValue(j);
          if ( att!= NULL && val != NULL )
            {
            ss << att;
            ss << " = ";
            ss << val;
            ss << "\n";
            }
          }
        }
      }
    }
  
  this->TagViewer->SetTagText ( ss.str().c_str() );
  this->TagViewer->DisplayTagViewWindow();
}


//---------------------------------------------------------------------------
void vtkFetchMIGUI::ShowAllTagView()
{
  if ( this->MRMLScene == NULL )
    {
    //TODO vtkErrorMacro();
    return;
    }
  if ( this->ResourceList == NULL )
    {
    //TODO vtkErrorMacro();
    return;
    }
  if ( this->ApplicationGUI == NULL )
    {
    //TODO vtkErrorMacro();
    return;
    }

  if ( this->TagViewer == NULL )
    {
    int px, py;
    this->TagViewer = vtkFetchMITagViewWidget::New();
    this->TagViewer->SetParent ( this->GetApplicationGUI()->GetMainSlicerWindow() );
    this->TagViewer->Create();
    vtkKWTkUtilities::GetWidgetCoordinates(this->GetServerMenuButton(), &px, &py);
    this->TagViewer->GetTagViewWindow()->SetPosition ( px + 10, py + 10) ;
    this->TagViewer->SetTagTitle ("Tags for scene and all data:");
    }

  std::stringstream ss;
  vtkMRMLStorableNode *node;
  vtkTagTable *t = NULL;

  //--- figure out the text
//  int dnum = this->TaggedDataList->GetNumberOfSelectedItems();
  int dnum = this->TaggedDataList->GetNumberOfItems();
  int i, j;
  int numtags;
  const char *nodeID;
  const char *att;
  const char *val;
  for (i=0; i<dnum; i++)
    {
//    nodeID = this->TaggedDataList->GetNthSelectedDataTarget(i);
    nodeID = this->TaggedDataList->GetMultiColumnList()->GetWidget()->GetCellText(i,4);
    if ( nodeID != NULL )
      {
      //--- get tags on the data.
      ss << "\n";
      ss << "**";
      ss << nodeID;
      ss << ":**\n";
      if ( !(strcmp (nodeID, "Scene description")))
        {
        t = this->MRMLScene->GetUserTagTable();
        }
      else
        {
        node = vtkMRMLStorableNode::SafeDownCast ( this->MRMLScene->GetNodeByID(nodeID));
        if ( node != NULL )
          {
          t = node->GetUserTagTable();
          }
        }
      if ( t != NULL )
        {
        numtags = t->GetNumberOfTags();
        for ( j=0; j <numtags; j++)
          {
          att = t->GetTagAttribute(j);
          val = t->GetTagValue(j);
          if ( att!= NULL && val != NULL )
            {
            ss << att;
            ss << " = ";
            ss << val;
            ss << "\n";
            }
          }
        }
      }
    }
  
  this->TagViewer->SetTagText ( ss.str().c_str() );
  this->TagViewer->DisplayTagViewWindow();

}


//---------------------------------------------------------------------------
void vtkFetchMIGUI::DeleteSelectedResourcesFromServer()
{
  int deleteError = 0;

  if ( this->MRMLScene == NULL )
    {
    vtkErrorMacro ("vtkFetchMIGUI:DeleteSelectedResourcesFromServer got NULL MRMLScene" );
    return;
    }
  if ( this->ResourceList == NULL)
    {
    vtkErrorMacro ("vtkFetchMIGUI:DeleteSelectedResourcesFromServer got NULL ResourceList" );
    return;
    }
  if ( this->Logic == NULL )
    {
    vtkErrorMacro ("vtkFetchMIGUI:DeleteSelectedResourcesFromServer got NULL Logic" );
    return;
    }
  
  //--- This method loops through all resources selected in the ResourceList.
  //--- For each selected resource, the DeleteResourceFromServer method on the Logic
  //--- is called, which in turn requests the DeleteResource method on the appropriate handler.
  //--- Then handler returns 0 if an error occurs, and the logic passes this back.
  //--- For each resource successfully deleted, the ResourceList is updated; and if any
  //--- resource deletion causes an error, a message dialog is posted warning the user
  //--- that not all resources may have been deleted.

  //--- try to post a message....

  int num = this->ResourceList->GetNumberOfSelectedItems();
  // loop over all selected data; 
  int retval = 1;
  int r;
  std::string deleteURI;
  for (int i=(num-1); i>=0; i-- )
    {
    if ( this->ResourceList->GetNthSelectedURI(i) != NULL && strcmp(this->ResourceList->GetNthSelectedURI(i), "" ) )
      {
      deleteURI.clear();
      deleteURI =  this->ResourceList->GetNthSelectedURI(i);
      
      vtkKWMessageDialog *dialog = vtkKWMessageDialog::New();
      dialog->SetParent ( this->GetApplicationGUI()->GetMainSlicerWindow() );
      dialog->SetStyleToOkOtherCancel();
      dialog->SetOKButtonText ( "Scene & its Data" );
      dialog->SetOtherButtonText ( "Scene File Only" );
      dialog->SetText ("What would you like to delete?");
      dialog->Create();
      dialog->Invoke();
      int rval = dialog->GetStatus();
      dialog->Delete();

      if ( rval == vtkKWMessageDialog::StatusOK )
        {
        if ( this->GetApplication() )
          {
          vtkSlicerApplication* app = vtkSlicerApplication::SafeDownCast(this->GetApplication() );
          if ( app )
            {
            vtkSlicerApplicationGUI *appGUI = app->GetApplicationGUI();
            if ( appGUI )
              {
              if (appGUI->GetMainSlicerWindow() )
                {
                vtkSlicerWaitMessageWidget *wm = vtkSlicerWaitMessageWidget::New();
                wm->SetParent ( appGUI->GetMainSlicerWindow() );
                wm->Create();
                wm->SetText ("Deleting scene and it's referenced datasets from the server (may take a little while)...");
                wm->DisplayWindow();
                retval = this->Logic->DeleteSceneFromServer ( deleteURI.c_str() );
                wm->SetText ("Deleting scene and it's referenced datasets from the server (may take a little while)... done.");
                wm->WithdrawWindow();
                wm->Delete();
                this->SetStatusText ("");
                }
              }
            }
          }
        else
          {
          retval = this->Logic->DeleteSceneFromServer ( deleteURI.c_str() );
          }
        }
      if ( rval == vtkKWMessageDialog::StatusOther )
        {
        if ( this->GetApplication() )
          {
          vtkSlicerApplication* app = vtkSlicerApplication::SafeDownCast(this->GetApplication() );
          if ( app )
            {
            vtkSlicerApplicationGUI *appGUI = app->GetApplicationGUI();
            if ( appGUI )
              {
              if (appGUI->GetMainSlicerWindow() )
                {
                vtkSlicerWaitMessageWidget *wm = vtkSlicerWaitMessageWidget::New();
                wm->SetParent ( appGUI->GetMainSlicerWindow() );
                wm->Create();
                wm->SetText ("Deleting scene file from the server (may take a little while)...");
                wm->DisplayWindow();
                retval = this->Logic->DeleteResourceFromServer ( deleteURI.c_str() );
                wm->SetText ("Deleting scene file from the server (may take a little while)... done.");
                wm->WithdrawWindow();
                wm->Delete();
                this->SetStatusText ("");
                }
              }
            }
          }
        else
          {
          retval = this->Logic->DeleteResourceFromServer ( deleteURI.c_str() );
          }
        }
      if ( retval != 1 )
        {
        deleteError = 1;
        }
      else
        {
        // update the ResourceList: find the row for this uri and delete the row.
        std::string uriTarget;
        for ( unsigned int j=0; j < this->Logic->GetNumberOfURIsDeletedOnServer(); j++ )
          {
          uriTarget.clear();
          uriTarget = this->Logic->URIsDeletedOnServer[j];
          r = this->ResourceList->GetRowForURI( uriTarget.c_str() );
          if ( r >= 0 )
            {
            this->ResourceList->GetMultiColumnList()->GetWidget()->DeleteRow ( r );
            }
          }
        }
      }
    }
  
  if ( deleteError )
    {
    //Pop up error message; not all resources were deleted.
    vtkKWMessageDialog *dialog = vtkKWMessageDialog::New();
    dialog->SetParent ( this->GetApplicationGUI()->GetMainSlicerWindow() );
    dialog->SetStyleToMessage();
    dialog->SetText ("Warning: there was a problem deleting some of the selected resources; please refresh your resource query to determine the resources' status.");
    dialog->Create();
    dialog->Invoke();
    dialog->Delete();
    return;
    }
}


//---------------------------------------------------------------------------
void vtkFetchMIGUI::RemoveTagFromSelectedData()
{
  if ( this->MRMLScene == NULL )
    {
    //TODO vtkErrorMacro();
    return;
    }
  if ( this->QueryList == NULL)
    {
    //TODO vtkErrorMacro();
    return;
    }
  if ( this->ResourceList == NULL )
    {
    //TODO vtkErrorMacro();
    return;
    }
  
  int j;
  int dnum;
  vtkMRMLStorableNode *node = NULL;
  vtkTagTable *t = NULL;
  std::string att = this->TaggedDataList->GetCurrentTagAttribute();
  std::string val = this->TaggedDataList->GetCurrentTagValue();

  if ( (att.c_str() != NULL) && (val.c_str() != NULL) &&
       (strcmp(att.c_str(), "")) && (strcmp(val.c_str(), "")) )
    {
    //--- apply to all selected data in TaggedDataList
    dnum = this->TaggedDataList->GetNumberOfSelectedItems();
    const char *nodeID;
    for (j=0; j<dnum; j++)
      {
      nodeID = this->TaggedDataList->GetNthSelectedDataTarget(j);
      if ( nodeID != NULL )
        {
        if ( !(strcmp(att.c_str(), "SlicerDataType")) )
          {
          //--- can't delete Slicer data type from data.
          vtkKWMessageDialog *dialog = vtkKWMessageDialog::New();
          dialog->SetParent ( this->GetApplicationGUI()->GetMainSlicerWindow() );
          dialog->SetStyleToMessage();
          dialog->SetText ("Slicer requires data to have a SlicerDataType tag; it cannot be deleted.");
          dialog->Create();
          dialog->Invoke();
          dialog->Delete();
          return;
          }
        if ( !(strcmp (nodeID, "Scene description")))
          {
          //--- want to delete tag on the scene 
          t = this->MRMLScene->GetUserTagTable();
          }
        else
          {
          //--- want to delete tag on the storable node.
          node = vtkMRMLStorableNode::SafeDownCast ( this->MRMLScene->GetNodeByID(nodeID));
          if ( node != NULL )
            {
            t = node->GetUserTagTable();
            }        
          }          
        //--- delete tag if it's in the tagtable.
        if ( t != NULL )
          {
          if (t->CheckTableForTag(att.c_str() ) >= 0 )
            {
            t->DeleteTag ( att.c_str() );
            }
          }
        }
      }
    }
}


//---------------------------------------------------------------------------
void vtkFetchMIGUI::TagSelectedData()
{
  if ( this->MRMLScene == NULL )
    {
    //TODO vtkErrorMacro();
    return;
    }
  if ( this->ResourceList == NULL )
    {
    //TODO vtkErrorMacro();
    return;
    }

  int j;
  int dnum;
  vtkMRMLStorableNode *node;
  vtkTagTable *t;
  std::string att = this->TaggedDataList->GetCurrentTagAttribute();
  std::string val = this->TaggedDataList->GetCurrentTagValue();

  if ( (att.c_str() != NULL) && (val.c_str() != NULL) &&
       (strcmp(att.c_str(), "")) && (strcmp(val.c_str(), "")) )
    {
    //--- apply to all selected data in TaggedDataList
    dnum = this->TaggedDataList->GetNumberOfSelectedItems();
    const char *nodeID;
    for (j=0; j<dnum; j++)
      {
      nodeID = this->TaggedDataList->GetNthSelectedDataTarget(j);
      if ( nodeID != NULL )
        {
        //--- tag the scene
        if ( !(strcmp (nodeID, "Scene description")))
          {
          t = this->MRMLScene->GetUserTagTable();
          if ( t != NULL )
            {
            //--- add current tag
            t->AddOrUpdateTag ( att.c_str(), val.c_str(), 1 );
            //--- enforce this tag on the scene.
            t->AddOrUpdateTag ( "SlicerDataType", "MRML", 1 );
            }
          }
        else
          {
          //--- or tag the storable node.
          node = vtkMRMLStorableNode::SafeDownCast ( this->MRMLScene->GetNodeByID(nodeID));
          if ( node != NULL )
            {
            t = node->GetUserTagTable();
            if ( t != NULL )
              {
              t->AddOrUpdateTag ( att.c_str(), val.c_str(), 1 );
              }
            }
          }
        }
      }
    }
}




//---------------------------------------------------------------------------
void vtkFetchMIGUI::BuildGUI ( ) 
{
  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
  vtkMRMLFetchMINode* n = vtkMRMLFetchMINode::New();
  if ( this->Logic != NULL )
    {
    this->Logic->GetMRMLScene()->RegisterNodeClass(n);
    if ( this->Logic->GetFetchMINode() == NULL )
      {
      this->Logic->SetFetchMINode ( NULL );
      this->Logic->SetFetchMINode( n );
      }
    }
  n->Delete();

  this->UIPanel->AddPage ( "FetchMI", "FetchMI", NULL );
  // ---
  // MODULE GUI FRAME 
  // configure a page for a volume loading UI for now.
  // later, switch on the modulesButton in the SlicerControlGUI
  // ---
    
  // HELP FRAME
  // Define your help text and build the help frame here.
  const char *about = "This work was supported by NA-MIC, NAC, BIRN, NCIGT, CTSC and the Slicer Community. See <a>http://www.slicer.org</a> for details. \n\n";
  
  const char *help = "**FetchMI** (Fetch Medical Informatics) is a 'sandbox' module for functionality to remotely upload, download and tag data. Its functionality will eventually be moved into Slicer's load and save panels and into a separate module for data markup.\n\n *** Select a server.** All subsequent operations will transact with the selected server.\n\n * The selected server will automatically be queried for the metadata it currently supports. The panel in the **Query for Scenes** tab will be populated with this metadata (attributes and available values). Update this panel time by clicking the **Refresh** button.\n\n * Select any tags you want to use to query for resources \n\n * Click the **Search** button (binoculars) to query the server for matching resources.\n\n * The **Browse Query Results & Download** tab will be populated by a list of resources matching the query.\n\n * Select a resource (MRML Scenes only) from that list. \n\n * Click the **Download** button and the scene will be downloaded to your current cache directory and loaded.\n\n * The **Tag Scene & Upload** tab shows all data in scene (and scene file).\n\n * Select or create new tags using the menubutton at the top of this panel, and click the **Apply Tags** button to associate to selected datasets. Click the **Remove Tags** button to dissociate selected tag from selected datasets. \n\n ***NOTE: for now, tags CANNOT include spaces or any special characters.**\n\n * Click the **Show All Tags** button to see all tags on all loaded datasets and scene.\n\n * Upload the current scene to the server by clicking the **Upload** button, after data is tagged. In this version, the entire scene and all datasets are uploaded.\n\n ***NOTE: remote uploads and downloads can be slow. Initiating these without access to the network or without access to the selected server may lead to unexpected application behavior.**\n\n";

  vtkKWWidget *page = this->UIPanel->GetPageWidget ( "FetchMI" );
  this->BuildHelpAndAboutFrame ( page, help, about );
  vtkKWLabel *NACLabel = vtkKWLabel::New();
  NACLabel->SetParent ( this->GetLogoFrame() );
  NACLabel->Create();
  NACLabel->SetImageToIcon ( this->GetAcknowledgementIcons()->GetNACLogo() );

  vtkKWLabel *NAMICLabel = vtkKWLabel::New();
  NAMICLabel->SetParent ( this->GetLogoFrame() );
  NAMICLabel->Create();
  NAMICLabel->SetImageToIcon ( this->GetAcknowledgementIcons()->GetNAMICLogo() );    

  vtkKWLabel *NCIGTLabel = vtkKWLabel::New();
  NCIGTLabel->SetParent ( this->GetLogoFrame() );
  NCIGTLabel->Create();
  NCIGTLabel->SetImageToIcon ( this->GetAcknowledgementIcons()->GetNCIGTLogo() );
    
  vtkKWLabel *BIRNLabel = vtkKWLabel::New();
  BIRNLabel->SetParent ( this->GetLogoFrame() );
  BIRNLabel->Create();
  BIRNLabel->SetImageToIcon ( this->GetAcknowledgementIcons()->GetBIRNLogo() );

  app->Script ( "grid %s -row 0 -column 0 -padx 2 -pady 2 -sticky w", NAMICLabel->GetWidgetName());
  app->Script ("grid %s -row 0 -column 1 -padx 2 -pady 2 -sticky w", NACLabel->GetWidgetName());
  app->Script ( "grid %s -row 1 -column 0 -padx 2 -pady 2 -sticky w",  BIRNLabel->GetWidgetName());
  app->Script ( "grid %s -row 1 -column 1 -padx 2 -pady 2 -sticky w",  NCIGTLabel->GetWidgetName());                  

  NACLabel->Delete();
  NAMICLabel->Delete();
  NCIGTLabel->Delete();
  BIRNLabel->Delete();
  
  // create icons
  this->FetchMIIcons = vtkFetchMIIcons::New();

  // server panel
  vtkKWFrame *serverFrame = vtkKWFrame::New();
  serverFrame->SetParent ( page );
  serverFrame->Create();
  this->Script ( "pack %s -side top -anchor nw -fill x -expand y -padx 2 -pady 2 -in %s",
                 serverFrame->GetWidgetName(), page->GetWidgetName() );

  vtkKWLabel *l1 = vtkKWLabel::New();
  l1->SetParent ( serverFrame );
  l1->Create();
  l1->SetText ( "Choose a server:" );
  this->ServerMenuButton = vtkKWMenuButton::New();
  this->ServerMenuButton->SetParent (serverFrame );
  this->ServerMenuButton->Create();
  this->ServerMenuButton->SetValue ( "none" );

  this->Notebook = vtkKWNotebook::New();
  this->Notebook->SetParent ( serverFrame );
  this->Notebook->Create();
  this->Notebook->UseFrameWithScrollbarsOn();
  this->Notebook->SetPageTabColor ( _br, _bg, _bb);
  this->Notebook->SetSelectedPageTabColor (1.0, 1.0, 1.0 );

  this->Notebook->AddPage ( "Query for Scenes", "Query the server for resources that match selected metadata." );
  this->Notebook->AddPage ( "Browse Query Results & Download", "Browse resources and select a scene for download." );
  this->Notebook->AddPage ( "Tag Scene & Upload", "Apply metadata to datasets and the scene and upload to selected server." );
  this->Notebook->RaisePage ("Query for Scenes");

  
  this->Script ( "grid %s -row 0 -column 0 -sticky e -padx 2 -pady 2", l1->GetWidgetName() );
  this->Script ( "grid %s -row 0 -column 1 -sticky ew -padx 2 -pady 2", this->ServerMenuButton->GetWidgetName() );
  this->Script ( "grid %s -row 1 -column 0 -columnspan 2 -sticky ew -padx 2 -pady 2", this->Notebook->GetWidgetName() );
  this->Script ( "grid columnconfigure %s 0 -weight 0", serverFrame->GetWidgetName() );
  this->Script ( "grid columnconfigure %s 1 -weight 1", serverFrame->GetWidgetName() );

  this->QueryList = vtkFetchMIQueryTermWidget::New();
  this->QueryList->SetParent ( this->Notebook->GetFrame ( "Query for Scenes") );
  this->QueryList->Create();
  this->QueryList->SetApplication ( app );
  this->QueryList->SetLogic ( this->Logic );
  this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2", this->QueryList->GetWidgetName() );

  this->ResourceList = vtkFetchMIFlatResourceWidget::New();
  this->ResourceList->SetParent ( this->Notebook->GetFrame ("Browse Query Results & Download") );
  this->ResourceList->Create();
  this->ResourceList->SetApplication ( app );
  this->ResourceList->SetLogic ( this->Logic );
  this->ResourceList->SetMRMLScene ( this->MRMLScene );
  this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2", this->ResourceList->GetWidgetName() );

  this->TaggedDataList = vtkFetchMIResourceUploadWidget::New();
  this->TaggedDataList->SetParent ( this->Notebook->GetFrame ( "Tag Scene & Upload" ) );
  this->TaggedDataList->Create();
  this->TaggedDataList->SetApplication ( app );
  this->TaggedDataList->SetLogic ( this->Logic );
  this->TaggedDataList->SetMRMLScene ( this->MRMLScene );
  this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2", this->TaggedDataList->GetWidgetName() );

  l1->Delete();
  serverFrame->Delete();

  this->UpdateGUI();
//  this->Logic->CreateTemporaryFiles();
  this->Logic->InitializeInformatics();
  this->LoadTclPackage();
}


//---------------------------------------------------------------------------
void vtkFetchMIGUI::Init ( )
{
}
