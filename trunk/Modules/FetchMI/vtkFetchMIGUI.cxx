#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"
#include "vtkCommand.h"

#include "vtkFetchMIGUI.h"

#include "vtkSlicerApplication.h"
#include "vtkSlicerApplicationLogic.h"
#include "vtkSlicerNodeSelectorWidget.h"

#include "vtkMRMLStorageNode.h"
#include "vtkMRMLVolumeNode.h"
#include "vtkMRMLModelStorageNode.h"
#include "vtkMRMLFreeSurferModelStorageNode.h"
#include "vtkMRMLNRRDStorageNode.h"
#include "vtkMRMLVolumeArchetypeStorageNode.h"
#include "vtkMRMLDiffusionTensorVolumeNode.h"
#include "vtkMRMLDiffusionWeightedVolumeNode.h"
#include "vtkMRMLScalarVolumeNode.h"

#include "vtkXNDTagTable.h"
#include "vtkHIDTagTable.h"
#include "vtkTagTable.h"

#include "vtkKWApplication.h"
#include "vtkKWWidget.h"
#include "vtkKWEntry.h"
#include "vtkKWEntryWithLabel.h"
#include "vtkKWLabel.h"
#include "vtkKWMenu.h"
#include "vtkKWMenuButton.h"
#include "vtkKWMenuButtonWithLabel.h"
#include "vtkKWFrame.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWPushButton.h"

#include "vtkFetchMIIcons.h"
#include "vtkKWMultiColumnList.h"
#include "vtkKWMultiColumnListWithScrollbars.h"
#include "vtkFetchMIQueryTermWidget.h"
#include "vtkFetchMIFlatResourceWidget.h"
#include "vtkFetchMIResourceUploadWidget.h"
#include "vtkFetchMITagViewWidget.h"

#include <map>
#include <string>
#include <vector>

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
  this->ServerMenuButton = NULL;
  this->AddServerEntry = NULL;
  this->FetchMIIcons = NULL;
  this->QueryTagsButton = NULL;
  this->FetchMINode = NULL;
  this->UpdatingGUI = 0;
  this->UpdatingMRML = 0;
  this->DataDirectoryName = NULL;
}

//----------------------------------------------------------------------------
vtkFetchMIGUI::~vtkFetchMIGUI()
{
    this->RemoveMRMLNodeObservers ( );
    this->RemoveLogicObservers ( );
    
    if ( this->QueryTagsButton )
      {
      this->QueryTagsButton->SetParent ( NULL );
      this->QueryTagsButton->Delete();
      this->QueryTagsButton = NULL;
      }
    if ( this->QueryList )
      {
      this->QueryList->SetParent ( NULL );
      this->QueryList->Delete();
      this->QueryList = NULL;
      }
    if ( this->ResourceList )
      {
      this->ResourceList->SetParent ( NULL );
      this->ResourceList->Delete();
      this->ResourceList = NULL;
      }
    if ( this->TaggedDataList )
      {
      this->TaggedDataList->SetParent ( NULL );
      this->TaggedDataList->Delete();
      this->TaggedDataList = NULL;
      }
    
    if ( this->AddServerButton )
      {
      this->AddServerButton->SetParent ( NULL );
      this->AddServerButton->Delete();
      this->AddServerButton = NULL;
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

    this->UpdatingMRML = 0;
    this->UpdatingGUI = 0;
    this->SetDataDirectoryName ( NULL );

    this->Logic = NULL;
    vtkSetAndObserveMRMLNodeMacro( this->FetchMINode, NULL );
}

//----------------------------------------------------------------------------
void vtkFetchMIGUI::TearDownGUI ( )
{
  if ( this->FetchMINode )
    {
    vtkSetAndObserveMRMLNodeMacro( this->FetchMINode, NULL );
    }
  this->QueryList->RemoveWidgetObservers();
  this->ResourceList->RemoveWidgetObservers();
  this->TaggedDataList->RemoveWidgetObservers();
  this->RemoveGUIObservers ( );
  this->Logic->SetFetchMINode ( NULL );
  this->SetLogic ( NULL );
  this->SetAndObserveMRMLScene ( NULL );
}


//----------------------------------------------------------------------------
void vtkFetchMIGUI::PrintSelf(ostream& os, vtkIndent indent)
{
  
}

//---------------------------------------------------------------------------
void vtkFetchMIGUI::AddGUIObservers ( ) 
{
  this->QueryList->AddWidgetObservers();
  this->QueryList->AddObserver(vtkFetchMIQueryTermWidget::TagChangedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->QueryList->AddObserver(vtkFetchMIQueryTermWidget::QuerySubmittedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->ResourceList->AddWidgetObservers();
  this->TaggedDataList->AddObserver(vtkFetchMIResourceUploadWidget::TagSelectedDataEvent, (vtkCommand *)this->GUICallbackCommand);
  this->TaggedDataList->AddObserver(vtkFetchMIResourceUploadWidget::ShowAllTagViewEvent, (vtkCommand *)this->GUICallbackCommand);
  this->TaggedDataList->AddWidgetObservers();
  this->QueryTagsButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->ServerMenuButton->GetMenu()->AddObserver ( vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
//  this->AddServerEntry->AddObserver ( vtkKWEntry::EntryValueChangedEvent, (vtkCommand *)this->GUICallbackCommand);
//  this->AddServerButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );

}



//---------------------------------------------------------------------------
void vtkFetchMIGUI::RemoveGUIObservers ( )
{
  this->QueryList->RemoveWidgetObservers();
  this->QueryList->RemoveObservers(vtkFetchMIQueryTermWidget::TagChangedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->QueryList->RemoveObservers(vtkFetchMIQueryTermWidget::QuerySubmittedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->ResourceList->RemoveWidgetObservers();
  this->TaggedDataList->RemoveObservers(vtkFetchMIResourceUploadWidget::TagSelectedDataEvent, (vtkCommand *)this->GUICallbackCommand);
  this->TaggedDataList->RemoveObservers(vtkFetchMIResourceUploadWidget::ShowAllTagViewEvent, (vtkCommand *)this->GUICallbackCommand);
  this->TaggedDataList->RemoveWidgetObservers();
  this->QueryTagsButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->ServerMenuButton->GetMenu()->RemoveObservers (vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
//  this->AddServerEntry->RemoveObservers (vtkKWEntry::EntryValueChangedEvent, (vtkCommand *)this->GUICallbackCommand);
//  this->AddServerButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );

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
    return;
    }

  vtkKWPushButton *b = vtkKWPushButton::SafeDownCast ( caller );
  vtkKWEntry *e = vtkKWEntry::SafeDownCast ( caller );
  vtkKWMenu *m = vtkKWMenu::SafeDownCast ( caller );
  vtkFetchMIResourceUploadWidget *w = vtkFetchMIResourceUploadWidget::SafeDownCast ( caller );
  vtkFetchMIQueryTermWidget *q= vtkFetchMIQueryTermWidget::SafeDownCast ( caller );

  if ( w != NULL )
    {
    if ( (w== this->TaggedDataList) && (event == vtkFetchMIResourceUploadWidget::TagSelectedDataEvent) )
      {
      this->TagSelectedData();
      }
    else if ( (w== this->TaggedDataList) && (event == vtkFetchMIResourceUploadWidget::ShowAllTagViewEvent) )
      {
      this->ShowAllTagView();
      }
    }
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

  if ( b && event == vtkKWPushButton::InvokedEvent )
    {
    if ( b == this->AddServerButton )
      {
      if ( this->GetAddServerEntry()->GetValue() != NULL )
        {
        this->FetchMINode->AddNewServer (this->GetAddServerEntry()->GetValue() );
        }
      }
    else if ( b == this->QueryTagsButton )
      {
      this->Logic->QueryServerForTags();
      // TODO: temporary fix for HID which we are
      // not yet querying for available tags. Just
      // repopulate from default tags in FetchMINode
      const char *svctype = this->GetFetchMINode()->GetSelectedServiceType();
      if ( svctype == NULL )
        {
        vtkErrorMacro ( "vtkFetchMIGUI: got null service type" );
        return;
        }
      if ( !(strcmp (svctype, "HID")))
        {
        this->UpdateTagTableFromMRML();
        }
      }
    }

  if ( e && event == vtkKWEntry::EntryValueChangedEvent )
    {
    if (e == this->AddServerEntry )
      {
      if ( e->GetValue() != NULL )
        {
        this->FetchMINode->AddNewServer(e->GetValue());
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
          this->FetchMINode->SetServer ( this->ServerMenuButton->GetValue() );
          this->UpdateTagTableFromMRML();
          }
        }
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
    const char *att;
    const char *val;
    int i, row;
    for (i=0; i < t->GetNumberOfTags(); i++ )
      {
      att = t->GetTagAttribute(i);
      val = t->GetTagValue(i);
      this->ResourceList->AddNewItem (att, val);
      row = this->ResourceList->GetRowForAttribute ( att );
      if ( row >= 0 && (t->IsTagSelected(att)) )
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
    vtkErrorMacro ("FetchMIGUI: UpdateTagTableFromMRML got a NULL FetchMINode." );
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

  vtkMRMLNode *node;
  int nnodes = this->MRMLScene->GetNumberOfNodesByClass("vtkMRMLVolumeNode");
  int n;
  int row = this->TaggedDataList->GetMultiColumnList()->GetWidget()->GetNumberOfRows();
  const char *dtype;
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
  int nnodes = this->MRMLScene->GetNumberOfNodesByClass("vtkMRMLModelNode");
  int n;
  int row = this->TaggedDataList->GetMultiColumnList()->GetWidget()->GetNumberOfRows();
  const char *dtype;
  
  for (n=0; n<nnodes; n++)
    {
    node = this->MRMLScene->GetNthNodeByClass(n, "vtkMRMLModelNode");
    if (node->GetHideFromEditors()) 
      {
      continue;
      }
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
    else
      {
      dtype = "VTKModel";
      }


    this->TaggedDataList->AddNewItem ( node->GetID(), dtype );
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
  
  const char *svctype = this->GetFetchMINode()->GetSelectedServiceType();
  if (svctype == NULL)
    {
    return;
    }

  int num = this->QueryList->GetMultiColumnList()->GetWidget()->GetNumberOfRows();
  std::string att;
  std::string val;
  int sel;
  //--- update the FetchMINode, depending on what service is selected.
  if ( !strcmp ( "XND", svctype ))
    {
    vtkXNDTagTable *t;
    if (this->FetchMINode->GetTagTableCollection()->FindTagTableByName ( "XNDTags" ) != NULL)
      {
      t = vtkXNDTagTable::SafeDownCast ( this->FetchMINode->GetTagTableCollection()->FindTagTableByName ( "XNDTags" ));
      if ( t == NULL )
        {
        // TODO: vtkErrorMacro
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
    }
  else if ( !strcmp ( "HID", svctype))
    {
    vtkHIDTagTable *t;
    if (this->FetchMINode->GetTagTableCollection()->FindTagTableByName ( "HIDTags" ) != NULL)
      {
      t = vtkHIDTagTable::SafeDownCast ( this->FetchMINode->GetTagTableCollection()->FindTagTableByName ( "HIDTags" ));
      if ( t == NULL )
        {
        // TODO: vtkErrorMacro
        return;
        }
      for ( int i=0; i < num; i++ )
        {
        att = this->QueryList->GetAttributeOfItem ( i );
        val = this->QueryList->GetValueOfItem ( i );
        if ( this->QueryList->IsItemSelected ( i ) )
          {
          t->AddOrUpdateTag ( att.c_str(), val.c_str(), 1 );
          }
        else
          {
          t->AddOrUpdateTag ( att.c_str(), val.c_str(), 0 );
          }
        }
      }
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
  
  const char *svctype = this->GetFetchMINode()->GetSelectedServiceType();
  if (svctype == NULL)
    {
    return;
    }
    
  //--- clear the table
  this->QueryList->DeleteAllItems();

  //--- now repopulate it from the FetchMINode, depending on
  //--- which service is selected.

  if ( !strcmp ( "XND", svctype ))
    {
    if (this->FetchMINode->GetTagTableCollection()->FindTagTableByName ( "XNDTags" ) != NULL)
      {
      vtkXNDTagTable *t = vtkXNDTagTable::SafeDownCast ( this->FetchMINode->GetTagTableCollection()->FindTagTableByName ( "XNDTags" ));
      if ( t != NULL )
        {
        const char *att;
        const char *val;
        int i, row;
        for (i=0; i < t->GetNumberOfTags(); i++ )
          {
          att = t->GetTagAttribute(i);
          val = t->GetTagValue(i);
          this->QueryList->AddNewItem (att, val );
          row = this->QueryList->GetRowForAttribute ( att );
          if ( row >= 0 && (t->IsTagSelected(att)) )
            {
            this->QueryList->SelectRow(row);              
            }
          }
        }
      }
    }

  else if ( !strcmp ( "HID", svctype))
    {
    if (this->FetchMINode->GetTagTableCollection()->FindTagTableByName ( "HIDTags" ) != NULL)
      {
      vtkHIDTagTable *t = vtkHIDTagTable::SafeDownCast ( this->FetchMINode->GetTagTableCollection()->FindTagTableByName ( "HIDTags" ));
      if ( t != NULL )
        {
        const char *att;
        const char *val;
        int i, row;
        for (i=0; i < t->GetNumberOfTags(); i++ )
          {
          att = t->GetTagAttribute(i);
          val = t->GetTagValue(i);
          this->QueryList->AddNewItem (att, val );
          row = this->QueryList->GetRowForAttribute ( att );
          if ( row >= 0 && (t->IsTagSelected(att)) )
            {
            this->QueryList->SelectRow(row);              
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
  // if parameter node has been changed externally, update GUI widgets with new values
  vtkMRMLFetchMINode* node = vtkMRMLFetchMINode::SafeDownCast(caller);

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
    if (event == vtkMRMLFetchMINode::TagResponseReadyEvent )
      {
      this->UpdateTagTableFromMRML();
      }
    if (event == vtkMRMLFetchMINode::ResourceResponseReadyEvent )
      {
      this->UpdateResourceTableFromMRML();
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
      std::string s;
      int l = this->FetchMINode->KnownServers.size();
      for (int i=0; i < l; i ++ )
        {
        s = this->FetchMINode->KnownServers[i];
        this->ServerMenuButton->GetMenu()->AddRadioButton ( s.c_str() );      
        }
      //TODO: hook up these commands!
      this->ServerMenuButton->GetMenu()->AddSeparator();
      this->ServerMenuButton->GetMenu()->AddCommand("Add New XND Server");
      this->ServerMenuButton->GetMenu()->AddCommand("Add New HID Server");
      
      //--- select active server in the ServerMenuButton
      if ( this->FetchMINode->GetSelectedServer() != NULL )
        {
        this->ServerMenuButton->SetValue ( this->FetchMINode->GetSelectedServer() );
        }
      else
        {
        this->ServerMenuButton->SetValue ( "<none>" );
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

  vtkFetchMITagViewWidget *viewer = vtkFetchMITagViewWidget::New();
  viewer->SetParent ( this->GetApplicationGUI()->GetMainSlicerWindow() );
  viewer->Create();

  viewer->SetTagTitle ("Tags for all currently selected data:");
  std::stringstream ss;
  vtkMRMLStorableNode *node;
  vtkTagTable *t;

  //--- figure out the text
  int dnum = this->TaggedDataList->GetNumberOfSelectedItems();
  int i, j;
  int numtags;
  const char *nodeID;
  const char *att;
  const char *val;
  for (i=0; i<dnum; i++)
    {
    nodeID = this->TaggedDataList->GetNthSelectedDataTarget(i);
    if ( nodeID != NULL )
      {
      //--- tag the data.
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
  
  viewer->SetTagText ( ss.str().c_str() );
  viewer->DisplayTagViewWindow();
}


//---------------------------------------------------------------------------
void vtkFetchMIGUI::TagSelectedData()
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
  
  //--- get all selected tags in QueryList
  vtkMRMLStorableNode *node;
  vtkTagTable *t;
  
  int i, j;
  int dnum;
  int num = this->QueryList->GetNumberOfSelectedItems();
  std::string att;
  std::string val;
  for ( i=0; i < num; i++)
    {
    att = this->QueryList->GetNthSelectedAttribute(i);
    val = this->QueryList->GetNthSelectedValue(i);

    if ( att.c_str() != NULL && val.c_str() != NULL )
      {
      //--- apply to all selected data in TaggedDataList
      dnum = this->TaggedDataList->GetNumberOfSelectedItems();
      const char *nodeID;
      for (j=0; j<dnum; j++)
        {
        nodeID = this->TaggedDataList->GetNthSelectedDataTarget(j);
        if ( nodeID != NULL )
          {
          //--- tag the data.
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
  const char *help = "FetchMI (Medical Informatics) help.\n\n *** Select a server\n\n  *** Query the server for tags. If server has any defined,  it'll fill up the top listbox\n\n *** You can add attributes for tags in the top listbox\n\n *** Select the tags you want to use in your query\n\n *** Click the spyglass to search the server for matching resources.\n\n *** Get a big blob of resources back (because something's wrong with my query string) and listed in the second listbox.\n\n *** Select a MRML scene file from that list\n\n *** And click download (and nothing happens yet.)\n\n more logic to come....\n\n ***The bottom listbox should initialize with all data in scene (and scene file) just like the savedatawidget.\n\n ***Each time a node added or deleted event occurs, it updates\n\n ***user selects tags in top box, and applies them to selected datasets in bottom box using the Apply tags button. Tags are preserved in node. (tagtable will be added to the scene file so scene tags can be preserved too).\n\n ***user can click tag-view icon to show all tags on any individual dataset  or scene in a popup widget.\n\n *** user can upload selected data or scene to selected server. (in our first pass, just scene).";

  const char *about = "This work was supported by NA-MIC, NAC, BIRN, NCIGT, and the Slicer Community. See <a>http://www.slicer.org</a> for details. \n\n";
  vtkKWWidget *page = this->UIPanel->GetPageWidget ( "FetchMI" );
  this->BuildHelpAndAboutFrame ( page, help, about );

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
  this->ServerMenuButton->SetValue ( "<none>" );
  this->QueryTagsButton = vtkKWPushButton::New();
  this->QueryTagsButton->SetParent ( serverFrame );
  this->QueryTagsButton->Create();
  this->QueryTagsButton->SetBorderWidth ( 0 );
  this->QueryTagsButton->SetReliefToFlat();
  this->QueryTagsButton->SetImageToIcon ( this->FetchMIIcons->GetQueryTagsIcon() );
  this->QueryTagsButton->SetBalloonHelpString ( "Query for tags that the selected web service supports.");  
  
/*
  vtkKWLabel *l2 = vtkKWLabel::New();
  l2->SetParent ( serverFrame );
  l2->Create();
  l2->SetText ( "Add new (XND) server:" );
  this->AddServerEntry = vtkKWEntry::New();
  this->AddServerEntry->SetParent ( serverFrame );
  this->AddServerEntry->Create ();
  this->AddServerEntry->SetValue ( "" );
  this->AddServerButton = vtkKWPushButton::New();
  this->AddServerButton->SetParent ( serverFrame );
  this->AddServerButton->Create();
  this->AddServerButton->SetBorderWidth ( 0 );
  this->AddServerButton->SetReliefToFlat();  
  this->AddServerButton->SetImageToIcon ( this->FetchMIIcons->GetAddNewIcon() );
  this->AddServerButton->SetBalloonHelpString ( "Add a new XNAT Desktop server to the menu" );
  this->Script ( "grid %s -row 0 -column 0 -sticky e -padx 2 -pady 2", l2->GetWidgetName() );
  this->Script ( "grid %s -row 0 -column 1 -sticky ew -padx 2 -pady 2", this->AddServerEntry->GetWidgetName() );
  this->Script ( "grid %s -row 0 -column 2 -sticky w -padx 2 -pady 2", this->AddServerButton->GetWidgetName() );
  this->Script ( "grid %s -row 1 -column 0 -sticky e -padx 2 -pady 2", l1->GetWidgetName() );
  this->Script ( "grid %s -row 1 -column 1 -sticky ew -padx 2 -pady 2", this->ServerMenuButton->GetWidgetName() );
  this->Script ( "grid %s -row 1 -column 2 -sticky w -padx 2 -pady 2", this->QueryTagsButton->GetWidgetName() );
  this->Script ( "grid columnconfigure %s 0 -weight 0", serverFrame->GetWidgetName() );
  this->Script ( "grid columnconfigure %s 1 -weight 1", serverFrame->GetWidgetName() );
  this->Script ( "grid columnconfigure %s 2 -weight 0", serverFrame->GetWidgetName() );
*/
  
  this->Script ( "grid %s -row 0 -column 0 -sticky e -padx 2 -pady 2", l1->GetWidgetName() );
  this->Script ( "grid %s -row 0 -column 1 -sticky ew -padx 2 -pady 2", this->ServerMenuButton->GetWidgetName() );
  this->Script ( "grid %s -row 0 -column 2 -sticky w -padx 2 -pady 2", this->QueryTagsButton->GetWidgetName() );
  this->Script ( "grid columnconfigure %s 0 -weight 0", serverFrame->GetWidgetName() );
  this->Script ( "grid columnconfigure %s 1 -weight 1", serverFrame->GetWidgetName() );
  this->Script ( "grid columnconfigure %s 2 -weight 0", serverFrame->GetWidgetName() );
  
  // Query Frame
  vtkSlicerModuleCollapsibleFrame *queryFrame = vtkSlicerModuleCollapsibleFrame::New ( );
  queryFrame->SetParent(page);
  queryFrame->Create();
  queryFrame->SetLabelText("Query Webservices");
  queryFrame->ExpandFrame();
  app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
    queryFrame->GetWidgetName(), page->GetWidgetName());

  // Download Frame
  vtkSlicerModuleCollapsibleFrame *resourceFrame = vtkSlicerModuleCollapsibleFrame::New ( );
  resourceFrame->SetParent(page);
  resourceFrame->Create();
  resourceFrame->SetLabelText("Browse Results & Download");
  resourceFrame->ExpandFrame();
  app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
    resourceFrame->GetWidgetName(), page->GetWidgetName());

  // Tag & Upload Frame
  vtkSlicerModuleCollapsibleFrame *descriptionFrame = vtkSlicerModuleCollapsibleFrame::New ( );
  descriptionFrame->SetParent(page);
  descriptionFrame->Create();
  descriptionFrame->SetLabelText("Describe Data & Upload");
  descriptionFrame->ExpandFrame();
  app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
    descriptionFrame->GetWidgetName(), page->GetWidgetName());

  this->QueryList = vtkFetchMIQueryTermWidget::New();
  this->QueryList->SetParent ( queryFrame->GetFrame() );
  this->QueryList->Create();
  this->QueryList->SetLogic ( this->Logic );
  this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2", this->QueryList->GetWidgetName() );

  this->ResourceList = vtkFetchMIFlatResourceWidget::New();
  this->ResourceList->SetParent ( resourceFrame->GetFrame() );
  this->ResourceList->Create();
  this->ResourceList->SetLogic ( this->Logic );
  this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2", this->ResourceList->GetWidgetName() );

  this->TaggedDataList = vtkFetchMIResourceUploadWidget::New();
  this->TaggedDataList->SetParent ( descriptionFrame->GetFrame() );
  this->TaggedDataList->Create();
  this->TaggedDataList->SetLogic ( this->Logic );
  this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2", this->TaggedDataList->GetWidgetName() );

  // Clean up.
  l1->Delete();
//  l2->Delete();  this->Script ( "grid columnconfigure %s 0 -weight 0", serverFrame->GetWidgetName() );
  this->Script ( "grid columnconfigure %s 1 -weight 1", serverFrame->GetWidgetName() );
  this->Script ( "grid columnconfigure %s 2 -weight 0", serverFrame->GetWidgetName() );
  serverFrame->Delete();
  queryFrame->Delete();
  resourceFrame->Delete();
  descriptionFrame->Delete();  this->Script ( "grid columnconfigure %s 0 -weight 0", serverFrame->GetWidgetName() );
  this->Script ( "grid columnconfigure %s 1 -weight 1", serverFrame->GetWidgetName() );
  this->Script ( "grid columnconfigure %s 2 -weight 0", serverFrame->GetWidgetName() );

  this->UpdateGUI();
  this->Logic->CreateTemporaryFiles();
}


//---------------------------------------------------------------------------
void vtkFetchMIGUI::Init ( )
{
}
