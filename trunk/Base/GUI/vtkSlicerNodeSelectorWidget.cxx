/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkSlicerNodeSelectorWidget.cxx,v $
Date:      $Date: 2006/03/17 15:10:10 $
Version:   $Revision: 1.2 $

=========================================================================auto=*/

#include "vtkSlicerNodeSelectorWidget.h"


#include "vtkKWMenu.h"
#include "vtkKWMenuButton.h"
#include "vtkKWMenuButtonWithSpinButtons.h"

#include <sstream>

//----------------------------------------------------------------------------
vtkStandardNewMacro( vtkSlicerNodeSelectorWidget );
vtkCxxRevisionMacro(vtkSlicerNodeSelectorWidget, "$Revision: 1.33 $");


//----------------------------------------------------------------------------
// Description:
// the MRMLCallback is a static function to relay modified events from the 
// observed mrml scene back into the logic layer for further processing
// - this can also end up calling observers of the logic (i.e. in the GUI)
//
static void MRMLCallback(vtkObject *__mrmlscene, unsigned long eid, void *__clientData, void *callData)
{
  static int inMRMLCallback = 0;

  if (inMRMLCallback)
    {
    vtkErrorWithObjectMacro (__mrmlscene, << "*********MRMLCallback called recursively?" << endl);
    return;
    }
  inMRMLCallback = 1;

  vtkMRMLScene *mrmlscene = static_cast<vtkMRMLScene *>(__mrmlscene); // Not used, since it is ivar

  vtkSlicerNodeSelectorWidget *self = reinterpret_cast<vtkSlicerNodeSelectorWidget *>(__clientData);

  self->UpdateMenu();

  inMRMLCallback = 0;
}


//----------------------------------------------------------------------------
vtkSlicerNodeSelectorWidget::vtkSlicerNodeSelectorWidget()
{
  this->NewNodeCount = 0;
  this->NewNodeEnabled = 0;
  this->MRMLScene      = NULL;
  this->MRMLCallbackCommand = vtkCallbackCommand::New();
  this->MRMLCallbackCommand->SetClientData( reinterpret_cast<void *> (this) );
  this->MRMLCallbackCommand->SetCallback(MRMLCallback);
}

//----------------------------------------------------------------------------
vtkSlicerNodeSelectorWidget::~vtkSlicerNodeSelectorWidget()
{
  this->SetMRMLScene ( NULL );
  if (this->MRMLCallbackCommand)
    {
    this->MRMLCallbackCommand->Delete();
    this->MRMLCallbackCommand = NULL;
    }
}

//----------------------------------------------------------------------------
void vtkSlicerNodeSelectorWidget::SetMRMLScene( vtkMRMLScene *MRMLScene)
{
  if ( this->MRMLScene )
    {
    this->MRMLScene->RemoveObserver( this->MRMLCallbackCommand );
    this->MRMLScene->Delete ( );
    this->MRMLScene = NULL;
    //    this->MRMLScene->Delete();
    }
  
  this->MRMLScene = MRMLScene;

  if ( this->MRMLScene )
    {
    this->MRMLScene->Register(this);
    this->MRMLScene->AddObserver( vtkCommand::ModifiedEvent, this->MRMLCallbackCommand );
    }

  this->UpdateMenu();
}

//----------------------------------------------------------------------------
void vtkSlicerNodeSelectorWidget::UpdateMenu()
{
    if (this->NodeClasses.size() == 0)
      {
      return;
      }
    vtkKWMenuButton *mb = this->GetWidget()->GetWidget();
    vtkKWMenu *m = mb->GetMenu();

    vtkMRMLNode *oldSelectedNode = this->GetSelected();
    m->DeleteAllItems();
    int count = 0;
    int c=0;

    if (this->NewNodeEnabled)
    {
      for (c=0; c < this->GetNumberOfNodeClasses(); c++)
      {
        const char *name = this->MRMLScene->GetTagByClassName(this->GetNodeClass(c));
        std::stringstream ss;
        ss << "Create New " << name;
        
        std::stringstream sc;
        sc << "ProcessNewNodeCommand " << this->GetNodeClass(c);

        this->GetWidget()->GetWidget()->GetMenu()->AddRadioButton(ss.str().c_str());
        this->GetWidget()->GetWidget()->GetMenu()->SetItemCommand(count++, this, sc.str().c_str() );
        this->GetWidget()->GetWidget()->SetValue(ss.str().c_str());
      }
    }

    vtkMRMLNode *node = NULL;
    vtkMRMLNode *selectedNode = NULL;
    this->MRMLScene->InitTraversal();
    bool selected = false;
    for (c=0; c < this->GetNumberOfNodeClasses(); c++)
    {
      const char *className = this->GetNodeClass(c);
      while ( (node = this->MRMLScene->GetNextNodeByClass(className) ) != NULL)
      {
        std::stringstream sc;
        sc << "ProcessCommand " << node->GetID();

        this->GetWidget()->GetWidget()->GetMenu()->AddRadioButton(node->GetName());
        this->GetWidget()->GetWidget()->GetMenu()->SetItemCommand(count++, this, sc.str().c_str());
        if (oldSelectedNode == node)
        {
          selectedNode = node;
          selected = true;
        }
        else if (!selected)
        {  
          selectedNode = node;
          selected = true;
        }
      }
    }
    if (selectedNode != NULL)
      {
      this->GetWidget()->GetWidget()->SetValue(selectedNode->GetName());
      this->SelectedID = std::string(selectedNode->GetID());
      }
}

//----------------------------------------------------------------------------
vtkMRMLNode *vtkSlicerNodeSelectorWidget::GetSelected()
{
  vtkMRMLNode *node = this->MRMLScene->GetNodeByID (this->SelectedID.c_str());
  return node;
}

//----------------------------------------------------------------------------
void vtkSlicerNodeSelectorWidget::ProcessNewNodeCommand(char *className)
{
  vtkMRMLNode *node = NULL;
  vtkKWMenuButton *mb = this->GetWidget()->GetWidget();
  vtkKWMenu *m = mb->GetMenu();

  if (className)
    {
    node = this->MRMLScene->CreateNodeByClass( className );
    if (node == NULL)
    {
      return;
    }
    node->SetScene(this->MRMLScene);
    std::stringstream ss;
    ss << this->MRMLScene->GetTagByClassName(className) << NewNodeCount++;
    
    node->SetName(ss.str().c_str());
    node->SetID(this->MRMLScene->GetUniqueIDByClass(className));
    this->MRMLScene->AddNode(node);
    }
  this->SetSelected(node);
}


//----------------------------------------------------------------------------
void vtkSlicerNodeSelectorWidget::ProcessCommand(char *slectedId)
{
  this->SelectedID = std::string(slectedId);

  this->InvokeEvent(vtkSlicerNodeSelectorWidget::NodeSelectedEvent, NULL);
}


//----------------------------------------------------------------------------
void vtkSlicerNodeSelectorWidget::SetSelected(vtkMRMLNode *node)
{
  if ( node != NULL) 
    {
    vtkKWMenuButton *m = this->GetWidget()->GetWidget();
    if ( !strcmp ( m->GetValue(), node->GetName() ) )
      {
      return; // no change, don't propogate events
      }

    // new value, set it and notify observers
    m->SetValue(node->GetName());
    this->SetBalloonHelpString(node->GetName());
    this->SelectedID = std::string(node->GetID());
    this->InvokeEvent(vtkSlicerNodeSelectorWidget::NodeSelectedEvent, NULL);
    }
}

//----------------------------------------------------------------------------
void vtkSlicerNodeSelectorWidget::SetSelectedNew(const char *className)
{
  if (this->NewNodeEnabled) 
    {
    const char *name = this->MRMLScene->GetTagByClassName(className);
    std::stringstream ss;
    ss << "Create New " << name;
    this->GetWidget()->GetWidget()->SetValue(ss.str().c_str());
    this->SetBalloonHelpString("Create a new Node");
    }
}
//----------------------------------------------------------------------------
void vtkSlicerNodeSelectorWidget::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

