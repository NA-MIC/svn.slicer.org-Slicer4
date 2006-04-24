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
    cout << "*********MRMLCallback called recursively?" << endl;
    return;
    }
  inMRMLCallback = 1;

  cout << "In MRMLCallback" << endl;

  vtkMRMLScene *mrmlscene = static_cast<vtkMRMLScene *>(__mrmlscene); // Not used, since it is ivar

  vtkSlicerNodeSelectorWidget *self = reinterpret_cast<vtkSlicerNodeSelectorWidget *>(__clientData);

  self->UpdateMenu();

  inMRMLCallback = 0;
}


//----------------------------------------------------------------------------
vtkSlicerNodeSelectorWidget::vtkSlicerNodeSelectorWidget()
{
  this->NodeClass = NULL;
  this->MRMLScene      = NULL;
  this->MRMLCallbackCommand = vtkCallbackCommand::New();
  this->MRMLCallbackCommand->SetClientData( reinterpret_cast<void *> (this) );
  this->MRMLCallbackCommand->SetCallback(MRMLCallback);
}

//----------------------------------------------------------------------------
vtkSlicerNodeSelectorWidget::~vtkSlicerNodeSelectorWidget()
{
  this->SetNodeClass(NULL);

  if (this->MRMLScene)
    {
    this->MRMLScene->Delete();
    this->MRMLScene = NULL;
    }
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
    this->MRMLScene->Delete();
    }
  
  this->MRMLScene = MRMLScene;

  if ( this->MRMLScene )
    {
    this->MRMLScene->Register(this);
    this->MRMLScene->AddObserver( vtkCommand::ModifiedEvent, this->MRMLCallbackCommand );
    }
}

//----------------------------------------------------------------------------
void vtkSlicerNodeSelectorWidget::UpdateMenu()
{
    if (this->NodeClass == NULL)
      {
      return;
      }

    vtkKWMenuButton *mb = this->GetWidget()->GetWidget();
    vtkKWMenu *m = mb->GetMenu();

    int index;
    m->DeleteAllItems();

    vtkMRMLNode *node;
    int nth_rank = 0;
    this->MRMLScene->InitTraversal();
    while ( (node = this->MRMLScene->GetNextNodeByClass(this->NodeClass) ) != NULL)
      {
      // TODO: figure out how to use the ID instead of the name as the menu indicator
      this->GetWidget()->GetWidget()->GetMenu()->AddRadioButton(node->GetName());
      m->SetItemSelectedValueAsInt(0, nth_rank++);
      }


}

//----------------------------------------------------------------------------
vtkMRMLNode *vtkSlicerNodeSelectorWidget::GetSelected()
{
    vtkKWMenuButton *mb = this->GetWidget()->GetWidget();
    vtkKWMenu *m = mb->GetMenu();

    int nth_rank = m->GetItemSelectedValueAsInt(
      m->GetIndexOfItem(mb->GetValue()));
    vtkMRMLNode *n = this->MRMLScene->GetNthNodeByClass (nth_rank, this->NodeClass);
    return n;
}

//----------------------------------------------------------------------------
void vtkSlicerNodeSelectorWidget::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "MRMLScene: " << this->MRMLScene << endl;
}

