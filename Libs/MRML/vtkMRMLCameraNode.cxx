/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLCameraNode.cxx,v $
Date:      $Date: 2006/03/03 22:26:39 $
Version:   $Revision: 1.3 $

=========================================================================auto=*/
#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"
#include "vtkCallbackCommand.h"

#include "vtkMRMLCameraNode.h"
#include "vtkMRMLViewNode.h"
#include "vtkMRMLScene.h"

//------------------------------------------------------------------------------
vtkMRMLCameraNode* vtkMRMLCameraNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLCameraNode");
  if(ret)
    {
    return (vtkMRMLCameraNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLCameraNode;
}

//-----------------------------------------------------------------------------

vtkMRMLNode* vtkMRMLCameraNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLCameraNode");
  if(ret)
    {
    return (vtkMRMLCameraNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLCameraNode;
}


//----------------------------------------------------------------------------
vtkMRMLCameraNode::vtkMRMLCameraNode()
{
  //this->SingletonTag = const_cast<char *>("vtkMRMLCameraNode");

  this->HideFromEditors = 0;

  this->ActiveTag = NULL;
  this->Camera = NULL;
  vtkCamera *camera = vtkCamera::New();

  camera->SetPosition(0, 500, 0);
  camera->SetFocalPoint(0, 0, 0);
  camera->SetViewUp(0, 0, 1);

  this->SetAndObserveCamera(camera); 
  camera->Delete();
 }

//----------------------------------------------------------------------------
vtkMRMLCameraNode::~vtkMRMLCameraNode()
{
  this->SetAndObserveCamera(NULL);
}

//----------------------------------------------------------------------------
void vtkMRMLCameraNode::WriteXML(ostream& of, int nIndent)
{
  // Write all attributes not equal to their defaults
  
  Superclass::WriteXML(of, nIndent);

  vtkIndent indent(nIndent);

  double *position = this->GetPosition();
  of << indent << " position=\"" << position[0] << " "
    << position[1] << " "
    << position[2] << "\"";

  double *focalPoint = this->GetFocalPoint();
  of << indent << " focalPoint=\"" << focalPoint[0] << " "
    << focalPoint[1] << " "
    << focalPoint[2] << "\"";

  double *viewUp = this->GetViewUp();
    of << indent << " viewUp=\"" << viewUp[0] << " "
      << viewUp[1] << " "
      << viewUp[2] << "\"";

  of << indent << " parallelProjection=\"" << (this->GetParallelProjection() ? "true" : "false") << "\"";

  of << indent << " parallelScale=\"" << this->GetParallelScale() << "\"";

  if (this->ActiveTag)
    {
    of << indent << " activetag=\"" << this->ActiveTag << "\"";
    }
}

//----------------------------------------------------------------------------
void vtkMRMLCameraNode::ReadXMLAttributes(const char** atts)
{
  int disabledModify = this->StartModify();

  Superclass::ReadXMLAttributes(atts);

  const char* attName;
  const char* attValue;
  while (*atts != NULL) 
    {
    attName = *(atts++);
    attValue = *(atts++);
    if (!strcmp(attName, "position")) 
      {
      std::stringstream ss;
      ss << attValue;
      double Position[3];
      ss >> Position[0];
      ss >> Position[1];
      ss >> Position[2];
      this->SetPosition(Position);
      }
    else if (!strcmp(attName, "focalPoint")) 
      {
      std::stringstream ss;
      ss << attValue;
      double FocalPoint[3];
      ss >> FocalPoint[0];
      ss >> FocalPoint[1];
      ss >> FocalPoint[2];
      this->SetFocalPoint(FocalPoint);
      }
    else if (!strcmp(attName, "viewUp")) 
      {
      std::stringstream ss;
      ss << attValue;
      double ViewUp[3];
      ss >> ViewUp[0];
      ss >> ViewUp[1];
      ss >> ViewUp[2];
      this->SetViewUp(ViewUp);
      }
    else if (!strcmp(attName, "parallelProjection")) 
      {
      if (!strcmp(attValue,"true")) 
        {
        this->SetParallelProjection(1);
        }
      else
        {
        this->SetParallelProjection(0);
        }
      }
    else if (!strcmp(attName, "parallelScale")) 
      {
      std::stringstream ss;
      ss << attValue;
      double parallelScale;
      ss >> parallelScale;
      this->SetParallelScale(parallelScale);
      }
    else if (!strcmp(attName, "activetag")) 
      {
      this->SetActiveTag(attValue);
      }
    else if (!strcmp(attName, "active")) 
      {
      // Legacy, was replaced by active tag, try to set ActiveTag instead
      // to link to the main viewer
      if (!this->ActiveTag && this->Scene)
        {
        vtkMRMLViewNode *vnode = vtkMRMLViewNode::SafeDownCast(
          this->Scene->GetNthNodeByClass(0, "vtkMRMLViewNode")); 
        if (vnode)
        {
          this->SetActiveTag(vnode->GetName());
        }
        }
      }
    }  
    this->EndModify(disabledModify);

}


//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, ID
void vtkMRMLCameraNode::Copy(vtkMRMLNode *anode)
{
  int disabledModify = this->StartModify();

  Superclass::Copy(anode);
  vtkMRMLCameraNode *node = (vtkMRMLCameraNode *) anode;


  this->SetPosition(node->GetPosition());
  this->SetFocalPoint(node->GetFocalPoint());
  this->SetViewUp(node->GetViewUp());
  this->SetParallelProjection(node->GetParallelProjection());
  this->SetParallelScale(node->GetParallelScale());
  //this->ActiveTag = node->GetActiveTag();

  this->EndModify(disabledModify);

}

//----------------------------------------------------------------------------
void vtkMRMLCameraNode::PrintSelf(ostream& os, vtkIndent indent)
{
  int idx;
  
  Superclass::PrintSelf(os,indent);

  os << "Position:\n";
  for (idx = 0; idx < 2; ++idx)
    {
    os << indent << ", " << (this->GetPosition())[idx];
    }
  os << "FocalPoint:\n";
  for (idx = 0; idx < 2; ++idx)
    {
    os << indent << ", " << (this->GetFocalPoint())[idx];
    }
  os << "ViewUp:\n";
  for (idx = 0; idx < 2; ++idx)
    {
    os << indent << ", " << (this->GetViewUp())[idx];
    }
  os << indent << "ActiveTag: " <<
    (this->ActiveTag ? this->ActiveTag : "(none)") << "\n";
}

//----------------------------------------------------------------------------
void vtkMRMLCameraNode::SetAndObserveCamera(vtkCamera *camera)
{
  if (this->Camera != NULL)
    {
    this->SetCamera(NULL);
    }
  this->SetCamera(camera);
  if ( this->Camera )
    {
    vtkEventBroker::GetInstance()->AddObservation (
      this->Camera, vtkCommand::ModifiedEvent, this, this->MRMLCallbackCommand );
    }
}


//---------------------------------------------------------------------------
void vtkMRMLCameraNode::ProcessMRMLEvents ( vtkObject *caller,
                                            unsigned long event, 
                                            void *callData )
{
  Superclass::ProcessMRMLEvents ( caller, event, callData );

  if (this->Camera != NULL && this->Camera == vtkCamera::SafeDownCast(caller) &&
      event ==  vtkCommand::ModifiedEvent)
    {
    this->InvokeEvent(vtkCommand::ModifiedEvent, NULL);
    }
}

//---------------------------------------------------------------------------
void vtkMRMLCameraNode::SetActiveTag(const char *_arg) 
{
  if (this->ActiveTag == NULL && _arg == NULL) 
    { 
    return;
    }

  if (this->ActiveTag && _arg && 
    (!strcmp(this->ActiveTag, _arg))) 
    {
    return;
    }

  // If a camera is already using that new tag, let's find it and assign it our
  // old tag later on if it's not ourself (thus performing a swap).

  vtkMRMLCameraNode *previous_owner = this->FindActiveTagInScene(_arg);
  if (previous_owner && previous_owner != this)
    {
    previous_owner->SetActiveTag(NULL);
    }
  std::string previous_active_tag(this->ActiveTag ? this->ActiveTag : "");

  if (this->ActiveTag) 
    { 
    delete [] this->ActiveTag; 
    }

  if (_arg)
    {
    this->ActiveTag = new char[strlen(_arg) + 1];
    strcpy(this->ActiveTag, _arg);
    }
  else
    {
    this->ActiveTag = NULL;
    }

  this->Modified();

  this->InvokeEvent(vtkMRMLCameraNode::ActiveTagModifiedEvent, NULL);

  // Swap with previous owner

  if (previous_owner && previous_owner != this)
    {
    previous_owner->SetActiveTag(
      previous_active_tag.size() ? previous_active_tag.c_str() : NULL);
    }
}

//----------------------------------------------------------------------------
void vtkMRMLCameraNode::RemoveActiveTagInScene(const char *tag)
{
  if (this->Scene == NULL || tag == NULL)
    {
    return;
    }

  vtkMRMLCameraNode *node = NULL;
  int nnodes = this->Scene->GetNumberOfNodesByClass("vtkMRMLCameraNode");
  for (int n = 0; n < nnodes; n++)
    {
    node = vtkMRMLCameraNode::SafeDownCast (
       this->Scene->GetNthNodeByClass(n, "vtkMRMLCameraNode"));
    if (node != this && 
        node->GetActiveTag() && 
        !strcmp(node->GetActiveTag(), tag))
      {
      node->SetActiveTag(NULL);
      }
    }
}

//----------------------------------------------------------------------------
vtkMRMLCameraNode* vtkMRMLCameraNode::FindActiveTagInScene(const char *tag)
{
  if (this->Scene == NULL || tag == NULL)
    {
    return NULL;
    }

  vtkMRMLCameraNode *node = NULL;
  int nnodes = this->Scene->GetNumberOfNodesByClass("vtkMRMLCameraNode");
  for (int n = 0; n < nnodes; n++)
    {
    node = vtkMRMLCameraNode::SafeDownCast (
      this->Scene->GetNthNodeByClass(n, "vtkMRMLCameraNode"));
    if (node != this && 
        node->GetActiveTag() && 
        !strcmp(node->GetActiveTag(), tag))
      {
      return node;
      }
    }

  return NULL;
}
