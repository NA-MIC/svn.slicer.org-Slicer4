/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women\"s Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLVolumeDisplayNode.cxx,v $
Date:      $Date: 2006/03/17 15:10:10 $
Version:   $Revision: 1.2 $

=========================================================================auto=*/

#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"
#include "vtkCallbackCommand.h"

#include "vtkMRMLVolumeDisplayNode.h"
#include "vtkMRMLScene.h"

// Initialize static member that controls resampling -- 
// old comment: "This offset will be changed to 0.5 from 0.0 per 2/8/2002 Slicer 
// development meeting, to move ijk coordinates to voxel centers."


//------------------------------------------------------------------------------
vtkMRMLVolumeDisplayNode* vtkMRMLVolumeDisplayNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLVolumeDisplayNode");
  if(ret)
    {
    return (vtkMRMLVolumeDisplayNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLVolumeDisplayNode;
}

//----------------------------------------------------------------------------

vtkMRMLNode* vtkMRMLVolumeDisplayNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLVolumeDisplayNode");
  if(ret)
    {
    return (vtkMRMLVolumeDisplayNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLVolumeDisplayNode;
}

//----------------------------------------------------------------------------
vtkMRMLVolumeDisplayNode::vtkMRMLVolumeDisplayNode()
{
  // Strings
  this->Interpolate = 1;
  this->AutoWindowLevel = 1;
  this->Window = 256;
  this->Level = 128;
  this->AutoThreshold = 0;
  this->ApplyThreshold = 0;
  this->LowerThreshold = VTK_SHORT_MIN;
  this->UpperThreshold = VTK_SHORT_MAX;
  this->ColorNodeID = NULL;
  this->ColorNode = NULL;
}

//----------------------------------------------------------------------------
void vtkMRMLVolumeDisplayNode::SetDefaultColorMap()
 {
  // set up a default color node
  this->SetAndObserveColorNodeID("vtkMRMLColorNodeGrey");
  if (this->ColorNode == NULL)
    {
    vtkDebugMacro("vtkMRMLVolumeDisplayNode: FAILED setting default greyscale color node, it's still null\n")
    }
  else
    {
    vtkDebugMacro("vtkMRMLVolumeDisplayNode: set up the default color node as grey\n");
    }
}

//----------------------------------------------------------------------------
vtkMRMLVolumeDisplayNode::~vtkMRMLVolumeDisplayNode()
{
  this->SetAndObserveColorNodeID( NULL);
}

//----------------------------------------------------------------------------
void vtkMRMLVolumeDisplayNode::WriteXML(ostream& of, int nIndent)
{
  Superclass::WriteXML(of, nIndent);

  vtkIndent indent(nIndent);

  {
  std::stringstream ss;
  ss << this->Window;
  of << indent << "window=\"" << ss.str() << "\" ";
  }
  {
  std::stringstream ss;
  ss << this->Level;
  of << indent << "level=\"" << ss.str() << "\" ";
  }
  {
  std::stringstream ss;
  ss << this->UpperThreshold;
  of << indent << "upperThreshold=\"" << ss.str() << "\" ";
  }
  {    
  std::stringstream ss;
  ss << this->LowerThreshold;
  of << indent << "lowerThreshold=\"" << ss.str() << "\" ";
  }
  {   
  std::stringstream ss;
  ss << this->Interpolate;
  of << indent << "interpolate=\"" << ss.str() << "\" ";
  }
  {
  std::stringstream ss;
  ss << this->AutoWindowLevel;
  of << indent << "autoWindowLevel=\"" << ss.str() << "\" ";
  }
  {
  std::stringstream ss;
  ss << this->ApplyThreshold;
  of << indent << "applyThreshold=\"" << ss.str() << "\" ";
  }
  {
  std::stringstream ss;
  ss << this->AutoThreshold;
  of << indent << "autoThreshold=\"" << ss.str() << "\" ";
  }
  if (this->ColorNodeID != NULL) 
    {
    of << indent << "colorNodeRef=\"" << this->ColorNodeID << "\" ";
    }

}

//----------------------------------------------------------------------------
void vtkMRMLVolumeDisplayNode::UpdateReferenceID(const char *oldID, const char *newID)
{
  if (this->ColorNodeID && !strcmp(oldID, this->ColorNodeID))
    {
    this->SetColorNodeID(newID);
    }
}

//----------------------------------------------------------------------------
void vtkMRMLVolumeDisplayNode::ReadXMLAttributes(const char** atts)
{

  Superclass::ReadXMLAttributes(atts);

  const char* attName;
  const char* attValue;
  while (*atts != NULL) 
    {
    attName = *(atts++);
    attValue = *(atts++);
    if (!strcmp(attName, "window")) 
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->Window;
      }
    else if (!strcmp(attName, "level")) 
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->Level;
      }
    else if (!strcmp(attName, "upperThreshold")) 
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->UpperThreshold;
      }
    else if (!strcmp(attName, "lowerThreshold")) 
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->LowerThreshold;
      }
    else if (!strcmp(attName, "interpolate")) 
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->Interpolate;
      }
    else if (!strcmp(attName, "autoWindowLevel")) 
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->AutoWindowLevel;
      }
    else if (!strcmp(attName, "applyThreshold")) 
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->ApplyThreshold;
      }
    else if (!strcmp(attName, "autoThreshold")) 
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->AutoThreshold;
      }
    else if (!strcmp(attName, "colorNodeRef")) 
      {
      this->SetColorNodeID(attValue);
      this->Scene->AddReferencedNodeID(this->ColorNodeID, this);
      }

    }  
}

//----------------------------------------------------------------------------
// Copy the node\"s attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, VolumeID
void vtkMRMLVolumeDisplayNode::Copy(vtkMRMLNode *anode)
{
  Superclass::Copy(anode);
  vtkMRMLVolumeDisplayNode *node = (vtkMRMLVolumeDisplayNode *) anode;

  this->SetAutoWindowLevel(node->AutoWindowLevel);
  this->SetWindow(node->Window);
  this->SetLevel(node->Level);
  this->SetAutoThreshold(node->AutoThreshold);
  this->SetApplyThreshold(node->ApplyThreshold);
  this->SetUpperThreshold(node->UpperThreshold);
  this->SetLowerThreshold(node->LowerThreshold);
  this->SetInterpolate(node->Interpolate);
  this->SetColorNodeID(node->ColorNodeID);

}

//----------------------------------------------------------------------------
void vtkMRMLVolumeDisplayNode::PrintSelf(ostream& os, vtkIndent indent)
{
  
  Superclass::PrintSelf(os,indent);

  os << indent << "AutoWindowLevel:   " << this->AutoWindowLevel << "\n";
  os << indent << "Window:            " << this->Window << "\n";
  os << indent << "Level:             " << this->Level << "\n";
  os << indent << "AutoThreshold:     " << this->AutoThreshold << "\n";
  os << indent << "ApplyThreshold:    " << this->ApplyThreshold << "\n";
  os << indent << "UpperThreshold:    " << this->UpperThreshold << "\n";
  os << indent << "LowerThreshold:    " << this->LowerThreshold << "\n";
  os << indent << "Interpolate:       " << this->Interpolate << "\n";
  os << indent << "ColorNodeID: " <<
    (this->ColorNodeID ? this->ColorNodeID : "(none)") << "\n";

}

//-----------------------------------------------------------
void vtkMRMLVolumeDisplayNode::UpdateScene(vtkMRMLScene *scene)
{
   Superclass::UpdateScene(scene);

   this->SetAndObserveColorNodeID(this->GetColorNodeID());
}

//-----------------------------------------------------------
void vtkMRMLVolumeDisplayNode::UpdateReferences()
{
   Superclass::UpdateReferences();

  if (this->ColorNodeID != NULL && this->Scene->GetNodeByID(this->ColorNodeID) == NULL)
    {
    this->SetAndObserveColorNodeID(NULL);
    }
}

//----------------------------------------------------------------------------
vtkMRMLColorNode* vtkMRMLVolumeDisplayNode::GetColorNode()
{
  vtkMRMLColorNode* node = NULL;
  if (this->GetScene() && this->GetColorNodeID() )
    {
    vtkMRMLNode* cnode = this->GetScene()->GetNodeByID(this->ColorNodeID);
    node = vtkMRMLColorNode::SafeDownCast(cnode);
    }
  return node;
}

//----------------------------------------------------------------------------
void vtkMRMLVolumeDisplayNode::SetAndObserveColorNodeID(const char *colorNodeID)
{
  vtkSetAndObserveMRMLObjectMacro(this->ColorNode, NULL);

  this->SetColorNodeID(colorNodeID);

  vtkMRMLColorNode *cnode = this->GetColorNode();

  vtkSetAndObserveMRMLObjectMacro(this->ColorNode, cnode);

}

//---------------------------------------------------------------------------
void vtkMRMLVolumeDisplayNode::ProcessMRMLEvents ( vtkObject *caller,
                                           unsigned long event, 
                                           void *callData )
{
  Superclass::ProcessMRMLEvents(caller, event, callData);

  vtkMRMLColorNode *cnode = this->GetColorNode();
  if (cnode != NULL && cnode == vtkMRMLColorNode::SafeDownCast(caller) &&
      event ==  vtkCommand::ModifiedEvent)
    {
    this->InvokeEvent(vtkCommand::ModifiedEvent, NULL);
    }
  return;
}


