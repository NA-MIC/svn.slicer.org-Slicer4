#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkMRMLCameraBasedROINode.h"
#include "vtkMRMLScene.h"

//------------------------------------------------------------------------------
vtkCxxRevisionMacro ( vtkMRMLCameraBasedROINode, "$Revision: 1.0 $");


//------------------------------------------------------------------------------
vtkMRMLCameraBasedROINode* vtkMRMLCameraBasedROINode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLCameraBasedROINode");
  if(ret)
    {
      return (vtkMRMLCameraBasedROINode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLCameraBasedROINode;
}

//----------------------------------------------------------------------------

vtkMRMLNode* vtkMRMLCameraBasedROINode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLCameraBasedROINode");
  if(ret)
    {
      return (vtkMRMLCameraBasedROINode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLCameraBasedROINode;
}

//----------------------------------------------------------------------------
vtkMRMLCameraBasedROINode::vtkMRMLCameraBasedROINode()
{
  this->HideFromEditors = true;
  this->CameraNodeID = NULL;
  this->ROINodeID = NULL;
  this->ROIDistanceToCamera = 10;
  this->ROISize = 50;
}




//----------------------------------------------------------------------------
vtkMRMLCameraBasedROINode::~vtkMRMLCameraBasedROINode()
{
  this->SetCameraNodeID ( NULL );
  this->SetROINodeID ( NULL );
}


//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, VolumeID
void vtkMRMLCameraBasedROINode::Copy(vtkMRMLNode *anode)
{
  int disabledModify = this->StartModify();

  Superclass::Copy(anode);
  vtkMRMLCameraBasedROINode *node = vtkMRMLCameraBasedROINode::SafeDownCast(anode);
  if (node)
  {
    this->SetCameraNodeID(node->GetCameraNodeID());
    this->SetROINodeID(node->GetROINodeID());
    this->SetROIDistanceToCamera(node->GetROIDistanceToCamera());
    this->SetROISize(node->GetROISize());
  }
  this->EndModify(disabledModify);

}


//----------------------------------------------------------------------------
void vtkMRMLCameraBasedROINode::PrintSelf(ostream& os, vtkIndent indent)
{
  //TODO  
  vtkMRMLNode::PrintSelf(os,indent);
}


//----------------------------------------------------------------------------
void vtkMRMLCameraBasedROINode::WriteXML(ostream& of, int nIndent)
{
  Superclass::WriteXML(of, nIndent);

  vtkIndent indent(nIndent);

  if (this->CameraNodeID != NULL) 
    {
    of << indent << " cameraRef=\"" << this->CameraNodeID << "\"";
    }
  if (this->ROINodeID != NULL) 
    {
    of << indent << " ROIRef=\"" << this->ROINodeID << "\"";
    }

  of << indent << " ROIDistanceToCamera=\"" << this->ROIDistanceToCamera << "\"";
  of << indent << " ROISize=\"" << this->ROISize << "\"";


}


//----------------------------------------------------------------------------
void vtkMRMLCameraBasedROINode::ReadXMLAttributes(const char** atts)
{
  int disabledModify = this->StartModify();

  Superclass::ReadXMLAttributes(atts);

  const char* attName;
  const char* attValue;
  while (*atts != NULL) 
    {
    attName = *(atts++);
    attValue = *(atts++);
    if (!strcmp(attName, "cameraRef")) 
      {
      this->SetCameraNodeID(attValue);
      }
    else if (!strcmp(attName, "ROIRef")) 
      {
      this->SetROINodeID(attValue);
      } 
    else if (!strcmp(attName, "ROIDistanceToCamera")) 
      {
      std::stringstream ss;
      ss << attValue;
      ss >> ROIDistanceToCamera;
      }
    else if (!strcmp(attName, "ROISize")) 
      {
      std::stringstream ss;
      ss << attValue;
      ss >> ROISize;
      }
    }  

  this->EndModify(disabledModify);
}


