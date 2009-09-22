/*=auto=======================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights
  Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLEMSNode.cxx,v$
  Date:      $Date: 2006/01/06 17:56:51 $
  Version:   $Revision: 1.6 $
  Author:    $Nicolas Rannou (BWH), Sylvain Jaume (MIT)$

=======================================================================auto=*/

#include "vtkMRMLEMSNode.h"
#include <sstream>
#include "vtkMRMLScene.h"

#include <vtksys/ios/sstream>

//----------------------------------------------------------------------------
vtkMRMLEMSNode* vtkMRMLEMSNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret =
    vtkObjectFactory::CreateInstance("vtkMRMLEMSNode");

  if (ret)
  {
    return (vtkMRMLEMSNode*)ret;
  }

  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLEMSNode;
}

//----------------------------------------------------------------------------
vtkMRMLNode* vtkMRMLEMSNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLEMSNode");

  if(ret)
  {
    return (vtkMRMLEMSNode*)ret;
  }

  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLEMSNode;
}

//----------------------------------------------------------------------------
vtkMRMLEMSNode::vtkMRMLEMSNode()
{
  this->SegmenterNodeID               = NULL;
  this->TemplateFilename              = NULL;
  this->SaveTemplateAfterSegmentation = 0;
}

//----------------------------------------------------------------------------
vtkMRMLEMSNode::~vtkMRMLEMSNode()
{
  this->SetSegmenterNodeID(NULL);
  this->SetTemplateFilename(NULL);
}

//----------------------------------------------------------------------------
void vtkMRMLEMSNode::
WriteXML(ostream& of, int nIndent)
{
  Superclass::WriteXML(of, nIndent);

  vtkIndent indent(nIndent);

  of << indent
    << " SegmenterNodeID=\""
    << (this->SegmenterNodeID ? this->SegmenterNodeID : "NULL")
    << "\"";

  of << indent
    << " TemplateFilename=\""
    << (this->TemplateFilename ? this->TemplateFilename : "NULL")
    << "\"";

  of << indent
    << " SaveTemplateAfterSegmentation=\""
    << this->SaveTemplateAfterSegmentation
    << "\"";
}

//----------------------------------------------------------------------------
void vtkMRMLEMSNode::UpdateReferenceID(const char* oldID, const char* newID)
{
  if (this->SegmenterNodeID && !strcmp(oldID, this->SegmenterNodeID))
    {
    this->SetSegmenterNodeID(newID);
    }
}

//----------------------------------------------------------------------------
void vtkMRMLEMSNode::UpdateReferences()
{
  Superclass::UpdateReferences();

  if (this->SegmenterNodeID != NULL && this->Scene->GetNodeByID(this->
        SegmenterNodeID) == NULL)
    {
    this->SetSegmenterNodeID(NULL);
    }
}

//----------------------------------------------------------------------------
void vtkMRMLEMSNode::ReadXMLAttributes(const char** attrs)
{
  Superclass::ReadXMLAttributes(attrs);

  // we assume an even number of elements

  const char* key;
  const char* val;

  while (*attrs != NULL)
    {
    key = *attrs++;
    val = *attrs++;

    if (!strcmp(key, "SegmenterNodeID"))
      {
      this->SetSegmenterNodeID(val);
      //this->Scene->AddReferencedNodeID(this->SegmenterNodeID, this);
      }
    else if (!strcmp(key, "TemplateFilename"))
      {
      this->SetTemplateFilename(val);
      }
    else if (!strcmp(key, "SaveTemplateAfterSegmentation"))
      {
      vtksys_stl::stringstream ss;
      ss << val;
      ss >> this->SaveTemplateAfterSegmentation;
      }
    }
}

//----------------------------------------------------------------------------
void vtkMRMLEMSNode::Copy(vtkMRMLNode *rhs)
{
  Superclass::Copy(rhs);

  vtkMRMLEMSNode* node = (vtkMRMLEMSNode*) rhs;

  this->SetSegmenterNodeID(node->SegmenterNodeID);
  this->SetTemplateFilename(node->TemplateFilename);
  this->SetSaveTemplateAfterSegmentation(node->SaveTemplateAfterSegmentation);
}

//----------------------------------------------------------------------------
void vtkMRMLEMSNode::PrintSelf(ostream& os, vtkIndent indent)
{
  Superclass::PrintSelf(os, indent);

  os << indent << "SegmenterNodeID: " <<
    (this->SegmenterNodeID ? this->SegmenterNodeID : "(none)") << "\n";

  os << indent << "TemplateFilename: " <<
    (this->TemplateFilename ? this->TemplateFilename : "(none)") << "\n";

  os << indent << "SaveTemplateAfterSegmentation: "
     << this->SaveTemplateAfterSegmentation << "\n";
}

//----------------------------------------------------------------------------
vtkMRMLEMSSegmenterNode* vtkMRMLEMSNode::GetSegmenterNode()
{
  vtkMRMLEMSSegmenterNode* node = NULL;

  if (this->GetScene() && this->GetSegmenterNodeID() )
    {
    vtkMRMLNode* snode = this->GetScene()->GetNodeByID(this->SegmenterNodeID);
    node = vtkMRMLEMSSegmenterNode::SafeDownCast(snode);
    }

  return node;
}

