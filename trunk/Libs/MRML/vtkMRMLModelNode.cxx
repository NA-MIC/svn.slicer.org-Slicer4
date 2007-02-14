/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLModelNode.cxx,v $
Date:      $Date: 2006/03/03 22:26:39 $
Version:   $Revision: 1.3 $

=========================================================================auto=*/
#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"
#include "vtkCallbackCommand.h"

#include "vtkMRMLModelNode.h"
#include "vtkMRMLScene.h"


#include "vtkDataSetAttributes.h"
#include "vtkPointData.h"
#include "vtkCellData.h"
#include "vtkFloatArray.h"

//------------------------------------------------------------------------------
vtkMRMLModelNode* vtkMRMLModelNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLModelNode");
  if(ret)
    {
    return (vtkMRMLModelNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLModelNode;
}

//-----------------------------------------------------------------------------

vtkMRMLNode* vtkMRMLModelNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLModelNode");
  if(ret)
    {
    return (vtkMRMLModelNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLModelNode;
}


//----------------------------------------------------------------------------
vtkMRMLModelNode::vtkMRMLModelNode()
{
  this->StorageNodeID = NULL;
  this->DisplayNodeID = NULL;
  this->ModelDisplayNode = NULL;
  PolyData = NULL;
}

//----------------------------------------------------------------------------
vtkMRMLModelNode::~vtkMRMLModelNode()
{
  if (this->StorageNodeID) 
    {
    delete [] this->StorageNodeID;
    this->StorageNodeID = NULL;
    }
  this->SetAndObserveDisplayNodeID( NULL);

  this->SetAndObservePolyData(NULL);
}

//----------------------------------------------------------------------------
void vtkMRMLModelNode::WriteXML(ostream& of, int nIndent)
{
  // Write all attributes not equal to their defaults
  
  Superclass::WriteXML(of, nIndent);

  vtkIndent indent(nIndent);

   if (this->StorageNodeID != NULL) 
    {
    of << indent << "storageNodeRef=\"" << this->StorageNodeID << "\" ";
    }
  if (this->DisplayNodeID != NULL) 
    {
    of << indent << "displayNodeRef=\"" << this->DisplayNodeID << "\" ";
    }
}

//----------------------------------------------------------------------------
void vtkMRMLModelNode::UpdateReferenceID(const char *oldID, const char *newID)
{
  if (!strcmp(oldID, this->StorageNodeID))
    {
    this->SetStorageNodeID(newID);
    }
  if (!strcmp(oldID, this->DisplayNodeID))
    {
    this->SetDisplayNodeID(newID);
    }
}

//----------------------------------------------------------------------------
void vtkMRMLModelNode::ReadXMLAttributes(const char** atts)
{

  Superclass::ReadXMLAttributes(atts);

  const char* attName;
  const char* attValue;
  while (*atts != NULL) 
    {
    attName = *(atts++);
    attValue = *(atts++);
    if (!strcmp(attName, "storageNodeRef")) 
      {
      this->SetStorageNodeID(attValue);
      this->Scene->AddReferencedNodeID(this->StorageNodeID, this);
      }
    else if (!strcmp(attName, "displayNodeRef")) 
      {
      this->SetDisplayNodeID(attValue);
      this->Scene->AddReferencedNodeID(this->DisplayNodeID, this);
      }    
    }  
}


//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, ID
void vtkMRMLModelNode::Copy(vtkMRMLNode *anode)
{
  Superclass::Copy(anode);
  vtkMRMLModelNode *node = (vtkMRMLModelNode *) anode;

  this->SetStorageNodeID(node->StorageNodeID);
  this->SetDisplayNodeID(node->DisplayNodeID);
  this->SetPolyData(node->PolyData);
}

//----------------------------------------------------------------------------
void vtkMRMLModelNode::PrintSelf(ostream& os, vtkIndent indent)
{
  
  Superclass::PrintSelf(os,indent);

  os << indent << "StorageNodeID: " <<
    (this->StorageNodeID ? this->StorageNodeID : "(none)") << "\n";

  os << indent << "DisplayNodeID: " <<
    (this->DisplayNodeID ? this->DisplayNodeID : "(none)") << "\n";

  os << "\nPoly Data:\n";
  if (this->PolyData) 
    {
    this->PolyData->PrintSelf(os, indent.GetNextIndent());
    }
}

//-----------------------------------------------------------
void vtkMRMLModelNode::UpdateScene(vtkMRMLScene *scene)
{
   Superclass::UpdateScene(scene);

  if (this->GetStorageNodeID() == NULL) 
    {
    //vtkErrorMacro("No reference StorageNodeID found");
    return;
    }

  vtkMRMLNode* mnode = scene->GetNodeByID(this->StorageNodeID);
  if (mnode) 
    {
    vtkMRMLStorageNode *node  = dynamic_cast < vtkMRMLStorageNode *>(mnode);
    if (node->ReadData(this) == 0)
      {
      scene->SetErrorCode(1);
      std::string msg = std::string("Error reading model file ") + std::string(node->GetFileName());
      scene->SetErrorMessage(msg);
      }
    this->SetAndObservePolyData(this->GetPolyData());
    this->SetAndObserveDisplayNodeID(this->GetDisplayNodeID());
    }
}

//-----------------------------------------------------------
void vtkMRMLModelNode::UpdateReferences()
{
   Superclass::UpdateReferences();

  if (this->DisplayNodeID != NULL && this->Scene->GetNodeByID(this->DisplayNodeID) == NULL)
    {
    this->SetAndObserveDisplayNodeID(NULL);
    }
 if (this->StorageNodeID != NULL && this->Scene->GetNodeByID(this->StorageNodeID) == NULL)
    {
    this->SetStorageNodeID(NULL);
    }
}

vtkMRMLStorageNode* vtkMRMLModelNode::GetStorageNode()
{
  vtkMRMLStorageNode* node = NULL;
  if (this->GetScene() && this->GetStorageNodeID() )
    {
    vtkMRMLNode* snode = this->GetScene()->GetNodeByID(this->StorageNodeID);
    node = vtkMRMLStorageNode::SafeDownCast(snode);
    }
  return node;
}

//----------------------------------------------------------------------------
vtkMRMLModelDisplayNode* vtkMRMLModelNode::GetDisplayNode()
{
  vtkMRMLModelDisplayNode* node = NULL;
  if (this->GetScene() && this->GetDisplayNodeID() )
    {
    vtkMRMLNode* snode = this->GetScene()->GetNodeByID(this->DisplayNodeID);
    node = vtkMRMLModelDisplayNode::SafeDownCast(snode);
    }
  return node;
}

//----------------------------------------------------------------------------
void vtkMRMLModelNode::SetAndObserveDisplayNodeID(const char *displayNodeID)
{
  vtkSetAndObserveMRMLObjectMacro(this->ModelDisplayNode, NULL);

  this->SetDisplayNodeID(displayNodeID);

  vtkMRMLModelDisplayNode *dnode = this->GetDisplayNode();

  vtkSetAndObserveMRMLObjectMacro(this->ModelDisplayNode, dnode);

}

//----------------------------------------------------------------------------
void vtkMRMLModelNode::SetAndObservePolyData(vtkPolyData *polyData)
{
if (this->PolyData != NULL)
    {
    this->PolyData->RemoveObservers ( vtkCommand::ModifiedEvent, this->MRMLCallbackCommand );
    }

  unsigned long mtime1, mtime2;
  mtime1 = this->GetMTime();
  this->SetPolyData(polyData);
  mtime2 = this->GetMTime();

  if (this->PolyData != NULL)
    {
    this->PolyData->AddObserver ( vtkCommand::ModifiedEvent, this->MRMLCallbackCommand );
    }

  if (mtime1 != mtime2)
    {
    this->InvokeEvent( vtkMRMLModelNode::PolyDataModifiedEvent , this);
    }
}


//---------------------------------------------------------------------------
void vtkMRMLModelNode::ProcessMRMLEvents ( vtkObject *caller,
                                           unsigned long event, 
                                           void *callData )
{
  Superclass::ProcessMRMLEvents(caller, event, callData);

  vtkMRMLModelDisplayNode *dnode = this->GetDisplayNode();
  if (dnode != NULL && dnode == vtkMRMLModelDisplayNode::SafeDownCast(caller) &&
      event ==  vtkCommand::ModifiedEvent)
    {
    this->InvokeEvent(vtkMRMLModelNode::DisplayModifiedEvent, NULL);
    }
  else if (this->PolyData == vtkPolyData::SafeDownCast(caller) &&
    event ==  vtkCommand::ModifiedEvent)
    {
    this->ModifiedSinceRead = true;
    this->InvokeEvent(vtkMRMLModelNode::PolyDataModifiedEvent, NULL);
    }
  return;
}

//---------------------------------------------------------------------------
void vtkMRMLModelNode::AddPointScalars(vtkFloatArray *array)
{
  if (array == NULL)
    {
    return;
    }
  if (this->PolyData == NULL)
    {
    vtkErrorMacro("No poly data on model " << this->GetName());
    return;
    }
  
  int numScalars = this->PolyData->GetPointData()->GetNumberOfArrays();
  vtkDebugMacro("Model node has " << numScalars << " point scalars now, adding " << array->GetName());
  if (numScalars > 0)
    {
    // add array
    this->PolyData->GetPointData()->AddArray(array);
    } 
  else
    {
    // set the scalars
    this->PolyData->GetPointData()->SetScalars(array);
    }
}

//---------------------------------------------------------------------------
void vtkMRMLModelNode::AddCellScalars(vtkFloatArray *array)
{
  if (array == NULL)
    {
    return;
    }
  if (this->PolyData == NULL)
    {
    vtkErrorMacro("No poly data on model " << this->GetName());
    return;
    }
  
  int numScalars = this->PolyData->GetCellData()->GetNumberOfArrays();
  vtkDebugMacro("Model node has " << numScalars << " cell scalars now, adding " << array->GetName());
  if (numScalars > 0)
    {
    // add array
    this->PolyData->GetCellData()->AddArray(array);
    } 
  else
    {
    // set the scalars
    this->PolyData->GetCellData()->SetScalars(array);
    }
}

//---------------------------------------------------------------------------
void vtkMRMLModelNode::RemoveScalars(const char *scalarName)
{
  if (scalarName == NULL)
    {
    vtkErrorMacro("Scalar name is null");
    return;
    }
  if (this->PolyData == NULL)
    {
    vtkErrorMacro("No poly data on model " << this->GetName());
    return;
    }
  // try removing the array from the points first
  if (this->PolyData->GetPointData())
    {
    this->PolyData->GetPointData()->RemoveArray(scalarName);
    // it's a void method, how to check if it succeeded?
    }
  // try the cells
  if (this->PolyData->GetCellData())
    {
    this->PolyData->GetCellData()->RemoveArray(scalarName);
    }
}

/*
//---------------------------------------------------------------------------
const char * vtkMRMLModelNode::GetActivePointScalarName(const char *type)
{
  if (this->PolyData == NULL)
    {
    return "";
    }
  if (this->PolyData->GetPointData() == NULL)
    {
    return "";
    }
  if (type != NULL)
    {
    if (strcmp(type, "scalars") == 0)
      {
      return this->PolyData->GetPointData()->GetActiveAttribute(vtkDataSetAttributes::SCALARS);
      }
    else if (strcmp(type, "vectors") == 0)
      {
      }
    else if (strcmp(type, "normals") == 0)
      {
      }
    else if (strcmp(type, "tcoords") == 0)
      {
      }
    else if (strcmp(type, "tensors") == 0)
      {
      }
    else
      {
      vtkErrorMacro("Unknown point scalar type " << type);
      return "";
      }
    }
  else
    {
    // returning empty string for now
    vtkErrorMacro("Unspecified type not implemented.");
    return "";
    }
}

//---------------------------------------------------------------------------
const char * vtkMRMLModelNode::GetActiveCellScalarName(const char *type)
{
  if (this->PolyData == NULL)
    {
    return "";
    }
  if (this->PolyData->GetCellData() == NULL)
    {
    return "";
    }
}
*/

//---------------------------------------------------------------------------
int vtkMRMLModelNode::SetActiveScalars(const char *scalarName)
{
  int retval = -1;
  if (this->PolyData == NULL || scalarName == NULL)
    {
    vtkErrorMacro("No poly data on model " << this->GetName() << " or the scalar name is null");
    return retval;
    }
  
  // is it a point scalar?
  retval = this->SetActivePointScalars(scalarName);
  if (retval != -1)
    {
    vtkDebugMacro("Set active point scalars to " << scalarName << " (" <<
                  this->PolyData->GetPointData()->GetAttributeTypeAsString(retval) <<
                  ") on model " << this->GetName());
    vtkWarningMacro("Set the active point scalars to " << scalarName << ", the display node's active scalars = " << this->GetDisplayNode()->GetActiveScalarName());
    return retval;
    }
  // is it a cell scalar?
  retval =  this->SetActiveCellScalars(scalarName);
  if (retval != -1)
    {
    vtkDebugMacro("Set active cell scalars to " << scalarName << " (" <<
                  this->PolyData->GetCellData()->GetAttributeTypeAsString(retval) << ") on model " <<
                  this->GetName());
    return retval;
    }
  vtkWarningMacro("Unable to find scalar attribute " << scalarName << " on model " << this->GetName());
  return retval;
}

//---------------------------------------------------------------------------
int vtkMRMLModelNode::SetActivePointScalars(const char *scalarName)
{
  if (this->PolyData == NULL || scalarName == NULL)
    {
    vtkErrorMacro("No poly data on model " << this->GetName() << " or the scalar name is null");
    return -1;
    }
  if (this->PolyData->GetPointData() == NULL)
    {
    vtkWarningMacro("No point data on this model " << this->GetName());
    return -1;
    }
  // try the different attributes until find this array name
  if (this->PolyData->GetPointData()->SetActiveAttribute(scalarName, vtkDataSetAttributes::SCALARS) != -1)
    {
    return vtkDataSetAttributes::SCALARS;
    }
  if (this->PolyData->GetPointData()->SetActiveAttribute(scalarName, vtkDataSetAttributes::VECTORS) != -1)
    {
    return vtkDataSetAttributes::VECTORS;
    }
  if (this->PolyData->GetPointData()->SetActiveAttribute(scalarName, vtkDataSetAttributes::NORMALS) != -1)
    {
    return vtkDataSetAttributes::NORMALS;
    }
  if (this->PolyData->GetPointData()->SetActiveAttribute(scalarName, vtkDataSetAttributes::TCOORDS) != -1)
    {
    return vtkDataSetAttributes::TCOORDS;
    }
  if (this->PolyData->GetPointData()->SetActiveAttribute(scalarName, vtkDataSetAttributes::TENSORS) != -1)
    {
    return vtkDataSetAttributes::TENSORS;
    }
  return -1;
}

//---------------------------------------------------------------------------
int vtkMRMLModelNode::SetActiveCellScalars(const char *scalarName)
{
  if (this->PolyData == NULL || scalarName == NULL)
    {
    vtkErrorMacro("No poly data on model " << this->GetName() << " or the scalar name is null");
    return -1;
    }
  if (this->PolyData->GetCellData() == NULL)
    {
    vtkWarningMacro("No cell data on this model " << this->GetName());
    return -1;
    }

  // try the different attributes until find this array name
  if (this->PolyData->GetCellData()->SetActiveAttribute(scalarName, vtkDataSetAttributes::SCALARS) != -1)
    {
    return vtkDataSetAttributes::SCALARS;
    }
  if (this->PolyData->GetCellData()->SetActiveAttribute(scalarName, vtkDataSetAttributes::VECTORS) != -1)
    {
    return vtkDataSetAttributes::VECTORS;
    }
  if (this->PolyData->GetCellData()->SetActiveAttribute(scalarName, vtkDataSetAttributes::NORMALS) != -1)
    {
    return vtkDataSetAttributes::NORMALS;
    }
  if (this->PolyData->GetCellData()->SetActiveAttribute(scalarName, vtkDataSetAttributes::TCOORDS) != -1)
    {
    return vtkDataSetAttributes::TCOORDS;
    }
  if (this->PolyData->GetCellData()->SetActiveAttribute(scalarName, vtkDataSetAttributes::TENSORS) != -1)
    {
    return vtkDataSetAttributes::TENSORS;
    }
  return -1;
}
