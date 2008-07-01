/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLGlyphVolumeSliceDisplayNode.cxx,v $
Date:      $Date: 2006/03/03 22:26:39 $
Version:   $Revision: 1.3 $

=========================================================================auto=*/
#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"
#include "vtkCallbackCommand.h"

#include "vtkTransformPolyDataFilter.h"

#include "vtkMRMLGlyphVolumeSliceDisplayNode.h"
#include "vtkMRMLScene.h"

//------------------------------------------------------------------------------
vtkMRMLGlyphVolumeSliceDisplayNode* vtkMRMLGlyphVolumeSliceDisplayNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLGlyphVolumeSliceDisplayNode");
  if(ret)
    {
    return (vtkMRMLGlyphVolumeSliceDisplayNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLGlyphVolumeSliceDisplayNode;
}

//-----------------------------------------------------------------------------
vtkMRMLNode* vtkMRMLGlyphVolumeSliceDisplayNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLGlyphVolumeSliceDisplayNode");
  if(ret)
    {
    return (vtkMRMLGlyphVolumeSliceDisplayNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLGlyphVolumeSliceDisplayNode;
}


//----------------------------------------------------------------------------
vtkMRMLGlyphVolumeSliceDisplayNode::vtkMRMLGlyphVolumeSliceDisplayNode()
{


  this->ColorMode = this->colorModeScalar;
  
  this->SliceToXYTransformer = vtkTransformPolyDataFilter::New();

  this->SliceToXYTransform = vtkTransform::New();
  
  this->SliceToXYMatrix = vtkMatrix4x4::New();
  this->SliceToXYMatrix->Identity();
  this->SliceToXYTransform->PreMultiply();
  this->SliceToXYTransform->SetMatrix(this->SliceToXYMatrix);

  //this->SliceToXYTransformer->SetInput(this->GlyphGlyphFilter->GetOutput());
  this->SliceToXYTransformer->SetTransform(this->SliceToXYTransform);
}


//----------------------------------------------------------------------------
vtkMRMLGlyphVolumeSliceDisplayNode::~vtkMRMLGlyphVolumeSliceDisplayNode()
{
  this->RemoveObservers ( vtkCommand::ModifiedEvent, this->MRMLCallbackCommand );
  this->SliceToXYMatrix->Delete();
  this->SliceToXYTransform->Delete();
  this->SliceToXYTransformer->Delete();
}

//----------------------------------------------------------------------------
void vtkMRMLGlyphVolumeSliceDisplayNode::WriteXML(ostream& of, int nIndent)
{

  // Write all attributes not equal to their defaults
  
  Superclass::WriteXML(of, nIndent);

  vtkIndent indent(nIndent);

  of << indent << " colorMode =\"" << this->ColorMode << "\"";

}


//----------------------------------------------------------------------------
void vtkMRMLGlyphVolumeSliceDisplayNode::ReadXMLAttributes(const char** atts)
{
  Superclass::ReadXMLAttributes(atts);

  const char* attName;
  const char* attValue;
  while (*atts != NULL) 
    {
    attName = *(atts++);
    attValue = *(atts++);

    if (!strcmp(attName, "colorMode")) 
      {
      std::stringstream ss;
      ss << attValue;
      ss >> ColorMode;
      }

    }  


}


//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, ID
void vtkMRMLGlyphVolumeSliceDisplayNode::Copy(vtkMRMLNode *anode)
{
  Superclass::Copy(anode);
  vtkMRMLGlyphVolumeSliceDisplayNode *node = (vtkMRMLGlyphVolumeSliceDisplayNode *) anode;

  this->SetColorMode(node->ColorMode);
}

//----------------------------------------------------------------------------
void vtkMRMLGlyphVolumeSliceDisplayNode::PrintSelf(ostream& os, vtkIndent indent)
{
 //int idx;
  
  Superclass::PrintSelf(os,indent);
  os << indent << "ColorMode:             " << this->ColorMode << "\n";
}
//----------------------------------------------------------------------------
void vtkMRMLGlyphVolumeSliceDisplayNode::SetSliceGlyphRotationMatrix(vtkMatrix4x4 *matrix)
{
}

//----------------------------------------------------------------------------
void vtkMRMLGlyphVolumeSliceDisplayNode::SetSlicePositionMatrix(vtkMatrix4x4 *matrix)
{
//  if (this->GlyphGlyphFilter)
//    {
//    this->GlyphGlyphFilter->SetVolumePositionMatrix(matrix);
//    }
  this->SliceToXYMatrix->DeepCopy(matrix);
  this->SliceToXYMatrix->Invert();
  if (this->SliceToXYTransform)
    {
    this->SliceToXYTransform->SetMatrix(this->SliceToXYMatrix);
    }
}

//----------------------------------------------------------------------------
void vtkMRMLGlyphVolumeSliceDisplayNode::SetSliceImage(vtkImageData *image)
{
/*
    if (this->GlyphGlyphFilter)
    {
    this->GlyphGlyphFilter->SetInput(image);
    this->GlyphGlyphFilter->SetDimensions(image->GetDimensions());
    }
*/
}

//----------------------------------------------------------------------------
void vtkMRMLGlyphVolumeSliceDisplayNode::SetPolyData(vtkPolyData *glyphPolyData)
{
}

//----------------------------------------------------------------------------
vtkPolyData* vtkMRMLGlyphVolumeSliceDisplayNode::GetPolyData()
{
    return NULL;
}

//----------------------------------------------------------------------------
vtkPolyData* vtkMRMLGlyphVolumeSliceDisplayNode::GetPolyDataTransformedToSlice()
{
    return NULL;
}
//----------------------------------------------------------------------------
void vtkMRMLGlyphVolumeSliceDisplayNode::UpdatePolyDataPipeline() 
{
  vtkErrorMacro("Shouldn't be calling this");
}

//---------------------------------------------------------------------------
void vtkMRMLGlyphVolumeSliceDisplayNode::ProcessMRMLEvents ( vtkObject *caller,
                                           unsigned long event, 
                                           void *callData )
{
  Superclass::ProcessMRMLEvents(caller, event, callData);
  return;
}

//-----------------------------------------------------------
void vtkMRMLGlyphVolumeSliceDisplayNode::UpdateScene(vtkMRMLScene *scene)
{
   Superclass::UpdateScene(scene);
}

//-----------------------------------------------------------
void vtkMRMLGlyphVolumeSliceDisplayNode::UpdateReferences()
{
  Superclass::UpdateReferences();
}


//----------------------------------------------------------------------------
void vtkMRMLGlyphVolumeSliceDisplayNode::UpdateReferenceID(const char *oldID, const char *newID)
{
}




