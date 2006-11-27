/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLVolumeNode.cxx,v $
Date:      $Date: 2006/03/17 17:01:53 $
Version:   $Revision: 1.14 $

=========================================================================auto=*/

#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"

#include "vtkMRMLTensorVolumeNode.h"
#include "vtkMRMLScene.h"

//------------------------------------------------------------------------------
vtkMRMLTensorVolumeNode* vtkMRMLTensorVolumeNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLTensorVolumeNode");
  if(ret)
    {
    return (vtkMRMLTensorVolumeNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLTensorVolumeNode;
}

//----------------------------------------------------------------------------

vtkMRMLNode* vtkMRMLTensorVolumeNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLTensorVolumeNode");
  if(ret)
    {
    return (vtkMRMLTensorVolumeNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLTensorVolumeNode;
}

//----------------------------------------------------------------------------
vtkMRMLTensorVolumeNode::vtkMRMLTensorVolumeNode()
{
  for(int i=0; i<3; i++) 
    {
    for(int j=0; j<3; j++) 
      {
      this->MeasurementFrame[i][j] = (i == j) ? 1.0 : 0.0;
      }
    }
  this->Order = -1; //Tensor order   
}

//----------------------------------------------------------------------------
vtkMRMLTensorVolumeNode::~vtkMRMLTensorVolumeNode()
{
}

//----------------------------------------------------------------------------
void vtkMRMLTensorVolumeNode::WriteXML(ostream& of, int nIndent)
{
  Superclass::WriteXML(of, nIndent);
  
  vtkIndent indent(nIndent);
  std::stringstream ss;
  for(int i=0; i<3; i++)
    {
    for(int j=0; j<3; j++)
      {
      ss << this->MeasurementFrame[i][j] << " ";
      if ( i != 2 && j != 2 )
        {
        ss << "  ";
        }
      }
    }
    of << indent << "measurementFrame=\"" << ss.str() << "\" ";
}

//----------------------------------------------------------------------------
void vtkMRMLTensorVolumeNode::ReadXMLAttributes(const char** atts)
{

  Superclass::ReadXMLAttributes(atts);

  const char* attName;
  const char* attValue;
  while (*atts != NULL)
    {
    attName = *(atts++);
    attValue = *(atts++);

    if (!strcmp(attName, "measurementFrame"))
      {
      std::stringstream ss;
      double val;
      ss << attValue;
      for(int i=0; i<3; i++)
        {
        for(int j=0; j<3; j++)
          {
          ss >> val;
          this->MeasurementFrame[i][j] = val;
          }
        }
      }
  }

} 

//----------------------------------------------------------------------------
void vtkMRMLTensorVolumeNode::SetMeasurementFrame(const double mf[3][3])
{
  for (int i=0; i<3; i++) 
    {
    for (int j=0; j<3; j++) 
      {
      this->MeasurementFrame[i][j] = mf[i][j];
      }
    }
}

//----------------------------------------------------------------------------
void vtkMRMLTensorVolumeNode::GetMeasurementFrame(double mf[3][3])
{
  for (int i=0; i<3; i++) 
    {
    for (int j=0; j<3; j++) 
      {
      mf[i][j]= this->MeasurementFrame[i][j];
      }
    }
}

//----------------------------------------------------------------------------
void vtkMRMLTensorVolumeNode::SetMeasurementFrame(const double xr, const double xa, const double xs,
                           const double yr, const double ya, const double ys,
                           const double zr, const double za, const double zs)
{
  MeasurementFrame[0][0] = xr;
  MeasurementFrame[0][1] = xa;
  MeasurementFrame[0][2] = xs;
  MeasurementFrame[1][0] = yr;
  MeasurementFrame[1][1] = ya;
  MeasurementFrame[1][2] = ys;
  MeasurementFrame[2][0] = zr;
  MeasurementFrame[2][1] = za;
  MeasurementFrame[2][2] = zs;
}

//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, VolumeID
void vtkMRMLTensorVolumeNode::Copy(vtkMRMLNode *anode)
{
  Superclass::Copy(anode);
  vtkMRMLTensorVolumeNode *node = (vtkMRMLTensorVolumeNode *) anode;

  // Matrices
  for(int i=0; i<3; i++) 
    {
    for(int j=0; j<3; j++) 
      {
      this->MeasurementFrame[i][j] = node->MeasurementFrame[i][j];
      }
    }
  this->Order = node->Order;
}

//----------------------------------------------------------------------------
void vtkMRMLTensorVolumeNode::PrintSelf(ostream& os, vtkIndent indent)
{
  Superclass::PrintSelf(os,indent);
  os << "MeasurementFrame:\n";
  for(int i=0; i<3; i++) 
    {
    for(int j=0; j<3; j++) 
      {
      os << indent << " " << this->MeasurementFrame[i][j];
      }
      os << indent << "\n";
    }
  os << "\n";
  os << "Order: "<<this->Order;

}


 
