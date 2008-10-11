/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkOpenIGTLinkDaemonLogic.cxx,v $
Date:      $Date: 2006/03/17 15:10:10 $
Version:   $Revision: 1.2 $

=========================================================================auto=*/

#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"

#include "vtkOpenIGTLinkDaemonLogic.h"

#include "vtkMRMLScene.h"
#include "vtkMRMLScalarVolumeNode.h"

vtkOpenIGTLinkDaemonLogic* vtkOpenIGTLinkDaemonLogic::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkOpenIGTLinkDaemonLogic");
  if(ret)
    {
      return (vtkOpenIGTLinkDaemonLogic*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkOpenIGTLinkDaemonLogic;
}


//----------------------------------------------------------------------------
vtkOpenIGTLinkDaemonLogic::vtkOpenIGTLinkDaemonLogic()
{
  OpenIGTLinkDaemonNode = vtkMRMLOpenIGTLinkDaemonNode::New();
  OpenIGTLinkDaemon = vtkITKOpenIGTLinkDaemon::New();
}

//----------------------------------------------------------------------------
vtkOpenIGTLinkDaemonLogic::~vtkOpenIGTLinkDaemonLogic()
{
  this->OpenIGTLinkDaemon->Delete();
  this->OpenIGTLinkDaemonNode->Delete();
}

//----------------------------------------------------------------------------
void vtkOpenIGTLinkDaemonLogic::PrintSelf(ostream& os, vtkIndent indent)
{
  
}

void vtkOpenIGTLinkDaemonLogic::Apply()
{
  // chack if MRML node is present 
  if (this->OpenIGTLinkDaemonNode == NULL)
    {
    vtkErrorMacro("No input OpenIGTLinkDaemonNode found");
    return;
    }
  
  // find input volume
  vtkMRMLNode* inNode = this->GetMRMLScene()->GetNodeByID(this->OpenIGTLinkDaemonNode->GetInputVolumeRef());
  vtkMRMLScalarVolumeNode *inVolume =  dynamic_cast<vtkMRMLScalarVolumeNode *> (inNode);
  if (inVolume == NULL)
    {
    vtkErrorMacro("No input volume found");
    return;
    }
  
  this->OpenIGTLinkDaemon->SetInput(inVolume->GetImageData());
  
  
  // set filter parameters
  this->OpenIGTLinkDaemon->SetConductanceParameter(this->OpenIGTLinkDaemonNode->GetConductance());
  this->OpenIGTLinkDaemon->SetNumberOfIterations(this->OpenIGTLinkDaemonNode->GetNumberOfIterations());
  this->OpenIGTLinkDaemon->SetTimeStep(this->OpenIGTLinkDaemonNode->GetTimeStep());
  
  // find output volume
  vtkMRMLScalarVolumeNode *outVolume = NULL;
  if (this->OpenIGTLinkDaemonNode->GetOutputVolumeRef() != NULL)
    {
    vtkMRMLNode* outNode = this->GetMRMLScene()->GetNodeByID(this->OpenIGTLinkDaemonNode->GetOutputVolumeRef());
    outVolume =  dynamic_cast<vtkMRMLScalarVolumeNode *> (outNode);
    if (outVolume == NULL)
      {
      vtkErrorMacro("No output volume found with id= " << this->OpenIGTLinkDaemonNode->GetOutputVolumeRef());
      return;
      }
    }
  else 
    {
    // create new volume Node and add it to mrml scene
    this->GetMRMLScene()->SaveStateForUndo();
    outVolume = vtkMRMLScalarVolumeNode::New();
    this->GetMRMLScene()->AddNode(outVolume);  
    outVolume->Delete();
    }

  outVolume->SetImageData(this->OpenIGTLinkDaemon->GetOutput());
  this->OpenIGTLinkDaemon->Update();
}
