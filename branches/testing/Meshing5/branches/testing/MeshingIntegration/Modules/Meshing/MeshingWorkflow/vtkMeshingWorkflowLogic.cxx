/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMeshingWorkflowLogic.cxx,v $
Date:      $Date: 2006/03/17 15:10:10 $
Version:   $Revision: 1.2 $

=========================================================================auto=*/

#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"

#include "vtkMeshingWorkflowLogic.h"

#include "vtkMRMLScene.h"
#include "vtkMRMLScalarVolumeNode.h"

vtkMeshingWorkflowLogic* vtkMeshingWorkflowLogic::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMeshingWorkflowLogic");
  if(ret)
    {
      return (vtkMeshingWorkflowLogic*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMeshingWorkflowLogic;
}


//----------------------------------------------------------------------------
vtkMeshingWorkflowLogic::vtkMeshingWorkflowLogic()
{
  MeshingWorkflowNode = vtkMRMLMeshingWorkflowNode::New();

}

//----------------------------------------------------------------------------
vtkMeshingWorkflowLogic::~vtkMeshingWorkflowLogic()
{

  this->MeshingWorkflowNode->Delete();
}

//----------------------------------------------------------------------------
void vtkMeshingWorkflowLogic::PrintSelf(ostream& os, vtkIndent indent)
{
  
}

void vtkMeshingWorkflowLogic::Apply()
{
  // chack if MRML node is present 
  if (this->MeshingWorkflowNode == NULL)
    {
    vtkErrorMacro("No input MeshingWorkflowNode found");
    return;
    }
  
  // find input volume
  vtkMRMLNode* inNode = this->GetMRMLScene()->GetNodeByID(this->MeshingWorkflowNode->GetInputVolumeRef());
  vtkMRMLScalarVolumeNode *inVolume =  dynamic_cast<vtkMRMLScalarVolumeNode *> (inNode);
  if (inVolume == NULL)
    {
    vtkErrorMacro("No input volume found");
    return;
    }
  
  //this->MeshingWorkflow->SetInput(inVolume->GetImageData());
  
  
  // set filter parameters
  // *** we don't have a subobject like the slicer daemon example had
  //this->MeshingWorkflow->SetConductanceParameter(this->MeshingWorkflowNode->GetConductance());
  //this->MeshingWorkflow->SetNumberOfIterations(this->MeshingWorkflowNode->GetNumberOfIterations());
  //this->MeshingWorkflow->SetTimeStep(this->MeshingWorkflowNode->GetTimeStep());
  
  // find output volume
  vtkMRMLScalarVolumeNode *outVolume = NULL;
  if (this->MeshingWorkflowNode->GetOutputVolumeRef() != NULL)
    {
    vtkMRMLNode* outNode = this->GetMRMLScene()->GetNodeByID(this->MeshingWorkflowNode->GetOutputVolumeRef());
    outVolume =  dynamic_cast<vtkMRMLScalarVolumeNode *> (outNode);
    if (outVolume == NULL)
      {
      vtkErrorMacro("No output volume found with id= " << this->MeshingWorkflowNode->GetOutputVolumeRef());
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

  //outVolume->SetImageData(this->MeshingWorkflow->GetOutput());
  //this->MeshingWorkflow->Update();
}
