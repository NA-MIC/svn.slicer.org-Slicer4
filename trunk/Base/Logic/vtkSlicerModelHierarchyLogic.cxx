/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkSlicerModelHierarchyLogic.cxx,v $
  Date:      $Date: 2006/01/06 17:56:48 $
  Version:   $Revision: 1.58 $

=========================================================================auto=*/

#include "vtkObjectFactory.h"
#include "vtkCallbackCommand.h"

#include "vtkSlicerModelHierarchyLogic.h"

vtkCxxRevisionMacro(vtkSlicerModelHierarchyLogic, "$Revision: 1.9.12.1 $");
vtkStandardNewMacro(vtkSlicerModelHierarchyLogic);

//----------------------------------------------------------------------------
vtkSlicerModelHierarchyLogic::vtkSlicerModelHierarchyLogic()
{
  this->ModeHierarchylNodesMTime = 0;
}

//----------------------------------------------------------------------------
vtkSlicerModelHierarchyLogic::~vtkSlicerModelHierarchyLogic()
{
}

//----------------------------------------------------------------------------
void vtkSlicerModelHierarchyLogic::ProcessMRMLEvents(vtkObject * /*caller*/, 
                                            unsigned long /*event*/, 
                                            void * /*callData*/)
{
  // TODO: implement if needed
}

//----------------------------------------------------------------------------
int vtkSlicerModelHierarchyLogic::CreateModelToHierarchyMap()
{
  if (this->MRMLScene == NULL)
    {
    this->ModeHierarchylNodes.clear();
    }
  else if (this->MRMLScene->GetSceneModifiedTime() > this->ModeHierarchylNodesMTime)
  {
    this->ModeHierarchylNodes.clear();
    
    std::vector<vtkMRMLNode *> nodes;
    int nnodes = this->MRMLScene->GetNodesByClass("vtkMRMLModelHierarchyNode", nodes);
  
    for (int i=0; i<nnodes; i++)
      {
      vtkMRMLModelHierarchyNode *node =  vtkMRMLModelHierarchyNode::SafeDownCast(nodes[i]);
      if (node)
        {
        vtkMRMLModelNode *mnode = node->GetModelNode();
        if (mnode)
          {
          this->ModeHierarchylNodes[std::string(mnode->GetID())] = node;
          }
        }
      }
    this->ModeHierarchylNodesMTime = this->MRMLScene->GetSceneModifiedTime();
  }
  return ModeHierarchylNodes.size();
}

//---------------------------------------------------------------------------
vtkMRMLModelHierarchyNode* vtkSlicerModelHierarchyLogic::GetModelHierarchyNode(const char *modelNodeID)
{
  if (modelNodeID == NULL)
    {
    return NULL;
    }
  this->CreateModelToHierarchyMap();
  
  std::map<std::string, vtkMRMLModelHierarchyNode *>::iterator iter;
  
  iter = this->ModeHierarchylNodes.find(modelNodeID);
  if (iter != this->ModeHierarchylNodes.end())
    {
    return iter->second;
    }
  else
    {
    return NULL;
    }
  
}
