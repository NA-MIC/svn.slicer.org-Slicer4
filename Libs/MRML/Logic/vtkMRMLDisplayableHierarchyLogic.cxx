/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLDisplayableHierarchyLogic.cxx,v $
  Date:      $Date: 2010-02-15 16:35:35 -0500 (Mon, 15 Feb 2010) $
  Version:   $Revision: 12142 $

=========================================================================auto=*/

// MRMLLogic includes
#include "vtkMRMLDisplayableHierarchyLogic.h"

// MRML includes
#include "vtkMRMLDisplayableHierarchyNode.h"
#include "vtkMRMLDisplayableNode.h"
#include "vtkMRMLDisplayNode.h"

// VTK includes
#include <vtkNew.h>

vtkCxxRevisionMacro(vtkMRMLDisplayableHierarchyLogic, "$Revision: 12142 $");
vtkStandardNewMacro(vtkMRMLDisplayableHierarchyLogic);

//----------------------------------------------------------------------------
vtkMRMLDisplayableHierarchyLogic::vtkMRMLDisplayableHierarchyLogic()
{
}

//----------------------------------------------------------------------------
vtkMRMLDisplayableHierarchyLogic::~vtkMRMLDisplayableHierarchyLogic()
{
}

//----------------------------------------------------------------------------
void vtkMRMLDisplayableHierarchyLogic::SetMRMLSceneInternal(vtkMRMLScene* newScene)
{
  vtkNew<vtkIntArray> sceneEvents;
  sceneEvents->InsertNextValue(vtkMRMLScene::NodeRemovedEvent);
  this->SetAndObserveMRMLSceneEventsInternal(newScene, sceneEvents.GetPointer());
}

//----------------------------------------------------------------------------
void vtkMRMLDisplayableHierarchyLogic::OnMRMLSceneNodeRemoved(vtkMRMLNode* node)
{
  vtkMRMLDisplayableNode* displayableNode = vtkMRMLDisplayableNode::SafeDownCast(node);
  if (!displayableNode || this->GetMRMLScene()->IsBatchProcessing())
    {
    return;
    }
  // A displayable hierarchy node without children as well as a displayble
  // node is useless node. Delete it.
  vtkMRMLHierarchyNode* displayableHierarchyNode = vtkMRMLHierarchyNode::GetAssociatedHierarchyNode(this->GetMRMLScene(), node->GetID());
  if (displayableHierarchyNode &&
      displayableHierarchyNode->GetNumberOfChildrenNodes() == 0)
    {
    this->GetMRMLScene()->RemoveNode(displayableHierarchyNode);
    }
}


//----------------------------------------------------------------------------
char *vtkMRMLDisplayableHierarchyLogic::AddDisplayableHierarchyNodeForNode(vtkMRMLDisplayableNode *node)
{
  char *hierarchyNodeID = NULL;

  if (!node)
    {
    vtkErrorMacro("AddDisplayableHierarchyNodeForNode: null node!");
    return hierarchyNodeID;
    }
  if (!node->GetScene())
    {
    vtkErrorMacro("AddDisplayableHierarchyNodeForNode: node isn't in a scene!");
    return hierarchyNodeID;
    }
  vtkMRMLDisplayableHierarchyNode *hierarchyNode = NULL;
  hierarchyNode = vtkMRMLDisplayableHierarchyNode::New();
  // it's a stealth node:
  hierarchyNode->HideFromEditorsOn();

  // give it a unique name based on the node
  std::string hnodeName = std::string(node->GetName()) + std::string(" Hierarchy");
  hierarchyNode->SetName(node->GetScene()->GetUniqueNameByString(hnodeName.c_str()));

  node->GetScene()->AddNode(hierarchyNode);
  // with a parent node id of null, it's a child of the scene

  // now point to the  node, need disable modified event to avoid an assert in qMRMLSceneModel
  node->SetDisableModifiedEvent(1);
  hierarchyNode->SetDisplayableNodeID(node->GetID());
  node->SetDisableModifiedEvent(0);

  // save the id for return
  hierarchyNodeID = hierarchyNode->GetID();
  
  // clean up
  hierarchyNode->Delete();

  return hierarchyNodeID;
}

//----------------------------------------------------------------------------
bool vtkMRMLDisplayableHierarchyLogic::AddChildToParent(vtkMRMLDisplayableNode *child, vtkMRMLDisplayableNode *parent)
{
  if (!child)
    {
    vtkErrorMacro("AddChildToParent: null child node");
    return false;
    }
  if (!child->GetScene())
    {
    vtkErrorMacro("AddChildToParent: child is not in a scene");
    return false;
    }
  if (!parent)
    {
    vtkErrorMacro("AddChildToParent: null parent node");
    return false;
    }
  if (!parent->GetScene())
    {
    vtkErrorMacro("AddChildToParent: parent is not in a scene");
    return false;
    }

  // does the parent already have a hierarchy node associated with it?
  char *parentHierarchyNodeID = NULL;
  vtkMRMLHierarchyNode *hierarchyNode = vtkMRMLHierarchyNode::GetAssociatedHierarchyNode(parent->GetScene(), parent->GetID());
  if (!hierarchyNode)
    {
    // create one and add to the scene
    parentHierarchyNodeID = this->AddDisplayableHierarchyNodeForNode(parent);
    }
  else
    {
    parentHierarchyNodeID = hierarchyNode->GetID();
    }
  if (!parentHierarchyNodeID)
    {
    vtkWarningMacro("AddChildToParent: unable to add or find a hierarchy node for the parent node " << parent->GetID() << ", so unable to place the child in a hierarchy");
    return false;
    }
  
  // does the child already have a hierarchy node associated with it?
  vtkMRMLHierarchyNode *childHierarchyNode = vtkMRMLHierarchyNode::GetAssociatedHierarchyNode(child->GetScene(), child->GetID());
  if (!childHierarchyNode)
    {
    char *childHierarchyNodeID = this->AddDisplayableHierarchyNodeForNode(child);
    if (childHierarchyNodeID)
      {
      vtkMRMLNode *mrmlNode = child->GetScene()->GetNodeByID(childHierarchyNodeID);
      if (mrmlNode)
        {
        childHierarchyNode = vtkMRMLHierarchyNode::SafeDownCast(mrmlNode);
        }
      }
    }
  if (childHierarchyNode)
    {
    std::cout << "AddChildToParent: parentHierarchyID = " << parentHierarchyNodeID << ", childHierarchyNodeID = " << childHierarchyNode->GetID() << std::endl;
    // disable modified events on the parent
    vtkMRMLNode *parentNode = childHierarchyNode->GetScene()->GetNodeByID(parentHierarchyNodeID);
    parentNode->SetDisableModifiedEvent(1);
    childHierarchyNode->SetParentNodeID(parentHierarchyNodeID);
    parentNode->SetDisableModifiedEvent(0);
    
    return true;
    }
  else
    {
    vtkWarningMacro("AddChildToParent: unable to add or find a hierarchy node for the child node " << child->GetID() << ", so unable to place it in a hierarchy");
    return false;
    }
  return false;
}
