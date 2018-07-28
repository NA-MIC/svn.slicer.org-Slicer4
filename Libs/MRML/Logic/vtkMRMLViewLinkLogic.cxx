/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLViewLinkLogic.cxx,v $
  Date:      $Date$
  Version:   $Revision$

=========================================================================auto=*/

// MRMLLogic includes
#include "vtkMRMLViewLinkLogic.h"
#include "vtkMRMLApplicationLogic.h"

// MRML includes
#include <vtkEventBroker.h>
#include <vtkMRMLCameraNode.h>
#include <vtkMRMLCrosshairNode.h>
#include <vtkMRMLScene.h>
#include <vtkMRMLViewNode.h>

// VTK includes
#include <vtkCamera.h>
#include <vtkCollection.h>
#include <vtkFloatArray.h>
#include <vtkMath.h>
#include <vtkMatrix4x4.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkTransform.h>

// STD includes
#include <cassert>

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkMRMLViewLinkLogic);

//----------------------------------------------------------------------------
vtkMRMLViewLinkLogic::vtkMRMLViewLinkLogic()
{
}

//----------------------------------------------------------------------------
vtkMRMLViewLinkLogic::~vtkMRMLViewLinkLogic()
{
}

//----------------------------------------------------------------------------
void vtkMRMLViewLinkLogic::SetMRMLSceneInternal(vtkMRMLScene * newScene)
{
  // List of events the slice logics should listen
  vtkNew<vtkIntArray> events;
  vtkNew<vtkFloatArray> priorities;

  float normalPriority = 0.0;
  // float highPriority = 0.5;

  // Events that use the default priority.  Don't care the order they
  // are triggered
  events->InsertNextValue(vtkMRMLScene::NodeAddedEvent);
  priorities->InsertNextValue(normalPriority);
  events->InsertNextValue(vtkMRMLScene::NodeRemovedEvent);
  priorities->InsertNextValue(normalPriority);

  this->SetAndObserveMRMLSceneEventsInternal(newScene, events.GetPointer(), priorities.GetPointer());

  this->ProcessMRMLSceneEvents(newScene, vtkCommand::ModifiedEvent, 0);
}

//----------------------------------------------------------------------------
void vtkMRMLViewLinkLogic::OnMRMLSceneNodeAdded(vtkMRMLNode* node)
{
  if (!node)
    {
    return;
    }
  if (node->IsA("vtkMRMLViewNode") || node->IsA("vtkMRMLCameraNode"))
    {
    vtkEventBroker::GetInstance()->AddObservation(
      node, vtkCommand::ModifiedEvent, this, this->GetMRMLNodesCallbackCommand());
    }
}

//----------------------------------------------------------------------------
void vtkMRMLViewLinkLogic::OnMRMLSceneNodeRemoved(vtkMRMLNode* node)
{
  if (!node)
    {
    return;
    }
  if (node->IsA("vtkMRMLViewNode") || node->IsA("vtkMRMLCameraNode"))
    {
    vtkEventBroker::GetInstance()->RemoveObservations(
      node, vtkCommand::ModifiedEvent, this, this->GetMRMLNodesCallbackCommand());
    }
}

//----------------------------------------------------------------------------
void vtkMRMLViewLinkLogic::OnMRMLNodeModified(vtkMRMLNode* node)
{
  // Update from CameraNode
  vtkMRMLCameraNode* cameraNode = vtkMRMLCameraNode::SafeDownCast(node);
  if (cameraNode && cameraNode->GetID() &&
      this->GetMRMLScene() && !this->GetMRMLScene()->IsBatchProcessing())
    {

    // if this is not the node that we are interacting with, short circuit
    if (!cameraNode->GetInteracting() || !cameraNode->GetInteractionFlags())
      {
      return;
      }

    // CameraNode was modified. Need to find the corresponding
    // ViewNode to check whether operations are linked
    vtkMRMLViewNode* viewNode = vtkMRMLViewNode::SafeDownCast
      (this->GetMRMLScene()->GetNodeByID(cameraNode->GetActiveTag()));

    if (viewNode && viewNode->GetLinkedControl())
      {
      this->BroadcastCameraNodeEvent(cameraNode);
      }
    else
      {
      // camera node changed and views are not linked. Do not broadcast.
      return;
      }
    }

  // Update from viewNode
  vtkMRMLViewNode* viewNode = vtkMRMLViewNode::SafeDownCast(node);
  if (viewNode && viewNode->GetID() &&
      this->GetMRMLScene() && !this->GetMRMLScene()->IsBatchProcessing())
    {
    // if this is not the node that we are interacting with, short circuit
    if (!viewNode->GetInteracting()
        || !viewNode->GetInteractionFlags())
      {
      return;
      }

    if (viewNode && viewNode->GetLinkedControl())
      {
      // view node changed and views are linked. Broadcast.
      this->BroadcastViewNodeEvent(viewNode);
      }
    else
      {
      // view node changed and views are not linked. Do not broadcast.
      }
    }
}


//----------------------------------------------------------------------------
void vtkMRMLViewLinkLogic::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  vtkIndent nextIndent;
  nextIndent = indent.GetNextIndent();
}

//----------------------------------------------------------------------------
void vtkMRMLViewLinkLogic::BroadcastCameraNodeEvent(vtkMRMLCameraNode *cameraNode)
{
  // only broadcast a camera node event if we are not already actively
  // broadcasting events and we are actively interacting with the node
  if (!cameraNode || !cameraNode->GetInteracting())
    {
    return;
    }

  vtkMRMLViewNode* viewNode = vtkMRMLViewNode::SafeDownCast
    (this->GetMRMLScene()->GetNodeByID(cameraNode->GetActiveTag()));

  if (!viewNode)
    {
    return;
    }

  int requiredViewGroup = viewNode->GetViewGroup();
  vtkMRMLCameraNode* sNode;
  vtkCollectionSimpleIterator it;
  vtkSmartPointer<vtkCollection> nodes;
  nodes.TakeReference(this->GetMRMLScene()->GetNodesByClass("vtkMRMLCameraNode"));
  for (nodes->InitTraversal(it);
      (sNode = vtkMRMLCameraNode::SafeDownCast(nodes->GetNextItemAsObject(it)));)
    {
    if (!sNode || sNode == cameraNode)
      {
      continue;
      }

    vtkMRMLViewNode* sViewNode = vtkMRMLViewNode::SafeDownCast
      (this->GetMRMLScene()->GetNodeByID(cameraNode->GetActiveTag()));

    if (!sViewNode || sViewNode->GetViewGroup() != requiredViewGroup)
      {
      continue;
      }

    // RenderMode selection
    if (cameraNode->GetInteractionFlags() == vtkMRMLCameraNode::LookFromAxis)
      {
      vtkCamera* sCamera = sNode->GetCamera();
      vtkCamera* Camera = cameraNode->GetCamera();
      if (sCamera && Camera)
        {
        int wasModifying = sNode->StartModify();
        sCamera->SetPosition(Camera->GetPosition());
        sCamera->SetViewUp(Camera->GetViewUp());
        sCamera->ComputeViewPlaneNormal();
        sCamera->OrthogonalizeViewUp();
        sNode->EndModify(wasModifying);
        sNode->InvokeCustomModifiedEvent(vtkMRMLCameraNode::ResetCameraClippingEvent);
        }
      }
    // ZoomIn
    else if (cameraNode->GetInteractionFlags() == vtkMRMLCameraNode::ZoomInFlag)
      {
      vtkCamera* sCamera = sNode->GetCamera();
      // The zoom factor value is defined in ethe constructor of ctkVTKRenderView
      double zoomFactor = 0.05;
      if (sCamera)
        {
        int wasModifying = sNode->StartModify();
        if (sCamera->GetParallelProjection())
          {
          sCamera->SetParallelScale(sCamera->GetParallelScale() / (1 + zoomFactor));
          }
        else
          {
          sCamera->Dolly(1 + zoomFactor);
          }
        sNode->EndModify(wasModifying);
        sNode->InvokeCustomModifiedEvent(vtkMRMLCameraNode::ResetCameraClippingEvent);
        }
      }
    // ZoomOut
    else if (cameraNode->GetInteractionFlags() == vtkMRMLCameraNode::ZoomOutFlag)
      {
      vtkCamera* sCamera = sNode->GetCamera();
      // The zoom factor value is defined in ethe constructor of ctkVTKRenderView
      double zoomFactor = -0.05;
      if (sCamera)
        {
        int wasModifying = sNode->StartModify();
        if (sCamera->GetParallelProjection())
          {
          sCamera->SetParallelScale(sCamera->GetParallelScale() / (1 + zoomFactor));
          }
        else
          {
          sCamera->Dolly(1 + zoomFactor);
          }
        sNode->EndModify(wasModifying);
        sNode->InvokeCustomModifiedEvent(vtkMRMLCameraNode::ResetCameraClippingEvent);
        }
      }
    // Reset Focal Point
    else if (cameraNode->GetInteractionFlags() == vtkMRMLCameraNode::CenterFlag)
      {
      vtkCamera* sCamera = sNode->GetCamera();
      vtkCamera* Camera = cameraNode->GetCamera();
      if (sCamera && Camera)
        {
        int wasModifying = sNode->StartModify();
        sCamera->SetFocalPoint(Camera->GetFocalPoint());
        sCamera->ComputeViewPlaneNormal();
        sCamera->OrthogonalizeViewUp();
        sNode->EndModify(wasModifying);
        sNode->InvokeCustomModifiedEvent(vtkMRMLCameraNode::ResetCameraClippingEvent);
        }
      }
    // update camera to modification to vtkCamera (i.e., mouse interaction)
    else if (cameraNode->GetInteractionFlags() == vtkMRMLCameraNode::vtkCameraFlag)
      {
      vtkCamera* sCamera = sNode->GetCamera();
      vtkCamera* Camera = cameraNode->GetCamera();
      if (sCamera && Camera)
        {
        int wasModifying = sNode->StartModify();
        sCamera->SetPosition(Camera->GetPosition());
        sCamera->SetFocalPoint(Camera->GetFocalPoint());
        sCamera->SetViewUp(Camera->GetViewUp());
        sNode->EndModify(wasModifying);
        sNode->InvokeCustomModifiedEvent(vtkMRMLCameraNode::ResetCameraClippingEvent);
        }
      }
    //
    // End of the block for broadcasting parameters and commands
    // that do not require the orientation to match
    //
    }
}

//----------------------------------------------------------------------------
void vtkMRMLViewLinkLogic::BroadcastViewNodeEvent(vtkMRMLViewNode *viewNode)
{
  // only broadcast a view node event if we are not already actively
  // broadcasting events and we actively interacting with the node
  if (!viewNode || !viewNode->GetInteracting())
    {
    return;
    }

  int requiredViewGroup = viewNode->GetViewGroup();
  vtkMRMLViewNode* vNode;
  vtkCollectionSimpleIterator it;
  vtkSmartPointer<vtkCollection> nodes;
  nodes.TakeReference(this->GetMRMLScene()->GetNodesByClass("vtkMRMLViewNode"));

  for (nodes->InitTraversal(it);
      (vNode = vtkMRMLViewNode::SafeDownCast(nodes->GetNextItemAsObject(it)));)
    {
    if (!vNode || vNode == viewNode || vNode->GetViewGroup() != requiredViewGroup)
      {
      continue;
      }

    // RenderMode selection
    if (viewNode->GetInteractionFlags() == vtkMRMLViewNode::RenderModeFlag)
      {
      vNode->SetRenderMode(viewNode->GetRenderMode());
      }
    // AnimationMode selection
    else if (viewNode->GetInteractionFlags() == vtkMRMLViewNode::AnimationModeFlag)
      {
      vNode->SetAnimationMode(viewNode->GetAnimationMode());
      }
    // Box visibility
    else if (viewNode->GetInteractionFlags() == vtkMRMLViewNode::BoxVisibleFlag)
      {
      vNode->SetBoxVisible(viewNode->GetBoxVisible());
      }
    // Box labels visibility
    else if (viewNode->GetInteractionFlags() == vtkMRMLViewNode::BoxLabelVisibileFlag)
      {
      vNode->SetAxisLabelsVisible(viewNode->GetAxisLabelsVisible());
      }
    // Background color
    else if (viewNode->GetInteractionFlags() == vtkMRMLViewNode::BackgroundColorFlag)
      {
      int wasModifying = vNode->StartModify();
      // The ThreeDView displayable manager will change the background color of
      // the renderer.

      vNode->SetBackgroundColor(viewNode->GetBackgroundColor());
      vNode->SetBackgroundColor2(viewNode->GetBackgroundColor2());
      vNode->EndModify(wasModifying);
      }
    // Stereo type
    else if (viewNode->GetInteractionFlags() == vtkMRMLViewNode::StereoTypeFlag)
      {
      vNode->SetStereoType(viewNode->GetStereoType());
      }
    // Orientation marker type
    else if (viewNode->GetInteractionFlags() == vtkMRMLViewNode::OrientationMarkerTypeFlag)
      {
      vNode->SetOrientationMarkerType(viewNode->GetOrientationMarkerType());
      }
    // Orientation marker size
    else if (viewNode->GetInteractionFlags() == vtkMRMLViewNode::OrientationMarkerSizeFlag)
      {
      vNode->SetOrientationMarkerSize(viewNode->GetOrientationMarkerSize());
      }
    // Ruler type
    else if (viewNode->GetInteractionFlags() == vtkMRMLViewNode::RulerTypeFlag)
      {
      vNode->SetRulerType(viewNode->GetRulerType());
      }
    // Ruler type
    else if (viewNode->GetInteractionFlags() == vtkMRMLViewNode::UseDepthPeelingFlag)
      {
      vNode->SetUseDepthPeeling(viewNode->GetUseDepthPeeling());
      }
    // FPS visibility
    else if (viewNode->GetInteractionFlags() == vtkMRMLViewNode::FPSVisibleFlag)
      {
      vNode->SetFPSVisible(viewNode->GetFPSVisible());
      }
    }
}
