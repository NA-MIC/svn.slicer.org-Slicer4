/*==============================================================================

  Program: 3D Slicer

  Copyright (c) Kitware Inc.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  This file was originally developed by Michael Jeulin-Lagarrigue, Kitware Inc.
  and was partially funded by NIH grant 3P41RR013218-12S1

==============================================================================*/

// TransformsReformatWidget includes
#include "vtkSlicerReformatLogic.h"

// MRML includes
#include "vtkMRMLScene.h"
#include "vtkMRMLSliceCompositeNode.h"
#include "vtkMRMLSliceLogic.h"
#include "vtkMRMLSliceNode.h"
#include "vtkMRMLVolumeNode.h"

// VTK includes
#include <vtkMath.h>
#include <vtkMatrix4x4.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkTransform.h>

// STD includes
#include <cassert>

//------------------------------------------------------------------------------
vtkStandardNewMacro(vtkSlicerReformatLogic);

//------------------------------------------------------------------------------
vtkSlicerReformatLogic::vtkSlicerReformatLogic()
{
}

//------------------------------------------------------------------------------
vtkSlicerReformatLogic::~vtkSlicerReformatLogic()
{
}

//------------------------------------------------------------------------------
void vtkSlicerReformatLogic::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//------------------------------------------------------------------------------
void vtkSlicerReformatLogic::
SetSliceOrigin(vtkMRMLSliceNode* node, double x, double y, double z)
{
  if (!node)
    {
    return;
    }

  // Clamp the position given the volume
  double bounds[6];
  this->GetVolumeBounds(node, bounds);

  x = std::max(bounds[0], std::min(x, bounds[1]));
  y = std::max(bounds[2], std::min(y, bounds[3]));
  z = std::max(bounds[4], std::min(z, bounds[5]));

  // Set the new position
  node->GetSliceToRAS()->SetElement(0, 3, x);
  node->GetSliceToRAS()->SetElement(1, 3, y);
  node->GetSliceToRAS()->SetElement(2, 3, z);
  node->UpdateMatrices();
}

//------------------------------------------------------------------------------
void vtkSlicerReformatLogic::
SetSliceOrigin(vtkMRMLSliceNode* node, double position[3])
{
  double x = position[0];
  double y = position[1];
  double z = position[2];

  this->SetSliceOrigin(node, x, y, z);
}

//------------------------------------------------------------------------------
void vtkSlicerReformatLogic::
SetSliceNormal(vtkMRMLSliceNode* node, double x, double y, double z)
{
  if (!node)
    {
    return;
    }

  double normal[3];
  normal[0] = x;
  normal[1] = y;
  normal[2] = z;
  vtkMath::Normalize(normal);

  // Set the new normal
  double cross[3], dot, rotation;
  vtkNew<vtkTransform> transform;
  vtkMatrix4x4* sliceToRAS = node->GetSliceToRAS();
  double nodeNormal[3] = {sliceToRAS->GetElement(0,2),
                           sliceToRAS->GetElement(1,2),
                           sliceToRAS->GetElement(2,2)};

  // Keep track of the current position
  double nodePosition[3];
  nodePosition[0] = sliceToRAS->GetElement(0,3);
  nodePosition[1] = sliceToRAS->GetElement(1,3);
  nodePosition[2] = sliceToRAS->GetElement(2,3);

  // Reset position
  sliceToRAS->SetElement(0,3,0);
  sliceToRAS->SetElement(1,3,0);
  sliceToRAS->SetElement(2,3,0);

  // Rotate the sliceNode to match the planeWidget normal
  vtkMath::Cross(nodeNormal, normal, cross);
  dot = vtkMath::Dot(nodeNormal, normal);
  // Clamp the dot product
  dot = (dot < -1.0) ? -1.0 : (dot > 1.0 ? 1.0 : dot);
  rotation = vtkMath::DegreesFromRadians(acos(dot));

  // Apply the rotation
  transform->PostMultiply();
  transform->SetMatrix(sliceToRAS);
  transform->RotateWXYZ(rotation,cross);
  transform->GetMatrix(sliceToRAS);

  // Reinsert the position
  sliceToRAS->SetElement(0, 3, nodePosition[0]);
  sliceToRAS->SetElement(1, 3, nodePosition[1]);
  sliceToRAS->SetElement(2, 3, nodePosition[2]);
  node->UpdateMatrices();
}

//------------------------------------------------------------------------------
void vtkSlicerReformatLogic::
SetSliceNormal(vtkMRMLSliceNode* node, double normal[3])
{
  double x = normal[0];
  double y = normal[1];
  double z = normal[2];

  this->SetSliceNormal(node, x, y, z);
}

//------------------------------------------------------------------------------
void vtkSlicerReformatLogic::GetVolumeBounds(vtkMRMLSliceNode* node,
                                             double bounds[6])
{
  vtkMRMLSliceCompositeNode* sliceCompositeNode =
    vtkMRMLSliceLogic::GetSliceCompositeNode(node);

  if (!node || !sliceCompositeNode)
    {
    return;
    }

  const char* volumeNodeID = 0;
  if (!volumeNodeID)
    {
    volumeNodeID = sliceCompositeNode ? sliceCompositeNode->GetBackgroundVolumeID() : 0;
    }
  if (!volumeNodeID)
    {
    volumeNodeID = sliceCompositeNode ? sliceCompositeNode->GetForegroundVolumeID() : 0;
    }
  if (!volumeNodeID)
    {
    volumeNodeID = sliceCompositeNode ? sliceCompositeNode->GetLabelVolumeID() : 0;
    }

  vtkMRMLVolumeNode* volumeNode = vtkMRMLVolumeNode::SafeDownCast(
   this->GetMRMLScene()->GetNodeByID(volumeNodeID));

  if (volumeNode)
    {
    double dimensions[3], center[3];
    vtkMRMLSliceLogic::GetVolumeRASBox(volumeNode, dimensions, center);
    bounds[0] = center[0] - dimensions[0] / 2;
    bounds[1] = center[0] + dimensions[0] / 2;
    bounds[2] = center[1] - dimensions[1] / 2;
    bounds[3] = center[1] + dimensions[1] / 2;
    bounds[4] = center[2] - dimensions[2] / 2;
    bounds[5] = center[2] + dimensions[2] / 2;
    }
}

//------------------------------------------------------------------------------
void vtkSlicerReformatLogic::GetCenterFromBounds(double bounds[6],
                                                 double center[3])
{
  center[0] = (bounds[0] + bounds[1]) / 2;
  center[1] = (bounds[2] + bounds[3]) / 2;
  center[2] = (bounds[4] + bounds[5]) / 2;
}
