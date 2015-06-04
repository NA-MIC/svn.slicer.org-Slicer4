/*==============================================================================

  Copyright (c) Laboratory for Percutaneous Surgery (PerkLab)
  Queen's University, Kingston, ON, Canada. All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  This file was originally developed by Andras Lasso, PerkLab, Queen's University
  and was supported through the Applied Cancer Research Unit program of Cancer Care
  Ontario with funds provided by the Ontario Ministry of Health and Long-Term Care

==============================================================================*/

// MRML includes
#include "vtkMRMLLabelMapVolumeNode.h"
#include "vtkMRMLLabelMapVolumeDisplayNode.h"
#include "vtkMRMLScene.h"

// VTK includes
#include <vtkDataArray.h>
#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>

//----------------------------------------------------------------------------
vtkMRMLNodeNewMacro(vtkMRMLLabelMapVolumeNode);

//----------------------------------------------------------------------------
vtkMRMLLabelMapVolumeNode::vtkMRMLLabelMapVolumeNode()
{
}

//----------------------------------------------------------------------------
vtkMRMLLabelMapVolumeNode::~vtkMRMLLabelMapVolumeNode()
{
}

//-----------------------------------------------------------
void vtkMRMLLabelMapVolumeNode::CreateNoneNode(vtkMRMLScene *scene)
{
  vtkNew<vtkImageData> id;
  id->SetDimensions(1, 1, 1);
#if (VTK_MAJOR_VERSION <= 5)
  id->SetNumberOfScalarComponents(1);
  id->SetScalarType(VTK_SHORT);
  id->AllocateScalars();
#else
  id->AllocateScalars(VTK_SHORT, 1);
#endif
  id->GetPointData()->GetScalars()->FillComponent(0, 0);

  vtkNew<vtkMRMLLabelMapVolumeNode> n;
  n->SetName("None");
  // the scene will set the id
  n->SetAndObserveImageData(id.GetPointer());
  scene->AddNode(n.GetPointer());
}

//----------------------------------------------------------------------------
void vtkMRMLLabelMapVolumeNode::CreateDefaultDisplayNodes()
{
  if (vtkMRMLLabelMapVolumeDisplayNode::SafeDownCast(this->GetDisplayNode())!=NULL)
    {
    // display node already exists
    return;
    }
  if (this->GetScene()==NULL)
    {
    vtkErrorMacro("vtkMRMLLabelMapVolumeNode::CreateDefaultDisplayNodes failed: scene is invalid");
    return;
    }
  vtkNew<vtkMRMLLabelMapVolumeDisplayNode> dispNode;
  dispNode->SetAndObserveColorNodeID ("vtkMRMLColorTableNodeLabels");
  this->GetScene()->AddNode(dispNode.GetPointer());
  this->SetAndObserveDisplayNodeID(dispNode->GetID());
}
