#include "vtkCameraBasedROILogic.h"

#include "vtkCamera.h"
#include "vtkTransform.h"
#include "vtkMatrix4x4.h"
#include "vtkMath.h"

#include "vtkMRMLScene.h"
#include "vtkMRMLNode.h"
#include "vtkMRMLCameraNode.h"
#include "vtkMRMLROINode.h"
#include "vtkMRMLTransformNode.h"

#include "vtkMRMLCameraBasedROINode.h"


vtkCameraBasedROILogic* vtkCameraBasedROILogic::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkCameraBasedROILogic");
  if(ret)
    {
      return (vtkCameraBasedROILogic*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkCameraBasedROILogic;
}



//----------------------------------------------------------------------------
vtkCameraBasedROILogic::vtkCameraBasedROILogic()
{
   
}



//----------------------------------------------------------------------------
vtkCameraBasedROILogic::~vtkCameraBasedROILogic()
{ 
}

//----------------------------------------------------------------------------
void vtkCameraBasedROILogic::UpdateROI(vtkMRMLCameraBasedROINode *paramNode)
{ 
  if (paramNode == NULL || paramNode->GetCameraNodeID() == NULL ||
      paramNode->GetROINodeID() == NULL) 
    {
    return;
    }
  vtkMRMLCameraNode *cameraNode = vtkMRMLCameraNode::SafeDownCast(
    this->GetMRMLScene()->GetNodeByID(paramNode->GetCameraNodeID()));
  vtkMRMLROINode *ROINode = vtkMRMLROINode::SafeDownCast(
    this->GetMRMLScene()->GetNodeByID(paramNode->GetROINodeID()));

  vtkCamera *camera = cameraNode->GetCamera();
  if (!camera) 
    {
    return;
    }

  double pos[4];
  double foc[4];
  pos[3]=1; 
  foc[3]=1; 

  camera->GetPosition(pos);
  camera->GetFocalPoint(foc);

  vtkMatrix4x4 *mat = vtkMatrix4x4::New();
  mat->Identity();

  vtkTransform *userTransform = 
      vtkTransform::SafeDownCast(camera->GetUserViewTransform());
  if (userTransform)
    {
    userTransform->GetMatrix(mat);
    }

  double pos1[4];
  mat->MultiplyPoint(pos, pos1);
  double foc1[4];
  mat->MultiplyPoint(foc, foc1);

  int i;
  for (i=0; i<3; i++)
    {
    pos[i] = foc1[i] - pos1[i];
    }

  vtkMath::Norm(pos);

  for (i=0; i<3; i++)
    {
    pos1[i] = pos1[i] + paramNode->GetROIDistanceToCamera()*pos[i];
    }

  double rad = paramNode->GetROISize()/2;

  ROINode->SetXYZ(pos1);
  ROINode->SetRadiusXYZ(rad,rad,rad);

  mat->Delete();
}
