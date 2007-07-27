#include <string>
#include <sstream>

#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkPolyData.h"
#include "vtkMatrix4x4.h"
#include "vtkRenderWindowInteractor.h"

#include "vtkSlicerViewerWidget.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerColor.h"
#include "vtkSlicerGUILayout.h"
#include "vtkSlicerViewerInteractorStyle.h"

#include "vtkActor.h"
#include "vtkFollower.h"
#include "vtkProperty.h"
#include "vtkTexture.h"
#include "vtkRenderer.h"
#include "vtkCamera.h"
#include "vtkPolyDataMapper.h"
#include "vtkTransformPolyDataFilter.h"
#include "vtkCellArray.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include "vtkOutlineSource.h"
#include "vtkVectorText.h"
#include "vtkRenderWindow.h"
#include "vtkImplicitBoolean.h"
#include "vtkPlane.h"
#include "vtkClipPolyData.h"

#include "vtkMRMLModelNode.h"
#include "vtkMRMLModelDisplayNode.h"
#include "vtkMRMLTransformNode.h"
#include "vtkMRMLLinearTransformNode.h"
#include "vtkMRMLScene.h"
#include "vtkMRMLClipModelsNode.h"
#include "vtkMRMLModelHierarchyNode.h"
#include "vtkMRMLColorNode.h"
#include "vtkMRMLProceduralColorNode.h"

#include "vtkSlicerModelHierarchyLogic.h"

#include "vtkKWWidget.h"

// for picking
#include "vtkWorldPointPicker.h"
#include "vtkPropPicker.h"
#include "vtkCellPicker.h"
#include "vtkPointPicker.h"

//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkSlicerViewerWidget );
vtkCxxRevisionMacro ( vtkSlicerViewerWidget, "$Revision: 1.0 $");


//---------------------------------------------------------------------------
vtkSlicerViewerWidget::vtkSlicerViewerWidget ( )
{
  this->MainViewer = NULL;  
  this->RenderPending = 0;  
  this->ViewerFrame = NULL;
  this->ProcessingMRMLEvent = 0;
  this->UpdateFromMRMLRequested = 0;

  this->CameraNode = NULL;

  this->ClipModelsNode = NULL;
  this->RedSliceNode = NULL;
  this->GreenSliceNode = NULL;
  this->YellowSliceNode = NULL;

  this->SlicePlanes = NULL;
  this->RedSlicePlane = NULL;
  this->GreenSlicePlane = NULL;
  this->YellowSlicePlane = NULL;

  this->ViewNode = NULL;
  this->BoxAxisActor = NULL;

  this->SceneClosing = false;

  this->ModelHierarchiesPresent = false;
  this->ModelHierarchyLogic = vtkSlicerModelHierarchyLogic::New();

  this->ApplicationLogic = NULL;
  this->WorldPointPicker = vtkWorldPointPicker::New();
  this->PropPicker = vtkPropPicker::New();
  this->CellPicker = vtkCellPicker::New();
  this->PointPicker = vtkPointPicker::New();
  this->ResetPick();
}




//---------------------------------------------------------------------------
vtkSlicerViewerWidget::~vtkSlicerViewerWidget ( )
{
  this->RemoveMRMLObservers();

  vtkSetMRMLNodeMacro(this->ClipModelsNode, NULL);
  vtkSetMRMLNodeMacro(this->CameraNode, NULL);
  vtkSetMRMLNodeMacro(this->ViewNode, NULL);

  vtkSetMRMLNodeMacro(this->RedSliceNode, NULL);
  vtkSetMRMLNodeMacro(this->GreenSliceNode, NULL);
  vtkSetMRMLNodeMacro(this->YellowSliceNode, NULL);
  
  if (this->MainViewer)
    {
    vtkSlicerViewerInteractorStyle *iStyle;
    iStyle = vtkSlicerViewerInteractorStyle::SafeDownCast (this->MainViewer->GetRenderWindowInteractor()->GetInteractorStyle());
    iStyle->SetApplicationLogic ( NULL);
    this->SetMRMLScene ( NULL );
    this->MainViewer->RemoveAllViewProps ( );
    }

//  this->SlicePlanes->RemoveFunction (this->RedSlicePlane);
//  this->SlicePlanes->RemoveFunction (this->GreenSlicePlane);
//  this->SlicePlanes->RemoveFunction (this->YellowSlicePlane);
  this->SlicePlanes->Delete();
  this->SlicePlanes = NULL;
  this->RedSlicePlane->Delete();
  this->RedSlicePlane = NULL;
  this->GreenSlicePlane->Delete();
  this->GreenSlicePlane = NULL;
  this->YellowSlicePlane->Delete();
  this->YellowSlicePlane = NULL;

  if (this->BoxAxisActor)
    {
    this->BoxAxisActor->Delete();
    this->BoxAxisActor = NULL;
    }
  for (unsigned int i=0; i<this->AxisLabelActors.size(); i++)
    {
    this->AxisLabelActors[i]->SetCamera ( NULL );
    this->AxisLabelActors[i]->Delete();
    }
  this->AxisLabelActors.clear();

  if (this->MainViewer)
    {

    this->MainViewer->SetParent ( NULL );
    this->MainViewer->Delete();
    this->MainViewer = NULL;
    }

  // release the DisplayedModelActors
  /*
    std::map< const char *, vtkActor * >::iterator dmIter;
  for (dmIter = this->DisplayedModelActors.begin();
       dmIter != this->DisplayedModelActors.end();
       dmIter++)
    {
    if (dmIter->second != NULL)
      {
      std::cout << "Deleting " << dmIter->first << endl;
      dmIter->second->Delete();
      }
    }
  */
  this->DisplayedModelActors.clear();
  
  this->ViewerFrame->SetParent ( NULL );
  this->ViewerFrame->Delete ( );
  this->ViewerFrame = NULL;


  if (this->WorldPointPicker)
    {
    this->WorldPointPicker->Delete();
    this->WorldPointPicker = NULL;
    }
  if (this->PropPicker)
    {
    this->PropPicker->Delete();
    this->PropPicker = NULL;
    }
  if (this->CellPicker)
    {
    this->CellPicker->Delete();
    this->CellPicker = NULL;
    }
  if (this->PointPicker)
    {
    this->PointPicker->Delete();
    this->PointPicker = NULL;
    }
  this->ApplicationLogic = NULL; 

  if (this->ModelHierarchyLogic)
    {
    this->ModelHierarchyLogic->Delete();
    }
}

//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::PrintSelf ( ostream& os, vtkIndent indent )
{
    this->vtkObject::PrintSelf ( os, indent );

    os << indent << "vtkSlicerViewerWidget: " << this->GetClassName ( ) << "\n";

    os << indent << "RenderPending = " << this->RenderPending << "\n";
    os << indent << "ProcessingMRMLEvent = " << this->ProcessingMRMLEvent << "\n";
    
    os << indent << "ClipType = " << this->ClipType << "\n";
    os << indent << "RedSliceClipState = " << this->RedSliceClipState << "\n";
    os << indent << "YellowSliceClipState = " << this->YellowSliceClipState << "\n";
    os << indent << "GreenSliceClipState = " << this->GreenSliceClipState << "\n";
    os << indent << "ClippingOn = " << (this->ClippingOn ? "true" : "false") << "\n";

    os << indent << "ModelHierarchiesPresent = " << this->ModelHierarchiesPresent << "\n";
    os << indent << "SceneClosing = " << this->SceneClosing << "\n";
    
    os << indent << "PickedNodeName = " << this->PickedNodeName.c_str() << "\n";
    os << indent << "PickedRAS = (" << this->PickedRAS[0] << ", " << this->PickedRAS[1] << ", "<< this->PickedRAS[2] << ")\n";
    os << indent << "PickedCellID = " << this->PickedCellID << "\n";
    os << indent << "PickedPointID = " << this->PickedPointID << "\n";
    // print widgets?
}

//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::ProcessWidgetEvents ( vtkObject *caller,
                                                  unsigned long event, 
                                                  void *callData )
{
} 

//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::CreateClipSlices()
{
  this->SlicePlanes = vtkImplicitBoolean::New();
  this->SlicePlanes->SetOperationTypeToIntersection();
  this->RedSlicePlane = vtkPlane::New();
  this->GreenSlicePlane = vtkPlane::New();
  this->YellowSlicePlane = vtkPlane::New();

  this->ClipType = vtkMRMLClipModelsNode::ClipIntersection;
  this->RedSliceClipState = vtkMRMLClipModelsNode::ClipOff;
  this->YellowSliceClipState = vtkMRMLClipModelsNode::ClipOff;
  this->GreenSliceClipState = vtkMRMLClipModelsNode::ClipOff;
  this->ClippingOn = false;
}

//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::CreateAxis()
{
  vtkOutlineSource *boxSource = vtkOutlineSource::New();
  vtkPolyDataMapper *boxMapper = vtkPolyDataMapper::New();
  boxMapper->SetInput( boxSource->GetOutput() );

  boxMapper->Update();
   
  this->BoxAxisActor = vtkActor::New();
  this->BoxAxisActor->SetMapper( boxMapper );
  this->BoxAxisActor->SetPickable(0);
  this->BoxAxisActor->SetScale(100, 100, 100);
  this->BoxAxisActor->GetProperty()->SetColor( 1.0, 0.0, 1.0 );

  this->AxisLabelActors.clear();
  std::vector<std::string> labels;
  labels.push_back("R");
  labels.push_back("A");
  labels.push_back("S");
  labels.push_back("L");
  labels.push_back("P");
  labels.push_back("I");
  
  for (unsigned int i=0; i<labels.size(); i++)
    {
    vtkVectorText *axisText = vtkVectorText::New();
    axisText->SetText(labels[i].c_str());
    vtkPolyDataMapper *axisMapper = vtkPolyDataMapper::New();
    axisMapper->SetInput(axisText->GetOutput());
    axisText->Delete();
    vtkFollower *axisActor = vtkFollower::New();

    axisActor->SetMapper(axisMapper);
    axisMapper->Delete();
    axisActor->SetScale(1,1,1); 
    axisActor->SetPickable (0);

    this->AxisLabelActors.push_back(axisActor);
    
    axisActor->GetProperty()->SetColor(1, 1, 1);
    axisActor->GetProperty()->SetDiffuse (0.0);
    axisActor->GetProperty()->SetAmbient (1.0);
    axisActor->GetProperty()->SetSpecular (0.0);
  }
  double fov = 200;
  double pos = fov * 0.6;

  this->AxisLabelActors[0]->SetPosition(pos,0,0);
  this->AxisLabelActors[1]->SetPosition(0,pos,0);
  this->AxisLabelActors[2]->SetPosition(0,0,pos);
  this->AxisLabelActors[3]->SetPosition(-pos,0,0);
  this->AxisLabelActors[4]->SetPosition(0,-pos,0);
  this->AxisLabelActors[5]->SetPosition(0,0,-pos);

  boxSource->Delete();
  boxMapper->Delete();
}


//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::ColorAxisLabelActors ( double r, double g, double b)
{
  for (unsigned int i=0; i<this->AxisLabelActors.size(); i++)
    {
    this->AxisLabelActors[i]->GetProperty()->SetColor ( r, g, b );
    }

}



//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::AddAxisActors()
{
  if (this->MainViewer)
    {
    if (this->BoxAxisActor)
      {
      this->MainViewer->AddViewProp( this->BoxAxisActor);
      }
    for (unsigned int i=0; i<this->AxisLabelActors.size(); i++)
      {
      this->AxisLabelActors[i]->SetCamera(this->MainViewer->GetRenderer()->GetActiveCamera());
      this->MainViewer->AddViewProp( this->AxisLabelActors[i]);
      }
    }
}

//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::UpdateAxis()
{
  this->UpdateViewNode();
  if (this->ViewNode == NULL) 
    {
    return;
    }
  double fov = this->ViewNode->GetFieldOfView();
  this->BoxAxisActor->SetScale(fov/2, fov/2, fov/2);
  this->BoxAxisActor->SetVisibility(this->ViewNode->GetBoxVisible());

  double pos = fov * 0.6;
  double letterSize = this->ViewNode->GetLetterSize();
  double scale = fov * letterSize;

  for (unsigned int i=0; i<AxisLabelActors.size(); i++)
    {
    this->AxisLabelActors[i]->SetScale(scale,scale,scale);
    this->AxisLabelActors[i]->SetVisibility(this->ViewNode->GetAxisLabelsVisible());
    this->AxisLabelActors[i]->SetCamera(this->MainViewer->GetRenderer()->GetActiveCamera());
    }

  this->AxisLabelActors[0]->SetPosition(pos,0,0);
  this->AxisLabelActors[1]->SetPosition(0,pos,0);
  this->AxisLabelActors[2]->SetPosition(0,0,pos);
  this->AxisLabelActors[3]->SetPosition(-pos,0,0);
  this->AxisLabelActors[4]->SetPosition(0,-pos,0);
  this->AxisLabelActors[5]->SetPosition(0,0,-pos);

}

//---------------------------------------------------------------------------
int vtkSlicerViewerWidget::UpdateClipSlicesFormMRML()
{
  if (this->MRMLScene == NULL)
    {
    return 0;
    }

  // update ClipModels node
  vtkMRMLClipModelsNode *clipNode = vtkMRMLClipModelsNode::SafeDownCast(this->MRMLScene->GetNthNodeByClass(0, "vtkMRMLClipModelsNode"));
  if (clipNode != this->ClipModelsNode) 
    {
    vtkSetAndObserveMRMLNodeMacro(this->ClipModelsNode, clipNode);
    }

  if (this->ClipModelsNode == NULL)
    {
    return 0;
    }

  // update Slice nodes
  vtkMRMLSliceNode *node= NULL;
  vtkMRMLSliceNode *nodeRed= NULL;
  vtkMRMLSliceNode *nodeGreen= NULL;
  vtkMRMLSliceNode *nodeYellow= NULL;
  int nnodes = this->MRMLScene->GetNumberOfNodesByClass("vtkMRMLSliceNode");
  for (int n=0; n<nnodes; n++)
    {
    node = vtkMRMLSliceNode::SafeDownCast (
          this->MRMLScene->GetNthNodeByClass(n, "vtkMRMLSliceNode"));
    // TODO use perhaps SliceLogic to get the name instead of "Red" etc.
    if (!strcmp(node->GetLayoutName(), "Red"))
      {
      nodeRed = node;
      }
    else if (!strcmp(node->GetLayoutName(), "Green"))
      {
      nodeGreen = node;
      }
    else if (!strcmp(node->GetLayoutName(), "Yellow"))
      {
      nodeYellow = node;
      }
    node = NULL;
    }

  if (nodeRed != this->RedSliceNode)
    {
    vtkSetAndObserveMRMLNodeMacro(this->RedSliceNode, nodeRed);
    }
  if (nodeGreen != this->GreenSliceNode)
   {
   vtkSetAndObserveMRMLNodeMacro(this->GreenSliceNode, nodeGreen);
   }
  if (nodeYellow != this->YellowSliceNode)
   {
   vtkSetAndObserveMRMLNodeMacro(this->YellowSliceNode, nodeYellow);
   }

  if (this->RedSliceNode == NULL || this->GreenSliceNode == NULL || this->YellowSliceNode == NULL)
    {
    return 0;
    }

  int modifiedState = 0;

  if ( this->ClipModelsNode->GetClipType() != this->ClipType)
  {
    modifiedState = 1;
    this->ClipType = this->ClipModelsNode->GetClipType();
    if (this->ClipType == vtkMRMLClipModelsNode::ClipIntersection) 
      {
      this->SlicePlanes->SetOperationTypeToIntersection();
      }
    else if (this->ClipType == vtkMRMLClipModelsNode::ClipUnion) 
      {
      this->SlicePlanes->SetOperationTypeToUnion();
      }
    else 
      {
      vtkErrorMacro("vtkMRMLClipModelsNode:: Invalid Clip Type");
      }
  }

  if (this->ClipModelsNode->GetRedSliceClipState() != this->RedSliceClipState)
    {
    if (this->RedSliceClipState == vtkMRMLClipModelsNode::ClipOff)
      {
      this->SlicePlanes->AddFunction(this->RedSlicePlane);
      }
    else if (this->ClipModelsNode->GetRedSliceClipState() == vtkMRMLClipModelsNode::ClipOff)
      {
      this->SlicePlanes->RemoveFunction(this->RedSlicePlane);
      }
    modifiedState = 1;
    this->RedSliceClipState = this->ClipModelsNode->GetRedSliceClipState();
    }

  if (this->ClipModelsNode->GetGreenSliceClipState() != this->GreenSliceClipState)
    {
    if (this->GreenSliceClipState == vtkMRMLClipModelsNode::ClipOff)
      {
      this->SlicePlanes->AddFunction(this->GreenSlicePlane);
      }
    else if (this->ClipModelsNode->GetGreenSliceClipState() == vtkMRMLClipModelsNode::ClipOff)
      {
      this->SlicePlanes->RemoveFunction(this->GreenSlicePlane);
      }
    modifiedState = 1;
    this->GreenSliceClipState = this->ClipModelsNode->GetGreenSliceClipState();
    }

  if (this->ClipModelsNode->GetYellowSliceClipState() != this->YellowSliceClipState)
    {
    if (this->YellowSliceClipState == vtkMRMLClipModelsNode::ClipOff)
      {
      this->SlicePlanes->AddFunction(this->YellowSlicePlane);
      }
    else if (this->ClipModelsNode->GetYellowSliceClipState() == vtkMRMLClipModelsNode::ClipOff)
      {
      this->SlicePlanes->RemoveFunction(this->YellowSlicePlane);
      }
    modifiedState = 1;
    this->YellowSliceClipState = this->ClipModelsNode->GetYellowSliceClipState();
    }

  // compute clipping on/off
  if (this->ClipModelsNode->GetRedSliceClipState() == vtkMRMLClipModelsNode::ClipOff &&
      this->ClipModelsNode->GetGreenSliceClipState() == vtkMRMLClipModelsNode::ClipOff &&
      this->ClipModelsNode->GetYellowSliceClipState() == vtkMRMLClipModelsNode::ClipOff )
    {
    this->ClippingOn = false;
    }
  else
    {
    this->ClippingOn = true;
    }

  // set slice plane normals and origins
  vtkMatrix4x4 *sliceMatrix = NULL;
  double normal[3];
  double origin[3];
  int i;

  sliceMatrix = this->RedSliceNode->GetSliceToRAS();
  for (i=0; i<3; i++) 
    {
    normal[i] = sliceMatrix->GetElement(i,2);
    if (this->RedSliceClipState == vtkMRMLClipModelsNode::ClipNegativeSpace)
      {
      normal[i] = - normal[i];
      }
    origin[i] = sliceMatrix->GetElement(i,3);
    }
  this->RedSlicePlane->SetNormal(normal);
  this->RedSlicePlane->SetOrigin(origin);

  sliceMatrix = this->GreenSliceNode->GetSliceToRAS();
  for (i=0; i<3; i++) 
    {
    normal[i] = sliceMatrix->GetElement(i,2);
    if (this->GreenSliceClipState == vtkMRMLClipModelsNode::ClipNegativeSpace)
      {
      normal[i] = - normal[i];
      }
    origin[i] = sliceMatrix->GetElement(i,3);
    }
  this->GreenSlicePlane->SetNormal(normal);
  this->GreenSlicePlane->SetOrigin(origin);

  sliceMatrix = this->YellowSliceNode->GetSliceToRAS();
  for (i=0; i<3; i++) 
    {
    normal[i] = sliceMatrix->GetElement(i,2);
    if (this->YellowSliceClipState == vtkMRMLClipModelsNode::ClipNegativeSpace)
      {
      normal[i] = - normal[i];
      }
    origin[i] = sliceMatrix->GetElement(i,3);
    }
  this->YellowSlicePlane->SetNormal(normal);
  this->YellowSlicePlane->SetOrigin(origin);

  return modifiedState;
}

//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::ProcessMRMLEvents ( vtkObject *caller,
                                                unsigned long event, 
                                                void *callData )
{ 
  if (this->ProcessingMRMLEvent != 0 )
    {
    return;
    }

  this->ProcessingMRMLEvent = event;

  vtkDebugMacro("ProcessMRMLEvents: processing event " << event);
   
  if (event == vtkMRMLScene::SceneCloseEvent )
    {
    this->SceneClosing = true;
    this->RemoveModelProps();
    this->RemoveHierarchyObservers();
    this->RemoveModelObservers();
    }
  else 
    {
    this->SceneClosing = false;
    }

  if ( vtkMRMLScene::SafeDownCast(caller) == this->MRMLScene 
    && (event == vtkMRMLScene::NodeAddedEvent || event == vtkMRMLScene::NodeRemovedEvent ) )
    {
    vtkMRMLNode *node = (vtkMRMLNode*) (callData);
    if (node != NULL && node->IsA("vtkMRMLModelNode") )
      {
      if (event == vtkMRMLScene::NodeRemovedEvent)
        {
        std::string sid(node->GetID());
        this->RemoveDispalyedID(sid);
        }
      this->UpdateFromMRMLRequested = 1;
      this->RequestRender();
      //this->UpdateFromMRML();
      }
    else if (node != NULL && node->IsA("vtkMRMLModelHierarchyNode") )
      {
      this->UpdateModelHierarchies();
      this->UpdateFromMRMLRequested = 1;
      this->RequestRender();
      //this->UpdateFromMRML();
      }
    else if (node != NULL && node->IsA("vtkMRMLClipModelsNode") )
      {
      if (event == vtkMRMLScene::NodeAddedEvent)
        {
        vtkSetAndObserveMRMLNodeMacro(this->ClipModelsNode, node);
        }
      else if (event == vtkMRMLScene::NodeRemovedEvent)
        {
        vtkSetMRMLNodeMacro(this->ClipModelsNode, NULL);
        }
      this->UpdateFromMRMLRequested = 1;
      this->RequestRender();
      //this->UpdateFromMRML();
      }
    }
  else if (vtkMRMLCameraNode::SafeDownCast(caller) != NULL &&
           event == vtkCommand::ModifiedEvent)
    {
    vtkDebugMacro("ProcessingMRML: got a camera node modified event");
    this->UpdateCameraNode();
    this->RequestRender();
    }
  else if (vtkMRMLViewNode::SafeDownCast(caller) != NULL &&
           event == vtkCommand::ModifiedEvent)
    {
    vtkDebugMacro("ProcessingMRML: got a view node modified event");
    this->UpdateViewNode();
    this->RequestRender();
    }
  else if (vtkMRMLModelNode::SafeDownCast(caller) != NULL)
    {
    // check for events on a model node
    vtkMRMLModelNode *modelNode = vtkMRMLModelNode::SafeDownCast(caller);
    if (this->DisplayedModelActors.find(modelNode->GetID()) != this->DisplayedModelActors.end() &&
       (event == vtkCommand::ModifiedEvent ||
        event == vtkMRMLModelNode::PolyDataModifiedEvent ||
        event == vtkMRMLModelNode::DisplayModifiedEvent) )
      {
      // if the node is already cached with an actor process only this one
      // if it was not visible and is still not visible do nothing
      vtkMRMLModelDisplayNode *dnode = this->GetModelDisplayNode(modelNode);
      if (!(dnode && dnode->GetVisibility() == 0 && this->GetDisplayedModelsVisibility(modelNode) == 0))
        {
        this->UpdateClipSlicesFormMRML();
        this->UpdateModifiedModel(modelNode);
        this->RequestRender( );
        }
      }
    else
      {
      this->UpdateFromMRMLRequested = 1;
      this->RequestRender();
      //this->UpdateFromMRML();
      }
    }
  else if (vtkMRMLClipModelsNode::SafeDownCast(caller) != NULL &&
           event == vtkCommand::ModifiedEvent)
    {
    this->UpdateFromMRMLRequested = 1;
    this->RequestRender();
    //this->UpdateFromMRML();
    }
  else if (vtkMRMLSliceNode::SafeDownCast(caller) != NULL &&
           event == vtkCommand::ModifiedEvent && (this->UpdateClipSlicesFormMRML() || this->ClippingOn))
    {
    this->UpdateFromMRMLRequested = 1;
    this->RequestRender();
    //this->UpdateFromMRML();
    }
  else if (vtkMRMLModelHierarchyNode::SafeDownCast(caller) &&
           event == vtkCommand::ModifiedEvent)
    {
    this->UpdateModelHierarchies();
    this->UpdateFromMRMLRequested = 1;
    this->RequestRender();
    //this->UpdateFromMRML();
    }
  else
    {
    vtkDebugMacro("ProcessMRMLEvents: unhandled event " << event << " " << ((event == 31) ? "ModifiedEvent" : "not ModifiedEvent"));
    if (vtkMRMLScene::SafeDownCast(caller) == this->MRMLScene) { vtkDebugMacro("\ton the mrml scene"); }
    if (vtkMRMLNode::SafeDownCast(caller) != NULL) { vtkDebugMacro("\tmrml node id = " << vtkMRMLNode::SafeDownCast(caller)->GetID()); }
    }
  
  this->ProcessingMRMLEvent = 0;
}
//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::UpdateCameraNode()
{
  if (this->SceneClosing)
    {
    return;
    }
  // find an active camera
  // or any camera if none is active
  vtkMRMLCameraNode *node = NULL;
  int nnodes = this->MRMLScene->GetNumberOfNodesByClass("vtkMRMLCameraNode");
  for (int n=0; n<nnodes; n++)
    {
    node = vtkMRMLCameraNode::SafeDownCast (
       this->MRMLScene->GetNthNodeByClass(n, "vtkMRMLCameraNode"));
    if (node->GetActive())
      {
      break;
      }
    }

  if ( this->CameraNode != NULL && node != NULL && this->CameraNode != node)
    {
    // local CameraNode is out of sync with the scene
    this->SetAndObserveCameraNode (NULL);
    }
  if ( this->CameraNode != NULL && this->MRMLScene->GetNodeByID(this->CameraNode->GetID()) == NULL)
    {
    // local node not in the scene
    this->SetAndObserveCameraNode (NULL);
    }
  if ( this->CameraNode == NULL )
    {
    if ( node == NULL )
      {
      // no camera in the scene and local
      // create an active camera
      node = vtkMRMLCameraNode::New();
      node->SetActive(1);
      this->MRMLScene->AddNode(node);
      node->Delete();
      }
    this->SetAndObserveCameraNode (node);
    }
 
  vtkRenderWindowInteractor *rwi = this->MainViewer->GetRenderWindowInteractor();
  if (rwi)
    {
    vtkInteractorObserver *iobs = rwi->GetInteractorStyle();
    vtkSlicerViewerInteractorStyle *istyle = vtkSlicerViewerInteractorStyle::SafeDownCast(iobs);
    if (istyle)
      {
      istyle->SetCameraNode(this->CameraNode);
      }
    }
  this->MainViewer->GetRenderer()->SetActiveCamera(this->CameraNode->GetCamera());
}


//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::UpdateViewNode()
{
  if (this->SceneClosing)
    {
    return;
    }

  vtkMRMLViewNode *node =  vtkMRMLViewNode::SafeDownCast (
       this->MRMLScene->GetNthNodeByClass(0, "vtkMRMLViewNode"));

  if ( this->ViewNode != NULL && node != NULL && this->ViewNode != node)
    {
    // local ViewNode is out of sync with the scene
    this->SetAndObserveViewNode (NULL);
    }
  if ( this->ViewNode != NULL && this->MRMLScene->GetNodeByID(this->ViewNode->GetID()) == NULL)
    {
    // local node not in the scene
    this->SetAndObserveViewNode (NULL);
    }
  if ( this->ViewNode == NULL )
    {
    if ( node == NULL )
      {
      // no view in the scene and local
      // create an active camera
      node = vtkMRMLViewNode::New();
      this->MRMLScene->AddNode(node);
      node->Delete();
      }
    this->SetAndObserveViewNode (node);
    }
 
}


//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::RemoveWidgetObservers ( ) 
{
}

//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::CreateWidget ( )
{
  if (this->IsCreated())
    {
    vtkErrorMacro(<< this->GetClassName() << " already created");
    return;
    }
  
  // Call the superclass to create the whole widget
  
  this->Superclass::CreateWidget();

  this->ViewerFrame = vtkKWFrame::New ( );
  this->ViewerFrame->SetParent ( this->GetParent ( ) );
  this->ViewerFrame->Create ( );
  
  this->MainViewer = vtkKWRenderWidget::New ( );  
  this->MainViewer->SetParent (this->ViewerFrame );
  this->MainViewer->Create ( );

  // make a Slicer viewer interactor style to process our events
  // look at the InteractorStyle to get our events
  vtkRenderWindowInteractor *rwi = this->MainViewer->GetRenderWindowInteractor();
  if (rwi)
    {
    vtkSlicerViewerInteractorStyle *iStyle = vtkSlicerViewerInteractorStyle::New();
    iStyle->SetViewerWidget( this );
    iStyle->SetApplicationLogic ( this->ApplicationLogic );
    rwi->SetInteractorStyle (iStyle);
    iStyle->Delete();
    }


  // Set the viewer's minimum dimension to be the modifiedState as that for
  // the three main Slice viewers.
  this->MainViewer->GetRenderer()->GetActiveCamera()->ParallelProjectionOff();
  if ( this->GetApplication() != NULL )
    {
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();      
    this->MainViewer->SetWidth ( app->GetMainLayout()->GetSliceViewerMinDim() );
    this->MainViewer->SetHeight ( app->GetMainLayout()->GetSliceViewerMinDim() );
    }
    

  // observe scene for add/remove nodes
  vtkIntArray *events = vtkIntArray::New();
  events->InsertNextValue(vtkMRMLScene::SceneCloseEvent);
  //events->InsertNextValue(vtkMRMLScene::NewSceneEvent);
  events->InsertNextValue(vtkCommand::ModifiedEvent);
  events->InsertNextValue(vtkMRMLScene::NodeAddedEvent);
  events->InsertNextValue(vtkMRMLScene::NodeRemovedEvent);
  this->SetAndObserveMRMLSceneEvents(this->MRMLScene, events);
  events->Delete();

  this->CreateClipSlices();

  this->CreateAxis();

  //this->PackWidget ( );
  this->MainViewer->ResetCamera ( );
}

//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::UpdateFromMRML()
{
  this->UpdateAxis();

  this->UpdateCameraNode();

  this->UpdateClipSlicesFormMRML();

  this->RemoveModelProps ( );
  
  this->UpdateModelsFromMRML();

  this->RequestRender ( );

  this->UpdateFromMRMLRequested = 0;
}

//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::UpdateModelsFromMRML()
{
  vtkMRMLScene *scene = this->GetMRMLScene();
  vtkMRMLNode *node = NULL;
  std::vector<vtkMRMLModelNode *> slices;

  // find volume slices
  bool clearDisplayedModels = false;
  int nnodes = scene->GetNumberOfNodesByClass("vtkMRMLModelNode");
  for (int n=0; n<nnodes; n++)
    {
    node = scene->GetNthNodeByClass(n, "vtkMRMLModelNode");
    vtkMRMLModelNode *model = vtkMRMLModelNode::SafeDownCast(node);
    // render slices last so that transparent objects are rendered in front of them
    if (!strcmp(model->GetName(), "Red Volume Slice") ||
        !strcmp(model->GetName(), "Green Volume Slice") ||
        !strcmp(model->GetName(), "Yellow Volume Slice"))
      {
      slices.push_back(model);
      if (this->DisplayedModelActors.find(model->GetID()) == this->DisplayedModelActors.end() )
        {
        clearDisplayedModels = true;
        }
      }
    }

  if (clearDisplayedModels)
    {
    this->MainViewer->RemoveAllViewProps();
    this->RemoveModelObservers();
    this->RemoveHierarchyObservers();
    this->DisplayedModelActors.clear();
    this->DisplayedModelNodes.clear();
    this->DisplayedModelsClipState.clear();
    this->DisplayedModelsVisibility.clear();
    this->AddAxisActors();
    this->UpdateModelHierarchies();
    }

  // render slices first
  for (unsigned int i=0; i<slices.size(); i++)
    {
    vtkMRMLModelNode *model = slices[i];
    // add nodes that are not in the list yet
    if (this->DisplayedModelActors.find(model->GetID()) == this->DisplayedModelActors.end() )
      {
      this->UpdateModel(model);
      } 
    //vtkActor *actor = this->DisplayedModelActors.find(model->GetID())->second;
    vtkActor *actor = this->DisplayedModelActors[ model->GetID() ];
    this->SetModelDisplayProperty(model, actor);
    }
  
  // render the rest of the models
  int nmodels = scene->GetNumberOfNodesByClass("vtkMRMLModelNode");
  for (int n=0; n<nmodels; n++)
    {
    vtkMRMLModelNode *model = vtkMRMLModelNode::SafeDownCast(scene->GetNthNodeByClass(n, "vtkMRMLModelNode"));
    // render slices last so that transparent objects are rendered in fron of them
    if (!strcmp(model->GetName(), "Red Volume Slice") ||
        !strcmp(model->GetName(), "Green Volume Slice") ||
        !strcmp(model->GetName(), "Yellow Volume Slice"))
      {
      continue;
      }
    this->UpdateModifiedModel(model);
    } // end while

}

//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::UpdateModifiedModel(vtkMRMLModelNode *model)
{
  this->UpdateModel(model);
  //vtkActor *actor = this->DisplayedModelActors.find(model->GetID())->second;
  vtkActor *actor = this->DisplayedModelActors[ model->GetID() ];
  this->SetModelDisplayProperty(model, actor);
}

//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::UpdateModelPolyData(vtkMRMLModelNode *model)
{
  vtkMRMLModelDisplayNode *modelDisplayNode = this->GetModelDisplayNode(model);
  vtkActor* actor = NULL;
  std::map<std::string, vtkActor *>::iterator ait;

  ait = this->DisplayedModelActors.find(model->GetID());
  if (ait == this->DisplayedModelActors.end() )
    {
    actor = vtkActor::New();
    }
  else
    {
    actor = (*ait).second;
    std::map<std::string, int>::iterator cit = this->DisplayedModelsClipState.find(model->GetID());
    if (modelDisplayNode && cit != this->DisplayedModelsClipState.end() && cit->second == modelDisplayNode->GetClipping())
      {
      this->DisplayedModelsVisibility[model->GetID()] = modelDisplayNode->GetVisibility();
      return;
      }
    }

  vtkClipPolyData *clipper = NULL;
  if (this->ClippingOn && modelDisplayNode != NULL && modelDisplayNode->GetClipping())
    {
    clipper = vtkClipPolyData::New();
    clipper->SetClipFunction(this->SlicePlanes);
    clipper->SetValue( 0.0);
    }

  vtkPolyDataMapper *mapper = vtkPolyDataMapper::New ();
  if (clipper)
    {
    clipper->SetInput ( model->GetPolyData() );
    clipper->Update();
    mapper->SetInput ( clipper->GetOutput() );
    }
  else
    {
    mapper->SetInput ( model->GetPolyData() );
    }

 
  actor->SetMapper( mapper );
  mapper->Delete();

  if (ait == this->DisplayedModelActors.end())
    {
    this->MainViewer->AddViewProp( actor );
    this->DisplayedModelActors[model->GetID()] = actor;
    this->DisplayedModelNodes[std::string(model->GetID())] = model;
    if (modelDisplayNode)
      {
      this->DisplayedModelsVisibility[model->GetID()] = modelDisplayNode->GetVisibility();
      }
    else
      {
      this->DisplayedModelsVisibility[model->GetID()] = 1;
      }
    actor->Delete();
    }

  if (clipper)
    {
    this->DisplayedModelsClipState[model->GetID()] = 1;
    clipper->Delete();
    }
  else
    {
    this->DisplayedModelsClipState[model->GetID()] = 0;
    }
}

//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::UpdateModel(vtkMRMLModelNode *model)
{
  this->UpdateModelPolyData(model);

  // observe polydata
  if (!model->HasObserver( vtkMRMLModelNode::PolyDataModifiedEvent, this->MRMLCallbackCommand ))
    {
    model->AddObserver ( vtkMRMLModelNode::PolyDataModifiedEvent, this->MRMLCallbackCommand );
    }
  // observe display node
  if (!model->HasObserver ( vtkMRMLModelNode::DisplayModifiedEvent, this->MRMLCallbackCommand ))
    {
    model->AddObserver ( vtkMRMLModelNode::DisplayModifiedEvent, this->MRMLCallbackCommand );
    }

  if (!model->HasObserver ( vtkMRMLTransformableNode::TransformModifiedEvent, this->MRMLCallbackCommand ) )
    {
    model->AddObserver ( vtkMRMLTransformableNode::TransformModifiedEvent, this->MRMLCallbackCommand );
    }
}

//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::CheckModelHierarchies()
{
  if (this->MRMLScene == NULL)
    {
    return;
    }
  this->ModelHierarchyLogic->SetMRMLScene(this->MRMLScene);
  int nnodes = this->ModelHierarchyLogic->CreateModelToHierarchyMap();
  this->ModelHierarchiesPresent = nnodes > 0 ? true:false;
}

//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::AddHierarchiyObservers()
{
   if (this->MRMLScene == NULL)
    {
    return;
    }
  vtkMRMLModelHierarchyNode *node;
  int nnodes = this->MRMLScene->GetNumberOfNodesByClass("vtkMRMLModelHierarchyNode");

  for (int n=0; n<nnodes; n++)
    {
    node = vtkMRMLModelHierarchyNode::SafeDownCast (
          this->MRMLScene->GetNthNodeByClass(n, "vtkMRMLModelHierarchyNode"));
    bool found = false;
    std::map<std::string, int>::iterator iter;
    // search for matching string (can't use find, since it would look for 
    // matching pointer not matching content)
    for(iter=this->RegisteredModelHierarchies.begin(); iter != this->RegisteredModelHierarchies.end(); iter++) 
      {
      if ( iter->first.c_str() && !strcmp( iter->first.c_str(), node->GetID() ) )
        {
        found = true;
        break;
        }
      }
    if (!found)
      {
      node->AddObserver ( vtkCommand::ModifiedEvent, this->MRMLCallbackCommand );
      this->RegisteredModelHierarchies[node->GetID()] = 0;
      }
    }
}

//----------------------------
vtkMRMLModelDisplayNode* vtkSlicerViewerWidget::GetModelDisplayNode(vtkMRMLModelNode *model)
{
  vtkMRMLModelDisplayNode* dnode = NULL;
  if (this->ModelHierarchiesPresent)
    {
    vtkMRMLModelHierarchyNode* mhnode = NULL;
    mhnode = this->ModelHierarchyLogic->GetModelHierarchyNode(model->GetID());
    if (mhnode) 
      {
      mhnode = mhnode->GetUnExpandedParentNode();
      }
    if (mhnode) 
      {
      dnode = mhnode->GetDisplayNode();
      }
    else
      {
      dnode = model->GetModelDisplayNode();
      }
    }
  else 
    {
    dnode = model->GetModelDisplayNode();
    }
  return dnode;
}


//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::RequestRender()
{
  if (this->GetRenderPending())
    {
    return;
    }

  this->SetRenderPending(1);
  this->Script("after idle \"%s Render\"", this->GetTclName());
}

//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::Render()
{
  if (this->UpdateFromMRMLRequested)
    {
    this->UpdateFromMRML();
    }

  this->MainViewer->Render();
  vtkDebugMacro("vtkSlicerViewerWidget::Render called render" << endl);
  this->SetRenderPending(0);
}


//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::RemoveModelProps()
{
  std::map<std::string, vtkActor *>::iterator iter;
  std::map<std::string, int>::iterator clipIter;
  std::vector<std::string> removedIDs;
  for(iter=this->DisplayedModelActors.begin(); iter != this->DisplayedModelActors.end(); iter++) 
    {
    vtkMRMLModelNode *model = vtkMRMLModelNode::SafeDownCast(this->GetMRMLScene()->GetNodeByID(iter->first));
    if (model == NULL)
      {
      this->MainViewer->RemoveViewProp(iter->second);
      removedIDs.push_back(iter->first);
      }
    else
      {
      vtkMRMLModelDisplayNode *modelDisplayNode = this->GetModelDisplayNode(model);
      int clipModel = 0;
      if (modelDisplayNode != NULL)
        {
        clipModel = modelDisplayNode->GetClipping();
        }
      clipIter = DisplayedModelsClipState.find(iter->first);
      if (clipIter == DisplayedModelsClipState.end())
        {
        vtkErrorMacro ("vtkSlicerViewerWidget::RemoveModelProps() Unknown clip state\n");
        }
      else 
        {
        if (clipIter->second && !this->ClippingOn ||
          this->ClippingOn && clipIter->second != clipModel)
          {
          this->MainViewer->RemoveViewProp(iter->second);
          removedIDs.push_back(iter->first);
          }     
        }
      }
    }
  for (unsigned int i=0; i< removedIDs.size(); i++)
    {
    this->RemoveDispalyedID(removedIDs[i]);
    }
}

void vtkSlicerViewerWidget::RemoveDispalyedID(std::string &id)
{
  std::map<std::string, vtkMRMLModelNode *>::iterator modelIter;
  this->DisplayedModelActors.erase(id);
  this->DisplayedModelsClipState.erase(id);
  this->DisplayedModelsVisibility.erase(id);
  modelIter = this->DisplayedModelNodes.find(id);
  if(modelIter != this->DisplayedModelNodes.end())
    {
    this->RemoveModelObservers(modelIter->second);
    this->DisplayedModelNodes.erase(modelIter->first);
    }
}

int vtkSlicerViewerWidget::GetDisplayedModelsVisibility(vtkMRMLModelNode *model)
{
  int visibility = 1;
  
  std::map<std::string, int>::iterator iter;
  iter = this->DisplayedModelsVisibility.find(model->GetID());
  if (iter != this->DisplayedModelsVisibility.end())
    {
    visibility = iter->second;
    }

  return visibility;
}

//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::RemoveMRMLObservers()
{
  this->RemoveModelObservers();
  this->RemoveHierarchyObservers();

  //this->RemoveFiducialObservers();

  this->SetAndObserveMRMLScene(NULL);
}

//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::RemoveModelObservers()
{
  std::map<std::string, vtkActor *>::iterator iter;
  std::map<std::string, vtkMRMLModelNode *>::iterator modelIter;
  for(iter=this->DisplayedModelActors.begin(); iter != this->DisplayedModelActors.end(); iter++) 
    {
    vtkMRMLModelNode *model = vtkMRMLModelNode::SafeDownCast(this->GetMRMLScene()->GetNodeByID(iter->first));
    if (model != NULL)
      {
      model->RemoveObservers ( vtkMRMLModelNode::PolyDataModifiedEvent, this->MRMLCallbackCommand );
      model->RemoveObservers ( vtkMRMLModelNode::DisplayModifiedEvent, this->MRMLCallbackCommand );
      model->RemoveObservers ( vtkMRMLTransformableNode::TransformModifiedEvent, this->MRMLCallbackCommand );
      }
    modelIter = this->DisplayedModelNodes.find(std::string(iter->first));
    if(modelIter != this->DisplayedModelNodes.end())
      {
      this->RemoveModelObservers(modelIter->second);
      }
    }
  this->DisplayedModelActors.clear();
  this->DisplayedModelNodes.clear();
  this->DisplayedModelsClipState.clear();
  this->DisplayedModelsVisibility.clear();

}

//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::RemoveModelObservers( vtkMRMLModelNode *model)
{
  if (model != NULL)
    {
    model->RemoveObservers ( vtkMRMLModelNode::PolyDataModifiedEvent, this->MRMLCallbackCommand );
    model->RemoveObservers ( vtkMRMLModelNode::DisplayModifiedEvent, this->MRMLCallbackCommand );
    model->RemoveObservers ( vtkMRMLTransformableNode::TransformModifiedEvent, this->MRMLCallbackCommand );
    }
}

//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::RemoveHierarchyObservers()
{
  std::map<std::string, int>::iterator iter;
  for(iter=this->RegisteredModelHierarchies.begin(); iter != this->RegisteredModelHierarchies.end(); iter++) 
    {
    vtkMRMLModelHierarchyNode *node = vtkMRMLModelHierarchyNode::SafeDownCast(this->GetMRMLScene()->GetNodeByID(iter->first));
    if (node)
      {
      node->RemoveObservers ( vtkCommand::ModifiedEvent, this->MRMLCallbackCommand );
      }
    }
  RegisteredModelHierarchies.clear();
}


//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::SetModelDisplayProperty(vtkMRMLModelNode *model,  vtkActor *actor)
{
  vtkMRMLTransformNode* tnode = model->GetParentTransformNode();
  vtkMatrix4x4* transformToWorld = vtkMatrix4x4::New();
  transformToWorld->Identity();
  if (tnode != NULL && tnode->IsLinear())
    {
    vtkMRMLLinearTransformNode *lnode = vtkMRMLLinearTransformNode::SafeDownCast(tnode);
    lnode->GetMatrixTransformToWorld(transformToWorld);
    }
  actor->SetUserMatrix(transformToWorld);
  transformToWorld->Delete();
 
  vtkMRMLModelDisplayNode *dnode = this->GetModelDisplayNode(model);
  if (dnode != NULL)
    {
    actor->SetVisibility(dnode->GetVisibility());
    this->DisplayedModelsVisibility[model->GetID()] = dnode->GetVisibility();

    actor->GetMapper()->SetScalarVisibility(dnode->GetScalarVisibility());
    // if the scalars are visible, set active scalars, try to get the lookup
    // table
    if (dnode->GetScalarVisibility())
      {
      if (dnode->GetColorNode() != NULL)
        {
        if (dnode->GetColorNode()->GetLookupTable() != NULL)
        {
        actor->GetMapper()->SetLookupTable(dnode->GetColorNode()->GetLookupTable());
        }
        else if (dnode->GetColorNode()->IsA("vtkMRMLProceduralColorNode") &&
                 vtkMRMLProceduralColorNode::SafeDownCast(dnode->GetColorNode())->GetColorTransferFunction() != NULL)
          {
          actor->GetMapper()->SetLookupTable((vtkScalarsToColors*)(vtkMRMLProceduralColorNode::SafeDownCast(dnode->GetColorNode())->GetColorTransferFunction()));
          }
        }
      
      if (dnode->GetActiveScalarName() != NULL)
        {
        model->SetActiveScalars(dnode->GetActiveScalarName(), "Scalars");
        }
      // set the scalar range
      actor->GetMapper()->SetScalarRange(dnode->GetScalarRange());
      }
    
    actor->GetProperty()->SetColor(dnode->GetColor());
    actor->GetProperty()->SetOpacity(dnode->GetOpacity());
    actor->GetProperty()->SetAmbient(dnode->GetAmbient());
    actor->GetProperty()->SetDiffuse(dnode->GetDiffuse());
    actor->GetProperty()->SetSpecular(dnode->GetSpecular());
    actor->GetProperty()->SetSpecularPower(dnode->GetPower());
    if (dnode->GetTextureImageData() != NULL)
      {
      if (actor->GetTexture() == NULL)
        {
        vtkTexture *texture = vtkTexture::New();
        texture->SetInterpolate(1);
        actor->SetTexture(texture);
        texture->Delete();
        }
      actor->GetTexture()->SetInput(dnode->GetTextureImageData());
      }
    else
      {
      actor->SetTexture(NULL);
      }
    }
}


//---------------------------------------------------------------------------
  // Description:
  // return the current actor corresponding to a give MRML ID
vtkActor *
vtkSlicerViewerWidget::GetActorByID (const char *id)
{
  if ( !id )
    {
    return (NULL);
    }

  std::map<std::string, vtkActor *>::iterator iter;
  // search for matching string (can't use find, since it would look for 
  // matching pointer not matching content)
  for(iter=this->DisplayedModelActors.begin(); iter != this->DisplayedModelActors.end(); iter++) 
    {
    if ( iter->first.c_str() && !strcmp( iter->first.c_str(), id ) )
      {
      return (iter->second);
      }
    }
  return (NULL);
}

//---------------------------------------------------------------------------
  // Description:
  // return the ID for the given actor 
const char *
vtkSlicerViewerWidget::GetIDByActor (vtkActor *actor)
{
  if ( !actor )
    {
    return (NULL);
    }

  std::map<std::string, vtkActor *>::iterator iter;
  for(iter=this->DisplayedModelActors.begin(); iter != this->DisplayedModelActors.end(); iter++) 
    {
    if ( iter->second && ( iter->second == actor ) )
      {
      return (iter->first.c_str());
      }
    }
  return (NULL);
}



//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::PackWidget ( vtkKWFrame *f )
{
  this->Script  ("pack %s -side left -fill both -expand y -padx 0 -pady 0 -in %s",
                 this->ViewerFrame->GetWidgetName ( ), f->GetWidgetName() );
  this->Script  ("pack %s -side top -anchor c  -fill both -expand y -padx 0 -pady 0",
                 this->MainViewer->GetWidgetName ( ) );
}


//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::GridWidget ( vtkKWFrame *f, int row, int col )
{
  this->Script ( "grid configure %s -in %s -row %d -column %d -stick news -padx 0 -pady 0",
                 this->ViewerFrame->GetWidgetName(), f->GetWidgetName(), row, col );
//  this->Script  ("grid %s -row %d -column %d -sticky news -padx 0 -pady 0 -in %s",
//                 this->ViewerFrame->GetWidgetName ( ), row, col, f->GetWidgetName()  );
  this->Script  ("pack %s -side top -anchor c  -fill both -expand y -padx 0 -pady 0",
                 this->MainViewer->GetWidgetName ( ) );
}

//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::UnpackWidget ( )
{
  this->Script ( "pack forget %s ", this->MainViewer->GetWidgetName ( ) );
  this->Script ( "pack forget %s ", this->ViewerFrame->GetWidgetName ( ) );
}

  
//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::UngridWidget ( )
{
  this->Script ( "grid forget %s ", this->MainViewer->GetWidgetName ( ) );
  this->Script ( "pack forget %s ", this->ViewerFrame->GetWidgetName ( ) );
}

//---------------------------------------------------------------------------
void vtkSlicerViewerWidget::ResetPick()
{
  double zero[3] = {0.0, 0.0, 0.0};
  this->PickedNodeName = std::string("");
  this->SetPickedRAS(zero);
  this->SetPickedCellID(-1);
  this->SetPickedPointID(-1);
}

//---------------------------------------------------------------------------
int vtkSlicerViewerWidget::Pick(int x, int y)
{
  double RASPoint[3] = {0.0, 0.0, 0.0};
  double pickPoint[3] = {0.0, 0.0, 0.0};

  // reset the pick vars
  this->ResetPick();
  
  vtkRenderer *ren;
  if (this->GetMainViewer() != NULL)
    {
    ren = this->GetMainViewer()->GetRenderer();
    }
  else
    {
    vtkErrorMacro("Pick: unable to get renderer\n");
    return 0;
    }
   // get the current renderer's size
  int *renSize = ren->GetSize();
  // resize the interactor?
  
  // pass the event's display point to the world point picker
  double displayPoint[3];
  displayPoint[0] = x;
  displayPoint[1] = renSize[1] - y;
  displayPoint[2] = 0.0;

  if (this->CellPicker->Pick(displayPoint, ren))
    {
    this->CellPicker->GetPickPosition(pickPoint);
    this->SetPickedCellID(this->CellPicker->GetCellId());
    // get the pointer to the poly data that the cell was in
    vtkPolyData *polyData = vtkPolyData::SafeDownCast(this->CellPicker->GetDataSet());
    if (polyData != NULL)
      {
      // now find the model this poly data belongs to
      std::map<std::string, vtkMRMLModelNode *>::iterator modelIter;
      for (modelIter = this->DisplayedModelNodes.begin();
           modelIter != this->DisplayedModelNodes.end();
           modelIter++)
        {
        vtkDebugMacro("Checking model " << modelIter->first.c_str() << "'s polydata");
        if (modelIter->second != NULL)
          {
          if (modelIter->second->GetPolyData() == polyData)
            {
            vtkDebugMacro("Found matching poly data, pick was on model " << modelIter->first.c_str());
            this->PickedNodeName = modelIter->first;
            
            // figure out the closest vertex in the picked cell to the picked RAS
            // point. Only doing this on model nodes for now.
            vtkCell *cell = polyData->GetCell(this->GetPickedCellID());
            if (cell != NULL)
              {
              int numPoints = cell->GetNumberOfPoints();
              int closestPointId = -1;
              double closestDistance = 0.0l;
              for (int p = 0; p < numPoints; p++)
                {
                int pointId = cell->GetPointId(p);
                double *pointCoords = polyData->GetPoint(pointId);
                if (pointCoords != NULL)
                  {
                  double distance = sqrt(pow(pointCoords[0]-pickPoint[0], 2) +
                                         pow(pointCoords[1]-pickPoint[1], 2) +
                                         pow(pointCoords[2]-pickPoint[2], 2));
                  if (p == 0 ||
                      distance < closestDistance)
                    {
                    closestDistance = distance;
                    closestPointId = pointId;
                    }
                  }
                }
              vtkDebugMacro("Pick: found closest point id = " << closestPointId << ", distance = " << closestDistance);
              this->SetPickedPointID(closestPointId);
              }
            continue;
            }
          }
        }
      }
    }
  else
    {
    return 0;
    }
  /**
  if (this->PropPicker->PickProp(x, y, ren))
    {
    this->PropPicker->GetPickPosition(pickPoint);
    }
  else
    {
    return 0;
    }
    **/

  // world point picker's Pick always returns 0
  /**
  this->WorldPointPicker->Pick(displayPoint, ren);
  this->WorldPointPicker->GetPickPosition(pickPoint);
  **/

  // translate world to RAS
  for (int p = 0; p < 3; p++)
    {
    RASPoint[p] = pickPoint[p];
    }
  
  // now set up the class vars
  this->SetPickedRAS( RASPoint );

  return 1;
}     

