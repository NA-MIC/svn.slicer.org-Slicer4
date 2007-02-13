/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkSlicerVolumesLogic.cxx,v $
  Date:      $Date: 2006/01/06 17:56:48 $
  Version:   $Revision: 1.58 $

=========================================================================auto=*/

#include "vtkObjectFactory.h"
#include "vtkCallbackCommand.h"
#include <vtksys/SystemTools.hxx> 

#include "vtkImageThreshold.h"

#include "vtkSlicerVolumesLogic.h"
#include "vtkSlicerColorLogic.h"

#include "vtkMRMLScalarVolumeNode.h"
#include "vtkMRMLVectorVolumeNode.h"
#include "vtkMRMLVolumeArchetypeStorageNode.h"

#ifdef USE_TEEM
  #include "vtkMRMLNRRDStorageNode.h"
#endif

#include "vtkMRMLDiffusionTensorVolumeNode.h"
#include "vtkMRMLDiffusionWeightedVolumeNode.h"
#include "vtkMRMLVolumeDisplayNode.h"
#include "vtkMRMLVectorVolumeDisplayNode.h"
#include "vtkMRMLDiffusionTensorVolumeDisplayNode.h"
#include "vtkMRMLDiffusionWeightedVolumeDisplayNode.h"

vtkCxxRevisionMacro(vtkSlicerVolumesLogic, "$Revision: 1.9.12.1 $");
vtkStandardNewMacro(vtkSlicerVolumesLogic);

//----------------------------------------------------------------------------
vtkSlicerVolumesLogic::vtkSlicerVolumesLogic()
{
  this->ActiveVolumeNode = NULL;
}

//----------------------------------------------------------------------------
vtkSlicerVolumesLogic::~vtkSlicerVolumesLogic()
{
  if (this->ActiveVolumeNode != NULL)
    {
    this->ActiveVolumeNode->Delete();
    this->ActiveVolumeNode = NULL;
    }
}

//----------------------------------------------------------------------------
void vtkSlicerVolumesLogic::ProcessMRMLEvents(vtkObject *caller, 
                                            unsigned long event, 
                                            void *callData)
{
}

//----------------------------------------------------------------------------
void vtkSlicerVolumesLogic::ProcessLogicEvents(vtkObject *caller, 
                                            unsigned long event, 
                                            void *callData)
{
  if (event ==  vtkCommand::ProgressEvent) 
    {
    this->InvokeEvent ( vtkCommand::ProgressEvent,callData );
    }
}

//----------------------------------------------------------------------------
void vtkSlicerVolumesLogic::SetActiveVolumeNode(vtkMRMLVolumeNode *activeNode)
{
  vtkSetMRMLNodeMacro(this->ActiveVolumeNode, activeNode );
  this->Modified();
}
#ifdef USE_TEEM
//----------------------------------------------------------------------------
vtkMRMLVolumeNode* vtkSlicerVolumesLogic::AddArchetypeVolume (char* filename, int centerImage, int labelMap, const char* volname)
{
  vtkMRMLVolumeNode *volumeNode = NULL;
  vtkMRMLVolumeDisplayNode *displayNode = NULL;

  vtkMRMLScalarVolumeNode *scalarNode = vtkMRMLScalarVolumeNode::New();
  vtkMRMLVectorVolumeNode *vectorNode = vtkMRMLVectorVolumeNode::New();
  vtkMRMLDiffusionTensorVolumeNode *tensorNode = vtkMRMLDiffusionTensorVolumeNode::New();
  vtkMRMLDiffusionWeightedVolumeNode *dwiNode = vtkMRMLDiffusionWeightedVolumeNode::New();

  // Instanciation of the two I/O mechanism
  vtkMRMLNRRDStorageNode *storageNode1 = vtkMRMLNRRDStorageNode::New();
  vtkMRMLVolumeArchetypeStorageNode *storageNode2 = vtkMRMLVolumeArchetypeStorageNode::New();
  vtkMRMLStorageNode *storageNode = NULL;
  
  storageNode1->SetFileName(filename);
  storageNode1->SetCenterImage(centerImage);
  storageNode1->AddObserver(vtkCommand::ProgressEvent,  this->LogicCallbackCommand);

  storageNode2->SetFileName(filename);
  storageNode2->SetCenterImage(centerImage);
  storageNode2->AddObserver(vtkCommand::ProgressEvent,  this->LogicCallbackCommand);

  // Try to read first with NRRD reader (look if file is a dwi or a tensor)
  cout<<"TEST DWI: "<< storageNode1->ReadData(dwiNode)<<endl;

  if (storageNode1->ReadData(dwiNode))
    {
    cout<<"DWI HAS BEEN READ"<<endl;
    displayNode = vtkMRMLDiffusionWeightedVolumeDisplayNode::New();
    // Give a chance to set/update displayNode
    volumeNode =  dwiNode;
    storageNode = storageNode1;
    cout<<"Done setting volumeNode to class: "<<volumeNode->GetClassName()<<endl;
    }
  else if (storageNode1->ReadData(tensorNode))
    {
    cout<<"Tensor HAS BEEN READ"<<endl;
    displayNode = vtkMRMLDiffusionTensorVolumeDisplayNode::New();
    volumeNode = tensorNode;
    storageNode = storageNode1;
    }
  else if (storageNode1->ReadData(vectorNode))
    {
    cout<<"Vector HAS BEEN READ WITH NRRD READER"<<endl;
    displayNode = vtkMRMLVectorVolumeDisplayNode::New();
    volumeNode = vectorNode;
    storageNode = storageNode1;
    }
  else if (storageNode2->ReadData(scalarNode))
    {
    cout<<"Scalar HAS BEEN READ WITH ARCHTYPE READER"<<endl;
    displayNode = vtkMRMLVolumeDisplayNode::New();
    scalarNode->SetLabelMap(labelMap);
    volumeNode = scalarNode;
    storageNode = storageNode2;
    }
  else if (storageNode2->ReadData(vectorNode))
    {
    cout<<"Vector HAS BEEN READ WITH ARCHTYPE READER"<<endl;
    displayNode = vtkMRMLVectorVolumeDisplayNode::New();
    volumeNode = vectorNode;
    storageNode = storageNode2;
    }

  storageNode1->RemoveObservers(vtkCommand::ProgressEvent,  this->LogicCallbackCommand);
  storageNode2->RemoveObservers(vtkCommand::ProgressEvent,  this->LogicCallbackCommand);

  if (volumeNode != NULL)
    {
    if (volname == NULL)
      {
      const vtksys_stl::string fname(filename);
      vtksys_stl::string name = vtksys::SystemTools::GetFilenameName(fname);
      volumeNode->SetName(name.c_str());
      }
    else
      {
      volumeNode->SetName(volname);
      }

    this->GetMRMLScene()->SaveStateForUndo();
    cout<<"Setting scene info"<<endl;
    volumeNode->SetScene(this->GetMRMLScene());
    storageNode->SetScene(this->GetMRMLScene());
    displayNode->SetScene(this->GetMRMLScene());
  
    //should we give the user the chance to modify this?.
    double range[2];
    cout<<"Set basic display info"<<endl;
    volumeNode->GetImageData()->GetScalarRange(range);
    displayNode->SetLowerThreshold(range[0]);
    displayNode->SetUpperThreshold(range[1]);
    displayNode->SetWindow(range[1] - range[0]);
    displayNode->SetLevel(0.5 * (range[1] + range[0]) );

    cout<<"Adding node.."<<endl;
    this->GetMRMLScene()->AddNode(storageNode);  
    this->GetMRMLScene()->AddNode(displayNode);  

    //displayNode->SetDefaultColorMap();
    vtkSlicerColorLogic *colorLogic = vtkSlicerColorLogic::New();
    displayNode->SetAndObserveColorNodeID(colorLogic->GetDefaultVolumeColorNodeID());
    colorLogic->Delete();
    
    volumeNode->SetStorageNodeID(storageNode->GetID());
    volumeNode->SetAndObserveDisplayNodeID(displayNode->GetID());

    cout<<"Name vol node "<<volumeNode->GetClassName()<<endl;
    cout<<"Display node "<<displayNode->GetClassName()<<endl;
    this->GetMRMLScene()->AddNode(volumeNode);
    cout<<"Node added to scene"<<endl;

    this->SetActiveVolumeNode(volumeNode);

    this->Modified();
    }

  scalarNode->Delete();
  vectorNode->Delete();
  dwiNode->Delete();
  tensorNode->Delete();
  storageNode1->Delete();
  storageNode2->Delete();
  if (displayNode)
    {
    displayNode->Delete();
    }
  return volumeNode;
}

//----------------------------------------------------------------------------
int vtkSlicerVolumesLogic::SaveArchetypeVolume (char* filename, vtkMRMLVolumeNode *volumeNode)
{
  if (volumeNode == NULL || filename == NULL)
    {
    return 0;
    }
  
  vtkMRMLNRRDStorageNode *storageNode1 = NULL;
  vtkMRMLVolumeArchetypeStorageNode *storageNode2 = NULL;
  vtkMRMLStorageNode *storageNode = NULL;
  vtkMRMLStorageNode *snode = volumeNode->GetStorageNode();

  if (snode != NULL)
    {
    storageNode2 = vtkMRMLVolumeArchetypeStorageNode::SafeDownCast(snode);
    storageNode1 = vtkMRMLNRRDStorageNode::SafeDownCast(snode);
    }

  // Use NRRD writer if we are dealing with DWI, DTI or vector volumes

  if (volumeNode->IsA("vtkMRMLDiffusionWeightedVolumeNode") ||
      volumeNode->IsA("vtkMRMLDiffusionTensorVolumeNode") ||
      volumeNode->IsA("vtkMRMLVectorVolumeNode"))
    {

    if (storageNode1 == NULL)
      {
      storageNode1 = vtkMRMLNRRDStorageNode::New();
      storageNode1->SetScene(this->GetMRMLScene());
      this->GetMRMLScene()->AddNode(storageNode1);
      volumeNode->SetStorageNodeID(storageNode1->GetID());
      storageNode1->Delete();
      }

    storageNode1->SetFileName(filename);
    storageNode = storageNode1;
    }
  else
    {
    if (storageNode2 == NULL)
      {
      storageNode2 = vtkMRMLVolumeArchetypeStorageNode::New();
      storageNode2->SetScene(this->GetMRMLScene());
      this->GetMRMLScene()->AddNode(storageNode2);
      volumeNode->SetStorageNodeID(storageNode2->GetID());
      storageNode2->Delete();
      }

    storageNode2->SetFileName(filename);
    storageNode = storageNode2;
    }

  int res = storageNode->WriteData(volumeNode);
  return res;
}



#else

//----------------------------------------------------------------------------
vtkMRMLVolumeNode* vtkSlicerVolumesLogic::AddArchetypeVolume (char* filename, int centerImage, int labelMap, const char* volname)
{
  vtkMRMLVolumeNode *volumeNode = NULL;
  
  vtkMRMLScalarVolumeNode *scalarNode = vtkMRMLScalarVolumeNode::New();
  vtkMRMLVectorVolumeNode *vectorNode = vtkMRMLVectorVolumeNode::New();
  
  vtkMRMLVolumeDisplayNode *displayNode = vtkMRMLVolumeDisplayNode::New();
  vtkMRMLVolumeArchetypeStorageNode *storageNode = vtkMRMLVolumeArchetypeStorageNode::New();

  storageNode->SetFileName(filename);
  storageNode->SetCenterImage(centerImage);
  storageNode->AddObserver(vtkCommand::ProgressEvent,  this->LogicCallbackCommand);

  if (storageNode->ReadData(scalarNode))
    {
    scalarNode->SetLabelMap(labelMap);
    volumeNode = scalarNode;
    }
  else if (storageNode->ReadData(vectorNode))
    {
    // cannot read scalar data, try vector
    volumeNode = vectorNode;
    }

  storageNode->RemoveObservers(vtkCommand::ProgressEvent,  this->LogicCallbackCommand);

  if (volumeNode != NULL)
    {
    if (volname == NULL)
      {
      const vtksys_stl::string fname(filename);
      vtksys_stl::string name = vtksys::SystemTools::GetFilenameName(fname);
      volumeNode->SetName(name.c_str());
      }
    else
      {
      volumeNode->SetName(volname);
      }

    this->GetMRMLScene()->SaveStateForUndo();

    volumeNode->SetScene(this->GetMRMLScene());
    storageNode->SetScene(this->GetMRMLScene());
    displayNode->SetScene(this->GetMRMLScene());

    double range[2];
    volumeNode->GetImageData()->GetScalarRange(range);
    displayNode->SetLowerThreshold(range[0]);
    displayNode->SetUpperThreshold(range[1]);
    displayNode->SetWindow(range[1] - range[0]);
    displayNode->SetLevel(0.5 * (range[1] + range[0]) );

    this->GetMRMLScene()->AddNode(storageNode);  
    this->GetMRMLScene()->AddNode(displayNode);
    int isLabelMap = 0;
    if (vtkMRMLScalarVolumeNode::SafeDownCast(volumeNode))
      {
      isLabelMap = vtkMRMLScalarVolumeNode::SafeDownCast(volumeNode)->GetLabelMap();
      }
    //displayNode->SetDefaultColorMap(isLabelMap);
    vtkSlicerColorLogic *colorLogic = vtkSlicerColorLogic::New();
      //vtkSlicerColorGUI::SafeDownCast(vtkSlicerApplication::SafeDownCast(this->GetApplication())->GetModuleGUIByName("Color"))->GetLogic();
    if (colorLogic)
      {
      if (isLabelMap)
        {
        displayNode->SetAndObserveColorNodeID(colorLogic->GetDefaultLabelMapColorNodeID());
        }
      else
        {
        displayNode->SetAndObserveColorNodeID(colorLogic->GetDefaultVolumeColorNodeID());
        }
      colorLogic->Delete();
      }
   
    volumeNode->SetStorageNodeID(storageNode->GetID());
    volumeNode->SetAndObserveDisplayNodeID(displayNode->GetID());    

    this->GetMRMLScene()->AddNode(volumeNode);  

    this->SetActiveVolumeNode(volumeNode);
    
    this->Modified();  
    }

  scalarNode->Delete();
  vectorNode->Delete();
  storageNode->Delete();
  displayNode->Delete();

  return volumeNode;
}

//----------------------------------------------------------------------------
int vtkSlicerVolumesLogic::SaveArchetypeVolume (char* filename, vtkMRMLVolumeNode *volumeNode)
{
  if (volumeNode == NULL || filename == NULL)
    {
    return 0;
    }
  
  vtkMRMLVolumeArchetypeStorageNode *storageNode = NULL;
  vtkMRMLStorageNode *snode = volumeNode->GetStorageNode();
  if (snode != NULL)
    {
    storageNode = vtkMRMLVolumeArchetypeStorageNode::SafeDownCast(snode);
    }
  if (storageNode == NULL)
    {
    storageNode = vtkMRMLVolumeArchetypeStorageNode::New();
    storageNode->SetScene(this->GetMRMLScene());
    this->GetMRMLScene()->AddNode(storageNode);  
    volumeNode->SetStorageNodeID(storageNode->GetID());
    storageNode->Delete();
    }

  storageNode->SetFileName(filename);

  int res = storageNode->WriteData(volumeNode);

  
  return res;
}

//----------------------------------------------------------------------------
vtkMRMLScalarVolumeNode *vtkSlicerVolumesLogic::CreateLabelVolume (vtkMRMLScene *scene, vtkMRMLVolumeNode *volumeNode, char *name)
{
  if ( volumeNode == NULL ) 
    {
    return NULL;
    }

  // create a display node
  vtkMRMLVolumeDisplayNode *labelDisplayNode  = vtkMRMLVolumeDisplayNode::New();

  scene->AddNode(labelDisplayNode);

  // create a volume node as copy of source volume
  vtkMRMLScalarVolumeNode *labelNode = vtkMRMLScalarVolumeNode::New();
  labelNode->Copy(volumeNode);
  labelNode->SetStorageNodeID(NULL);
  labelNode->SetModifiedSinceRead(1);
  labelNode->SetLabelMap(1);

  // set the display node to have a label map lookup table
  labelDisplayNode->SetAndObserveColorNodeID ("vtkMRMLColorTableNodeLabels");
  labelNode->SetName(name);
  labelNode->SetAndObserveDisplayNodeID( labelDisplayNode->GetID() );

  // make an image data of the same size and shape as the input volume,
  // but filled with zeros
  vtkImageThreshold *thresh = vtkImageThreshold::New();
  thresh->ReplaceInOn();
  thresh->ReplaceOutOn();
  thresh->SetInValue(0);
  thresh->SetOutValue(0);
  thresh->SetInput( volumeNode->GetImageData() );
  thresh->GetOutput()->Update();
  labelNode->SetAndObserveImageData( thresh->GetOutput() );
  thresh->Delete();

  // add the label volume to the scene
  scene->AddNode(labelNode);

  labelNode->Delete();
  labelDisplayNode->Delete();

  return (labelNode);
}


#endif

//----------------------------------------------------------------------------
void vtkSlicerVolumesLogic::PrintSelf(ostream& os, vtkIndent indent)
{
  this->vtkObject::PrintSelf(os, indent);

  os << indent << "vtkSlicerVolumesLogic:             " << this->GetClassName() << "\n";

  os << indent << "ActiveVolumeNode: " <<
    (this->ActiveVolumeNode ? this->ActiveVolumeNode->GetName() : "(none)") << "\n";
}

