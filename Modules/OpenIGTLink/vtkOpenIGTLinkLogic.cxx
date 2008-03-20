/*==========================================================================

Portions (c) Copyright 2008 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $HeadURL: $
Date:      $Date: $
Version:   $Revision: $

==========================================================================*/

#include "vtkObjectFactory.h"
#include "vtkCallbackCommand.h"

#include "vtkOpenIGTLinkLogic.h"

#include "vtkMRMLModelDisplayNode.h"
#include "vtkMRMLScalarVolumeNode.h"
#include "vtkMRMLLinearTransformNode.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerApplicationGUI.h"
#include "vtkSlicerColorLogic.h"

#include "vtkMultiThreader.h"

#include "vtkIGTLConnector.h"


vtkCxxRevisionMacro(vtkOpenIGTLinkLogic, "$Revision: 1.9.12.1 $");
vtkStandardNewMacro(vtkOpenIGTLinkLogic);

//---------------------------------------------------------------------------
vtkOpenIGTLinkLogic::vtkOpenIGTLinkLogic()
{

    this->SliceDriver0 = vtkOpenIGTLinkLogic::SLICE_DRIVER_USER;
    this->SliceDriver1 = vtkOpenIGTLinkLogic::SLICE_DRIVER_USER;
    this->SliceDriver2 = vtkOpenIGTLinkLogic::SLICE_DRIVER_USER;


    // If the following code doesn't work, slice nodes should be obtained from application GUI
    this->SliceNode0 = NULL;
    this->SliceNode1 = NULL;
    this->SliceNode2 = NULL;

    /*
    this->SliceNode0 = this->GetApplication()->GetApplicationGUI()->GetMainSliceLogic0()->GetSliceNode();
    this->SliceNode1 = this->GetApplication()->GetApplicationGUI()->GetMainSliceLogic1()->GetSliceNode();
    this->SliceNode2 = this->GetApplication()->GetApplicationGUI()->GetMainSliceLogic2()->GetSliceNode();
    */
    /*
    this->Logic0 = appGUI->GetMainSliceGUI0()->GetLogic();
    this->Logic1 = appGUI->GetMainSliceGUI1()->GetLogic();
    this->Logic2 = appGUI->GetMainSliceGUI2()->GetLogic();
    */
                                         
    this->NeedRealtimeImageUpdate0 = 0;
    this->NeedRealtimeImageUpdate1 = 0;
    this->NeedRealtimeImageUpdate2 = 0;

    this->ImagingControl = 0;

#ifdef USE_NAVITRACK
    this->OpenTrackerStream   = vtkOpenIGTLinkDataStream::New();
    this->RealtimeVolumeNode = NULL;
#endif

    // Timer Handling

    this->DataCallbackCommand = vtkCallbackCommand::New();
    this->DataCallbackCommand->SetClientData( reinterpret_cast<void *> (this) );
    this->DataCallbackCommand->SetCallback(vtkOpenIGTLinkLogic::DataCallback);

#ifdef USE_NAVITRACK
    this->OpenTrackerStream->AddObserver(vtkCommand::ModifiedEvent, this->DataCallbackCommand);
#endif 

    this->ConnectorList.clear();
    this->ConnectorPrevStateList.clear();

}


//---------------------------------------------------------------------------
vtkOpenIGTLinkLogic::~vtkOpenIGTLinkLogic()
{

    if (this->DataCallbackCommand)
    {
      this->DataCallbackCommand->Delete();
    }

#ifdef USE_NAVITRACK
    if (this->OpenTrackerStream)
    {
      this->OpenTrackerStream->RemoveObservers( vtkCommand::ModifiedEvent, this->DataCallbackCommand );
      this->OpenTrackerStream->Delete();
    }
#endif

}


//---------------------------------------------------------------------------
void vtkOpenIGTLinkLogic::PrintSelf(ostream& os, vtkIndent indent)
{
    this->vtkObject::PrintSelf(os, indent);

    os << indent << "vtkOpenIGTLinkLogic:             " << this->GetClassName() << "\n";

}


//---------------------------------------------------------------------------
void vtkOpenIGTLinkLogic::DataCallback(vtkObject *caller, 
                                       unsigned long eid, void *clientData, void *callData)
{
    vtkOpenIGTLinkLogic *self = reinterpret_cast<vtkOpenIGTLinkLogic *>(clientData);
    vtkDebugWithObjectMacro(self, "In vtkOpenIGTLinkLogic DataCallback");
    self->UpdateAll();
}


//---------------------------------------------------------------------------
void vtkOpenIGTLinkLogic::UpdateAll()
{

}


//---------------------------------------------------------------------------
int vtkOpenIGTLinkLogic::CheckConnectorsStatusUpdates()
{

  //----------------------------------------------------------------
  // Find state change in the connectors

  int nCon = GetNumberOfConnectors();
  int updated = 0;

  for (int i = 0; i < nCon; i ++)
    {
      if (this->ConnectorPrevStateList[i] != this->ConnectorList[i]->GetState())
        {
          updated = 1;
          this->ConnectorPrevStateList[i] = this->ConnectorList[i]->GetState();
        }
    }

  return updated;

}



//---------------------------------------------------------------------------
void vtkOpenIGTLinkLogic::AddConnector()
{
  vtkIGTLConnector* connector = vtkIGTLConnector::New();
  connector->SetName("connector");
  this->ConnectorList.push_back(connector);
  this->ConnectorPrevStateList.push_back(-1);

  std::cerr << "Number of Connectors: " << this->ConnectorList.size() << std::endl;
}

//---------------------------------------------------------------------------
void vtkOpenIGTLinkLogic::DeleteConnector(int id)
{
  if (id >= 0 && id < this->ConnectorList.size())
    {
      this->ConnectorList[id]->Delete();
      this->ConnectorList.erase(this->ConnectorList.begin() + id);
      this->ConnectorPrevStateList.erase(this->ConnectorPrevStateList.begin() + id);
    }
}

//---------------------------------------------------------------------------
int vtkOpenIGTLinkLogic::GetNumberOfConnectors()
{
  return this->ConnectorList.size();
}

vtkIGTLConnector* vtkOpenIGTLinkLogic::GetConnector(int id)
{
  if (id >= 0 && id < GetNumberOfConnectors())
    {
      return this->ConnectorList[id];
    }
  else
    {
      return NULL;
    }
}


//---------------------------------------------------------------------------
vtkMRMLVolumeNode* vtkOpenIGTLinkLogic::AddVolumeNode(const char* volumeNodeName)
{

    std::cerr << "AddVolumeNode(): called." << std::endl;

    vtkMRMLVolumeNode *volumeNode = NULL;

    if (volumeNode == NULL)  // if real-time volume node has not been created
    {

        //vtkMRMLVolumeDisplayNode *displayNode = NULL;
        vtkMRMLScalarVolumeDisplayNode *displayNode = NULL;
        vtkMRMLScalarVolumeNode *scalarNode = vtkMRMLScalarVolumeNode::New();
        vtkImageData* image = vtkImageData::New();

        float fov = 300.0;
        image->SetDimensions(256, 256, 1);
        image->SetExtent(0, 255, 0, 255, 0, 0 );
        image->SetSpacing( fov/256, fov/256, 10 );
        image->SetOrigin( -fov/2, -fov/2, -0.0 );
        image->SetScalarTypeToShort();
        image->AllocateScalars();
        
        short* dest = (short*) image->GetScalarPointer();
        if (dest)
        {
          memset(dest, 0x00, 256*256*sizeof(short));
          image->Update();
        }
        
        /*
        vtkSlicerSliceLayerLogic *reslice = vtkSlicerSliceLayerLogic::New();
        reslice->SetUseReslice(0);
        */
        scalarNode->SetAndObserveImageData(image);

        
        /* Based on the code in vtkSlicerVolumeLogic::AddHeaderVolume() */
        //displayNode = vtkMRMLVolumeDisplayNode::New();
        displayNode = vtkMRMLScalarVolumeDisplayNode::New();
        scalarNode->SetLabelMap(0);
        volumeNode = scalarNode;
        
        if (volumeNode != NULL)
        {
            volumeNode->SetName(volumeNodeName);
            this->GetMRMLScene()->SaveStateForUndo();
            
            vtkDebugMacro("Setting scene info");
            volumeNode->SetScene(this->GetMRMLScene());
            displayNode->SetScene(this->GetMRMLScene());
            
            
            double range[2];
            vtkDebugMacro("Set basic display info");
            volumeNode->GetImageData()->GetScalarRange(range);
            range[0] = 0.0;
            range[1] = 256.0;
            displayNode->SetLowerThreshold(range[0]);
            displayNode->SetUpperThreshold(range[1]);
            displayNode->SetWindow(range[1] - range[0]);
            displayNode->SetLevel(0.5 * (range[1] - range[0]) );
            
            vtkDebugMacro("Adding node..");
            this->GetMRMLScene()->AddNode(displayNode);
            
            //displayNode->SetDefaultColorMap();
            vtkSlicerColorLogic *colorLogic = vtkSlicerColorLogic::New();
            displayNode->SetAndObserveColorNodeID(colorLogic->GetDefaultVolumeColorNodeID());
            //colorLogic->Delete();
            
            volumeNode->SetAndObserveDisplayNodeID(displayNode->GetID());
            
            vtkDebugMacro("Name vol node "<<volumeNode->GetClassName());
            vtkDebugMacro("Display node "<<displayNode->GetClassName());
            
            this->GetMRMLScene()->AddNode(volumeNode);
            vtkDebugMacro("Node added to scene");
        }

        //scalarNode->Delete();
        /*
        if (displayNode)
        {
            displayNode->Delete();
        }
        */

    }
    return volumeNode;
}

