
#include "vtkIGTDataManager.h"
#include "vtkIGTMatrixState.h"
#include "vtkObjectFactory.h"
#include "vtkMRMLModelNode.h"
#include "vtkMRMLModelDisplayNode.h"
#include "vtkCylinderSource.h"


void vtkIGTDataManager::RegisterStreamDevice (int streamType, vtkIGTDataStream* datastream)
{
    // streamType: 0 - matrix; 1 - image 
    //vtkIGTImageState *p_image;
    this->StreamTypes.push_back(streamType);
    this->RegisteredDataStreams.push_back(datastream);
    CreateMRMLNode(streamType);
}





void vtkIGTDataManager::CreateMRMLNode(int streamType)
{
    switch (streamType) {
        case IGT_MATRIX_STREAM:
            {

            vtkMRMLModelNode *modelNode = vtkMRMLModelNode::New();
            vtkMRMLModelDisplayNode *dispNode = vtkMRMLModelDisplayNode::New();

            this->MRMLScene->SaveStateForUndo();
            this->MRMLScene->AddNode(dispNode);
            this->MRMLScene->AddNode(modelNode);  

            dispNode->SetScene(this->MRMLScene);

            int size = this->MRMLIds.size();
            char name[20];
            sprintf(name, "matrix_%d", size);

            modelNode->SetName(name);
            modelNode->SetScene(this->MRMLScene);
            modelNode->SetAndObserveDisplayNodeID(dispNode->GetID());  
            this->MRMLIds.push_back(modelNode->GetID());

            vtkCylinderSource *cylinder = vtkCylinderSource::New();
            cylinder->SetRadius(1.5);
            cylinder->SetHeight(100);
            modelNode->SetAndObservePolyData(cylinder->GetOutput());
            this->Modified();  

            // modelNode->Delete();
            cylinder->Delete();
            // displayNode->Delete();
            }
            break;

        case IGT_IMAGE_STREAM:
            break;
        default:
            break;
    }

}



char *vtkIGTDataManager::GetMRMLId(int index) 
{
    return this->MRMLIds.at(index);
}


/*
void vtkIGTDataManager::UpdateMatrixData(int index, vtkIGTMatrixState state)
{
    vtkIGTDataStream  *stream = this->RegisteredDataStreams.at(index);
    // stream->SetMatrixState(state);

}
*/


