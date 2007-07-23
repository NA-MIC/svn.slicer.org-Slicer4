
#include "vtkIGTDataManager.h"
#include "vtkObjectFactory.h"
#include "vtkMRMLModelNode.h"
#include "vtkMRMLModelDisplayNode.h"
#include "vtkMRMLLinearTransformNode.h"
#include "vtkCylinderSource.h"
#include "vtkSphereSource.h"
#include "vtkAppendPolyData.h"


#include <string>


vtkStandardNewMacro(vtkIGTDataManager);
vtkCxxRevisionMacro(vtkIGTDataManager, "$Revision: 1.0 $");

int vtkIGTDataManager::index = 0;

vtkIGTDataManager::vtkIGTDataManager()
{
    this->MRMLScene = NULL;
}


vtkIGTDataManager::~vtkIGTDataManager()
{

}


const char *vtkIGTDataManager::RegisterStream(int streamType)
{
    std::string id("");

    // streamType: 0 - matrix; 1 - image 
    switch (streamType) {
        case IGT_MATRIX_STREAM:
            {

            vtkMRMLModelNode *modelNode = vtkMRMLModelNode::New();
            vtkMRMLModelDisplayNode *dispNode = vtkMRMLModelDisplayNode::New();
            vtkMRMLLinearTransformNode *transform = vtkMRMLLinearTransformNode::New();
            dispNode->SetVisibility(0);

            this->MRMLScene->SaveStateForUndo();
            this->MRMLScene->AddNode(dispNode);
            this->MRMLScene->AddNode(transform);
            this->MRMLScene->AddNode(modelNode);  

            dispNode->SetScene(this->MRMLScene);

            char name[20];
            sprintf(name, "igt_matrix_%d", index);

            modelNode->SetName(name);
            modelNode->SetHideFromEditors(1);
            modelNode->SetScene(this->MRMLScene);
            modelNode->SetAndObserveDisplayNodeID(dispNode->GetID());  
            modelNode->SetAndObserveTransformNodeID(transform->GetID());  
            id = std::string(modelNode->GetID());

            // Cylinder represents the locator stick
            vtkCylinderSource *cylinder = vtkCylinderSource::New();
            cylinder->SetRadius(1.5);
            cylinder->SetHeight(100);
            cylinder->Update();
            // Sphere represents the locator tip 
            vtkSphereSource *sphere = vtkSphereSource::New();
            sphere->SetRadius(3.0);
            sphere->SetCenter(0, -50, 0);
            sphere->Update();

            vtkAppendPolyData *apd = vtkAppendPolyData::New();
            apd->AddInput(sphere->GetOutput());
            apd->AddInput(cylinder->GetOutput());
            apd->Update();

            modelNode->SetAndObservePolyData(apd->GetOutput());
            this->Modified();  
            this->MRMLScene->Modified();

            modelNode->Delete();
            cylinder->Delete();
            sphere->Delete();
            apd->Delete();
            dispNode->Delete();
            transform->Delete();
            }
            break;

        case IGT_IMAGE_STREAM:
            break;
        default:
            break;
    }

    return id.c_str();
}



//Philip Mewes: 07/20/2007 This Locator has an extention to show were the needle is going to advance

const char *vtkIGTDataManager::RegisterStream_new(int streamType)
{
    std::string id_extend("");

    // streamType: 0 - matrix; 1 - image 
    switch (streamType) {
        case IGT_MATRIX_STREAM:
            {

            vtkMRMLModelNode *modelNode_extend = vtkMRMLModelNode::New();
            vtkMRMLModelDisplayNode *dispNode_extend = vtkMRMLModelDisplayNode::New();
            vtkMRMLLinearTransformNode *transform_extend = vtkMRMLLinearTransformNode::New();
            dispNode_extend->SetVisibility(0);

            this->MRMLScene->SaveStateForUndo();
            this->MRMLScene->AddNode(dispNode_extend);
            this->MRMLScene->AddNode(transform_extend);
            this->MRMLScene->AddNode(modelNode_extend);  

            dispNode_extend->SetScene(this->MRMLScene);

            char name[20];
            sprintf(name, "igt_matrix_%d", index);

            modelNode_extend->SetName(name);
            modelNode_extend->SetHideFromEditors(1);
            modelNode_extend->SetScene(this->MRMLScene);
            modelNode_extend->SetAndObserveDisplayNodeID(dispNode_extend->GetID());  
            modelNode_extend->SetAndObserveTransformNodeID(transform_extend->GetID());  
            id_extend = std::string(modelNode_extend->GetID());

            // Cylinder represents the locator stick
            vtkCylinderSource *cylinder_new = vtkCylinderSource::New();
            cylinder_new->SetRadius(10.5);
            cylinder_new->SetHeight(100);
            cylinder_new->Update();    
            // Sphere represents the locator tip 
            vtkSphereSource *sphere_new = vtkSphereSource::New();
            sphere_new->SetRadius(50.0);
            sphere_new->SetCenter(0, -50, 0);
            sphere_new->Update();

            vtkAppendPolyData *apd_extend = vtkAppendPolyData::New();
            apd_extend->AddInput(sphere_new->GetOutput());
            apd_extend->AddInput(cylinder_new->GetOutput());
            apd_extend->Update();

            modelNode_extend->SetAndObservePolyData(apd_extend->GetOutput());
            this->Modified();  
            this->MRMLScene->Modified();

            modelNode_extend->Delete();
            cylinder_new->Delete();
            sphere_new->Delete();
            apd_extend->Delete();
            dispNode_extend->Delete();
            transform_extend->Delete();
            }
            break;

        case IGT_IMAGE_STREAM:
            break;
        default:
            break;
    }

    return id_extend.c_str();
}






void vtkIGTDataManager::PrintSelf(ostream& os, vtkIndent indent)
{
}

